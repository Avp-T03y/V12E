"""Live 'past performance' features: prior_excess_* computed at scoring time.

The model needs, for each new order, how late its country / vendor / fulfillment route have
been recently compared with everyone. This module computes those values from a delivery
history with EXACTLY the preprocessing notebook's rules (V12E_preprocessing_template_v3, §6):

  * only shipments DELIVERED before the order's cutoff count (cutoff = decision date,
    capped at the scheduled date), within the last 365 days;
  * prior_late_rate_<key> = (late_in_key + M * global_rate) / (n_in_key + M), M = 20;
  * global_rate = late rate of all deliveries in the same window (0.115 when none);
  * the order itself is excluded if it appears in the history;
  * prior_known_<key> = 1 if the key had any delivery in the window;
  * prior_excess_<key> = prior_late_rate_<key> - global_rate.

tests/test_feature_parity.py checks that this reproduces the training tables exactly.
"""
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_DAYS = 365
SMOOTH_M = 20.0
FALLBACK_PRIOR = 0.115
KEYS = ["country", "vendor", "fulfillment_route"]


def load_scms(path):
    """Delivery history from the raw SCMS CSV (one row per shipment)."""
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    scheduled = pd.to_datetime(df["scheduled delivery date"], errors="coerce", format="%d-%b-%y")
    delivered = pd.to_datetime(df["delivered to client date"], errors="coerce", format="%d-%b-%y")
    return pd.DataFrame({
        "order_id": (df["id"] if "id" in df.columns else pd.Series(range(len(df)))).astype(str),
        "country": df["country"].astype(str),
        "vendor": df["vendor"].astype(str),
        "fulfillment_route": df["fulfill via"].astype(str),
        "delivered_date": delivered,
        "late": (delivered > scheduled).astype(int),
    }).dropna(subset=["delivered_date"])


class DeliveryHistory:
    """Completed deliveries; answers 'what did each lane look like before this cutoff?'."""

    def __init__(self, deliveries):
        d = deliveries[["order_id", "country", "vendor", "fulfillment_route", "delivered_date", "late"]].copy()
        d["order_id"] = d["order_id"].astype(str)
        d["delivered_date"] = pd.to_datetime(d["delivered_date"]).values.astype("datetime64[D]")
        self.d = d.reset_index(drop=True)
        self._index()

    @classmethod
    def from_scms(cls, path):
        return cls(load_scms(path))

    def _index(self):
        d = self.d
        o = np.argsort(d["delivered_date"].values, kind="stable")
        self.g_dates = d["delivered_date"].values[o]
        self.g_cum = np.concatenate([[0], np.cumsum(d["late"].values[o])])
        self.by_key = {}
        for key in KEYS:
            groups = {}
            for val, g in d.groupby(key):
                go = np.argsort(g["delivered_date"].values, kind="stable")
                groups[val] = (g["delivered_date"].values[go], np.concatenate([[0], np.cumsum(g["late"].values[go])]))
            self.by_key[key] = groups
        self.self_lookup = dict(zip(d["order_id"], zip(d["delivered_date"].values, d["late"].values)))

    def add(self, order_id, country, vendor, fulfillment_route, delivered_date, late):
        """Record a completed delivery (e.g. from POST /deliveries); rebuilds the index."""
        row = pd.DataFrame([{"order_id": str(order_id), "country": str(country), "vendor": str(vendor),
                             "fulfillment_route": str(fulfillment_route),
                             "delivered_date": pd.Timestamp(delivered_date), "late": int(late)}])
        row["delivered_date"] = row["delivered_date"].values.astype("datetime64[D]")
        self.d = pd.concat([self.d[self.d.order_id != str(order_id)], row], ignore_index=True)
        self._index()

    @staticmethod
    def _window(dates, cum, cutoff, w):
        hi = np.searchsorted(dates, cutoff, "left")
        lo = np.searchsorted(dates, cutoff - w, "left")
        return hi - lo, cum[hi] - cum[lo]

    def features(self, orders, cutoff_col=None):
        """prior_late_rate_*, prior_known_*, prior_global_rate, prior_excess_* for each order.

        orders needs order_id, country, vendor, fulfillment_route and a cutoff: `cutoff_col` if
        given, else decision_date capped at scheduled_delivery_date.
        """
        o = orders.reset_index(drop=True)
        if cutoff_col:
            cut = pd.to_datetime(o[cutoff_col])
        else:
            dd, sd = pd.to_datetime(o["decision_date"]), pd.to_datetime(o["scheduled_delivery_date"])
            cut = dd.where(dd <= sd, sd)
        cut = cut.values.astype("datetime64[D]")
        w = np.timedelta64(WINDOW_DAYS, "D")
        out = {f"prior_{k}_{key}": np.zeros(len(o)) for key in KEYS for k in ("late_rate", "excess")}
        out.update({f"prior_known_{key}": np.zeros(len(o), dtype=int) for key in KEYS})
        out["prior_global_rate"] = np.zeros(len(o))
        for i in range(len(o)):
            c = cut[i]
            me = self.self_lookup.get(str(o.at[i, "order_id"]))
            self_in = me is not None and (me[0] < c) and (me[0] >= c - w)
            gn, gs = self._window(self.g_dates, self.g_cum, c, w)
            if self_in:
                gn, gs = gn - 1, gs - me[1]
            gp = gs / gn if gn > 0 else FALLBACK_PRIOR
            out["prior_global_rate"][i] = gp
            for key in KEYS:
                grp = self.by_key[key].get(str(o.at[i, key]))
                n, s = self._window(*grp, c, w) if grp is not None else (0, 0)
                if self_in and grp is not None and self.d.loc[self.d.order_id == str(o.at[i, "order_id"]), key].iloc[0] == str(o.at[i, key]):
                    n, s = n - 1, s - me[1]
                rate = (s + SMOOTH_M * gp) / (n + SMOOTH_M)
                out[f"prior_late_rate_{key}"][i] = rate
                out[f"prior_known_{key}"][i] = int(n > 0)
                out[f"prior_excess_{key}"][i] = rate - gp
        return pd.DataFrame(out)


def planned_lead_time(orders):
    """scheduled - decision date in days; negative (data error) becomes missing, as in training."""
    lead = (pd.to_datetime(orders["scheduled_delivery_date"]) - pd.to_datetime(orders["decision_date"])).dt.days
    return lead.where(lead >= 0)
