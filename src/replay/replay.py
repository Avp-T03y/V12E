"""Demo replay: send historical TEST orders (2014-2015) to the scoring API, one at a time, in order-date order.

This is a replay of historical data, not a live feed. Each order goes in with only the fields a new
order has when it is placed. Labels (late, days late) and precomputed features are never sent: the API
computes planned lead time and the past-performance features itself from the delivery history, using
only deliveries completed before the order date ("as_of").

Run inside Docker Compose (Ethan's setup: data mounted at /data, API service called "api"):
  docker compose --profile demo up replay
Run on a laptop against a local API:
  python src/replay/replay.py --input data/test_v3.csv --api http://localhost:8000 --interval 1 --limit 40

Settings (flag or environment variable):
  --input     REPLAY_INPUT     default $DATA_DIR/test_v3.csv, else data/test_v3.csv
  --api       API_URL          default http://api:8000 inside Docker, else http://localhost:8000
  --interval  REPLAY_INTERVAL  seconds between orders (default 2; raise it if Granite is slow)
  --limit     REPLAY_LIMIT     stop after N orders (0 = all)
  --start     REPLAY_START     first order date, YYYY-MM-DD
  --loop      REPLAY_LOOP=1    start again from the beginning when finished
  V12E_API_KEY                 sent as X-API-Key when the API requires it
"""
import argparse
import os
import signal
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# Only what a real new order contains.
SEND = ["order_id", "decision_date", "scheduled_delivery_date", "quantity",
        "country", "vendor", "fulfillment_route", "shipment_mode", "product_group"]
REQUIRED = ["order_id", "decision_date", "scheduled_delivery_date", "quantity",
            "country", "vendor", "fulfillment_route"]
# Columns that must never leave this script: outcomes and anything computed from them.
NEVER_SEND_PREFIXES = ("label_", "days_late", "prior_", "planned_lead_time")


def default_input():
    data_dir = os.environ.get("DATA_DIR")
    if data_dir and (Path(data_dir) / "test_v3.csv").exists():
        return str(Path(data_dir) / "test_v3.csv")
    return "data/test_v3.csv"


def default_api():
    return "http://api:8000" if Path("/.dockerenv").exists() else "http://localhost:8000"


def wait_for(api, headers, timeout=180):
    """Block until /health answers with a loaded model."""
    t0, last = time.time(), ""
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{api}/health", headers=headers, timeout=5)
            if r.ok and r.json().get("status") == "ok":
                h = r.json()
                print(f"API ready: model {h.get('model_version')} ({h.get('role')}), backend {h.get('backend')}, "
                      f"machine {h.get('machine')}, granite {h.get('granite')}", flush=True)
                return h
            last = f"HTTP {r.status_code}: {r.text[:120]}"
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep(3)
    sys.exit(f"API at {api} not ready after {timeout}s (last answer: {last})")


def load_orders(path, start="", limit=0):
    df = pd.read_csv(path, dtype={"order_id": str})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"{path} is missing columns: {missing}")
    for c in SEND:                                   # optional columns may be absent
        if c not in df.columns:
            df[c] = None
    df = df[df.decision_date.notna()].copy()
    df["decision_date"] = pd.to_datetime(df.decision_date).dt.strftime("%Y-%m-%d")
    df["scheduled_delivery_date"] = pd.to_datetime(df.scheduled_delivery_date).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["decision_date", "order_id"], kind="stable")
    if start:
        df = df[df.decision_date >= start]
    if limit:
        df = df.head(limit)
    return df[SEND]


def to_body(row):
    body = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    assert not any(k.startswith(NEVER_SEND_PREFIXES) for k in body), "label or feature column in request"
    body["order_id"] = str(body["order_id"])
    body["quantity"] = float(body["quantity"])
    body["as_of"] = body["decision_date"]            # replay clock = the day the order was placed
    return body


def post_score(api, body, headers, retries=3):
    """POST /score; retries on connection errors and 5xx, gives up on 4xx (bad order)."""
    for attempt in range(retries):
        try:
            r = requests.post(f"{api}/score", json=body, headers=headers, timeout=90)
            if r.ok:
                return r.json(), None
            if r.status_code < 500:
                return None, f"HTTP {r.status_code} {r.text[:160]}"
            err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            err = type(e).__name__
        time.sleep(2 * (attempt + 1))
    return None, err


def main():
    ap = argparse.ArgumentParser(description="Replay historical test orders into the scoring API.")
    ap.add_argument("--input", default=os.environ.get("REPLAY_INPUT") or default_input())
    ap.add_argument("--api", default=os.environ.get("API_URL") or default_api())
    ap.add_argument("--interval", type=float, default=float(os.environ.get("REPLAY_INTERVAL", "2")))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("REPLAY_LIMIT", "0")))
    ap.add_argument("--start", default=os.environ.get("REPLAY_START", ""))
    ap.add_argument("--loop", action="store_true", default=os.environ.get("REPLAY_LOOP") == "1")
    args = ap.parse_args()
    api = args.api.rstrip("/")
    headers = {"X-API-Key": os.environ["V12E_API_KEY"]} if os.environ.get("V12E_API_KEY") else {}

    orders = load_orders(args.input, args.start, args.limit)
    if orders.empty:
        sys.exit("no orders to replay (check --input and --start)")
    health = wait_for(api, headers)
    if str(health.get("role", "")).startswith("production"):
        print("WARNING: the API loaded the production model (trained through 2015), so the 2014-2015 replay "
              "is in-sample. Set MODEL_FILE=v12e_eval_model.pkl for an honest demo.", flush=True)

    stats = {"sent": 0, "failed": 0, "bands": {}}

    def summary(*_):
        b = ", ".join(f"{k} {v}" for k, v in sorted(stats["bands"].items()))
        print(f"\nreplay stopped: {stats['sent']} scored ({b or 'none'}), {stats['failed']} failed", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGINT, summary)
    signal.signal(signal.SIGTERM, summary)

    rounds = 0
    while True:
        rounds += 1
        print(f"replaying {len(orders)} historical orders, {orders.decision_date.iloc[0]} to "
              f"{orders.decision_date.iloc[-1]}, one every {args.interval}s (pass {rounds})", flush=True)
        for row in orders.to_dict("records"):
            body = to_body(row)
            res, err = post_score(api, body, headers)
            if res is None:
                stats["failed"] += 1
                print(f"{body['decision_date']}  order {body['order_id']:>8}  FAILED  {err}", flush=True)
            else:
                stats["sent"] += 1
                band = res.get("risk_band", "?")
                stats["bands"][band] = stats["bands"].get(band, 0) + 1
                rank = res.get("risk_percentile")
                print(f"{body['decision_date']}  order {body['order_id']:>8}  {band:<8}  "
                      f"score {res.get('risk_score', 0):.3f}  "
                      f"{'rank P' + format(rank, '.0f') + '  ' if rank is not None else ''}"
                      f"runway {res.get('runway_days')}d", flush=True)
            time.sleep(args.interval)
        if not args.loop:
            break
    summary()


if __name__ == "__main__":
    main()
