"""V12 Engines — baseline models for the delivery-delay early-warning engine.

Implements the baselines in Playbook v2 §3 on the v2 feature tables, plus the variants
needed to interpret them before the Snap ML step.

  Late / on-time (classifier baselines)
    no_skill                      every shipment gets the training late rate
    route_lookup                  score = prior_late_rate_fulfillment_route (what programs do today)
    logistic_all_years            original Playbook baseline: 16 features, trained on 2006-2013
    logistic_all_years_no_ddk     same without decision_date_known
    logistic_2010plus             trained on 2010-2013 only (see "era" below), 15 features
    logistic_priors_only_2010plus trained on 2010-2013, numeric features + prior flags only
                                  (no one-hot country / vendor / site identities)
  v3 tables only (prior_excess_* = lane prior minus the overall late rate in the same window):
    route_excess_lookup           score = prior_excess_fulfillment_route
    logistic_relative_2010plus    as logistic_2010plus, raw priors replaced by prior_excess_*
    logistic_relative_priors_only_2010plus
                                  quantity, lead time, prior_excess_* and prior flags only

  Days late (regressor baselines, evaluated on late test shipments only)
    global_median, route_median, linear_late   each fitted on all years and on 2010+

Why the 2010+ variants
  2006-2009 is a different era: late rate 0-4% (vs 8-24% after) and the decision date is
  missing for most rows, so decision_date_known is ~0 there and 1 for every 2010+ and test
  row. A model trained on all years partly learns "old era = on time", which never applies
  to a live order.

Protocol (no peeking at test)
  - fit on training rows scheduled <= 2012, tune on the 2013 validation slice
  - refit on all training rows (<= 2013) of the same era, evaluate ONCE on test (2014-2015)

Metrics (all per split; read them against the split's late rate)
  pr_auc            area under precision-recall; a random score gets the late rate
  lift              pr_auc / late rate; 1.0 = no better than random
  recall_top10 /
  precision_top10   flag the top 10% of the whole split by score. Ties are split evenly
                    (expected value under random tie-breaking), so a coarse score is not
                    helped or hurt by row order. Random = recall 0.10, precision = late rate.
  recall_monthly10  flag the top 10% of each order month (decision-date month, else the
                    scheduled month), the way a live alert queue would run. Unaffected by
                    score drift across months.
  flagged@thr /     the fixed score threshold chosen on 2013 validation (top 10% there),
  recall@thr        applied unchanged. If flagged share moves far from 0.10 on test, the
                    score distribution drifted and a fixed cutoff will not hold alert volume.

Usage:
  python baselines.py --data . --out ./models/baselines              # v3 tables if present
  python baselines.py --data . --out ./models/baselines --tables v2  # original tables
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAL = ["country", "vendor", "fulfillment_route", "shipment_mode",
               "product_group", "managed_by", "manufacturing_site"]
NUMERIC = ["quantity", "planned_lead_time_days",
           "prior_late_rate_country", "prior_late_rate_vendor", "prior_late_rate_fulfillment_route"]
FLAGS = ["decision_date_known", "prior_known_country", "prior_known_vendor",
         "prior_known_fulfillment_route"]
PRIOR_FLAGS = [f for f in FLAGS if f != "decision_date_known"]
BASE_NUMERIC = ["quantity", "planned_lead_time_days"]
RELATIVE = ["prior_excess_country", "prior_excess_vendor", "prior_excess_fulfillment_route"]
FEATURES = CATEGORICAL + NUMERIC + FLAGS          # the 16 Playbook features
VALIDATION_YEAR = 2013
MODERN_ERA_START = 2010                           # decision dates recorded from 2010 on
ALERT_SHARE = 0.10                                # top ~10% by risk
MIN_CATEGORY_COUNT = 20                           # rarer values -> "infrequent" bucket


# ---------------------------------------------------------------- data
def load(data_dir: Path, tables: str):
    dates = ["scheduled_delivery_date", "decision_date"]
    train = pd.read_csv(data_dir / f"train_{tables}.csv", parse_dates=dates)
    test = pd.read_csv(data_dir / f"test_{tables}.csv", parse_dates=dates)
    for df in (train, test):
        df[CATEGORICAL] = df[CATEGORICAL].astype(str)
        df["year"] = df["scheduled_delivery_date"].dt.year
        # alert batch = month the order was placed; scheduled month when that is unknown
        anchor = df["decision_date"].fillna(df["scheduled_delivery_date"])
        df["alert_month"] = anchor.dt.to_period("M").astype(str)
    return train, test


def era(df, start_year):
    return df if start_year is None else df[df["year"] >= start_year]


def splits(train, start_year=None):
    """fit (<=2012), val (2013), refit (<=2013) for the given training era."""
    t = era(train, start_year)
    return t[t.year < VALIDATION_YEAR], t[t.year == VALIDATION_YEAR], t


# ---------------------------------------------------------------- metrics
def expected_top_k(y, score, k):
    """Expected (true positives, flagged) when flagging the k highest scores, ties split evenly."""
    if k <= 0 or len(y) == 0:
        return 0.0, 0
    k = min(k, len(y))
    order = np.argsort(-score, kind="stable")
    cut = score[order[k - 1]]
    above = score > cut
    tied = score == cut
    take_from_tied = k - above.sum()
    tp = y[above].sum() + take_from_tied * y[tied].mean()
    return float(tp), k


def classification_report(y, score, threshold, months=None):
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    pos = max(1.0, y.sum())
    late_rate = float(y.mean())
    pr_auc = float(average_precision_score(y, score))
    out = {"n": int(len(y)), "late": int(y.sum()), "late_rate": round(late_rate, 4),
           "pr_auc": round(pr_auc, 4), "lift": round(pr_auc / late_rate, 2) if late_rate else None}

    k = max(1, int(round(ALERT_SHARE * len(y))))
    tp, flagged = expected_top_k(y, score, k)
    out["recall_top10"] = round(tp / pos, 4)
    out["precision_top10"] = round(tp / flagged, 4)

    if months is not None:
        months = np.asarray(months)
        tp_m = flagged_m = 0.0
        for m in np.unique(months):
            idx = months == m
            km = max(1, int(round(ALERT_SHARE * idx.sum())))
            t, f = expected_top_k(y[idx], score[idx], km)
            tp_m += t
            flagged_m += f
        out["recall_monthly10"] = round(tp_m / pos, 4)
        out["precision_monthly10"] = round(tp_m / flagged_m, 4)

    if threshold is None:                     # constant score: nothing to threshold
        return out
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp_ = confusion_matrix(y.astype(int), pred, labels=[0, 1]).ravel()
    n = len(y)
    out.update({
        "threshold": round(float(threshold), 4),
        "flagged_at_threshold": round(float(pred.mean()), 4),
        "recall_at_threshold": round(tp_ / max(1, tp_ + fn), 4),
        "precision_at_threshold": round(tp_ / max(1, tp_ + fp), 4),
        "confusion": {"tp": int(tp_), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "error_breakdown": {"correct": round((tp_ + tn) / n, 4),
                            "type1": round(fp / n, 4),     # false alarm
                            "type2": round(fn / n, 4)},    # missed late shipment (costliest)
    })
    return out


def alert_threshold(val_scores):
    """Score at the 90th percentile of validation scores (top 10% flagged)."""
    return float(np.quantile(val_scores, 1 - ALERT_SHARE))


def report_pair(val, val_score, test, test_score, thr):
    return {"validation": classification_report(val.label_late, val_score, thr, val.alert_month),
            "test": classification_report(test.label_late, test_score, thr, test.alert_month)}


# ---------------------------------------------------------------- classifier baselines
def no_skill(train, test):
    fit, val, refit = splits(train)
    rate = refit["label_late"].mean()
    r = report_pair(val, np.full(len(val), rate), test, np.full(len(test), rate), None)
    r["note"] = f"constant score {rate:.4f}; PR-AUC equals each split's late rate"
    return r


def route_lookup(train, test):
    _, val, _ = splits(train)
    col = "prior_late_rate_fulfillment_route"
    r = report_pair(val, val[col].values, test, test[col].values, alert_threshold(val[col]))
    r["note"] = ("score = leakage-safe rolling 365-day late rate of the shipment's fulfillment route "
                 "(only 2 routes, so it mostly tracks the recent overall late rate)")
    return r


def route_excess_lookup(train, test):
    _, val, _ = splits(train)
    col = "prior_excess_fulfillment_route"
    r = report_pair(val, val[col].values, test, test[col].values, alert_threshold(val[col]))
    r["note"] = "score = route prior minus the overall late rate in the same 365-day window"
    return r


def _one_hot_encoder(**kwargs):
    """Compatibility helper for older/newer scikit-learn OneHotEncoder APIs."""
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)


def preprocessor(categorical, numeric, flags):
    signed = [c for c in numeric if c in RELATIVE]               # can be negative: no log
    numeric = [c for c in numeric if c not in RELATIVE]
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),            # planned_lead_time NaN -> train median
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    parts = []
    if categorical:
        parts.append(("cat", _one_hot_encoder(handle_unknown="infrequent_if_exist",
                                              min_frequency=MIN_CATEGORY_COUNT), list(categorical)))
    parts.append(("num", num, list(numeric)))
    if signed:
        parts.append(("rel", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), signed))
    if flags:
        parts.append(("flag", "passthrough", list(flags)))
    return ColumnTransformer(parts)


def logistic_pipeline(categorical, numeric, flags):
    return Pipeline([("pre", preprocessor(categorical, numeric, flags)),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0))])


def logistic(train, test, start_year=None, categorical=CATEGORICAL, numeric=NUMERIC, flags=FLAGS):
    feats = list(categorical) + list(numeric) + list(flags)
    fit, val, refit = splits(train, start_year)

    pipe = logistic_pipeline(categorical, numeric, flags).fit(fit[feats], fit.label_late)
    val_score = pipe.predict_proba(val[feats])[:, 1]
    thr = alert_threshold(val_score)                            # chosen on validation only

    final = logistic_pipeline(categorical, numeric, flags).fit(refit[feats], refit.label_late)
    test_score = final.predict_proba(test[feats])[:, 1]

    r = report_pair(val, val_score, test, test_score, thr)
    r["training_rows"] = {"fit": int(len(fit)), "refit": int(len(refit)),
                          "years": f"{start_year or int(train.year.min())}-{VALIDATION_YEAR}"}
    r["features"] = feats
    names = final.named_steps["pre"].get_feature_names_out()
    coefs = final.named_steps["clf"].coef_[0]
    r["coefficients"] = {n.split("__", 1)[1]: round(float(c), 3)
                         for n, c in zip(names, coefs) if not n.startswith("cat__")}
    r["note"] = ("coefficients: standardized numeric/flag features of the refit model "
                 "(positive = riskier); class-weighted; one-hot (rare values grouped); log1p + median fill on numerics; "
                 "threshold from the <=2012 model's 2013 scores, applied to the refit model")
    return r


# ---------------------------------------------------------------- diagnostics
def single_feature_diagnostics(train, test):
    """PR-AUC of each numeric feature used directly as a score (sign chosen on validation)."""
    _, val, _ = splits(train)
    out = {}
    for col in [c for c in NUMERIC + RELATIVE + ["prior_global_rate"] if c in val]:
        v = val[col].fillna(val[col].median()).values
        sign = 1.0 if average_precision_score(val.label_late, v) >= \
            average_precision_score(val.label_late, -v) else -1.0
        t = test[col].fillna(val[col].median()).values
        out[col] = {"direction": "higher = riskier" if sign > 0 else "lower = riskier",
                    "pr_auc_validation": round(float(average_precision_score(val.label_late, sign * v)), 4),
                    "pr_auc_test": round(float(average_precision_score(test.label_late, sign * t)), 4)}
    return out


def late_rate_by_year(train, test):
    both = pd.concat([train, test])
    g = both.groupby("year").label_late
    return {int(y): {"rows": int(n), "late_rate": round(float(r), 4)}
            for y, n, r in zip(g.size().index, g.size().values, g.mean().values)}


# ---------------------------------------------------------------- days-late baselines
def days_late_baselines(train, test, start_year=None):
    tr = era(train, start_year)
    tr = tr[tr.label_late == 1]
    te = test[test.label_late == 1]
    out = {"n_late_train": int(len(tr)), "n_late_test": int(len(te))}

    g = tr.days_late.median()
    out["global_median"] = {"mae_days": round(mean_absolute_error(te.days_late, np.full(len(te), g)), 2),
                            "prediction": float(g)}

    by_route = tr.groupby("fulfillment_route").days_late.median()
    pred = te.fulfillment_route.map(by_route).fillna(g)
    out["route_median"] = {"mae_days": round(mean_absolute_error(te.days_late, pred), 2),
                           "prediction_by_route": {k: float(v) for k, v in by_route.items()}}

    flags = PRIOR_FLAGS if start_year else FLAGS
    feats = CATEGORICAL + NUMERIC + flags
    lin = Pipeline([("pre", preprocessor(CATEGORICAL, NUMERIC, flags)), ("reg", LinearRegression())])
    lin.fit(tr[feats], np.log1p(tr.days_late))
    pred = np.clip(np.expm1(lin.predict(te[feats])), 0, None)
    out["linear_late"] = {"mae_days": round(mean_absolute_error(te.days_late, pred), 2)}
    out["test_days_late_median"] = float(te.days_late.median())
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", type=Path)
    ap.add_argument("--out", default="./models/baselines", type=Path)
    ap.add_argument("--tables", choices=["v2", "v3"], default=None,
                    help="feature tables to read (default: v3 if train_v3.csv exists, else v2)")
    args = ap.parse_args()
    tables = args.tables or ("v3" if (args.data / "train_v3.csv").exists() else "v2")

    train, test = load(args.data, tables)
    m = MODERN_ERA_START
    classifier = {
        "no_skill": no_skill(train, test),
        "route_lookup": route_lookup(train, test),
        "logistic_all_years": logistic(train, test),
        "logistic_all_years_no_ddk": logistic(train, test, flags=PRIOR_FLAGS),
        f"logistic_{m}plus": logistic(train, test, start_year=m, flags=PRIOR_FLAGS),
        f"logistic_priors_only_{m}plus": logistic(train, test, start_year=m, categorical=[],
                                                  flags=PRIOR_FLAGS),
    }
    if tables == "v3":
        classifier["route_excess_lookup"] = route_excess_lookup(train, test)
        classifier[f"logistic_relative_{m}plus"] = logistic(
            train, test, start_year=m, numeric=BASE_NUMERIC + RELATIVE, flags=PRIOR_FLAGS)
        classifier[f"logistic_relative_priors_only_{m}plus"] = logistic(
            train, test, start_year=m, categorical=[], numeric=BASE_NUMERIC + RELATIVE,
            flags=PRIOR_FLAGS)
    report = {
        "tables": tables,
        "split": "fit <=2012 / validate 2013 / refit <=2013 / test 2014-2015",
        "late_rate_by_year": late_rate_by_year(train, test),
        "classifier": classifier,
        "single_feature_diagnostics": single_feature_diagnostics(train, test),
        "days_late": {"all_years": days_late_baselines(train, test),
                      f"{m}plus": days_late_baselines(train, test, start_year=m)},
        "metric_notes": {
            "lift": "pr_auc / late_rate; 1.0 = random",
            "recall_top10": "top 10% of the split, ties split evenly; random = 0.10",
            "recall_monthly10": "top 10% within each order month; random = 0.10",
            "flagged_at_threshold": "share flagged by the fixed 2013 threshold; 0.10 = no drift",
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "baseline_report.json"
    path.write_text(json.dumps(report, indent=2))

    # ------------------------------------------------ console summary
    def fmt(v, w=8):
        return f"{v:>{w}.3f}" if isinstance(v, (int, float)) and v is not None else f"{'n/a':>{w}}"

    print(f"tables: {tables}")
    print("late rate by scheduled year:  " + "  ".join(
        f"{y}:{d['late_rate']:.3f}" for y, d in report["late_rate_by_year"].items()))
    print()
    cols = [("PR-AUC", "pr_auc"), ("lift", "lift"), ("rec@top10", "recall_top10"),
            ("prec@top10", "precision_top10"), ("rec@month10", "recall_monthly10"),
            ("flag@thr", "flagged_at_threshold"), ("rec@thr", "recall_at_threshold")]
    print(f"{'model':<42}{'split':<6}" + "".join(f"{c:>12}" for c, _ in cols))
    for name, r in classifier.items():
        for split, tag in (("validation", "val"), ("test", "test")):
            mm = r[split]
            print(f"{name:<42}{tag:<6}" + "".join(fmt(mm.get(k), 12) for _, k in cols))
    v, t = classifier["no_skill"]["validation"], classifier["no_skill"]["test"]
    print(f"\nrandom ranking: lift 1.00, rec@top10 / rec@month10 0.100, "
          f"prec@top10 = late rate (val {v['late_rate']:.3f}, test {t['late_rate']:.3f}); "
          f"flag@thr 0.100 means the 2013 threshold still flags 10%")

    print("\nsingle numeric feature as a score (PR-AUC val -> test):")
    for col, d in report["single_feature_diagnostics"].items():
        print(f"  {col:<36}{d['pr_auc_validation']:.3f} -> {d['pr_auc_test']:.3f}   ({d['direction']})")

    print("\nlogistic coefficients on prior features (refit model, standardized; positive = riskier):")
    for name, r in classifier.items():
        c = {k: v for k, v in r.get("coefficients", {}).items() if k.startswith("prior_")
             and not k.startswith("prior_known")}
        if c:
            print(f"  {name:<40}" + "  ".join(f"{k.replace('prior_', '')}={v:+.2f}" for k, v in c.items()))

    print("\ndays-late MAE on late test shipments (lower is better):")
    for tag, d in report["days_late"].items():
        print(f"  trained on {tag:<10} n_late_test={d['n_late_test']}: global median "
              f"{d['global_median']['mae_days']} · route median {d['route_median']['mae_days']} · "
              f"linear {d['linear_late']['mae_days']}")
    print(f"\nreport written to {path}")


if __name__ == "__main__":
    main()
