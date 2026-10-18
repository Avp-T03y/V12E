"""V12 Engines — baseline models for the delivery-delay early-warning engine.

Implements the baselines in Playbook v2 §3 on the v2 feature tables:

  Late / on-time (classifier baselines)
    1. no_skill        every shipment gets the training late rate
    2. route_lookup    score = prior_late_rate_fulfillment_route (what programs do today)
    3. logistic        class-weighted logistic regression on the 16 features

  Days late (regressor baselines, evaluated on late shipments only)
    4. global_median   median days late among late training shipments
    5. route_median    same, per fulfillment route
    6. linear_late     linear regression on log1p(days_late), late training rows only

Protocol (no peeking at test):
  - fit on training rows scheduled <= 2012, tune on the 2013 validation slice
  - operating threshold = score at the 90th percentile of validation scores
    (Playbook: surface the top ~10% by risk)
  - refit on all training rows (<= 2013), evaluate ONCE on test (2014-2015)

Usage:
  python baselines.py --data /srv/v12e/data --out /srv/v12e/models/baselines
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
FEATURES = CATEGORICAL + NUMERIC + FLAGS          # the 16 Playbook features
VALIDATION_YEAR = 2013
ALERT_SHARE = 0.10                                # top ~10% by risk
MIN_CATEGORY_COUNT = 20                           # rarer values -> "infrequent" bucket


# ---------------------------------------------------------------- data
def load(data_dir: Path):
    train = pd.read_csv(data_dir / "train_v2.csv", parse_dates=["scheduled_delivery_date"])
    test = pd.read_csv(data_dir / "test_v2.csv", parse_dates=["scheduled_delivery_date"])
    for df in (train, test):
        df[CATEGORICAL] = df[CATEGORICAL].astype(str)
    year = train["scheduled_delivery_date"].dt.year
    fit, val = train[year < VALIDATION_YEAR], train[year == VALIDATION_YEAR]
    return train, fit, val, test


# ---------------------------------------------------------------- metrics
def classification_report(y, score, threshold):
    """Playbook /metrics set for one model on one split."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    out = {"pr_auc": round(float(average_precision_score(y, score)), 4),
           "late_rate": round(float(y.mean()), 4)}

    # recall among the top 10% highest-risk shipments of this split
    # (undefined for a constant score: any "top 10%" would just be an arbitrary tie-break)
    if np.ptp(score) == 0:
        out["recall_top10pct"] = None
    else:
        k = max(1, int(round(ALERT_SHARE * len(y))))
        top = np.argsort(-score, kind="stable")[:k]
        out["recall_top10pct"] = round(float(y[top].sum() / max(1, y.sum())), 4)

    if threshold is None:                     # constant scores (no-skill): no ranking to threshold
        return out
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    out.update({
        "threshold": round(float(threshold), 4),
        "flagged_share": round(float(pred.mean()), 4),
        "recall_at_threshold": round(tp / max(1, tp + fn), 4),
        "precision_at_threshold": round(tp / max(1, tp + fp), 4),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "error_breakdown": {"correct": round((tp + tn) / n, 4),
                            "type1": round(fp / n, 4),     # false alarm
                            "type2": round(fn / n, 4)},    # missed late shipment (costliest)
    })
    return out


def alert_threshold(val_scores):
    """Score at the 90th percentile of validation scores (top 10% flagged)."""
    return float(np.quantile(val_scores, 1 - ALERT_SHARE))


# ---------------------------------------------------------------- classifier baselines
def no_skill(fit, val, test, train):
    rate = train["label_late"].mean()
    return {"validation": classification_report(val.label_late, np.full(len(val), rate), None),
            "test": classification_report(test.label_late, np.full(len(test), rate), None),
            "note": f"constant score {rate:.4f}; PR-AUC equals each split's late rate"}


def route_lookup(fit, val, test, train):
    col = "prior_late_rate_fulfillment_route"
    thr = alert_threshold(val[col])
    return {"validation": classification_report(val.label_late, val[col], thr),
            "test": classification_report(test.label_late, test[col], thr),
            "note": "score = leakage-safe prior late rate of the shipment's fulfillment route"}


def logistic_pipeline(flags=FLAGS):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),            # planned_lead_time NaN -> train median
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=MIN_CATEGORY_COUNT, sparse_output=False), CATEGORICAL),
        ("num", numeric, NUMERIC),
        ("flag", "passthrough", list(flags)),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))])


def logistic(fit, val, test, train, drop_flag=False):
    flags = [f for f in FLAGS if not (drop_flag and f == "decision_date_known")]
    feats = CATEGORICAL + NUMERIC + flags
    pipe = logistic_pipeline(flags)
    pipe.fit(fit[feats], fit.label_late)
    val_score = pipe.predict_proba(val[feats])[:, 1]
    thr = alert_threshold(val_score)                            # chosen on validation only

    final = logistic_pipeline(flags)
    final.fit(train[feats], train.label_late)                  # refit on all <= 2013
    test_score = final.predict_proba(test[feats])[:, 1]
    return {"validation": classification_report(val.label_late, val_score, thr),
            "test": classification_report(test.label_late, test_score, thr),
            "note": "class-weighted; one-hot (rare values grouped); log1p + median fill on numerics; "
                    "threshold from the <=2012 model's 2013 scores, applied to the refit model"}


# ---------------------------------------------------------------- days-late baselines
def days_late_baselines(train, test):
    tr = train[train.label_late == 1]
    te = test[test.label_late == 1]
    out = {"n_late_train": int(len(tr)), "n_late_test": int(len(te))}

    g = tr.days_late.median()
    out["global_median"] = {"mae_days": round(mean_absolute_error(te.days_late, np.full(len(te), g)), 2),
                            "prediction": float(g)}

    by_route = tr.groupby("fulfillment_route").days_late.median()
    pred = te.fulfillment_route.map(by_route).fillna(g)
    out["route_median"] = {"mae_days": round(mean_absolute_error(te.days_late, pred), 2),
                           "prediction_by_route": {k: float(v) for k, v in by_route.items()}}

    pre = logistic_pipeline().named_steps["pre"]
    lin = Pipeline([("pre", pre), ("reg", LinearRegression())])
    lin.fit(tr[FEATURES], np.log1p(tr.days_late))
    pred = np.clip(np.expm1(lin.predict(te[FEATURES])), 0, None)
    out["linear_late"] = {"mae_days": round(mean_absolute_error(te.days_late, pred), 2)}
    out["note"] = "evaluated on late test shipments only: 'how late, if late'"
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/srv/v12e/data", type=Path)
    ap.add_argument("--out", default="/srv/v12e/models/baselines", type=Path)
    args = ap.parse_args()

    train, fit, val, test = load(args.data)
    report = {
        "split": "fit <=2012 / validate 2013 / refit <=2013 / test 2014-2015",
        "rows": {"fit": len(fit), "validation": len(val), "train": len(train), "test": len(test)},
        "late": {"fit": int(fit.label_late.sum()), "validation": int(val.label_late.sum()),
                 "train": int(train.label_late.sum()), "test": int(test.label_late.sum())},
        "classifier": {
            "no_skill": no_skill(fit, val, test, train),
            "route_lookup": route_lookup(fit, val, test, train),
            "logistic": logistic(fit, val, test, train),
            "logistic_without_decision_date_known": logistic(fit, val, test, train, drop_flag=True),
        },
        "days_late": days_late_baselines(train, test),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baseline_report.json").write_text(json.dumps(report, indent=2))

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "  n/a"
    print(f"{'model':<40}{'split':<12}{'PR-AUC':>8}{'flagged':>9}{'recall@thr':>12}{'prec@thr':>10}{'recall top10%':>15}")
    for name, r in report["classifier"].items():
        for split in ("validation", "test"):
            m = r[split]
            print(f"{name:<40}{split:<12}{fmt(m['pr_auc']):>8}{fmt(m.get('flagged_share')):>9}"
                  f"{fmt(m.get('recall_at_threshold')):>12}{fmt(m.get('precision_at_threshold')):>10}"
                  f"{fmt(m['recall_top10pct']):>15}")
    d = report["days_late"]
    print(f"\ndays-late MAE on {d['n_late_test']} late test shipments: "
          f"global median {d['global_median']['mae_days']} · route median {d['route_median']['mae_days']} · "
          f"linear {d['linear_late']['mae_days']}")
    print(f"\nreport written to {args.out / 'baseline_report.json'}")


if __name__ == "__main__":
    main()
