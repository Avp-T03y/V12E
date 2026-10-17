"""V12 Engines — boosted-tree prototype for the delivery-delay early-warning engine.

Snap ML ready: uses snapml.BoostingMachineClassifier / BoostingMachineRegressor when the
snapml package is importable (LinuxONE), otherwise scikit-learn's HistGradientBoosting
with the same feature set, folds and metrics (local development).

What it does
  1. Features (decided from the baseline runs, see baselines.py):
       quantity, planned_lead_time_days (+ missing flag),
       prior_excess_country / _vendor / _fulfillment_route  (lane prior minus overall rate),
       prior_known_country / _vendor / _fulfillment_route
     Not used: one-hot country/vendor/site identities (memorize 2010-13, fail on test),
     raw prior_late_rate_* (lag the year-level late rate and flip sign), decision_date_known.
  2. Training era 2010+ (2006-09 has a 0-4% late rate and mostly missing decision dates).
  3. Rolling-origin validation, no test rows involved:
       train 2010       -> validate 2011
       train 2010-2011  -> validate 2012
       train 2010-2012  -> validate 2013
     Each tree config and the logistic benchmark (logistic_relative_priors_only_2010plus)
     are scored on all three folds; the config with the best mean recall_monthly10 wins
     (mean lift breaks ties).
  4. Refit the chosen config on 2010-2013 and score the 2014-2015 test set ONCE.
  5. Alert rule: top 10% of each order month (no fixed score threshold; scores drift).
  6. Days late (late shipments only): boosted regressor on log1p(days_late) vs the global
     median, on the same folds and then once on test.

Outputs (in --out):
  boosted_report.json   folds, chosen config, test metrics, permutation importances
  test_alerts.csv       one row per test shipment: score, monthly rank, flagged, outcome
  model.pkl             the refit classifier (pickle; Snap ML models may not pickle)

Usage:
  python train_boosted.py --data . --out ./models/boosted
  python train_boosted.py --data . --out ./models/boosted --backend sklearn   # force local
"""
import argparse
import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error

from baselines import (BASE_NUMERIC, PRIOR_FLAGS, RELATIVE, classification_report,
                       expected_top_k, load, logistic_pipeline)

TREE_FEATURES = BASE_NUMERIC + ["lead_time_missing"] + RELATIVE + PRIOR_FLAGS
LOGIT_NUMERIC = BASE_NUMERIC + RELATIVE
ERA_START, VAL_YEARS, TEST_YEARS = 2010, [2011, 2012, 2013], [2014, 2015]
ALERT_SHARE = 0.10
SEED = 0

GRID = {"n_rounds": [100, 300], "learning_rate": [0.03, 0.1], "max_depth": [2, 3, 4]}


# ---------------------------------------------------------------- backend
def snapml_available():
    try:
        import snapml  # noqa: F401
        return True
    except Exception:
        return False


def make_model(backend, task, n_rounds, learning_rate, max_depth):
    if backend == "snapml":
        import snapml
        cls = snapml.BoostingMachineClassifier if task == "clf" else snapml.BoostingMachineRegressor
        kw = dict(num_round=n_rounds, learning_rate=learning_rate, max_depth=max_depth,
                  random_state=SEED, subsample=0.8, colsample_bytree=0.8, lambda_l2=1.0,
                  n_threads=-1, objective="logloss" if task == "clf" else "mse")
        return cls(**kw)
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    cls = HistGradientBoostingClassifier if task == "clf" else HistGradientBoostingRegressor
    return cls(max_iter=n_rounds, learning_rate=learning_rate, max_depth=max_depth,
               min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)


def fit(model, X, y, weight=None):
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if weight is None:
        return model.fit(X, y)
    try:
        return model.fit(X, y, sample_weight=np.asarray(weight, dtype=np.float32))
    except TypeError:                         # backend without sample weights
        return model.fit(X, y)


def predict_score(model, X):
    X = np.ascontiguousarray(X, dtype=np.float32)
    p = model.predict_proba(X)
    p = np.asarray(p)
    return p[:, 1] if p.ndim == 2 else p


def predict_value(model, X):
    return np.asarray(model.predict(np.ascontiguousarray(X, dtype=np.float32)), dtype=float)


# ---------------------------------------------------------------- features
def tree_matrix(df, lead_median):
    X = pd.DataFrame(index=df.index)
    X["quantity"] = np.log1p(df["quantity"].astype(float))
    lead = df["planned_lead_time_days"].astype(float)
    X["planned_lead_time_days"] = lead.fillna(lead_median)
    X["lead_time_missing"] = lead.isna().astype(float)
    for c in RELATIVE + PRIOR_FLAGS:
        X[c] = df[c].astype(float)
    return X[TREE_FEATURES].values


def balanced_weights(y):
    y = np.asarray(y)
    pos = max(1, y.sum())
    neg = max(1, len(y) - y.sum())
    return np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))


def monthly_flags(score, months):
    """1 for shipments in the top 10% of their order month (ties broken by order)."""
    s = pd.Series(score)
    flags = np.zeros(len(s), dtype=int)
    for _, idx in pd.Series(np.arange(len(s))).groupby(np.asarray(months)):
        k = max(1, int(round(ALERT_SHARE * len(idx))))
        top = s.iloc[idx.values].sort_values(ascending=False, kind="stable").index[:k]
        flags[top] = 1
    return flags


# ---------------------------------------------------------------- folds
def folds(train):
    t = train[train.year >= ERA_START]
    for vy in VAL_YEARS:
        yield vy, t[t.year < vy], t[t.year == vy]


def score_metrics(y, score, months):
    m = classification_report(y, score, None, months)
    return {k: m[k] for k in ("late_rate", "pr_auc", "lift", "recall_top10", "precision_top10",
                              "recall_monthly10", "precision_monthly10")}


def summarize(per_fold):
    keys = ["pr_auc", "lift", "recall_top10", "recall_monthly10", "precision_monthly10"]
    return {f"mean_{k}": round(float(np.mean([f[k] for f in per_fold.values()])), 4) for k in keys} | \
           {"min_recall_monthly10": round(float(min(f["recall_monthly10"] for f in per_fold.values())), 4)}


def cv_logistic(train):
    feats = LOGIT_NUMERIC + PRIOR_FLAGS
    out = {}
    for vy, tr, va in folds(train):
        pipe = logistic_pipeline([], LOGIT_NUMERIC, PRIOR_FLAGS).fit(tr[feats], tr.label_late)
        out[vy] = score_metrics(va.label_late, pipe.predict_proba(va[feats])[:, 1], va.alert_month)
    return out


def cv_tree(train, backend, params):
    out = {}
    for vy, tr, va in folds(train):
        med = tr.planned_lead_time_days.median()
        model = make_model(backend, "clf", **params)
        fit(model, tree_matrix(tr, med), tr.label_late, balanced_weights(tr.label_late))
        out[vy] = score_metrics(va.label_late, predict_score(model, tree_matrix(va, med)),
                                va.alert_month)
    return out


def permutation_importance(model, X, y, months, repeats=5):
    rng = np.random.default_rng(SEED)
    base = score_metrics(y, predict_score(model, X), months)
    out = {}
    for j, name in enumerate(TREE_FEATURES):
        drops_pr, drops_rec = [], []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            m = score_metrics(y, predict_score(model, Xp), months)
            drops_pr.append(base["pr_auc"] - m["pr_auc"])
            drops_rec.append(base["recall_monthly10"] - m["recall_monthly10"])
        out[name] = {"pr_auc_drop": round(float(np.mean(drops_pr)), 4),
                     "recall_monthly10_drop": round(float(np.mean(drops_rec)), 4)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["pr_auc_drop"]))


# ---------------------------------------------------------------- days late
def days_late(train, test, backend, params):
    def late(df):
        return df[df.label_late == 1]
    res = {"folds": {}}
    for vy, tr, va in folds(train):
        tr, va = late(tr), late(va)
        if len(tr) < 20 or len(va) == 0:
            continue
        med = tr.planned_lead_time_days.median()
        m = make_model(backend, "reg", **params)
        fit(m, tree_matrix(tr, med), np.log1p(tr.days_late))
        pred = np.clip(np.expm1(predict_value(m, tree_matrix(va, med))), 0, None)
        res["folds"][vy] = {"mae_boosted": round(mean_absolute_error(va.days_late, pred), 2),
                            "mae_global_median": round(mean_absolute_error(
                                va.days_late, np.full(len(va), tr.days_late.median())), 2)}
    era = late(train[train.year >= ERA_START])
    te = late(test)
    med = era.planned_lead_time_days.median()
    m = make_model(backend, "reg", **params)
    fit(m, tree_matrix(era, med), np.log1p(era.days_late))
    pred = np.clip(np.expm1(predict_value(m, tree_matrix(te, med))), 0, None)
    res["test"] = {"n_late": int(len(te)),
                   "mae_boosted": round(mean_absolute_error(te.days_late, pred), 2),
                   "mae_global_median": round(mean_absolute_error(
                       te.days_late, np.full(len(te), era.days_late.median())), 2)}
    res["params"] = params
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", type=Path)
    ap.add_argument("--out", default="./models/boosted", type=Path)
    ap.add_argument("--backend", choices=["auto", "snapml", "sklearn"], default="auto")
    args = ap.parse_args()
    backend = ("snapml" if snapml_available() else "sklearn") if args.backend == "auto" else args.backend
    if backend == "snapml" and not snapml_available():
        raise SystemExit("snapml is not installed: pip install snapml, or use --backend sklearn")

    train, test = load(args.data, "v3")
    t0 = time.time()
    print(f"backend: {backend} · features: {', '.join(TREE_FEATURES)}")
    print(f"rolling folds (validate on {VAL_YEARS}), training era {ERA_START}+, test untouched\n")

    # ------------------------------------------------ 1. model selection on rolling folds
    candidates = {"logistic_relative_priors_only (benchmark)": cv_logistic(train)}
    keys = list(GRID)
    configs = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    for p in configs:
        candidates[f"trees r={p['n_rounds']} lr={p['learning_rate']} d={p['max_depth']}"] = cv_tree(train, backend, p)
    summary = {name: summarize(f) for name, f in candidates.items()}

    print(f"{'candidate':<44}" + "".join(f"{'rec@m10 ' + str(y):>15}" for y in VAL_YEARS) +
          f"{'mean':>8}{'min':>8}{'lift':>8}")
    for name, f in candidates.items():
        s = summary[name]
        print(f"{name:<44}" + "".join(f"{f[y]['recall_monthly10']:>15.3f}" for y in VAL_YEARS) +
              f"{s['mean_recall_monthly10']:>8.3f}{s['min_recall_monthly10']:>8.3f}{s['mean_lift']:>8.2f}")

    tree_names = [n for n in candidates if n.startswith("trees")]
    best_name = max(tree_names, key=lambda n: (summary[n]["mean_recall_monthly10"], summary[n]["mean_lift"]))
    best = configs[tree_names.index(best_name)]
    bench = summary["logistic_relative_priors_only (benchmark)"]
    beats = summary[best_name]["mean_recall_monthly10"] > bench["mean_recall_monthly10"]
    print(f"\nchosen on validation: {best_name} "
          f"(mean rec@month10 {summary[best_name]['mean_recall_monthly10']:.3f} vs benchmark "
          f"{bench['mean_recall_monthly10']:.3f}: {'beats' if beats else 'does NOT beat'} it)")

    # ------------------------------------------------ 2. refit on 2010-2013, score test once
    era = train[train.year >= ERA_START]
    med = era.planned_lead_time_days.median()
    model = make_model(backend, "clf", **best)
    fit(model, tree_matrix(era, med), era.label_late, balanced_weights(era.label_late))
    X_test = tree_matrix(test, med)
    test_score = predict_score(model, X_test)
    tree_test = score_metrics(test.label_late, test_score, test.alert_month)

    feats = LOGIT_NUMERIC + PRIOR_FLAGS
    logit = logistic_pipeline([], LOGIT_NUMERIC, PRIOR_FLAGS).fit(era[feats], era.label_late)
    logit_test = score_metrics(test.label_late, logit.predict_proba(test[feats])[:, 1], test.alert_month)

    no_skill = score_metrics(test.label_late, np.full(len(test), era.label_late.mean()), test.alert_month)
    route = score_metrics(test.label_late, test.prior_excess_fulfillment_route.values, test.alert_month)

    print("\nTEST 2014-2015 (scored once):")
    print(f"{'model':<44}{'PR-AUC':>8}{'lift':>8}{'rec@top10':>11}{'rec@m10':>9}{'prec@m10':>10}")
    for name, m in (("boosted trees (chosen)", tree_test), ("logistic_relative_priors_only", logit_test),
                    ("route_excess_lookup (reference)", route), ("no_skill", no_skill)):
        print(f"{name:<44}{m['pr_auc']:>8.3f}{m['lift']:>8.2f}{m['recall_top10']:>11.3f}"
              f"{m['recall_monthly10']:>9.3f}{m['precision_monthly10']:>10.3f}")

    # ------------------------------------------------ 3. importances on the last fold (no test)
    _, tr, va = list(folds(train))[-1]
    fmed = tr.planned_lead_time_days.median()
    fm = make_model(backend, "clf", **best)
    fit(fm, tree_matrix(tr, fmed), tr.label_late, balanced_weights(tr.label_late))
    importance = permutation_importance(fm, tree_matrix(va, fmed), va.label_late.values, va.alert_month.values)
    print(f"\npermutation importance on the {VAL_YEARS[-1]} fold (drop when shuffled):")
    for k, v in importance.items():
        print(f"  {k:<34} PR-AUC {v['pr_auc_drop']:+.3f}   rec@month10 {v['recall_monthly10_drop']:+.3f}")

    # ------------------------------------------------ 4. days late
    dl = days_late(train, test, backend, best)
    print("\ndays late (late shipments only), MAE in days, boosted vs global median:")
    for vy, f in dl["folds"].items():
        print(f"  validate {vy}: {f['mae_boosted']} vs {f['mae_global_median']}")
    print(f"  TEST (n={dl['test']['n_late']}): {dl['test']['mae_boosted']} vs {dl['test']['mae_global_median']}")

    # ------------------------------------------------ outputs
    args.out.mkdir(parents=True, exist_ok=True)
    alerts = test[["order_id", "alert_month", "scheduled_delivery_date", "fulfillment_route",
                   "country", "vendor", "label_late", "days_late"]].copy()
    alerts["risk_score"] = np.round(test_score, 5)
    alerts["flagged_monthly_top10"] = monthly_flags(test_score, test.alert_month.values)
    alerts.sort_values(["alert_month", "risk_score"], ascending=[True, False]).to_csv(
        args.out / "test_alerts.csv", index=False)
    try:
        with open(args.out / "model.pkl", "wb") as fh:
            pickle.dump({"model": model, "features": TREE_FEATURES, "lead_time_median": float(med),
                         "backend": backend, "params": best}, fh)
        saved = "model.pkl"
    except Exception as e:                    # some Snap ML builds do not pickle
        saved = f"not saved ({type(e).__name__})"

    report = {
        "backend": backend, "features": TREE_FEATURES, "training_era": f"{ERA_START}-2013",
        "validation": {"scheme": "rolling origin", "years": VAL_YEARS,
                       "candidates": {n: {"per_fold": {str(k): v for k, v in f.items()},
                                          "summary": summary[n]} for n, f in candidates.items()},
                       "chosen": best_name, "chosen_params": best,
                       "beats_logistic_benchmark": bool(beats)},
        "test": {"boosted": tree_test, "logistic_relative_priors_only": logit_test,
                 "route_excess_lookup": route, "no_skill": no_skill},
        "alert_rule": "top 10% of each order month",
        "permutation_importance_last_fold": importance,
        "days_late": dl,
        "model_file": saved,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (args.out / "boosted_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwritten: {args.out / 'boosted_report.json'}, {args.out / 'test_alerts.csv'}, model: {saved}")


if __name__ == "__main__":
    main()
