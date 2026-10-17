"""V12 Engines — delivery-delay early-warning model (final, Snap ML first).

One file, three commands (plus a small Python API for the scoring service):

  python v12e_model.py train --data data --out models
      Select a boosted-tree configuration on rolling yearly folds (2011, 2012, 2013),
      evaluate it ONCE on the 2014-2015 test set, then refit on all 2010-2015 data and
      save the production model.

  python v12e_model.py evaluate --model models --data data
      Reload the saved model, check its scores are identical to the ones recorded at save
      time, and write models/metrics.json (the /metrics payload for the API and dashboard).

  python v12e_model.py score --model models/v12e_model.pkl --input new_orders.csv --out alerts.csv
      Score new orders (same columns as the v3 feature tables) and flag the top 10% of
      each order month.

  Python API (used by src/api): load_artifact(), score_frame(), explain(), RollingAlertPolicy.

Two saved models (both written by train):
  v12e_eval_model.pkl  trained on 2010-2013. Use it for the demo replay of 2014-2015 orders,
                       which it has never seen (honest, out-of-sample scores).
  v12e_model.pkl       trained on 2010-2015. Use it for genuinely new orders.

Live alert rule: a rolling 30-day window. Each new order is ranked against the scores of
orders placed in the previous 30 days and flagged when it is in their top 10% (RollingAlertPolicy).
This needs no future orders, unlike the month-end "top 10% of each order month" used in the
offline evaluation; `evaluate` reports both.

Backend: snapml.BoostingMachineClassifier when snapml is installed (LinuxONE / IBM Z),
otherwise scikit-learn HistGradientBoostingClassifier (local development). Same features,
folds, selection rule and metrics either way; --backend forces one.

Design decisions (each traced to a measured result in the baseline runs)
  * Features: quantity, planned lead time, and lane priors expressed RELATIVE to the overall
    late rate in the same 365-day window (prior_excess_*). Raw priors lag the year-level late
    rate and flipped sign in every logistic fit; one-hot country/vendor/site identities
    memorized 2010-13 and fell below random on test. Both are excluded.
  * Training era 2010+: 2006-09 has a 0-4% late rate and mostly missing decision dates.
  * Alert rule: top 10% of each order month. Scores drift between years, so a fixed score
    threshold does not hold alert volume (it flagged 15-34% of test in the baselines).
  * Seed ensemble: each model is the average of SEEDS boosted models with row/column
    subsampling (Snap ML), which steadies a small, noisy dataset. With the scikit-learn
    backend the members are near-identical, which is harmless.
  * Days late: no regressor. Boosted and linear regressors did not beat the median on the
    folds or on test, so the model reports the median (and quartiles) of past delays.

Selection rule (fixed before running; uses validation folds only)
  1. Highest mean recall_monthly10 over the three folds.
  2. Among candidates within SELECTION_TOLERANCE of that best mean, the highest worst-fold
     recall_monthly10 (most consistent).
  3. Remaining ties: the simplest model (fewest rounds x depth), then the "core" feature set.

Metrics (read against each split's late rate)
  recall_monthly10     share of late shipments caught by flagging the top 10% of each
                       order month; random = 0.10. The headline metric.
  precision_monthly10  share of flagged shipments that were late; random = late rate.
  pr_auc / lift        precision-recall AUC and PR-AUC / late rate (1.0 = random).
  recall_top10         top 10% of the whole period (sensitive to drift between months).
  Ties in scores are split evenly (expected value), so row order never matters.
"""
import argparse
import itertools
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- configuration
BASE = ["quantity", "planned_lead_time_days"]
RELATIVE = ["prior_excess_country", "prior_excess_vendor", "prior_excess_fulfillment_route"]
PRIOR_FLAGS = ["prior_known_country", "prior_known_vendor", "prior_known_fulfillment_route"]
FEATURE_SETS = {
    "core": BASE + RELATIVE,                                        # 5 features
    "full": BASE + ["lead_time_missing"] + RELATIVE + PRIOR_FLAGS,  # 9 features
}
REQUIRED_COLUMNS = BASE + RELATIVE + PRIOR_FLAGS

ERA_START = 2010
VAL_YEARS = [2011, 2012, 2013]
LAST_TRAIN_YEAR = 2013
ALERT_SHARE = 0.10
SEEDS = (0, 1, 2)
SELECTION_TOLERANCE = 0.01

GRID = {
    "n_rounds": [100, 300],
    "learning_rate": [0.03, 0.1],
    "max_depth": [2, 3, 4],
    "feature_set": ["core", "full"],
}


# ---------------------------------------------------------------- data
def add_time_columns(df):
    df = df.copy()
    sched = pd.to_datetime(df["scheduled_delivery_date"])
    decided = pd.to_datetime(df["decision_date"]) if "decision_date" in df else pd.Series(pd.NaT, index=df.index)
    df["year"] = sched.dt.year
    # alert batch = month the order was placed; the scheduled month when that is unknown
    df["alert_month"] = decided.fillna(sched).dt.to_period("M").astype(str)
    return df


def load(data_dir):
    train = add_time_columns(pd.read_csv(Path(data_dir) / "train_v3.csv"))
    test = add_time_columns(pd.read_csv(Path(data_dir) / "test_v3.csv"))
    missing = [c for c in REQUIRED_COLUMNS + ["label_late", "days_late"] if c not in train]
    if missing:
        raise SystemExit(f"train_v3.csv is missing columns {missing}: build it with "
                         "preprocessing/V12E_preprocessing_template_v3.ipynb")
    return train, test


def matrix(df, feature_set, lead_median):
    X = pd.DataFrame(index=df.index)
    X["quantity"] = np.log1p(pd.to_numeric(df["quantity"], errors="coerce").fillna(0).clip(lower=0))
    lead = pd.to_numeric(df["planned_lead_time_days"], errors="coerce")
    X["planned_lead_time_days"] = lead.fillna(lead_median)
    X["lead_time_missing"] = lead.isna().astype(float)
    for c in RELATIVE + PRIOR_FLAGS:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return np.ascontiguousarray(X[FEATURE_SETS[feature_set]].values, dtype=np.float32)


def balanced_weights(y):
    y = np.asarray(y)
    pos, neg = max(1, y.sum()), max(1, len(y) - y.sum())
    return np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg)).astype(np.float32)


# ---------------------------------------------------------------- metrics
def expected_top_k(y, score, k):
    """Expected true positives when flagging the k highest scores, ties split evenly."""
    k = min(k, len(y))
    if k <= 0:
        return 0.0, 0
    cut = np.sort(score)[::-1][k - 1]
    above, tied = score > cut, score == cut
    return float(y[above].sum() + (k - above.sum()) * y[tied].mean()), k


def metrics(y, score, months):
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    months = np.asarray(months)
    pos = max(1.0, y.sum())
    rate = float(y.mean())
    pr_auc = float(average_precision_score(y, score))
    tp, k = expected_top_k(y, score, max(1, int(round(ALERT_SHARE * len(y)))))
    tp_m = flagged_m = 0.0
    for m in np.unique(months):
        idx = months == m
        t, f = expected_top_k(y[idx], score[idx], max(1, int(round(ALERT_SHARE * idx.sum()))))
        tp_m, flagged_m = tp_m + t, flagged_m + f
    return {"n": int(len(y)), "late": int(y.sum()), "late_rate": round(rate, 4),
            "pr_auc": round(pr_auc, 4), "lift": round(pr_auc / rate, 3) if rate else None,
            "recall_top10": round(tp / pos, 4), "precision_top10": round(tp / k, 4),
            "recall_monthly10": round(tp_m / pos, 4),
            "precision_monthly10": round(tp_m / flagged_m, 4) if flagged_m else None}


def monthly_flags(score, months):
    """1 for the top 10% of each order month (ties broken by row order)."""
    score, months = np.asarray(score, dtype=float), np.asarray(months)
    flags = np.zeros(len(score), dtype=int)
    for m in np.unique(months):
        idx = np.flatnonzero(months == m)
        k = max(1, int(round(ALERT_SHARE * len(idx))))
        flags[idx[np.argsort(-score[idx], kind="stable")[:k]]] = 1
    return flags


class RollingAlertPolicy:
    """Live alert rule: rank each order against orders placed in the previous WINDOW days.

    history: (order_id, order_date, score) of orders already scored. Orders are compared only
    with strictly earlier order dates, so the rule never uses the future. Falls back to a fixed
    reference distribution when the window holds fewer than min_window orders.
    """

    def __init__(self, ids, dates, scores, reference, window_days=30, min_window=30):
        self.window = np.timedelta64(window_days, "D")
        self.min_window = min_window
        self.reference = np.asarray(reference, dtype=float)
        self.known = {}
        for i, d, s in zip(ids, pd.to_datetime(pd.Series(dates)).values.astype("datetime64[D]"), scores):
            self.known[str(i)] = (d, float(s))
        self._rebuild()

    def _rebuild(self):
        items = sorted(self.known.values(), key=lambda t: t[0])
        self.dates = np.array([t[0] for t in items], dtype="datetime64[D]")
        self.scores = np.array([t[1] for t in items], dtype=float)

    def add(self, order_id, order_date, score):
        self.known[str(order_id)] = (np.datetime64(pd.Timestamp(order_date).date(), "D"), float(score))
        self._rebuild()

    def rank(self, score, order_date):
        """-> (band, percentile 0-100, window size, basis)."""
        t = np.datetime64(pd.Timestamp(order_date).date(), "D")
        lo, hi = np.searchsorted(self.dates, t - self.window, "left"), np.searchsorted(self.dates, t, "left")
        window = self.scores[lo:hi]
        if len(window) >= self.min_window:
            pct, basis = 100.0 * ((window < score).mean() + 0.5 * (window == score).mean()), "rolling_30d"
        else:
            pct = float(np.searchsorted(self.reference, score, side="right") - 1)
            basis = "reference"
        pct = min(max(pct, 0.0), 100.0)
        band = "high" if pct >= 90 else "elevated" if pct >= 75 else "normal"
        return band, round(pct, 1), int(len(window)), basis


def policy_from_artifact(art, window_days=30):
    h = art["score_history"]
    return RollingAlertPolicy(h["order_id"], h["order_date"], h["score"], art["score_reference"], window_days)


# ---------------------------------------------------------------- models
def snapml_available():
    try:
        import snapml  # noqa: F401
        return True
    except Exception:
        return False


def make_tree(backend, n_rounds, learning_rate, max_depth, seed, n_jobs):
    if backend == "snapml":
        from snapml import BoostingMachineClassifier
        return BoostingMachineClassifier(
            num_round=n_rounds, learning_rate=learning_rate, max_depth=max_depth,
            subsample=0.8, colsample_bytree=0.8, lambda_l2=1.0,
            use_histograms=True, random_state=seed, n_jobs=n_jobs)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=n_rounds, learning_rate=learning_rate, max_depth=max_depth,
        min_samples_leaf=20, l2_regularization=1.0, random_state=seed)


class SeedEnsemble:
    """Average of boosted classifiers trained with different seeds."""

    def __init__(self, backend, params, seeds=SEEDS, n_jobs=1):
        self.backend, self.params, self.seeds, self.n_jobs = backend, dict(params), tuple(seeds), n_jobs
        self.members = []

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float32)
        p = {k: v for k, v in self.params.items() if k != "feature_set"}
        self.members = []
        for s in self.seeds:
            m = make_tree(self.backend, seed=s, n_jobs=self.n_jobs, **p)
            m.fit(X, y, sample_weight=sample_weight)
            self.members.append(m)
        return self

    def predict_proba(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        p1 = np.mean([np.asarray(m.predict_proba(X))[:, 1] for m in self.members], axis=0)
        return np.column_stack([1 - p1, p1])

    def optimize_for_inference(self):
        """Snap ML only: pick the fastest tree format (uses the IBM Z AI accelerator if present)."""
        done = []
        for m in self.members:
            try:
                m.optimize_trees(tree_format="auto")
                done.append(str(getattr(m.booster_, "optimized_tree_format_", "auto")))
            except Exception as e:
                done.append(f"skipped ({type(e).__name__})")
        return done


def logistic_benchmark(X, y):
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(class_weight="balanced", max_iter=5000).fit(scaler.transform(X), y)
    return lambda Xn: lr.predict_proba(scaler.transform(Xn))[:, 1]


# ---------------------------------------------------------------- validation
def folds(train):
    era = train[train.year >= ERA_START]
    for vy in VAL_YEARS:
        yield vy, era[era.year < vy], era[era.year == vy]


def evaluate_candidate(train, backend, params, n_jobs):
    per_fold = {}
    for vy, tr, va in folds(train):
        med = float(tr.planned_lead_time_days.median())
        fs = params["feature_set"]
        model = SeedEnsemble(backend, params, n_jobs=n_jobs).fit(
            matrix(tr, fs, med), tr.label_late, balanced_weights(tr.label_late))
        per_fold[vy] = metrics(va.label_late, model.predict_proba(matrix(va, fs, med))[:, 1], va.alert_month)
    return per_fold


def evaluate_benchmark(train):
    per_fold = {}
    for vy, tr, va in folds(train):
        med = float(tr.planned_lead_time_days.median())
        score = logistic_benchmark(matrix(tr, "full", med), tr.label_late)
        per_fold[vy] = metrics(va.label_late, score(matrix(va, "full", med)), va.alert_month)
    return per_fold


def summarize(per_fold):
    rec = [f["recall_monthly10"] for f in per_fold.values()]
    return {"mean_recall_monthly10": round(float(np.mean(rec)), 4),
            "min_recall_monthly10": round(float(np.min(rec)), 4),
            "mean_precision_monthly10": round(float(np.mean([f["precision_monthly10"] for f in per_fold.values()])), 4),
            "mean_pr_auc": round(float(np.mean([f["pr_auc"] for f in per_fold.values()])), 4),
            "mean_lift": round(float(np.mean([f["lift"] for f in per_fold.values()])), 3)}


def select(candidates):
    """Pre-registered rule: best mean, then best worst fold within tolerance, then simplest."""
    best_mean = max(c["summary"]["mean_recall_monthly10"] for c in candidates)
    close = [c for c in candidates
             if c["summary"]["mean_recall_monthly10"] >= best_mean - SELECTION_TOLERANCE]
    return max(close, key=lambda c: (c["summary"]["min_recall_monthly10"],
                                     -c["params"]["n_rounds"] * c["params"]["max_depth"],
                                     c["params"]["feature_set"] == "core"))


def label(p):
    return f"trees {p['feature_set']:<4} r={p['n_rounds']:<3} lr={p['learning_rate']:<4} d={p['max_depth']}"


def permutation_importance(model, X, y, months, names, repeats=5):
    rng = np.random.default_rng(0)
    base = metrics(y, model.predict_proba(X)[:, 1], months)
    out = {}
    for j, name in enumerate(names):
        d_pr, d_rec = [], []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            m = metrics(y, model.predict_proba(Xp)[:, 1], months)
            d_pr.append(base["pr_auc"] - m["pr_auc"])
            d_rec.append(base["recall_monthly10"] - m["recall_monthly10"])
        out[name] = {"pr_auc_drop": round(float(np.mean(d_pr)), 4),
                     "recall_monthly10_drop": round(float(np.mean(d_rec)), 4)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["recall_monthly10_drop"]))


# ---------------------------------------------------------------- train command
def cmd_train(args):
    backend = resolve_backend(args.backend)
    t0 = time.time()
    train, test = load(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"backend: {backend} · seeds per model: {len(SEEDS)} · era {ERA_START}+ · "
          f"folds validate on {VAL_YEARS} · test untouched until step 2\n")

    # 1. model selection on rolling folds ---------------------------------------
    keys = list(GRID)
    configs = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    candidates = []
    for i, p in enumerate(configs, 1):
        f = evaluate_candidate(train, backend, p, args.n_jobs)
        candidates.append({"name": label(p), "params": p, "per_fold": f, "summary": summarize(f)})
        print(f"\r  fitted {i}/{len(configs)} candidates", end="", flush=True)
    print()
    bench_f = evaluate_benchmark(train)
    bench = {"name": "logistic benchmark (full features)", "per_fold": bench_f, "summary": summarize(bench_f)}
    chosen = select(candidates)

    header = f"\n{'candidate (validation folds only)':<40}" + "".join(f"{'m10 ' + str(y):>10}" for y in VAL_YEARS) + \
             f"{'mean':>8}{'min':>8}{'prec':>8}{'lift':>7}"
    print(header)
    ranked = sorted(candidates, key=lambda c: -c["summary"]["mean_recall_monthly10"])
    for c in [bench] + ranked:
        s = c["summary"]
        mark = "  <- chosen" if c is chosen else ""
        print(f"{c['name']:<40}" + "".join(f"{c['per_fold'][y]['recall_monthly10']:>10.3f}" for y in VAL_YEARS) +
              f"{s['mean_recall_monthly10']:>8.3f}{s['min_recall_monthly10']:>8.3f}"
              f"{s['mean_precision_monthly10']:>8.3f}{s['mean_lift']:>7.2f}{mark}")
    beats = chosen["summary"]["mean_recall_monthly10"] > bench["summary"]["mean_recall_monthly10"]
    print(f"\nchosen: {chosen['name']} — beats the logistic benchmark on the folds: {beats}")

    # 2. one evaluation on the 2014-2015 test set -------------------------------
    fs, p = chosen["params"]["feature_set"], chosen["params"]
    era = train[train.year >= ERA_START]
    med = float(era.planned_lead_time_days.median())
    eval_model = SeedEnsemble(backend, p, n_jobs=args.n_jobs).fit(
        matrix(era, fs, med), era.label_late, balanced_weights(era.label_late))
    test_score = eval_model.predict_proba(matrix(test, fs, med))[:, 1]
    results = {
        "boosted (chosen)": metrics(test.label_late, test_score, test.alert_month),
        "logistic benchmark": metrics(test.label_late, logistic_benchmark(
            matrix(era, "full", med), era.label_late)(matrix(test, "full", med)), test.alert_month),
        "route_excess_lookup": metrics(test.label_late, test.prior_excess_fulfillment_route.values,
                                       test.alert_month),
        "no_skill": metrics(test.label_late, np.full(len(test), era.label_late.mean()), test.alert_month),
    }
    print("\nTEST 2014-2015 (model trained on 2010-2013, scored once):")
    print(f"{'model':<26}{'rec@m10':>9}{'prec@m10':>10}{'PR-AUC':>8}{'lift':>7}{'rec@top10':>11}")
    for n, m in results.items():
        print(f"{n:<26}{m['recall_monthly10']:>9.3f}{m['precision_monthly10']:>10.3f}"
              f"{m['pr_auc']:>8.3f}{m['lift']:>7.2f}{m['recall_top10']:>11.3f}")

    alerts = test[["order_id", "alert_month", "scheduled_delivery_date", "country", "vendor",
                   "fulfillment_route", "label_late", "days_late"]].copy()
    alerts["risk_score"] = np.round(test_score, 5)
    alerts["flagged_monthly_top10"] = monthly_flags(test_score, test.alert_month.values)
    alerts.sort_values(["alert_month", "risk_score"], ascending=[True, False]).to_csv(
        out / "test_alerts.csv", index=False)

    def order_dates(df):
        return pd.to_datetime(df["decision_date"]).fillna(pd.to_datetime(df["scheduled_delivery_date"]))

    def history(model, df, lead_med):
        return {"order_id": df["order_id"].astype(str).tolist(),
                "order_date": order_dates(df).dt.strftime("%Y-%m-%d").tolist(),
                "score": model.predict_proba(matrix(df, fs, lead_med))[:, 1].round(6).tolist()}

    everything = pd.concat([era, test])
    X_era = matrix(era, fs, med)
    eval_artifact = {
        "model_version": f"{args.model_version}-eval", "role": "evaluation (trained 2010-2013; replay 2014-2015)",
        "model": eval_model, "backend": backend, "params": p, "feature_set": fs,
        "features": FEATURE_SETS[fs], "lead_time_median": med,
        "feature_medians": {n: float(np.median(X_era[:, j])) for j, n in enumerate(FEATURE_SETS[fs])},
        "score_reference": np.quantile(eval_model.predict_proba(matrix(era[era.year == era.year.max()], fs, med))[:, 1],
                                       np.linspace(0, 1, 101)).tolist(),
        "score_history": history(eval_model, everything, med),
        "days_late_if_late": {"median": float(era[era.label_late == 1].days_late.median()),
                              "q25": float(era[era.label_late == 1].days_late.quantile(.25)),
                              "q75": float(era[era.label_late == 1].days_late.quantile(.75))},
        "alert_share": ALERT_SHARE, "trained_on": f"{ERA_START}-{LAST_TRAIN_YEAR}", "n_train": int(len(era)),
        "tree_format": None,
    }
    try:
        with open(out / "v12e_eval_model.pkl", "wb") as fh:
            pickle.dump(eval_artifact, fh)
        np.save(out / "reload_check_eval.npy", np.asarray(test_score, dtype=np.float64))
    except Exception as e:
        print(f"evaluation model not saved: {type(e).__name__}: {e}")

    # 3. importances on the last validation fold (no test rows) -----------------
    _, tr, va = list(folds(train))[-1]
    fmed = float(tr.planned_lead_time_days.median())
    fm = SeedEnsemble(backend, p, n_jobs=args.n_jobs).fit(
        matrix(tr, fs, fmed), tr.label_late, balanced_weights(tr.label_late))
    importance = permutation_importance(fm, matrix(va, fs, fmed), va.label_late.values,
                                        va.alert_month.values, FEATURE_SETS[fs])
    print(f"\nwhat drives the score (shuffle one feature on the {VAL_YEARS[-1]} fold; drop in rec@m10 / PR-AUC):")
    for k, v in importance.items():
        print(f"  {k:<34}{v['recall_monthly10_drop']:+.3f}   {v['pr_auc_drop']:+.3f}")

    # 4. production model: same config, refit on everything labelled (2010-2015) -
    full = pd.concat([era, test])
    pmed = float(full.planned_lead_time_days.median())
    prod = SeedEnsemble(backend, p, n_jobs=args.n_jobs).fit(
        matrix(full, fs, pmed), full.label_late, balanced_weights(full.label_late))
    tree_format = prod.optimize_for_inference() if (backend == "snapml" and args.optimize_trees) else None
    late = full[full.label_late == 1].days_late
    X_full = matrix(full, fs, pmed)
    recent = full[full.year >= full.year.max() - 1]
    reference = prod.predict_proba(matrix(recent, fs, pmed))[:, 1]
    artifact = {
        "model_version": args.model_version, "role": "production (trained 2010-2015; new orders)",
        "feature_medians": {n: float(np.median(X_full[:, j])) for j, n in enumerate(FEATURE_SETS[fs])},
        "score_reference": np.quantile(reference, np.linspace(0, 1, 101)).tolist(),
        "model": prod, "backend": backend, "params": p, "feature_set": fs,
        "features": FEATURE_SETS[fs], "lead_time_median": pmed,
        "days_late_if_late": {"median": float(late.median()), "q25": float(late.quantile(.25)),
                              "q75": float(late.quantile(.75))},
        "alert_share": ALERT_SHARE, "trained_on": f"{ERA_START}-{int(full.year.max())}",
        "n_train": int(len(full)), "tree_format": tree_format,
    }
    artifact["score_history"] = history(prod, full, pmed)
    model_path = out / "v12e_model.pkl"
    try:
        with open(model_path, "wb") as fh:
            pickle.dump(artifact, fh)
        saved = str(model_path)
        # scores of the saved model on the test rows, for the save -> reload check in `evaluate`
        np.save(out / "reload_check.npy",
                prod.predict_proba(matrix(test, fs, pmed))[:, 1].astype(np.float64))
    except Exception as e:
        saved = f"not saved: {type(e).__name__}: {e}"
    pmml = []
    if backend == "snapml":
        for i, m in enumerate(prod.members):
            try:
                path = out / f"v12e_member{i}.pmml"
                m.export_model(str(path), output_type="pmml")
                pmml.append(str(path))
            except Exception as e:
                pmml.append(f"skipped ({type(e).__name__})")

    # 5. days late: report the distribution, not a model -------------------------
    dl_test = test[test.label_late == 1].days_late
    dl_era = era[era.label_late == 1].days_late
    days_late = {"median_2010_2013": float(dl_era.median()),
                 "test_mae_of_that_median": round(float((dl_test - dl_era.median()).abs().mean()), 2),
                 "production": artifact["days_late_if_late"]}

    report = {
        "model_version": args.model_version,
        "backend": backend, "seeds": list(SEEDS), "training_era": f"{ERA_START}+",
        "selection_rule": "max mean recall_monthly10; within %.2f, max worst-fold; then simplest" % SELECTION_TOLERANCE,
        "validation": {"years": VAL_YEARS, "benchmark": bench,
                       "candidates": sorted(candidates, key=lambda c: -c["summary"]["mean_recall_monthly10"]),
                       "chosen": chosen["name"], "chosen_params": p,
                       "beats_benchmark": bool(beats)},
        "test_2014_2015": results,
        "alert_rule": "top 10% of each order month",
        "importance_last_fold": importance,
        "days_late": days_late,
        "production_model": {"path": saved, "trained_on": artifact["trained_on"], "n": artifact["n_train"],
                             "tree_format": tree_format, "pmml": pmml},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    write_results_md(out / "RESULTS.md", report)

    d = days_late["production"]
    print(f"\ndays late if late: median {d['median']:.0f} days (middle half {d['q25']:.0f}-{d['q75']:.0f}); "
          f"no regressor beat the median, so none is shipped")
    print(f"\nproduction model {args.model_version} ({artifact['trained_on']}, n={artifact['n_train']}): {saved}")
    if tree_format:
        print(f"inference tree format: {tree_format}")
    print(f"written: {out / 'report.json'}, {out / 'RESULTS.md'}, {out / 'test_alerts.csv'} "
          f"· {report['runtime_seconds']}s")


def write_results_md(path, r):
    t = r["test_2014_2015"]
    lines = [
        "# V12 Engines — model results", "",
        f"Backend: **{r['backend']}** · chosen on validation: **{r['validation']['chosen']}** "
        f"· beats logistic benchmark on folds: **{r['validation']['beats_benchmark']}**", "",
        "## Test 2014-2015 (trained on 2010-2013, scored once)", "",
        "| model | recall @ monthly top 10% | precision | PR-AUC | lift |",
        "|---|---|---|---|---|",
    ]
    for n, m in t.items():
        lines.append(f"| {n} | {m['recall_monthly10']:.3f} | {m['precision_monthly10']:.3f} | "
                     f"{m['pr_auc']:.3f} | {m['lift']:.2f} |")
    b = t["boosted (chosen)"]
    lines += ["", f"Flagging 10% of each month's orders catches **{b['recall_monthly10']:.0%}** of late "
              f"shipments; **{b['precision_monthly10']:.0%}** of flags are late (base rate {b['late_rate']:.0%}).", "",
              "## Validation folds (recall @ monthly top 10%)", "",
              "| candidate | " + " | ".join(str(y) for y in r["validation"]["years"]) + " | mean | worst |",
              "|---|" + "---|" * (len(r["validation"]["years"]) + 2)]
    for c in [r["validation"]["benchmark"]] + r["validation"]["candidates"][:8]:
        s = c["summary"]
        lines.append(f"| {c['name']} | " + " | ".join(f"{c['per_fold'][y]['recall_monthly10']:.3f}"
                                                       for y in r["validation"]["years"]) +
                     f" | {s['mean_recall_monthly10']:.3f} | {s['min_recall_monthly10']:.3f} |")
    d = r["days_late"]["production"]
    lines += ["", "## Days late", "",
              f"Reported as the historical distribution: median {d['median']:.0f} days, middle half "
              f"{d['q25']:.0f}-{d['q75']:.0f}. No regressor beat the median on the folds or on test.", "",
              "## Caveats", "",
              "- Test was also viewed during baseline analysis, so treat test numbers as indicative.",
              "- Each validation year has roughly 130-230 late shipments; differences under ~0.02 are noise.", ""]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------- scoring API (used by src/api)
class _Unpickler(pickle.Unpickler):
    """Also loads artifacts saved by older versions, whose classes were pickled under __main__."""

    def find_class(self, module, name):
        if module == "__main__":
            module = __name__
        return super().find_class(module, name)


def load_artifact(model_dir_or_file, name="v12e_model.pkl"):
    path = Path(model_dir_or_file)
    if path.is_dir():
        path = path / name
    with open(path, "rb") as fh:
        return _Unpickler(fh).load()


def score_frame(art, df):
    """Risk scores (0-1, a ranking score: class weighting means it is not a calibrated probability)."""
    return art["model"].predict_proba(matrix(df, art["feature_set"], art["lead_time_median"]))[:, 1]


def risk_band(art, score):
    """Where a score sits in the recent score distribution of the production model."""
    pct = float(np.searchsorted(art["score_reference"], score, side="right") - 1)
    pct = min(max(pct, 0.0), 100.0)
    band = "high" if pct >= 90 else "elevated" if pct >= 75 else "normal"
    return band, round(pct, 1)


_LANE = {"prior_excess_country": ("country", "destination country"),
         "prior_excess_vendor": ("vendor", "vendor"),
         "prior_excess_fulfillment_route": ("fulfillment_route", "fulfillment route")}


def describe(feature, row):
    """Plain-language fact for one driver, built only from the row's own values."""
    if feature in _LANE:
        key, noun = _LANE[feature]
        if int(row.get(f"prior_known_{key}", 1) or 0) == 0:
            return f"no recent delivery history for this {noun}"
        v = float(row[feature]) * 100
        name = row.get(key)
        who = f"this {noun} ({name})" if isinstance(name, str) and name else f"this {noun}"
        rel = "above" if v > 0 else "below"
        pts = f"{abs(v):.0f} point" + ("" if round(abs(v)) == 1 else "s")
        return f"{who} has a late rate over the past year {pts} {rel} the overall rate"
    if feature in ("planned_lead_time_days", "lead_time_missing"):
        lead = pd.to_numeric(pd.Series([row.get("planned_lead_time_days")]), errors="coerce").iloc[0]
        return "planned lead time not recorded" if pd.isna(lead) else f"planned lead time of {lead:.0f} days"
    if feature == "quantity":
        return f"order quantity of {float(row['quantity']):,.0f} units"
    if feature.startswith("prior_known_"):
        noun = _LANE.get("prior_excess_" + feature[len("prior_known_"):], ("", feature))[1]
        return f"no recent delivery history for this {noun}" if int(row[feature]) == 0 else f"recent history available for this {noun}"
    return feature


def explain(art, df, top_k=3, min_effect=0.005):
    """Per-row drivers: reset one feature to its typical training value and measure the score change.

    Returns one list per row of {feature, effect, direction, text}, largest |effect| first.
    A lane driver is only reported when its direction matches its value (above the overall
    rate raises risk, below lowers it); the opposite comes from tree interactions and is not
    a reason an officer can act on.
    A missing planned lead time is always reported, because the model used a filled value.
    """
    names = art["features"]
    X = matrix(df, art["feature_set"], art["lead_time_median"])
    base = art["model"].predict_proba(X)[:, 1]
    effects = np.zeros((len(df), len(names)))
    for j, n in enumerate(names):
        Xr = X.copy()
        Xr[:, j] = art["feature_medians"][n]
        effects[:, j] = base - art["model"].predict_proba(Xr)[:, 1]
    rows = df.to_dict("records")
    out = []
    for i, row in enumerate(rows):
        order = np.argsort(-np.abs(effects[i]))
        drivers, seen = [], set()
        for j in order:
            e = float(effects[i, j])
            if len(drivers) >= top_k or abs(e) < min_effect:
                break
            n = names[j]
            if n in _LANE and int(row.get(f"prior_known_{_LANE[n][0]}", 1) or 0) == 1 \
                    and float(row[n]) * e < 0:
                continue        # e.g. "below the overall rate" raising risk: an interaction, not a reason
            text = describe(names[j], row)
            if text in seen:
                continue
            seen.add(text)
            drivers.append({"feature": names[j], "effect": round(e, 4),
                            "direction": "raises risk" if e > 0 else "lowers risk", "text": text})
        lead = pd.to_numeric(pd.Series([row.get("planned_lead_time_days")]), errors="coerce").iloc[0]
        if pd.isna(lead) and "planned lead time not recorded" not in seen:
            drivers.append({"feature": "planned_lead_time_days", "effect": None,
                            "direction": "unknown", "text": "planned lead time not recorded"})
        out.append(drivers)
    return out


# ---------------------------------------------------------------- evaluate command
def cmd_evaluate(args):
    mdir = Path(args.model)
    art = load_artifact(mdir)
    report = json.loads((mdir / "report.json").read_text())
    alerts = pd.read_csv(mdir / "test_alerts.csv")
    _, test = load(args.data)

    # 1. save -> reload check: identical scores on the test rows
    recorded = np.load(mdir / "reload_check.npy")
    now = score_frame(art, test)
    diff = float(np.max(np.abs(now - recorded))) if len(recorded) == len(now) else float("inf")
    reload_ok = diff <= 1e-6
    print(f"save -> reload check on {len(now)} test rows: max |diff| = {diff:.2e} -> "
          f"{'PASS' if reload_ok else 'FAIL'}")

    ev = load_artifact(mdir, "v12e_eval_model.pkl")
    ev_scores = score_frame(ev, test)
    ev_rec = np.load(mdir / "reload_check_eval.npy")
    ev_diff = float(np.max(np.abs(ev_scores - ev_rec))) if len(ev_rec) == len(ev_scores) else float("inf")
    ev_ok = ev_diff <= 1e-6
    print(f"evaluation model reload check: max |diff| = {ev_diff:.2e} -> {'PASS' if ev_ok else 'FAIL'}")
    reload_ok = reload_ok and ev_ok

    # 1b. live alert rule simulated on the test period, using only earlier orders at each step
    policy = policy_from_artifact(ev)
    t_dates = pd.to_datetime(test["decision_date"]).fillna(pd.to_datetime(test["scheduled_delivery_date"]))
    ranked = [policy.rank(s, d) for s, d in zip(ev_scores, t_dates)]
    live_flag = np.array([r[0] == "high" for r in ranked])
    yv = test.label_late.values
    live = {"rule": "top 10% of orders placed in the previous 30 days (live)",
            "flagged_share": round(float(live_flag.mean()), 4),
            "recall": round(float((live_flag & (yv == 1)).sum() / max(1, yv.sum())), 4),
            "precision": round(float((live_flag & (yv == 1)).sum() / max(1, live_flag.sum())), 4),
            "used_reference_fallback": int(sum(r[3] == "reference" for r in ranked))}
    print(f"live rolling-30-day rule on test: flags {live['flagged_share']:.1%} · recall {live['recall']:.1%} · "
          f"precision {live['precision']:.1%}")

    # 2. error breakdown of the monthly top-10% alert rule (evaluation model, 2014-2015)
    f, y = alerts.flagged_monthly_top10.values, alerts.label_late.values
    tp, fp = int(((f == 1) & (y == 1)).sum()), int(((f == 1) & (y == 0)).sum())
    fn, tn = int(((f == 0) & (y == 1)).sum()), int(((f == 0) & (y == 0)).sum())
    n = len(alerts)
    test_m = report["test_2014_2015"]
    metrics_payload = {
        "model_version": art.get("model_version"),
        "backend": art["backend"],
        "split": "train 2010-2013 (selection on rolling folds 2011/2012/2013) / test 2014-2015, scored once",
        "alert_rule": "live: top 10% of orders placed in the previous 30 days; "
                      "month-end top 10% shown as the best case",
        "live_alert_rule_test": live,
        "test": {
            "boosted": test_m["boosted (chosen)"],
            "baselines": {"logistic_benchmark": test_m["logistic benchmark"],
                          "route_excess_lookup": test_m["route_excess_lookup"],
                          "no_skill": test_m["no_skill"]},
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "error_breakdown": {"correct": round((tp + tn) / n, 4), "type1_false_alarm": round(fp / n, 4),
                                "type2_missed_late": round(fn / n, 4)},
            "flagged_share": round(f.mean(), 4),
        },
        "pr_curve": pr_curve(alerts.label_late.values, alerts.risk_score.values),
        "feature_importance": [{"feature": k, **v} for k, v in report["importance_last_fold"].items()],
        "validation": {"chosen": report["validation"]["chosen"],
                       "selection_rule": report["selection_rule"],
                       "benchmark": report["validation"]["benchmark"]["summary"],
                       "top_candidates": [{"name": c["name"], **c["summary"]}
                                          for c in report["validation"]["candidates"][:5]]},
        "days_late": report["days_late"],
        "production_model": {**report["production_model"], "model_version": art.get("model_version")},
        "reload_check": {"rows": int(len(now)), "max_abs_diff": diff, "eval_model_max_abs_diff": ev_diff,
                         "passed": reload_ok},
        "caveats": [
            "Historical replay of 2006-2015 data, not a live feed.",
            "risk_score ranks orders; it is not a calibrated probability.",
            "Days late is the historical median and middle half, not a per-order prediction.",
            "Test years were also viewed during the baseline study; treat test numbers as indicative.",
        ],
    }
    (mdir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, default=str))
    e = metrics_payload["test"]["error_breakdown"]
    print(f"error breakdown (test, monthly top 10%): correct {e['correct']:.1%} · "
          f"Type I {e['type1_false_alarm']:.1%} · Type II {e['type2_missed_late']:.1%}")
    print(f"written: {mdir / 'metrics.json'}")
    if not reload_ok:
        raise SystemExit("reload check FAILED: the saved model does not reproduce its scores")


def pr_curve(y, score, points=50):
    """Precision-recall points for the Model view chart (thresholds on the score quantiles)."""
    y, score = np.asarray(y), np.asarray(score)
    out = []
    for q in np.linspace(0.0, 0.99, points):
        thr = float(np.quantile(score, q))
        pred = score >= thr
        tp = int((pred & (y == 1)).sum())
        out.append({"threshold": round(thr, 5), "recall": round(tp / max(1, y.sum()), 4),
                    "precision": round(tp / max(1, pred.sum()), 4), "flagged_share": round(float(pred.mean()), 4)})
    return out


# ---------------------------------------------------------------- score command
def cmd_score(args):
    art = load_artifact(args.model)
    df = pd.read_csv(args.input)
    missing = [c for c in REQUIRED_COLUMNS + ["scheduled_delivery_date"] if c not in df]
    if missing:
        raise SystemExit(f"input is missing columns {missing} (build them with the v3 preprocessing)")
    df = add_time_columns(df)
    score = art["model"].predict_proba(matrix(df, art["feature_set"], art["lead_time_median"]))[:, 1]
    out = df.drop(columns=["year"]).copy()
    out["risk_score"] = np.round(score, 5)
    out["flagged_monthly_top10"] = monthly_flags(score, df.alert_month.values)
    d = art["days_late_if_late"]
    out["expected_days_late_if_late"] = d["median"]
    out["risk_band"] = [risk_band(art, s)[0] for s in score]
    out["top_drivers"] = ["; ".join(x["text"] for x in dr) for dr in explain(art, df)]
    out.sort_values(["alert_month", "risk_score"], ascending=[True, False]).to_csv(args.out, index=False)
    print(f"scored {len(out)} orders with {art['backend']} model ({art['trained_on']}); "
          f"{int(out.flagged_monthly_top10.sum())} flagged -> {args.out}")


# ---------------------------------------------------------------- cli
def resolve_backend(choice):
    if choice == "auto":
        return "snapml" if snapml_available() else "sklearn"
    if choice == "snapml" and not snapml_available():
        raise SystemExit("snapml is not importable here (see README: LinuxONE setup), or use --backend sklearn")
    return choice


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="select on folds, test once, save the production model")
    t.add_argument("--data", default="data")
    t.add_argument("--out", default="models")
    t.add_argument("--backend", choices=["auto", "snapml", "sklearn"], default="auto")
    t.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1, help="threads per model (Snap ML n_jobs)")
    t.add_argument("--optimize-trees", action="store_true",
                   help="Snap ML: convert the production trees to the fastest inference format")
    t.add_argument("--model-version", default=os.environ.get("MODEL_VERSION", "dev"))
    e = sub.add_parser("evaluate", help="reload check + metrics.json for the /metrics endpoint")
    e.add_argument("--model", default="models", help="directory written by train")
    e.add_argument("--data", default="data")
    s = sub.add_parser("score", help="score new orders with a saved model")
    s.add_argument("--model", default="models/v12e_model.pkl")
    s.add_argument("--input", required=True)
    s.add_argument("--out", default="alerts.csv")
    args = ap.parse_args(argv)
    {"train": cmd_train, "evaluate": cmd_evaluate, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    # Run through the importable module so pickled classes are saved as v12e_model.SeedEnsemble
    # (not __main__.SeedEnsemble), which the API and other importers can load.
    import v12e_model
    sys.exit(v12e_model.main())
