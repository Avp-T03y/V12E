"""V12 Engines — delivery-delay early-warning model (final, Snap ML first).

One file, two commands:

  python v12e_model.py train --data data --out models
      Select a boosted-tree configuration on rolling yearly folds (2011, 2012, 2013),
      evaluate it ONCE on the 2014-2015 test set, then refit on all 2010-2015 data and
      save the production model.

  python v12e_model.py score --model models/v12e_model.pkl --input new_orders.csv --out alerts.csv
      Score new orders (same columns as the v3 feature tables) and flag the top 10% of
      each order month.

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
    artifact = {
        "model": prod, "backend": backend, "params": p, "feature_set": fs,
        "features": FEATURE_SETS[fs], "lead_time_median": pmed,
        "days_late_if_late": {"median": float(late.median()), "q25": float(late.quantile(.25)),
                              "q75": float(late.quantile(.75))},
        "alert_share": ALERT_SHARE, "trained_on": f"{ERA_START}-{int(full.year.max())}",
        "n_train": int(len(full)), "tree_format": tree_format,
    }
    model_path = out / "v12e_model.pkl"
    try:
        with open(model_path, "wb") as fh:
            pickle.dump(artifact, fh)
        saved = str(model_path)
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
    print(f"\nproduction model ({artifact['trained_on']}, n={artifact['n_train']}): {saved}")
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


# ---------------------------------------------------------------- score command
def cmd_score(args):
    with open(args.model, "rb") as fh:
        art = pickle.load(fh)
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
    s = sub.add_parser("score", help="score new orders with a saved model")
    s.add_argument("--model", default="models/v12e_model.pkl")
    s.add_argument("--input", required=True)
    s.add_argument("--out", default="alerts.csv")
    args = ap.parse_args(argv)
    {"train": cmd_train, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
