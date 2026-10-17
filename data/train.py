#!/usr/bin/env python3
"""V12 Engines - Predictive ML training script.

Trains, evaluates and saves:
  * baseline: class-weighted Logistic Regression
  * classifier: boosted trees -> late probability       (Snap ML on LinuxONE)
  * regressor:  boosted trees -> days late               (Snap ML on LinuxONE)

Backend: Snap ML if `snapml` imports, otherwise scikit-learn HistGradientBoosting
(practice stand-in, same features). Force one with --backend snapml|sklearn.

Usage (on the pod):
    python src/training/train.py --data-dir /srv/v12e/data --model-dir /srv/v12e/models

Outputs in --model-dir:
    classifier.pkl  regressor.pkl  encoder.joblib  model_meta.json
    metrics.json (the Playbook section 4 /metrics payload)
    evaluation_report.md  evaluation_report.json  feature_importance.json
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Config - the 16 features, in the exact order of Playbook section 3 / 4.
# ----------------------------------------------------------------------------
CAT_FEATURES = ["country", "vendor", "fulfillment_route", "shipment_mode",
                "product_group", "managed_by", "manufacturing_site"]
NUM_FEATURES = ["quantity", "planned_lead_time_days", "decision_date_known",
                "prior_late_rate_country", "prior_late_rate_vendor",
                "prior_late_rate_fulfillment_route",
                "prior_known_country", "prior_known_vendor",
                "prior_known_fulfillment_route"]
FEATURES = CAT_FEATURES + NUM_FEATURES            # 16, fixed order
assert len(FEATURES) == 16

TARGET_CLS, TARGET_REG = "label_late", "days_late"
NO_SKILL = 0.115
SPLIT_LABEL = "train<=2013 / test 2014-2015"      # exact string from Playbook section 4
MISSING_FILL = -1.0                               # planned_lead_time NaN -> -1 (+ decision_date_known flag)
SEED = 42


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load(path_base):
    """Load parquet (preferred) or CSV for the given base path without extension."""
    if os.path.exists(path_base + ".parquet"):
        df = pd.read_parquet(path_base + ".parquet")
    elif os.path.exists(path_base + ".csv"):
        df = pd.read_csv(path_base + ".csv", keep_default_na=False, na_values=[""],
                         parse_dates=["decision_date", "scheduled_delivery_date"])
    else:
        sys.exit(f"Missing {path_base}.parquet / .csv")
    return df.reset_index(drop=True)


def check_schema(df, name):
    missing = [c for c in FEATURES + [TARGET_CLS, TARGET_REG, "scheduled_delivery_date"]
               if c not in df.columns]
    if missing:
        sys.exit(f"{name}: missing columns {missing}")
    bad = df[FEATURES].drop(columns=["planned_lead_time_days"]).isna().sum()
    if bad.any():
        sys.exit(f"{name}: unexpected NaN in {bad[bad > 0].to_dict()}")


def make_encoder():
    """Ordinal-encode categoricals (unseen category -> -1). Same encoder is used at scoring time."""
    return ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_FEATURES)],
        remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")


def encode(enc, df, fit=False):
    X = df[FEATURES].copy()
    X["planned_lead_time_days"] = X["planned_lead_time_days"].fillna(MISSING_FILL)
    Xe = enc.fit_transform(X) if fit else enc.transform(X)
    return Xe[FEATURES].astype(np.float32)        # keep the fixed feature order


# ----------------------------------------------------------------------------
# Backend (Snap ML on LinuxONE, scikit-learn stand-in elsewhere)
# ----------------------------------------------------------------------------
def pick_backend(requested):
    if requested in ("auto", "snapml"):
        try:
            import snapml  # noqa: F401
            return "snapml"
        except Exception as e:                                          # pragma: no cover
            if requested == "snapml":
                sys.exit(f"snapml requested but not importable: {e}")
            print("[backend] snapml not importable -> scikit-learn stand-in")
    return "sklearn"


def new_classifier(backend):
    if backend == "snapml":
        from snapml import BoostingMachineClassifier
        return BoostingMachineClassifier(random_state=SEED, n_jobs=4)
    return HistGradientBoostingClassifier(random_state=SEED, max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=15, l2_regularization=1.0)


def new_regressor(backend):
    if backend == "snapml":
        from snapml import BoostingMachineRegressor
        return BoostingMachineRegressor(random_state=SEED, n_jobs=4)
    return HistGradientBoostingRegressor(random_state=SEED, max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=15, l2_regularization=1.0)


def balanced_weights(y):
    """Class weights so late (minority) rows count as much as on-time rows overall."""
    y = np.asarray(y)
    w_pos = 0.5 / max(y.mean(), 1e-9)
    w_neg = 0.5 / max(1 - y.mean(), 1e-9)
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


def fit_classifier(backend, X, y):
    m = new_classifier(backend)
    m.fit(X.values, np.asarray(y), sample_weight=balanced_weights(y))
    return m


def fit_regressor(backend, X, y):
    m = new_regressor(backend)
    m.fit(X.values, np.asarray(y, dtype=np.float32))
    return m


def predict_proba(model, X):
    p = np.asarray(model.predict_proba(X.values))
    return p[:, 1] if p.ndim == 2 else p


def predict_days(model, X):
    return np.maximum(np.rint(np.asarray(model.predict(X.values))), 0).astype(int)


def make_baseline():
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
         ("num", StandardScaler(), NUM_FEATURES)])
    return Pipeline([("pre", pre),
                     ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5))])


def baseline_frame(df):
    X = df[FEATURES].copy()
    X["planned_lead_time_days"] = X["planned_lead_time_days"].fillna(MISSING_FILL)
    return X


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def choose_threshold(y_true, p, recall_target):
    """Highest threshold that still reaches recall_target on the validation slice
    (= best precision for that recall). Falls back to the max-recall threshold."""
    y_true = np.asarray(y_true)
    order = np.argsort(-p)
    ps, ys = p[order], y_true[order]
    tp = np.cumsum(ys)
    recall = tp / max(ys.sum(), 1)
    ok = np.where(recall >= recall_target)[0]
    i = ok[0] if len(ok) else len(ps) - 1
    return float(ps[i])


def confusion(y_true, pred):
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    return dict(tp=int(((pred == 1) & (y_true == 1)).sum()),
                fp=int(((pred == 1) & (y_true == 0)).sum()),
                fn=int(((pred == 0) & (y_true == 1)).sum()),
                tn=int(((pred == 0) & (y_true == 0)).sum()))


def evaluate(y_true, p, thr):
    y_true = np.asarray(y_true)
    pred = (p >= thr).astype(int)
    c = confusion(y_true, pred)
    n = len(y_true)
    recall = c["tp"] / max(c["tp"] + c["fn"], 1)
    precision = c["tp"] / max(c["tp"] + c["fp"], 1)
    k = int(np.ceil(0.10 * n))
    top = np.argsort(-p)[:k]
    return dict(
        pr_auc=float(average_precision_score(y_true, p)),
        threshold=float(thr), recall=float(recall), precision=float(precision),
        confusion=c,
        error_breakdown=dict(correct=(c["tp"] + c["tn"]) / n,
                             type1=c["fp"] / n,          # false alarm
                             type2=c["fn"] / n),         # missed late (costly)
        recall_top10pct=float(y_true[top].sum() / max(y_true.sum(), 1)),
        precision_top10pct=float(y_true[top].mean()),
        flagged_share=float(pred.mean()),
    )


def permutation_importance(model_fn, X, y, n_repeats=5):
    """Drop in PR-AUC when one feature is shuffled (larger = more important)."""
    rng = np.random.default_rng(SEED)
    base = average_precision_score(y, model_fn(X))
    out = {}
    for f in FEATURES:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].values)
            drops.append(base - average_precision_score(y, model_fn(Xp)))
        out[f] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/srv/v12e/data")
    ap.add_argument("--model-dir", default="/srv/v12e/models")
    ap.add_argument("--backend", default="auto", choices=["auto", "snapml", "sklearn"])
    ap.add_argument("--recall-target", type=float, default=0.70,
                    help="recall the operating threshold must reach on the 2013 validation slice")
    ap.add_argument("--val-year", type=int, default=2013)
    args = ap.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    train = load(os.path.join(args.data_dir, "train"))
    test = load(os.path.join(args.data_dir, "test"))
    check_schema(train, "train"); check_schema(test, "test")
    print(f"train {train.shape} late={int(train[TARGET_CLS].sum())} | "
          f"test {test.shape} late={int(test[TARGET_CLS].sum())}")
    assert train["scheduled_delivery_date"].max() < test["scheduled_delivery_date"].min(), \
        "train/test overlap in time - do not re-split"

    backend = pick_backend(args.backend)
    print(f"[backend] {backend}")

    # ---- 1. threshold selection on a time-based validation slice (inside train) -------------
    yr = train["scheduled_delivery_date"].dt.year
    tr_in, val = train[yr < args.val_year], train[yr == args.val_year]
    enc_v = make_encoder(); Xtr_in = encode(enc_v, tr_in, fit=True); Xval = encode(enc_v, val)
    clf_v = fit_classifier(backend, Xtr_in, tr_in[TARGET_CLS])
    thr = choose_threshold(val[TARGET_CLS], predict_proba(clf_v, Xval), args.recall_target)
    base_v = make_baseline().fit(baseline_frame(tr_in), tr_in[TARGET_CLS])
    thr_base = choose_threshold(val[TARGET_CLS],
                                base_v.predict_proba(baseline_frame(val))[:, 1], args.recall_target)
    print(f"[threshold] chosen on {args.val_year} validation slice "
          f"(recall target {args.recall_target}): classifier={thr:.4f} baseline={thr_base:.4f}")

    # ---- 2. refit on ALL train, evaluate ONCE on test ---------------------------------------
    enc = make_encoder(); Xtr = encode(enc, train, fit=True); Xte = encode(enc, test)
    clf = fit_classifier(backend, Xtr, train[TARGET_CLS])
    reg = fit_regressor(backend, Xtr, train[TARGET_REG])
    base = make_baseline().fit(baseline_frame(train), train[TARGET_CLS])

    p_clf = predict_proba(clf, Xte)
    p_base = base.predict_proba(baseline_frame(test))[:, 1]
    yte = test[TARGET_CLS].values
    res_clf = evaluate(yte, p_clf, thr)
    res_base = evaluate(yte, p_base, thr_base)

    d_pred = predict_days(reg, Xte)
    d_true = test[TARGET_REG].values
    late = d_true > 0
    reg_res = dict(mae_all=float(mean_absolute_error(d_true, d_pred)),
                   mae_late_only=float(mean_absolute_error(d_true[late], d_pred[late])),
                   mae_zero_baseline_all=float(mean_absolute_error(d_true, np.zeros_like(d_true))),
                   mae_median_late_baseline=float(mean_absolute_error(
                       d_true, np.full_like(d_true, int(np.median(train.loc[train[TARGET_REG] > 0, TARGET_REG]))))),
                   mean_pred_days_on_late_rows=float(d_pred[late].mean()),
                   mean_true_days_on_late_rows=float(d_true[late].mean()))

    importance = permutation_importance(lambda X: predict_proba(clf, X), Xte, yte)

    # ---- 3. save + reload identity check ----------------------------------------------------
    stamp = hashlib.sha1(pd.util.hash_pandas_object(train[FEATURES + [TARGET_CLS]], index=False)
                         .values.tobytes()).hexdigest()[:8]
    model_version = f"v12e-{datetime.now(timezone.utc):%Y%m%d}-{backend}-{stamp}"

    paths = {k: os.path.join(args.model_dir, v) for k, v in
             dict(clf="classifier.pkl", reg="regressor.pkl", enc="encoder.joblib",
                  meta="model_meta.json").items()}
    joblib.dump(clf, paths["clf"]); joblib.dump(reg, paths["reg"]); joblib.dump(enc, paths["enc"])
    clf2, reg2, enc2 = joblib.load(paths["clf"]), joblib.load(paths["reg"]), joblib.load(paths["enc"])
    Xte2 = encode(enc2, test)
    same_cls = np.allclose(predict_proba(clf2, Xte2), p_clf, rtol=0, atol=0)
    same_reg = np.array_equal(predict_days(reg2, Xte2), d_pred)
    print(f"[reload check] classifier identical={same_cls} regressor identical={same_reg}")
    if not (same_cls and same_reg):
        sys.exit("FAILED: predictions changed after save -> reload")

    meta = dict(model_version=model_version, backend=backend, features=FEATURES,
                cat_features=CAT_FEATURES, missing_fill=MISSING_FILL,
                operating_threshold=thr, threshold_rule=f"recall>={args.recall_target} on {args.val_year} validation slice",
                trained_at=datetime.now(timezone.utc).isoformat(), train_rows=len(train),
                note="predicted days late is an estimate, not a guarantee")
    json.dump(meta, open(paths["meta"], "w"), indent=2)

    # ---- 4. outputs: /metrics payload (exact Playbook section 4 shape) + report -------------
    metrics = dict(late_rate=round(float(yte.mean()), 4), no_skill=NO_SKILL,
                   pr_auc=round(res_clf["pr_auc"], 4),
                   recall_at_threshold=round(res_clf["recall"], 4),
                   confusion=res_clf["confusion"],
                   error_breakdown={k: round(v, 4) for k, v in res_clf["error_breakdown"].items()},
                   split=SPLIT_LABEL)
    json.dump(metrics, open(os.path.join(args.model_dir, "metrics.json"), "w"), indent=2)
    json.dump(importance, open(os.path.join(args.model_dir, "feature_importance.json"), "w"), indent=2)

    report = dict(model_version=model_version, backend=backend, test_late_rate=float(yte.mean()),
                  classifier=res_clf, baseline_logreg=res_base, regressor=reg_res,
                  feature_importance_pr_auc_drop=importance)
    json.dump(report, open(os.path.join(args.model_dir, "evaluation_report.json"), "w"), indent=2)
    write_markdown(os.path.join(args.model_dir, "evaluation_report.md"), report, args.recall_target)

    print(json.dumps(metrics, indent=2))
    print(f"[done] model_version={model_version} -> {args.model_dir}")


def pct(x):
    return f"{100 * x:.1f}%"


def write_markdown(path, r, recall_target):
    c, b, g = r["classifier"], r["baseline_logreg"], r["regressor"]
    rows = lambda name, m: (f"| {name} | {m['pr_auc']:.3f} | {pct(m['recall'])} | {pct(m['precision'])} | "
                            f"{pct(m['recall_top10pct'])} | {pct(m['error_breakdown']['correct'])} | "
                            f"{pct(m['error_breakdown']['type1'])} | {pct(m['error_breakdown']['type2'])} |")
    lines = [
        f"# V12 Engines - model evaluation ({r['model_version']})", "",
        f"Backend: **{r['backend']}**. Test split: 2014-2015, late rate **{pct(r['test_late_rate'])}** "
        f"(no-skill PR-AUC = test late rate; project baseline {NO_SKILL}).",
        f"Operating threshold chosen on a 2013 validation slice inside train (recall >= {recall_target}); "
        "the test set was scored once.", "",
        "| Model | PR-AUC | Recall | Precision | Recall top 10% | Correct | Type I (false alarm) | Type II (missed late) |",
        "|---|---|---|---|---|---|---|---|", rows("Logistic baseline", b), rows("Boosted trees", c), "",
        f"Confusion (boosted trees): {c['confusion']}", "",
        "## Days-late regressor",
        f"- MAE all rows: **{g['mae_all']:.2f} d** (predict-0 baseline {g['mae_zero_baseline_all']:.2f} d)",
        f"- MAE on truly late rows: **{g['mae_late_only']:.2f} d** (predict-median-late baseline {g['mae_median_late_baseline']:.2f} d)",
        "- Predicted days late is an estimate, not a guarantee.", "",
        "## Feature importance (PR-AUC drop when shuffled, test set)"]
    lines += [f"- {k}: {v:.4f}" for k, v in r["feature_importance_pr_auc_drop"].items()]
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
