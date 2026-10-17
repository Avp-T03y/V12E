# V12 Engines — delivery-delay early-warning model

Flags the purchase orders most likely to be delivered late, at the moment the order is placed,
using the USAID Supply Chain Management System (SCMS) delivery history. Trains a boosted-tree
classifier with **IBM Snap ML** on LinuxONE (falls back to scikit-learn elsewhere) and produces a
monthly alert queue: the top 10% riskiest orders of each order month.

## Repository

| path | what it is |
|---|---|
| `v12e_model.py` | the model: `train` (select, evaluate, save) and `score` (new orders → alerts) |
| `data/train_v3.csv`, `data/test_v3.csv` | leakage-safe feature tables (train ≤2013, test 2014–2015) |
| `preprocessing/V12E_preprocessing_template_v3.ipynb` | builds the tables from the raw SCMS CSV, with leakage checks |
| `preprocessing/SCMS_Delivery_History_Dataset_20150929.csv` | raw public SCMS dataset |
| `analysis/baselines.py` | the baseline study that drove the design decisions |
| `requirements.txt`, `environment-linuxone.yml` | dependencies |

## Run on LinuxONE (Snap ML)

```bash
git clone <repo-url> && cd v12e-delay-model
conda env create -f environment-linuxone.yml      # Python 3.12 + numpy/pandas/sklearn + snapml
conda activate v12e
python -c "import snapml; print(snapml.__version__)"

python v12e_model.py train --data data --out models --optimize-trees
```

The first line of output must say `backend: snapml`. If the environment file fails, IBM's
recommended route works too:

```bash
conda create -n v12e python=3.12 numpy scipy pandas scikit-learn -y   # or python=3.10
conda activate v12e && pip install snapml
```

## Run locally (any machine, scikit-learn backend)

```bash
pip install numpy pandas scikit-learn
python v12e_model.py train --data data --out models --backend sklearn
```

Apple-silicon Macs and Python 3.13 have no Snap ML wheel; the scikit-learn backend uses the same
features, folds, selection rule and metrics.

## What `train` does

1. **Model selection on rolling yearly folds** (no test rows): train 2010 → validate 2011,
   2010–11 → 2012, 2010–12 → 2013. 24 candidates (rounds × learning rate × depth × feature set),
   each an average of 3 seeds, plus a logistic benchmark.
   **Selection rule, fixed in advance:** highest mean recall at the monthly top 10%; among those
   within 0.01 of it, the best worst year; then the simplest.
2. **One evaluation on 2014–2015 test** with the chosen model trained on 2010–2013.
3. **Feature importance** by permutation on the 2013 fold.
4. **Production model**: same configuration refit on all labelled data 2010–2015, saved to
   `models/v12e_model.pkl` (plus PMML per tree ensemble under Snap ML). `--optimize-trees` lets
   Snap ML pick its fastest inference format, including the IBM Z on-chip AI accelerator when present.
5. Writes `models/report.json`, `models/RESULTS.md` and `models/test_alerts.csv`.

## Score new orders

```bash
python v12e_model.py score --model models/v12e_model.pkl --input new_orders.csv --out alerts.csv
```

`new_orders.csv` needs the v3 feature columns (`quantity`, `planned_lead_time_days`,
`prior_excess_*`, `prior_known_*`, `scheduled_delivery_date`, and `decision_date` when known),
built with the same preprocessing. Output adds `risk_score`, `flagged_monthly_top10` and
`expected_days_late_if_late`.

## Design decisions (from `analysis/baselines.py`)

- **Relative lane priors.** Each lane's rolling 365-day late rate minus the overall late rate in
  the same window (`prior_excess_*`). Raw priors lag the year-level late rate (it swung
  0.16 → 0.24 → 0.08 → 0.18 over 2010–13), so logistic models learned negative weights on them.
- **No one-hot country / vendor / site identities.** They memorized 2010–13 and ranked test
  shipments below random.
- **Training era 2010+.** 2006–09 had a 0–4% late rate and mostly missing order dates.
- **Monthly top-10% alert rule** instead of a fixed score threshold, which drifted to flag
  15–34% of test orders.
- **Days late is reported, not modelled.** No regressor beat the historical median.

## Results so far (scikit-learn backend, laptop run of the previous prototype)

Test 2014–2015, trained on 2010–2013: flagging 10% of each month's orders caught **26%** of late
shipments with **36%** precision (base late rate 14%), against 22% / 30% for logistic regression
and 10% / 14% at random; PR-AUC 0.353 (lift 2.5). `models/RESULTS.md` holds the numbers from your
own run.

Caveats: the test years were also viewed during the baseline study, so treat test numbers as
indicative; each validation year has ~130–230 late shipments, so differences under ~0.02 are noise.

## Data

SCMS Delivery History Dataset, USAID (public). Label: delivered after the scheduled date.
Prior late rates use only shipments delivered before each order's decision date; the notebook's
§8b checks this by brute force.
