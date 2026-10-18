# Model / Decision Card: V12 Engines Delivery-Delay Predictor

> Figures come from the team's `metrics.json`, `health.json`, and `evidence.json` (Oct 18, 2026). Raw files: `docs/evidence/`.

## 1. Overview
- **Project:** V12 Engines, order-time delivery-delay early-warning engine for global health (HIV) supply chains
- **Model:** Snap ML 1.17.2 boosted-tree classifier. Selected config: `trees full r=100 lr=0.03 d=3`
- **Version labels:** the API reports `dev-eval` (evaluation model, file `v12e_eval_model.pkl`, trained on orders due 2010–2013, used by the demo). `metrics.json` reports `dev`. A production model (`v12e_model.pkl`, trained on 2010–2015, n = 7,305) exists but is not used in the replay. These are development labels, not release versions.
- **Deployment:** IBM LinuxONE (s390x, machine type 9175, IBM z17), z/VM guest, Docker Compose, container `api`
- **Purpose:** rank each order's delay risk at the moment it is placed, so a logistics officer can act. The model does not act on its own.

## 2. What it outputs
| Output | How it is produced | Notes |
|---|---|---|
| Risk rank (e.g. P94) | Snap ML score, ranked against orders placed in the previous 30 days | Top 10% = high, next 15% = elevated. **Not a calibrated probability** |
| Typical delay if late | Historical median and middle half of late orders due 2010–2013: 13 days (5–25) | **Not a per-order prediction.** No regressor shipped: none beat the median (its test MAE is 11.74 days) |
| Runway ("Promised in N days") | Promised delivery date minus order date | long >120 d · medium 45–120 d · short <45 d |
| Explanation | Template built from the score and top drivers | IBM Granite tested (about 2.9 s per sentence), **off in the demo** |

## 3. Data
- **Source:** USAID/PEPFAR SCMS shipments: 10,324 shipments, 2006–2015, 43 countries. HIV treatment drugs (ARVs) and test kits, plus a small malaria subset (46 rows).
- **Late rate:** 11.5% overall (train 10.7%, test 14.0%). From RDC 17.2% vs Direct Drop 5.3%; From RDC is 78% of late shipments.
- **Split (by promised delivery date):** trained on orders due 2010–2013, with selection on rolling folds due 2011, 2012, 2013; tested on orders due 2014–2015, scored once.
- **Headline evaluation set:** the 1,734 test orders **placed after 25 Feb 2014**, the last outcome in the training data (164 late, 9.5%). The other 811 (192 late, 23.7%) were placed earlier, so the model had learned from outcomes that came after they were placed. All-orders figures are shown for comparison.
- **Features (9):** `prior_excess_vendor`, `prior_excess_country`, `prior_excess_fulfillment_route`, `prior_known_vendor`, `prior_known_country`, `prior_known_fulfillment_route`, `quantity`, `planned_lead_time_days`, `lead_time_missing`. Past-performance priors use only shipments delivered before each order's cutoff (leakage-safe).
- **Reload check:** scores are identical after save → reload (max difference 0.0 across 2,545 rows).

## 4. Evaluation (test, orders due 2014–2015)

**Orders placed after the training data (live rule: top 10% of the previous 30 days; 1,734 orders, 164 late)**

| Approach | Flagged | Late caught | Alerts that were late | PR-AUC | Lift |
|---|---|---|---|---|---|
| Snap ML boosted trees (final) | 12.3% | 37.8% (62 of 164) | 29.1% (62 of 213) | 0.28 | 2.91× |
| Logistic regression | 12.0% | 38.4% | 30.3% | 0.27 | 2.81× |
| Route late-rate lookup | 17.0% | 44.5% | 24.8% | 0.28 | 2.92× |
| Vendor late-rate lookup | 14.1% | 44.5% | 29.9% | 0.30 | 3.14× |
| No skill (random ranking) | 10.0% | 10.0% | 9.5% | 0.09 | 1.00× |

No skill is a random ranking flagging the same top 10%; its alerts are late at the base rate. The final model is **comparable to simple baselines** on this set; the vendor lookup has a higher PR-AUC.

**All orders due 2014–2015 (2,545 orders, late rate 14.0%)**

| Approach | PR-AUC | Lift (month-end) |
|---|---|---|
| Snap ML boosted trees (final) | 0.36 | 2.55× |
| Logistic regression | 0.26 | 1.85× |
| Route late-rate lookup | 0.29 | 2.08× |
| No skill (random ranking) | 0.14 | 1.00× |

Live rule on all orders: flagged 12.2%, caught 26.4% (94 of 356), alerts that were late 30.3% (94 of 310).

**Confusion matrices (counts)**

| Cohort and rule | Caught (TP) | Missed (FN) | False alarms (FP) | Correct on-time (TN) |
|---|---|---|---|---|
| Placed after training data, live rule | 62 | 102 | 151 | 1,419 |
| Placed after training data, month-end best case | 63 | 101 | 111 | 1,459 |
| All 2014–2015, live rule | 94 | 262 | 216 | 1,973 |
| All 2014–2015, month-end best case | 98 | 258 | 161 | 2,028 |

**Error breakdown (shares of the test set)**

| Cohort and rule | Correct | Type I (false alarm) | Type II (missed late) |
|---|---|---|---|
| Placed after training data, live | 85.4% | 8.7% | 5.9% |
| Placed after training data, month-end | 87.8% | 6.4% | 5.8% |
| Flag nothing (reference) | 90.5% | 0% | 9.5% |
| All 2014–2015, live | 81.2% | 8.5% | 10.3% |
| All 2014–2015, month-end | 83.5% | 6.3% | 10.1% |
| Flag nothing (reference) | 86.0% | 0% | 14.0% |

"Correct" is lower than flagging nothing because most orders are on time, so accuracy is not our headline. We favor recall: a missed late shipment can mean a clinic stockout, while a false alarm costs a follow-up call.

**What drives the predictions** (drop in PR-AUC when a feature is shuffled): `prior_excess_vendor` +0.012, `prior_excess_country` +0.005, `planned_lead_time_days` +0.003; `quantity` and `prior_excess_fulfillment_route` slightly negative. These effects are small, so treat importance as indicative.

## 5. Intended use and out-of-scope
- **Intended:** decision support for logistics officers prioritizing which orders to review at order time.
- **Not for:** automated decisions, patient-level decisions, or guarantees of delivery times.

## 6. Limitations
- **Replay, not live:** the demo replays historical orders (placed from 2014-02-26) as if placed today. In deployment the model would retrain on current data.
- **Ranking, not probability:** the score is not calibrated.
- **Selection and test overlap:** model selection on the validation years has the same overlap with training outcomes and was not re-run. Test years were also viewed during the baseline study, so test numbers are indicative.
- **Cold start:** 98 headline-set orders (122 in all orders) arrived when fewer than 30 orders had been placed in the previous 30 days and were ranked against a fixed reference distribution.
- **Non-stationary:** late rates swing by year (for example 23.5% in 2011 vs 7.9% in 2012).
- **No dispatch date, in-transit location, or temperature**, so no transit-time or cold-chain modeling.
- **Impact** is an aim, not a measured outcome.

## 7. Platform and security
- Verified: s390x, machine type 9175 (IBM z17), z/VM 7.3.0, 3 vCPUs, 7.8 GB memory, Snap ML 1.17.2, Python 3.12.15, Docker Compose with 3 containers.
- Accelerator (NNPA): available (CPU flag), **not used by the model**.
- Latency: median 107.0 ms, p95 114.0 ms, max 144.6 ms over 976 orders, measured inside the API.
- **TLS: not enabled.** Only the web container is published, bound to `127.0.0.1:8080`; access is through the SSH tunnel. Quantum-safe cryptography: not demonstrated (roadmap).
- Data is de-identified and shipment-level. Officer reviews are logged.
