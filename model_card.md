# Model / Decision Card: V12 Engines Delivery-Delay Predictor

> Figures come from the finished dashboard's Model and Platform views (video, captured 2026-10-18). Fill the `[TBD]` items from the final `report.json`.

## 1. Overview
- **Project:** V12 Engines, order-time delivery-delay early-warning engine for global health (HIV) supply chains
- **Model:** Snap ML 1.17.2 boosted-tree classifier ("dev-eval"), trained on orders due 2010–2013. **Model version string:** `[TBD]`
- **Deployment:** IBM LinuxONE (s390x, machine type 9175 / IBM z17), Docker Compose, container `api`
- **Purpose:** rank each order's delay risk at the moment it is placed, so a logistics officer can act. The model does not act on its own.

## 2. What it outputs
| Output | How it is produced | Notes |
|---|---|---|
| Risk rank (e.g. P94) | Snap ML score, ranked against orders placed in the previous 30 days | Top 10% = high, next 15% = elevated. **Not a calibrated probability** |
| Typical delay if late (13 days; middle half 5–25) | Historical median and middle half of late orders due 2010–2013 | **Not a per-order prediction.** No regressor shipped: none beat the median |
| Runway ("Promised in N days") | Promised delivery date minus order date | long >120 d · medium 45–120 d · short <45 d |
| Explanation | Template built from the score and top drivers | IBM Granite tested (about 2.9 s per sentence), **off in the demo** |

## 3. Data
- **Source:** USAID/PEPFAR SCMS shipments: 10,324 shipments, 2006–2015, 43 countries. HIV treatment drugs (ARVs) and test kits, plus a small malaria subset (46 rows).
- **Late rate:** 11.5% overall (train 10.7%, test 14.0%). From RDC 17.2% vs Direct Drop 5.3%; From RDC is 78% of late shipments.
- **Split (by promised delivery date):** trained on orders due 2010–2013; tested on orders due 2014–2015 (2,545 orders, 356 late).
- **Headline evaluation set:** the 1,734 test orders **placed after 25 Feb 2014**, the last outcome in the training data (164 late, 9.5%). The other 811 were placed before it, so the model had learned from outcomes that came after they were placed.
- **Features:** order-time attributes and leakage-safe past-performance priors (only shipments delivered before each order's cutoff). Feature set: `[TBD: list from report.json]`.
- **Model selection:** three validation years (orders due 2011, 2012, 2013), each trained only on earlier orders. Rule fixed in advance: highest mean recall in the monthly top 10%, within 0.01 of the best worst-year, then simplest. Chosen: `trees full r=100 lr=0.03 d=3` (mean 16.1%, worst year 14.0%).

## 4. Evaluation (test, orders due 2014–2015)

**Orders placed after the training data (live 30-day rule; 1,734 orders, 164 late)**

| Approach | Flagged | Late caught | Alerts that were late | PR-AUC | Lift |
|---|---|---|---|---|---|
| Snap ML boosted trees (final) | 12.3% | 37.8% (62 of 164) | 29.1% (62 of 213) | 0.28 | 2.91× |
| Logistic regression | 12.0% | 38.4% | 30.3% | 0.27 | 2.81× |
| Route late-rate lookup | 17.0% | 44.5% | 24.8% | 0.28 | 2.92× |
| Vendor late-rate lookup | 14.1% | 44.5% | 29.9% | 0.30 | 3.14× |
| No skill | n/a | n/a | n/a | 0.09 | 1.00× |

The final model is **comparable to simple baselines** on this view (the vendor lookup has a higher PR-AUC).

**All orders due 2014–2015 (2,545 orders, late rate 14.0%)**

| Approach | PR-AUC | Lift (month-end) |
|---|---|---|
| Snap ML boosted trees (final) | 0.36 | 2.55× |
| Logistic regression | 0.26 | 1.85× |
| Route late-rate lookup | 0.29 | 2.08× |
| No skill | 0.14 | 1.00× |

**Error breakdown** (live rule, placed after training data; shares of 1,734, derived from the dashboard's counts): Correct 85.4% · Type I (false alarm) 8.7% · Type II (missed late) 5.9%. Month-end best case: caught 63, missed 101, false alarms 111, correct 1,459.

We favor recall because a missed late shipment can mean a clinic stockout, while a false alarm costs a follow-up call. Flagging more than 10% of orders catches more late ones at the cost of more false alarms.

## 5. Intended use and out-of-scope
- **Intended:** decision support for logistics officers prioritizing which orders to review at order time.
- **Not for:** automated decisions, patient-level decisions, or guarantees of delivery times.

## 6. Limitations
- **Replay, not live:** the demo replays historical orders as if placed today. In deployment the model would retrain on current data.
- **Ranking, not probability:** the score is not calibrated.
- **Test years were also viewed during the baseline study**, so test numbers are indicative.
- **Cold start:** 98 test orders arrived when fewer than 30 orders had been placed in the previous 30 days and were ranked against a fixed reference distribution. Some test countries and vendors do not appear in training.
- **Non-stationary:** late rates swing by year (for example 23.5% in 2011 vs 7.9% in 2012).
- **No dispatch date, in-transit location, or temperature**, so no transit-time or cold-chain modeling.
- **Impact** is an aim, not a measured outcome.

## 7. Platform and security (as shown on the Platform view)
- Verified: s390x, machine type 9175 (IBM z17), 3 vCPUs, 7.8 GB memory, Snap ML 1.17.2, Docker Compose with 3 containers.
- Accelerator (NNPA): available (CPU flag), **not used by the model** (Snap ML default tree format, not optimized).
- Latency: median 106.7 ms, p95 114.0 ms over 855 orders, measured inside the API.
- **TLS: not enabled.** Access is through the SSH tunnel only. Quantum-safe cryptography: not demonstrated (roadmap).
- Data is de-identified and shipment-level. Officer reviews are logged.
