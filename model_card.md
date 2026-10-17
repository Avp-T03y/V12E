# Model / Decision Card: V12 Engines Delivery-Delay Predictor

> Fill every `[TBD]` from the Model owner's real `/metrics` output and `model_version`. Do not estimate.

## 1. Overview
- **Project:** V12 Engines, order-time delivery-delay early-warning engine for global health (HIV) supply chains
- **Model version:** `[TBD: exact model_version string from /srv/v12e/models]`
- **Deployment target:** IBM LinuxONE (s390x, no GPU), Docker container `scoring-api`
- **Purpose:** at order time, estimate (a) the probability a shipment will be late and (b) how many days late. A logistics officer reviews the alert and decides; the model does not act on its own.

## 2. What it predicts
| Output | Model | Definition |
|---|---|---|
| `late_probability` (0–1) | Snap ML boosted-tree classifier | late = delivered date > scheduled delivery date |
| `predicted_days_late` (int) | Snap ML boosted-tree regressor | days late (0 if on time). **An estimate, not a guarantee** |
| `explanation`, `recommended_action` | IBM Granite (explanation layer) | Plain-language text grounded only in the model's `top_drivers`, matched to the runway tier |

Runway tiers: long (>120 days) · medium (45–120) · short (<45). Risk band is set by probability percentile; the dashboard focuses on the top ~10% to limit alert fatigue.

## 3. Data
- **Source:** USAID/PEPFAR SCMS shipment data: 10,324 shipments, 2006–2015, 43 countries. HIV treatment drugs (ARVs) and HIV test kits, plus a small malaria subset (46 rows).
- **Late rate:** 11.5% overall (no-skill PR-AUC baseline = 0.115). From RDC 17.2% vs Direct Drop 5.3%; From RDC is 78% of all late shipments.
- **Split (chronological, by scheduled date):** train ≤ 2013 (7,779 rows, 830 late) / test 2014–2015 (2,545 rows, 356 late). No shuffling or re-splitting.
- **Features (16, order-time only):** country, vendor, fulfillment_route, shipment_mode, product_group, managed_by, manufacturing_site, quantity, planned_lead_time_days, decision_date_known, prior_late_rate_{country, vendor, fulfillment_route}, prior_known_{country, vendor, fulfillment_route}.
- **Leakage control:** `prior_late_rate_*` uses only shipments delivered before each row's cutoff, within a 365-day window, smoothed toward the global prior (M = 20). Cold start falls back to 0.115 with `prior_known = 0`.
- **Excluded:** package weight (not known at order time).
- **Handling gaps:** blank shipment mode → "Unknown"; missing decision date → scheduled date with `decision_date_known = 0`.

## 4. Training and evaluation
- **Baseline:** class-weighted Logistic Regression.
- **Main models:** Snap ML boosted trees (classifier + regressor). Imbalance handled with class weights and threshold tuning.
- **Operating threshold:** `[TBD]`, chosen to favor recall.
- **Headline metrics (test split 2014–2015):** PR-AUC and recall at the threshold (plus precision), not accuracy.

| Metric | Baseline | Snap ML |
|---|---|---|
| PR-AUC (no-skill 0.115) | [TBD] | [TBD] |
| Recall at threshold | [TBD] | [TBD] |
| Precision at threshold | [TBD] | [TBD] |
| Recall in top 10% by risk | [TBD] | [TBD] |
| Days-late MAE | n/a | [TBD] days |

**Error breakdown (share of test set):** Correct [TBD %] · Type I, false alarm [TBD %] · Type II, missed late [TBD %]. Type II is the costliest (a missed warning can mean interrupted treatment), so we favor recall; Type I contributes to alert fatigue.

## 5. Intended use and out-of-scope
- **Intended:** decision support for logistics officers prioritizing which shipments to review at order time.
- **Not for:** automated decisions, patient-level decisions, or claims of guaranteed delivery times.

## 6. Limitations
- **Replay, not live:** the demo replays 2006–2015 shipments as if live. In deployment the models retrain on current data.
- **Non-stationary:** late rates swing by year (e.g. From RDC 5% → 37% → 10% → 35%), so performance on future data may differ.
- **Order-time features only:** no dispatch date, in-transit location, or temperature, so no transit-time or cold-chain modeling.
- **Partly missing anchor dates:** PO date is absent for 55.5% of rows; planned lead time is unavailable for ~19% (flagged, not dropped).
- **Explanations:** Granite output is validated against the JSON schema and the drivers; if it times out or fails validation, a templated message built from the numbers is returned.
- **Impact** is an aim and a proxy, not a measured outcome.

## 7. Governance and security
- De-identified, shipment-level data (no patient data).
- The human-in-the-loop log (status: New → Acknowledged → Actioned → Dismissed, reviewer, time) is the audit trail.
- Security items are listed as Verified only if enabled and checked by the Platform owner; quantum-safe is roadmap.
