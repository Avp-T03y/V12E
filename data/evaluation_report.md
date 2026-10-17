# V12 Engines - model evaluation (v12e-20261017-sklearn-8c040306)

Backend: **sklearn**. Test split: 2014-2015, late rate **14.0%** (no-skill PR-AUC = test late rate; project baseline 0.115).
Operating threshold chosen on a 2013 validation slice inside train (recall >= 0.7); the test set was scored once.

| Model | PR-AUC | Recall | Precision | Recall top 10% | Correct | Type I (false alarm) | Type II (missed late) |
|---|---|---|---|---|---|---|---|
| Logistic baseline | 0.197 | 51.4% | 21.9% | 7.0% | 67.5% | 25.7% | 6.8% |
| Boosted trees | 0.331 | 22.2% | 30.7% | 22.2% | 82.1% | 7.0% | 10.9% |

Confusion (boosted trees): {'tp': 79, 'fp': 178, 'fn': 277, 'tn': 2011}

## Days-late regressor
- MAE all rows: **4.77 d** (predict-0 baseline 2.52 d)
- MAE on truly late rows: **15.63 d** (predict-median-late baseline 11.96 d)
- Predicted days late is an estimate, not a guarantee.

## Feature importance (PR-AUC drop when shuffled, test set)
- fulfillment_route: 0.1052
- vendor: 0.0561
- prior_late_rate_vendor: 0.0384
- country: 0.0272
- prior_late_rate_country: 0.0166
- prior_late_rate_fulfillment_route: 0.0110
- shipment_mode: 0.0077
- quantity: 0.0021
- prior_known_vendor: 0.0009
- product_group: 0.0003
- managed_by: 0.0000
- decision_date_known: 0.0000
- prior_known_fulfillment_route: 0.0000
- prior_known_country: -0.0005
- planned_lead_time_days: -0.0042
- manufacturing_site: -0.0080
