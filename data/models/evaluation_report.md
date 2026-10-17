# V12 Engines - model evaluation (v12e-20261017-2008-snapml-8c040306)

Backend: **snapml**. Test split: 2014-2015, late rate **14.0%** (no-skill PR-AUC = test late rate; project baseline 0.115).
Operating threshold chosen on a 2013 validation slice inside train (recall >= 0.7); the test set was scored once.

| Model | PR-AUC | Recall | Precision | Recall top 10% | Correct | Type I (false alarm) | Type II (missed late) |
|---|---|---|---|---|---|---|---|
| Logistic baseline | 0.197 | 50.3% | 21.9% | 7.0% | 68.0% | 25.1% | 7.0% |
| Boosted trees | 0.344 | 28.1% | 37.7% | 26.7% | 83.5% | 6.5% | 10.1% |

Confusion (boosted trees): {'tp': 100, 'fp': 165, 'fn': 256, 'tn': 2024}

## Days-late regressor (if late, about how many days)
- Chosen by rolling validation on late rows of 2012-2013 (not test): **median** (validation MAE: median 16.82 d, trees_raw 26.93 d, trees_log 19.17 d)
- Test MAE on the 356 late shipments: **11.71 d** | global-median baseline 11.71 d | route-median baseline 12.24 d
- Predicted days late is an estimate, not a guarantee.

## Feature importance (PR-AUC drop when shuffled, test set)
- fulfillment_route: 0.1097
- prior_late_rate_vendor: 0.0412
- vendor: 0.0342
- country: 0.0318
- prior_late_rate_country: 0.0278
- quantity: 0.0255
- planned_lead_time_days: 0.0241
- prior_late_rate_fulfillment_route: 0.0169
- shipment_mode: 0.0108
- manufacturing_site: 0.0052
- product_group: 0.0041
- prior_known_vendor: 0.0014
- managed_by: 0.0000
- decision_date_known: 0.0000
- prior_known_fulfillment_route: 0.0000
- prior_known_country: -0.0003
