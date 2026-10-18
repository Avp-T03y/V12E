# V12 Engines - model evaluation (v12e-20261017-2215-snapml-92c49f21)

Backend: **snapml**. Test split: 2014-2015, late rate **14.0%** (no-skill PR-AUC = test late rate; project baseline 0.115).
Operating threshold chosen on a 2013 validation slice inside train (recall >= 0.7); the test set was scored once.

| Model | PR-AUC | Recall | Precision | Recall top 10% | Correct | Type I (false alarm) | Type II (missed late) |
|---|---|---|---|---|---|---|---|
| Logistic baseline | 0.200 | 51.1% | 22.0% | 5.9% | 67.8% | 25.3% | 6.8% |
| Boosted trees | 0.322 | 41.9% | 32.6% | 24.4% | 79.8% | 12.1% | 8.1% |

Confusion (boosted trees): {'tp': 149, 'fp': 308, 'fn': 207, 'tn': 1881}

## Days-late regressor (if late, about how many days)
- Chosen by rolling validation on late rows of 2012-2013 (not test): **median** (validation MAE: median 16.82 d, trees_raw 27.50 d, trees_log 17.99 d)
- Test MAE on the 356 late shipments: **11.71 d** | global-median baseline 11.71 d | route-median baseline 12.24 d
- Predicted days late is an estimate, not a guarantee.

## Feature importance (PR-AUC drop when shuffled, test set)
- fulfillment_route: 0.0975
- vendor: 0.0381
- prior_late_rate_vendor: 0.0298
- country: 0.0249
- planned_lead_time_days: 0.0236
- quantity: 0.0139
- shipment_mode: 0.0105
- prior_late_rate_fulfillment_route: 0.0088
- product_group: 0.0072
- prior_late_rate_country: 0.0059
- prior_known_vendor: 0.0008
- prior_known_country: 0.0002
- managed_by: 0.0000
- decision_date_known: 0.0000
- prior_known_fulfillment_route: 0.0000
- manufacturing_site: -0.0035
