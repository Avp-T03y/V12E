# V12 Engines — model results

Backend: **sklearn** · chosen on validation: **trees core r=100 lr=0.1  d=4** · beats logistic benchmark on folds: **True**

## Test 2014-2015 (trained on 2010-2013, scored once)

| model | recall @ monthly top 10% | precision | PR-AUC | lift |
|---|---|---|---|---|
| boosted (chosen) | 0.257 | 0.353 | 0.345 | 2.47 |
| logistic benchmark | 0.233 | 0.321 | 0.259 | 1.85 |
| route_excess_lookup | 0.262 | 0.359 | 0.291 | 2.08 |
| no_skill | 0.105 | 0.144 | 0.140 | 1.00 |

Flagging 10% of each month's orders catches **26%** of late shipments; **35%** of flags are late (base rate 14%).

## Validation folds (recall @ monthly top 10%)

| candidate | 2011 | 2012 | 2013 | mean | worst |
|---|---|---|---|---|---|
| logistic benchmark (full features) | 0.122 | 0.109 | 0.123 | 0.118 | 0.109 |
| trees full r=100 lr=0.03 d=3 | 0.105 | 0.173 | 0.192 | 0.157 | 0.105 |
| trees full r=300 lr=0.03 d=3 | 0.130 | 0.173 | 0.165 | 0.156 | 0.130 |
| trees core r=100 lr=0.1  d=4 | 0.158 | 0.139 | 0.171 | 0.156 | 0.139 |
| trees core r=100 lr=0.03 d=3 | 0.105 | 0.183 | 0.174 | 0.154 | 0.105 |
| trees full r=300 lr=0.1  d=2 | 0.122 | 0.178 | 0.161 | 0.153 | 0.122 |
| trees full r=100 lr=0.03 d=4 | 0.158 | 0.137 | 0.158 | 0.151 | 0.137 |
| trees core r=100 lr=0.03 d=4 | 0.158 | 0.135 | 0.157 | 0.150 | 0.135 |
| trees core r=300 lr=0.03 d=3 | 0.128 | 0.173 | 0.149 | 0.150 | 0.128 |

## Days late

Reported as the historical distribution: median 12 days, middle half 6-25. No regressor beat the median on the folds or on test.

## Caveats

- Test was also viewed during baseline analysis, so treat test numbers as indicative.
- Each validation year has roughly 130-230 late shipments; differences under ~0.02 are noise.
