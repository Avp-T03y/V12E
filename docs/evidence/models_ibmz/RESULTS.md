# V12 Engines — model results

Backend: **snapml** · chosen on validation: **trees full r=100 lr=0.03 d=3** · beats logistic benchmark on folds: **True**

## Test 2014-2015 (trained on 2010-2013, scored once)

| model | recall @ monthly top 10% | precision | PR-AUC | lift |
|---|---|---|---|---|
| boosted (chosen) | 0.271 | 0.372 | 0.356 | 2.55 |
| logistic benchmark | 0.233 | 0.321 | 0.259 | 1.85 |
| route_excess_lookup | 0.262 | 0.359 | 0.291 | 2.08 |
| no_skill | 0.105 | 0.144 | 0.140 | 1.00 |

Flagging 10% of each month's orders catches **27%** of late shipments; **37%** of flags are late (base rate 14%).

## Validation folds (recall @ monthly top 10%)

| candidate | 2011 | 2012 | 2013 | mean | worst |
|---|---|---|---|---|---|
| logistic benchmark (full features) | 0.122 | 0.109 | 0.123 | 0.118 | 0.109 |
| trees full r=300 lr=0.1  d=3 | 0.131 | 0.198 | 0.167 | 0.165 | 0.131 |
| trees full r=100 lr=0.1  d=4 | 0.134 | 0.188 | 0.167 | 0.163 | 0.134 |
| trees full r=100 lr=0.1  d=3 | 0.135 | 0.198 | 0.153 | 0.162 | 0.135 |
| trees core r=300 lr=0.1  d=4 | 0.126 | 0.163 | 0.194 | 0.161 | 0.126 |
| trees full r=100 lr=0.03 d=3 | 0.149 | 0.193 | 0.140 | 0.161 | 0.140 |
| trees core r=300 lr=0.1  d=3 | 0.131 | 0.178 | 0.156 | 0.155 | 0.131 |
| trees full r=300 lr=0.1  d=2 | 0.116 | 0.208 | 0.140 | 0.155 | 0.116 |
| trees core r=300 lr=0.1  d=2 | 0.134 | 0.183 | 0.140 | 0.152 | 0.134 |

## Days late

Reported as the historical distribution: median 12 days, middle half 6-25. No regressor beat the median on the folds or on test.

## Caveats

- Test was also viewed during baseline analysis, so treat test numbers as indicative.
- Each validation year has roughly 130-230 late shipments; differences under ~0.02 are noise.
