# Models and results produced on IBM LinuxONE

Copied from /srv/v12e/models on the team VM (IBM z17, machine type 9175, s390x, big-endian).
Trained and evaluated in the notebook container with Snap ML 1.17.2 via v12e_model.py.

- v12e_eval_model.pkl: trained on orders due 2010-2013; used by the live demo (replay).
- v12e_model.pkl: trained on orders due 2010-2015; for scoring new orders.
- metrics.json, report.json, RESULTS.md: evaluation results (headline: 1,734 orders placed after 2014-02-25).
- test_alerts.csv: alerts on the test set.
- reload_check*.npy: saved-and-reloaded scores; the reload check passed with difference 0.
- *.pmml: portable model exports.

The pickles were written on a big-endian machine; load them on s390x with v12e_model.py importable.
To regenerate: run `python v12e_model.py train` and then `evaluate` (see docs/DEPLOY.md or the README).
