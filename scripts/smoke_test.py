"""smoke_test.py - check a TabPFN-3.5 back-end end to end on the synthetic cohort.

    python scripts/smoke_test.py --backend api        # needs TABPFN_TOKEN (Prior Labs API)
    python scripts/smoke_test.py --backend local      # needs `pip install -r requirements-local.txt`
                                                      # and the TabPFN-3.5 weights (see README)
    python scripts/smoke_test.py --backend baseline   # no dependencies beyond the app's
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from copilot import calibration as C                                   # noqa: E402
from copilot.engine import ACTIVITY, DoseModel, load_context, split_examples  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["api", "local", "baseline"], default="api")
ap.add_argument("--target", default="kidneys")
ap.add_argument("--cv", action="store_true", help="also run the 5-fold live calibration check")
a = ap.parse_args()
backend = {"api": "tabpfn-3.5-api", "local": "tabpfn-3.5-local", "baseline": "physical-baseline"}[a.backend]

ctx, ex = split_examples(load_context(a.target))
t0 = time.time()
m = DoseModel(a.target, ctx, backend=backend).fit()
print(f"fit   {time.time() - t0:5.1f}s  backend={backend}  context rows={len(ctx)}  features={len(m.features)}")
case = ex.iloc[0].to_dict()
t0 = time.time()
r = m.predict(case, float(case[ACTIVITY]))
lo, hi = r["interval80_gy"]
assert np.all(np.diff(r["quantiles_gy"]) >= -1e-9) and lo <= r["median_gy"] <= hi
print(f"pred  {time.time() - t0:5.1f}s  median {r['median_gy']:.2f} Gy  80% [{lo:.2f}, {hi:.2f}]  "
      f"measured (synthetic) {case['dose_gy']:.2f} Gy  ",
      {k: (round(v['p_exceed'], 3), v['call']) for k, v in r["thresholds"].items()})
if a.cv:                                   # identical folds for every model, as in the app
    for b in dict.fromkeys([backend, "physical-baseline", "gbm-quantile", "gbm-tuned-conformal"]):
        t0 = time.time()
        Q, y = C.oof_quantiles(a.target, load_context(a.target), b, k=5)
        rep = C.report(Q, y)
        print(f"cv    {time.time() - t0:5.1f}s  {b:20s} cov50 {rep['cov50']:.2f}  cov80 {rep['cov80']:.2f}  "
              f"cov95 {rep['cov95']:.2f}  width95 x{rep['width95_fold']:.1f}  log R2 {rep['log_r2']:.3f}")
print("SMOKE_OK")
