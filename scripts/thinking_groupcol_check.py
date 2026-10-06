"""thinking_groupcol_check.py - does TabPFN-3.5 thinking mode with `group_col` fix lesion-level
interval coverage?

Known weakness (real cohort): lesion-level intervals are too narrow (nominal 95% covers 0.86-0.89),
because lesions of one patient share their error (ICC 0.78) and the model treats rows as
independent. tabpfn-client >= 0.6 offers `group_col` in thinking mode: "the rows of one group are
never split between training and validation" during the fit. This script asks one question, decided
before running: on the SYNTHETIC lesion table (200 synthetic patients, imposed within-patient
correlation), does thinking + group_col move the 80% / 95% coverage of held-out PATIENTS closer to
nominal than the default fit, on identical folds?

Cost: 5 thinking fits (the API allows a limited number per month) + 5 default fits. Synthetic data
only. Report whatever it shows - a null is a result.

    python scripts/thinking_groupcol_check.py            # needs TABPFN_TOKEN
    python scripts/thinking_groupcol_check.py --folds 3 --timeout 300
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from copilot import calibration as C                                     # noqa: E402
from copilot.engine import GROUP, LEVELS, feature_columns, load_context, log_target  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--timeout", type=float, default=600, help="thinking budget per fit, seconds")
ap.add_argument("--effort", default="medium", choices=["medium", "high"])
a = ap.parse_args()

from tabpfn_client import TabPFNRegressor  # noqa: E402

d = load_context("tumors")
y = log_target(d)
feats = feature_columns(d)
X = d[feats + [GROUP]].copy()
folds = list(GroupKFold(n_splits=a.folds, shuffle=True, random_state=0).split(X, y, d[GROUP]))


def run(thinking: bool):
    Q = np.full((len(d), len(LEVELS)), np.nan)
    t0 = time.time()
    for i, (tr, te) in enumerate(folds):
        if thinking:
            m = TabPFNRegressor.create_default_for_version(
                "v3.5", n_estimators=8, random_state=0, thinking_effort=a.effort,
                thinking_timeout_s=a.timeout, group_col=GROUP)
            Xtr, Xte = X.iloc[tr], X.iloc[te]
        else:
            m = TabPFNRegressor.create_default_for_version("v3.5", n_estimators=8, random_state=0)
            Xtr, Xte = X.iloc[tr][feats], X.iloc[te][feats]
        m.fit(Xtr, y[tr])
        qs = m.predict(Xte, output_type="quantiles", quantiles=[float(l) for l in LEVELS])
        Q[te] = np.maximum.accumulate(np.column_stack([np.asarray(q, float).ravel() for q in qs]), axis=1)
        print(f"  {'thinking+group_col' if thinking else 'default'} fold {i + 1}/{len(folds)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return C.report(Q, y)


out = {"default": run(False), "thinking_group_col": run(True)}
rows = [{"arm": k, "n": v["n"], "cover50": v["cov50"], "cover80": v["cov80"], "cover95": v["cov95"],
         "width95_fold": v["width95_fold"], "crps_log": v["crps_log"], "log_r2": v["log_r2"]}
        for k, v in out.items()]
res = pd.DataFrame(rows).round(3)
print(res.to_string(index=False))
dst = ROOT / "results" / "synthetic_experiments"
dst.mkdir(parents=True, exist_ok=True)
res.to_csv(dst / "thinking_groupcol_lesions.csv", index=False)
(dst / "thinking_groupcol_lesions.json").write_text(json.dumps(
    {"folds": a.folds, "effort": a.effort, "timeout_s": a.timeout, "data": "data/synthetic/tumors.csv",
     "note": "synthetic cohort; demonstrates a mechanism, not a study result"}, indent=2))
print(f"wrote {dst}")
