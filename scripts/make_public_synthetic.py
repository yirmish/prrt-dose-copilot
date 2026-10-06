"""make_public_synthetic.py - derive the PUBLIC demo tables from the synthetic cohort.

Input : <SYNTH_SOURCE>/<target>_synthetic.csv - the full-width Gaussian-copula cohort made by
        research/hackathon_v35_run_2026-10-06/make_synthetic_cohort.py. NOT published: it still
        carries the quasi-identifier columns this script removes. Set SYNTH_SOURCE to its folder.
Output: data/synthetic/<target>.csv  - what the app reads.

Two changes, both made before publication on 2026-10-06:
  1. Disclosure minimisation. Columns the app does not use and that act as quasi-identifiers in a
     small single-centre rare-disease cohort are dropped: age, sex, height, PET-to-therapy interval,
     PET uptake time, camera code. Body weight (used by the bone-marrow baseline) is clipped to
     >= 40 kg and rounded to 5 kg.
  2. Kurtosis definition. The source column `pet_kurtosis_excess` was produced by subtracting 3
     from the PET software's kurtosis on the assumption that it was Pearson kurtosis. The values
     violate the bound excess >= skewness^2 - 2 by up to ~2.7 units, i.e. the software value was
     already excess kurtosis. The subtraction is undone and the column renamed `pet_kurtosis`
     (excess kurtosis as reported by the PET software). Before the fix 55-98% of rows violated the
     bound; after it 0-12%, the residue being rows where the copula paired skewness and kurtosis
     values that never co-occur in real data; those are clamped to the bound.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("SYNTH_SOURCE", ROOT / "data" / "synthetic_source_2026-10-06"))
OUT = ROOT / "data" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)
DROP = ["gen_age_at_prrt_years", "gen_gender_code_1m_2f", "pt_height_m", "gen_days_pet_to_prrt",
        "gen_pet_uptake_time_minutes", "gen_pet_camera_code_1mi_2dr"]
TARGETS = ["tumor_burden", "tumors", "kidneys", "liver_healthy", "spleen", "bone_marrow"]

rows = []
for t in TARGETS:
    d = pd.read_csv(SRC / f"{t}_synthetic.csv")
    n_in = len(d)
    d = d.drop(columns=[c for c in DROP if c in d.columns])
    d["pt_weight_kg"] = (np.clip(d["pt_weight_kg"], 40, None) / 5).round() * 5
    d["pet_kurtosis"] = d.pop("pet_kurtosis_excess") + 3.0
    floor = d["pet_skewness"] ** 2 - 2
    viol = int((d["pet_kurtosis"] < floor - 1e-9).sum())
    # The copula samples skewness and kurtosis without their joint constraint, so a few synthetic
    # rows still fall below the bound after the fix: clamp them to it (counted in the report).
    d["pet_kurtosis"] = np.maximum(d["pet_kurtosis"], floor + 1e-5).round(5)
    cols = ["is_synthetic", "synthetic_patient_id"] + sorted(c for c in d.columns if c.startswith("pet_")) \
        + [c for c in d.columns if c.startswith("voi_")] + ["pt_weight_kg", "gen_prrt_net_injected_activity_gbq",
                                                           "dose_gy", "dose_gy_per_gbq"]
    d = d[cols]
    d.to_csv(OUT / f"{t}.csv", index=False)
    rows.append({"target": t, "rows_in": n_in, "rows_out": len(d), "columns_out": d.shape[1],
                 "kurtosis_rows_clamped_to_bound": viol,
                 "weight_min_kg": float(d.pt_weight_kg.min()), "weight_max_kg": float(d.pt_weight_kg.max())})
rep = pd.DataFrame(rows)
rep.to_csv(OUT / "public_derivation_report.csv", index=False)
print(rep.to_string(index=False))
