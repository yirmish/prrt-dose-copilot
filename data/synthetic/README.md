# Synthetic demonstration cohort - these rows are not patients

The app's context tables. Generated on 2026-10-06 in two steps:

1. `research/hackathon_v35_run_2026-10-06/make_synthetic_cohort.py` (seed 20261006) fitted a
   **Gaussian copula** to a compact feature set of the study's de-identified cohort and drew new
   rows: rank-based normal scores, rank-correlation matrix projected to positive definite,
   back-transformation through smoothed quantile functions built from the *interior* order
   statistics (observed minimum and maximum dropped), 2% jitter, 5 significant figures. Lesions of
   one synthetic patient are correlated (rho = 0.75, close to the measured ICC 0.78). That
   full-width output is **not published** (it still has the columns removed in step 2).
2. `scripts/make_public_synthetic.py` removed columns the app does not use that act as
   quasi-identifiers in a small single-centre cohort (age, sex, height, PET-to-therapy interval,
   uptake time, camera), clipped body weight at 40 kg and rounded it to 5 kg, and fixed the
   kurtosis definition (see the script's docstring). Report: `public_derivation_report.csv`.

| file | rows | unit |
|---|---|---|
| `tumor_burden.csv`, `kidneys.csv`, `liver_healthy.csv`, `spleen.csv`, `bone_marrow.csv` | 300 each | one synthetic treatment |
| `tumors.csv` | 761 | one synthetic lesion; 200 synthetic patients (`synthetic_patient_id`) |

Columns: the 17 per-ROI PET features (`pet_*`), whole-body / liver / spleen uptake, body weight,
administered activity (`gen_prrt_net_injected_activity_gbq`), the target `dose_gy`, and
`dose_gy_per_gbq`. `is_synthetic = 1` on every row.

**Fidelity** (`synthetic_fidelity.csv`): the 2-term log-log baseline explains about as much
variance in-sample on synthetic as on real data (e.g. tumour burden 0.51 vs 0.47, kidneys 0.35 vs
0.36); median and IQR of dose match. Higher-order structure, era / scanner effects and the real
missingness mechanism are **not** preserved.

**Privacy** (`synthetic_privacy_check.csv`, computed before step 2): zero exact row matches; every
synthetic row is farther from its nearest real row than real rows typically are from each other
(median distance 0.81-0.93 vs 0.41-0.72 in normal-score space). A handful of single dose values
coincide with a real dose at 4 decimals in the targets with few significant digits - rounding
coincidence in one column, not a copied record.

**Use.** Numbers obtained by fitting on these tables demonstrate the mechanism; they are **not**
the study's results. The study's results are in `results/real_cohort/`.
