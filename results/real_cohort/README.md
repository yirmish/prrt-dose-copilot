# Real-cohort results - aggregates only

Single centre, 86 cycle-1 ¹⁷⁷Lu-DOTATATE treatments in 82 patients, 309 lesions; absorbed dose from
post-therapy SPECT/CT (single-time-point method). Pre-therapy ⁶⁸Ga-DOTATATE PET features.
No row-level table is published. One figure, `fig_pred_interval_vs_measured_M1.png`, plots measured against
predicted dose as unlabelled points, without identifiers.

## `v35_2026-10-06/` - TabPFN-3.5 predictive distributions (run of 2026-10-06)

Design: 5 x 5-fold StratifiedGroupKFold **grouped by patient**, seeds 0-4 (the first 5 of the
protocol's 20 repeats); target log(Gy/GBq); TabPFN-3.5 (`tabpfn` 9.0.0, checkpoint
`tabpfn-v3.5-20260909.safetensors`, SHA-256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`),
8 estimators, softmax temperature 0.9, CPU, float32, **no tuning**, offline. Comparators (physical
baseline, CatBoost quantile heads, TabPFN-2) on **identical folds**. Settings were declared before
the run. **Status: 5 of 20 repeats - a preliminary measurement, to be
confirmed.**

| file | content |
|---|---|
| `V35_DISTRIBUTION_NUMBERS_2026-10-06.md` | M1 tables: accuracy, coverage, interval span, CRPS, PIT; paired deltas; decisive calls; identity gate |
| `v35_summary.csv`, `v35_paired_deltas.csv`, `v35_decisive_calls.csv`, `v35_reliability_curve.csv`, `v35_gate.csv` | the same, machine-readable (M1 = 17 PET features of the organ / lesion) |
| `fig_reliability_M1.png`, `fig_pred_interval_vs_measured_M1.png` | figures |
| `v35_M3_summary_from_results_raw.csv`, `v35_M3_comparisons_vs_M0.csv` | M3 (~165 features: all organs' PET + treatment + body). Medians over the 5 repeats of the run's per-repeat result files and its patient-clustered bootstrap against the physical baseline. Computed after `analyse_v35.py` (the M3 cells finished later). The comparison with CatBoost is in the `v35_M3_all_arms_*` files below |
| `v35_M3_all_arms_summary.csv`, `v35_M3_all_arms_paired_deltas.csv`, `v35_M3_all_arms_decisive_calls.csv`, `v35_M3_all_arms_reliability_curve.csv`, `v35_M3_all_arms_gate.csv`, `fig_reliability_M3.png` | M3, all arms: `analyse_v35.py` re-run on all 13 units after the M3 cells finished. The stored CatBoost and TabPFN-2 arms are on identical rows, identical folds (CV seeds 0-4) and the identical feature list (165 features; 164 for spleen), checked from the run manifests and recorded in the gate file. CatBoost is stored for 4 of the 6 M3 targets |
| `proper_scores__coverage_with_ci.csv`, `proper_scores__paired_crps_and_sharpness.csv`, `proper_scores__brier_threshold_probabilities.csv`, `proper_scores__risk_coverage_decisive_calls.csv`, `proper_scores__summary_stats.txt`, `proper_scores__flagged_rows.csv` | **Post-hoc** (written after the M3 point results were seen; not pre-registered; no model refitted), from the stored out-of-fold quantiles, M1 and M3: patient-clustered bootstrap (B = 2000) on coverage, paired CRPS and 95%-interval-width ratio, Brier score of P(dose > threshold), and decisive-call fraction and accuracy at cut-offs 0.6-0.95. Coverage here is the mean over repeats; the summary files give the median. Code: `research/hackathon_v35_run_2026-10-06/proper_scores_paired.py` |
| `run__run.log`, `run__run_config.json`, `run__summary_stats.txt` | provenance: versions, settings, rows in = rows out per unit |

## `validation/` - what the validation scheme does to R²

`leakage_summary.csv`: same model and data, random split (lesions of one patient on both sides)
vs split by patient, 50 repeats. Kidneys (about one row per patient) are the negative control.

## Known issues

- Lesion-level predictive intervals of TabPFN-3.5 are too narrow (nominal 95% covers 0.86-0.89)
  because lesions of one patient share their error (ICC 0.78).
- The study pipeline's `pet_kurtosis_excess` carries a constant -3 offset (the PET software
  already reports excess kurtosis). Tree models are invariant to it; for TabPFN's preprocessing the
  effect is expected to be negligible but was not tested. The public demo tables undo it.
