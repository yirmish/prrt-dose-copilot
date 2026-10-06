# Lesion-level dose prediction: TabPFN-3.5 against seven other models (real cohort, aggregates)

**Status: discovery result plus a registered robustness check that passed. Not yet confirmed on
independent patients.** Read the caveats before quoting.

## Design
- **Rows:** 307 lesions in 85 cycle-1 treatments of 81 patients (single centre). Treatment-level
  targets: 85 treatments (spleen 70).
- **Features:** the study's "M3" pre-therapy table, **140 columns**: the lesion's own PET statistics,
  PET statistics of the whole body and of other organs, treatment and body-size covariates. The feature
  manifest was frozen (SHA-256 `04a01a82…`) before any dose value was read.
- **Target:** log(absorbed dose / administered activity), dose from post-therapy SPECT/CT.
- **Validation:** 5 × 5-fold cross-validation **grouped by patient**, identical folds and rows for every
  model; patient-level paired bootstrap (B = 4000) for every difference.
- **Comparators, each nested-tuned** over a fixed 4-configuration grid by an inner patient-grouped
  3-fold CV on training rows only; predictive intervals by split-conformal calibration on the inner
  out-of-fold residuals:

  | model | fixed settings | inner grid |
  |---|---|---|
  | CatBoost | 300 iterations, lr 0.05 | depth {4, 6} × l2 {3, 10} |
  | XGBoost | 300 trees, lr 0.05, subsample/colsample 0.8 | max_depth {3, 6} × λ {1, 10} |
  | HistGradientBoosting | 300 iterations, lr 0.05 | max_leaf_nodes {8, 31} × l2 {0, 1} |
  | LightGBM | 300 trees, lr 0.05 | num_leaves {7, 31} × λ {0, 10} |
  | Random forest | 300 trees | max_features {0.33, 1.0} × min_samples_leaf {1, 5} |
  | Elastic net | standardised | α {0.03, 0.3} × l1_ratio {0.2, 0.8} |

- **TabPFN-3.5:** `tabpfn` 9.0.0, `ModelVersion.V3_5`, checkpoint SHA-256 `ece4d67e…f0be3`, 8 estimators
  (32 in the registered check), float32, CPU, **no tuning, native quantiles (no calibration step)**.
  TabPFN-2 (open weights) is run the same way.

## Results
- `r2_by_model_and_target.csv`: R² (mean over repeats of the pooled out-of-fold R², log scale) for all
  eight models and six targets. At **lesion level TabPFN-3.5 is first (0.451; best tuned tree 0.357)**.
  At treatment level (n = 85; spleen 70) it is first on kidneys, healthy liver and spleen, not on tumour
  burden or bone marrow; treatment-level differences are not resolvable at this sample size.
- `lesion_head_to_head_tabpfn35.csv`: paired differences, TabPFN-3.5 minus each model, with 95% patient
  bootstrap CIs, for overall R², **within-patient R²** (patient-centred observed vs predicted over
  patients with ≥ 2 lesions) and CRPS. Against every tuned model the CI of ΔR² excludes 0; ΔCRPS
  favours TabPFN-3.5 with CIs excluding 0; within-patient likewise except the elastic net (touches 0).
  Against TabPFN-2 nothing is resolvable.
- `lesion_components_and_calibration.csv`: between- and within-patient R², CRPS and interval coverage.
  TabPFN-3.5's **native** 80 / 95% intervals cover 0.80 / 0.94 with no calibration step; the tuned
  models need split-conformal calibration to reach nominal coverage.
- `n_estimators_8_vs_32.csv` + `Registration_HT_*.md`: the registered robustness check. The lesion-level
  advantage over CatBoost was registered as hypothesis H-T before re-running at 32 estimators; at 32 it
  is +0.079 [+0.011, +0.136] (verdict **Robust**); R² changes by ≤ 0.003 on every target.
- `fig_lesion_8models.png`: made by `scripts/make_evidence_figures.py` from these CSVs.

## Caveats (stated in the source report)
- **Post hoc.** The between-model comparison was not registered in advance; TabPFN-3.5 was the eighth
  model examined on these patients. The registration (`Registration_HT_…md`) fixed the robustness check
  (R-1, passed) and a confirmatory test on independent patients (C-1, not yet possible: ~100 new patients
  needed if the true advantage is +0.08).
- **Multiplicity.** 7 models × 6 targets × 3 metrics, unadjusted; the lesion-level pattern is uniform
  across all six conventional models.
- **Feature set matters.** With only the 17 PET features of the lesion itself, TabPFN-3.5 does *not*
  beat a tuned CatBoost at lesion level (`../v35_2026-10-06/v35_paired_deltas.csv`: −0.080 [−0.161,
  +0.011]); the advantage appears with the 140-feature table, where tuned trees do worse.
- **Two R² definitions.** The R² tables are the mean over repeats of pooled-OOF R²; the bootstrap Δ uses
  repeat-averaged predictions. They agree in sign and size.
- Row-level data are not public. Source: the study's amendment-R results report (2 Oct 2026, updated
  4 Oct with check R-1); numbers here are transcribed from it, without patient identifiers.
