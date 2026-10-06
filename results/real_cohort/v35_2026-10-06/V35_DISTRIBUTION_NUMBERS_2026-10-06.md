# TabPFN-3.5 predictive distributions on the PRRT cohort - numbers only

Generated 2026-10-06 14:46 by `analyse_v35.py` from the OOF quantile stores. **Not a protocol run** (5 of 20 repeats).

Target: log(Gy/GBq). Folds: 5-fold StratifiedGroupKFold on the person, seeds 0-4. `log_r2` is the median over repeats (min, max beside it). Comparators are the stored scaffold arms cut to the same 5 repeats (identical folds - see `v35_gate.csv`). `width95_fold` = exp(median 95% interval width): the multiplicative span of the interval (e.g. 6.0 means the upper bound is 6x the lower).

## 1. Accuracy and calibration

| model_size | target | arm | n_rows | n_persons | log_r2 | log_r2_min | log_r2_max | cov_50 | cov_80 | cov_95 | width80_fold | width95_fold | crps_log | pit_ks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | tumor_burden | CatBoost[phase12_catboost_fixed] | 86 | 82 | +0.428 | +0.359 | +0.438 | 0.105 | 0.302 | 0.500 | 1.65 | 2.26 | 0.439 | 0.316 |
| M1 | tumor_burden | M0(plan) | 86 | 82 | +0.445 | +0.408 | +0.459 | 0.477 | 0.814 | 0.953 | 5.84 | 15.03 | 0.392 | 0.054 |
| M1 | tumor_burden | TabPFN-V2@8[phase12_catboost_fixed] | 86 | 82 | +0.440 | +0.416 | +0.465 | 0.477 | 0.814 | 0.942 | 5.01 | 11.65 | 0.386 | 0.028 |
| M1 | tumor_burden | TabPFN-V3_5@8 | 86 | 82 | +0.456 | +0.438 | +0.467 | 0.465 | 0.791 | 0.942 | 4.78 | 11.00 | 0.384 | 0.045 |
| M1 | tumors | CatBoost[phase12_catboost_lesions] | 309 | 82 | +0.382 | +0.371 | +0.399 | 0.285 | 0.505 | 0.731 | 2.63 | 5.93 | 0.429 | 0.176 |
| M1 | tumors | M0(plan) | 309 | 82 | +0.372 | +0.360 | +0.382 | 0.492 | 0.819 | 0.955 | 6.20 | 16.44 | 0.404 | 0.023 |
| M1 | tumors | TabPFN-V3_5@8 | 309 | 82 | +0.291 | +0.234 | +0.308 | 0.411 | 0.683 | 0.893 | 4.39 | 10.24 | 0.437 | 0.061 |
| M1 | tumors_hepatic | M0(plan) | 194 | 63 | +0.308 | +0.298 | +0.321 | 0.510 | 0.804 | 0.943 | 6.36 | 17.10 | 0.418 | 0.044 |
| M1 | tumors_hepatic | TabPFN-V3_5@8 | 194 | 63 | +0.214 | +0.155 | +0.237 | 0.407 | 0.655 | 0.861 | 4.28 | 10.94 | 0.457 | 0.083 |
| M1 | kidneys | CatBoost[phase12_catboost_fixed] | 86 | 82 | +0.110 | +0.057 | +0.238 | 0.174 | 0.453 | 0.628 | 1.54 | 1.87 | 0.260 | 0.250 |
| M1 | kidneys | M0(organ3p) | 86 | 82 | +0.329 | +0.313 | +0.364 | 0.558 | 0.837 | 0.907 | 2.47 | 4.00 | 0.203 | 0.049 |
| M1 | kidneys | TabPFN-V2@8[phase12_catboost_fixed] | 86 | 82 | +0.257 | +0.208 | +0.319 | 0.500 | 0.791 | 0.919 | 2.37 | 3.84 | 0.214 | 0.045 |
| M1 | kidneys | TabPFN-V3_5@8 | 86 | 82 | +0.206 | +0.178 | +0.260 | 0.500 | 0.791 | 0.907 | 2.33 | 3.98 | 0.218 | 0.048 |
| M1 | liver_healthy | CatBoost[phase12_catboost_liver] | 86 | 82 | +0.210 | +0.068 | +0.253 | 0.105 | 0.221 | 0.407 | 1.43 | 1.88 | 0.434 | 0.381 |
| M1 | liver_healthy | M0(organ3p) | 86 | 82 | -0.009 | -0.040 | +0.028 | 0.442 | 0.791 | 0.953 | 5.88 | 15.19 | 0.406 | 0.062 |
| M1 | liver_healthy | TabPFN-V2@8[phase12_organ3p] | 86 | 82 | +0.266 | +0.231 | +0.295 | 0.547 | 0.802 | 0.930 | 4.50 | 10.20 | 0.346 | 0.069 |
| M1 | liver_healthy | TabPFN-V3_5@8 | 86 | 82 | +0.282 | +0.252 | +0.307 | 0.500 | 0.767 | 0.930 | 4.13 | 9.34 | 0.347 | 0.034 |
| M1 | spleen | M0(organ3p) | 72 | 68 | +0.120 | +0.065 | +0.141 | 0.472 | 0.806 | 0.958 | 3.57 | 7.04 | 0.285 | 0.065 |
| M1 | spleen | TabPFN-V2@8[phase12_spleen_gbq_m1] | 72 | 68 | -0.028 | -0.046 | +0.016 | 0.444 | 0.708 | 0.917 | 3.07 | 5.37 | 0.309 | 0.079 |
| M1 | spleen | TabPFN-V3_5@8 | 72 | 68 | +0.066 | +0.015 | +0.102 | 0.375 | 0.722 | 0.944 | 2.94 | 5.21 | 0.294 | 0.074 |
| M1 | bone_marrow | CatBoost[phase12_catboost_fixed] | 86 | 82 | +0.011 | -0.072 | +0.044 | 0.070 | 0.186 | 0.360 | 1.32 | 1.73 | 0.425 | 0.416 |
| M1 | bone_marrow | M0(organ3p) | 86 | 82 | +0.033 | -0.025 | +0.050 | 0.477 | 0.814 | 0.953 | 4.41 | 9.75 | 0.326 | 0.072 |
| M1 | bone_marrow | TabPFN-V2@8[phase12_catboost_fixed] | 86 | 82 | +0.033 | -0.076 | +0.065 | 0.465 | 0.779 | 0.977 | 4.07 | 8.72 | 0.327 | 0.059 |
| M1 | bone_marrow | TabPFN-V3_5@8 | 86 | 82 | +0.043 | -0.057 | +0.052 | 0.442 | 0.767 | 0.965 | 4.01 | 8.90 | 0.324 | 0.058 |


## 2. Paired differences in log R2 (person-clustered bootstrap, B=2000, on identical folds)

Margins: treatment level 0.15, lesion level 0.07 (0.09 with a clustered residual). A CI that includes 0, or a delta below the margin, is not a difference.

| model_size | target | comparison | delta_log_r2 | ci_lo | ci_hi | margin | ci_excludes_zero |
|---|---|---|---|---|---|---|---|
| M1 | bone_marrow | TabPFN-V3_5@8 minus M0(organ3p) | +0.009 | -0.106 | +0.132 | 0.15 | False |
| M1 | bone_marrow | TabPFN-V3_5@8 minus CatBoost[phase12_catboost_fixed] | -0.008 | -0.118 | +0.143 | 0.15 | False |
| M1 | bone_marrow | TabPFN-V3_5@8 minus TabPFN-V2@8[phase12_catboost_fixed] | +0.003 | -0.042 | +0.061 | 0.15 | False |
| M1 | kidneys | TabPFN-V3_5@8 minus M0(organ3p) | -0.110 | -0.203 | +0.018 | 0.15 | False |
| M1 | kidneys | TabPFN-V3_5@8 minus CatBoost[phase12_catboost_fixed] | +0.050 | -0.076 | +0.229 | 0.15 | False |
| M1 | kidneys | TabPFN-V3_5@8 minus TabPFN-V2@8[phase12_catboost_fixed] | -0.033 | -0.102 | +0.059 | 0.15 | False |
| M1 | liver_healthy | TabPFN-V3_5@8 minus M0(organ3p) | +0.274 | +0.085 | +0.458 | 0.15 | True |
| M1 | liver_healthy | TabPFN-V3_5@8 minus CatBoost[phase12_catboost_liver] | +0.082 | -0.031 | +0.196 | 0.15 | False |
| M1 | liver_healthy | TabPFN-V3_5@8 minus TabPFN-V2@8[phase12_organ3p] | +0.004 | -0.071 | +0.068 | 0.15 | False |
| M1 | spleen | TabPFN-V3_5@8 minus M0(organ3p) | -0.050 | -0.220 | +0.192 | 0.15 | False |
| M1 | spleen | TabPFN-V3_5@8 minus TabPFN-V2@8[phase12_spleen_gbq_m1] | +0.097 | +0.004 | +0.278 | 0.15 | True |
| M1 | tumor_burden | TabPFN-V3_5@8 minus M0(plan) | +0.027 | -0.063 | +0.126 | 0.15 | False |
| M1 | tumor_burden | TabPFN-V3_5@8 minus CatBoost[phase12_catboost_fixed] | +0.009 | -0.112 | +0.090 | 0.15 | False |
| M1 | tumor_burden | TabPFN-V3_5@8 minus TabPFN-V2@8[phase12_catboost_fixed] | +0.013 | -0.028 | +0.058 | 0.15 | False |
| M1 | tumors | TabPFN-V3_5@8 minus M0(plan) | -0.053 | -0.176 | +0.073 | 0.07 | False |
| M1 | tumors | TabPFN-V3_5@8 minus CatBoost[phase12_catboost_lesions] | -0.080 | -0.161 | +0.011 | 0.07 | False |
| M1 | tumors_hepatic | TabPFN-V3_5@8 minus M0(plan) | -0.085 | -0.248 | +0.057 | 0.07 | False |


## 3. Decisive calls at the scaffold's clinical thresholds

A call is decisive when the predictive distribution puts P(dose > threshold) >= 0.9 or <= 0.1. `decisive_accuracy` = share of decisive calls on the correct side.

| model_size | target | arm | threshold_gy | prevalence_above | decisive_fraction | decisive_accuracy | decisive_n_median |
|---|---|---|---|---|---|---|---|
| M1 | bone_marrow | TabPFN-V3_5@8 | 0.5 | 0.012 | 0.988 | 0.988 | 85.0 |
| M1 | bone_marrow | M0(organ3p) | 0.5 | 0.012 | 1.000 | 0.988 | 86.0 |
| M1 | bone_marrow | CatBoost[phase12_catboost_fixed] | 0.5 | 0.012 | 1.000 | 0.988 | 86.0 |
| M1 | bone_marrow | TabPFN-V2@8[phase12_catboost_fixed] | 0.5 | 0.012 | 1.000 | 0.988 | 86.0 |
| M1 | kidneys | TabPFN-V3_5@8 | 5.75 | 0.221 | 0.256 | 0.875 | 22.0 |
| M1 | kidneys | M0(organ3p) | 5.75 | 0.221 | 0.221 | 0.900 | 19.0 |
| M1 | kidneys | CatBoost[phase12_catboost_fixed] | 5.75 | 0.221 | 0.651 | 0.847 | 56.0 |
| M1 | kidneys | TabPFN-V2@8[phase12_catboost_fixed] | 5.75 | 0.221 | 0.256 | 0.909 | 22.0 |
| M1 | tumor_burden | TabPFN-V3_5@8 | 20.0 | 0.733 | 0.337 | 0.966 | 29.0 |
| M1 | tumor_burden | TabPFN-V3_5@8 | 30.0 | 0.593 | 0.186 | 0.938 | 16.0 |
| M1 | tumor_burden | TabPFN-V3_5@8 | 36.5 | 0.500 | 0.209 | 0.933 | 18.0 |
| M1 | tumor_burden | TabPFN-V3_5@8 | 40.0 | 0.465 | 0.233 | 0.952 | 20.0 |
| M1 | tumor_burden | M0(plan) | 20.0 | 0.733 | 0.267 | 0.958 | 23.0 |
| M1 | tumor_burden | M0(plan) | 30.0 | 0.593 | 0.186 | 0.875 | 16.0 |
| M1 | tumor_burden | M0(plan) | 36.5 | 0.500 | 0.174 | 0.933 | 15.0 |
| M1 | tumor_burden | M0(plan) | 40.0 | 0.465 | 0.244 | 0.950 | 21.0 |
| M1 | tumor_burden | CatBoost[phase12_catboost_fixed] | 20.0 | 0.733 | 0.733 | 0.867 | 63.0 |
| M1 | tumor_burden | CatBoost[phase12_catboost_fixed] | 30.0 | 0.593 | 0.698 | 0.836 | 60.0 |
| M1 | tumor_burden | CatBoost[phase12_catboost_fixed] | 36.5 | 0.500 | 0.663 | 0.772 | 57.0 |
| M1 | tumor_burden | CatBoost[phase12_catboost_fixed] | 40.0 | 0.465 | 0.674 | 0.811 | 58.0 |
| M1 | tumor_burden | TabPFN-V2@8[phase12_catboost_fixed] | 20.0 | 0.733 | 0.360 | 0.938 | 31.0 |
| M1 | tumor_burden | TabPFN-V2@8[phase12_catboost_fixed] | 30.0 | 0.593 | 0.198 | 0.882 | 17.0 |
| M1 | tumor_burden | TabPFN-V2@8[phase12_catboost_fixed] | 36.5 | 0.500 | 0.209 | 0.895 | 18.0 |
| M1 | tumor_burden | TabPFN-V2@8[phase12_catboost_fixed] | 40.0 | 0.465 | 0.244 | 0.909 | 21.0 |
| M1 | tumors | TabPFN-V3_5@8 | 20.0 | 0.738 | 0.401 | 0.857 | 124.0 |
| M1 | tumors | TabPFN-V3_5@8 | 30.0 | 0.595 | 0.311 | 0.833 | 96.0 |
| M1 | tumors | TabPFN-V3_5@8 | 36.2 | 0.502 | 0.259 | 0.797 | 80.0 |
| M1 | tumors | TabPFN-V3_5@8 | 40.0 | 0.414 | 0.275 | 0.819 | 85.0 |
| M1 | tumors | M0(plan) | 20.0 | 0.738 | 0.256 | 0.934 | 79.0 |
| M1 | tumors | M0(plan) | 30.0 | 0.595 | 0.113 | 0.871 | 35.0 |
| M1 | tumors | M0(plan) | 36.2 | 0.502 | 0.133 | 0.878 | 41.0 |
| M1 | tumors | M0(plan) | 40.0 | 0.414 | 0.168 | 0.942 | 52.0 |
| M1 | tumors | CatBoost[phase12_catboost_lesions] | 20.0 | 0.738 | 0.589 | 0.863 | 182.0 |
| M1 | tumors | CatBoost[phase12_catboost_lesions] | 30.0 | 0.595 | 0.456 | 0.859 | 141.0 |
| M1 | tumors | CatBoost[phase12_catboost_lesions] | 36.2 | 0.502 | 0.398 | 0.805 | 123.0 |
| M1 | tumors | CatBoost[phase12_catboost_lesions] | 40.0 | 0.414 | 0.398 | 0.828 | 123.0 |
| M1 | tumors_hepatic | TabPFN-V3_5@8 | 30.0 | 0.603 | 0.263 | 0.764 | 51.0 |
| M1 | tumors_hepatic | M0(plan) | 30.0 | 0.603 | 0.093 | 0.882 | 18.0 |


## 4. Identity gate

| target | size | check | stored_tag | same_y_model | max_abs_diff | identical |
|---|---|---|---|---|---|---|
| bone_marrow | M1 | M0 median, stored vs today (first 5 repeats) | gate_m0_plan | True | 2.46e-01 | False |
| bone_marrow | M1 | M0 median, stored vs today (first 5 repeats) | phase12_catboost_fixed | True | 2.46e-01 | False |
| bone_marrow | M1 | y_model identical: CatBoost | phase12_catboost_fixed | True |  | True |
| bone_marrow | M1 | y_model identical: TabPFN-V2@8 | phase12_catboost_fixed | True |  | True |
| kidneys | M1 | M0 median, stored vs today (first 5 repeats) | gate_m0_plan | True | 5.55e-01 | False |
| kidneys | M1 | M0 median, stored vs today (first 5 repeats) | phase12_catboost_fixed | True | 5.55e-01 | False |
| kidneys | M1 | M0 median, stored vs today (first 5 repeats) | phase12_organ3p | True | 0.00e+00 | True |
| kidneys | M1 | y_model identical: CatBoost | phase12_catboost_fixed | True |  | True |
| kidneys | M1 | y_model identical: TabPFN-V2@8 | phase12_catboost_fixed | True |  | True |
| liver_healthy | M1 | M0 median, stored vs today (first 5 repeats) | gate_m0_corrected | True | 5.55e-01 | False |
| liver_healthy | M1 | M0 median, stored vs today (first 5 repeats) | gate_m0_plan | True | 8.14e-01 | False |
| liver_healthy | M1 | M0 median, stored vs today (first 5 repeats) | phase12_catboost_liver | True | 5.55e-01 | False |
| liver_healthy | M1 | M0 median, stored vs today (first 5 repeats) | phase12_organ3p | True | 0.00e+00 | True |
| liver_healthy | M1 | y_model identical: CatBoost | phase12_catboost_liver | True |  | True |
| liver_healthy | M1 | y_model identical: TabPFN-V2@8 | phase12_organ3p | True |  | True |
| spleen | M1 | M0 median, stored vs today (first 5 repeats) | gate_m0_corrected | True | 4.62e-01 | False |
| spleen | M1 | M0 median, stored vs today (first 5 repeats) | gate_m0_plan | True | 3.70e-01 | False |
| spleen | M1 | M0 median, stored vs today (first 5 repeats) | phase12_spleen_gy | False | 2.22e+00 | False |
| spleen | M1 | M0 median, stored vs today (first 5 repeats) | phase12_spleen_topk | True | 4.62e-01 | False |
| spleen | M1 | y_model identical: TabPFN-V2@8 | phase12_spleen_gbq_m1 | True |  | True |
| tumor_burden | M1 | M0 median, stored vs today (first 5 repeats) | gate_m0_plan | True | 0.00e+00 | True |
| tumor_burden | M1 | M0 median, stored vs today (first 5 repeats) | phase12_catboost_fixed | True | 0.00e+00 | True |
| tumor_burden | M1 | M0 median, stored vs today (first 5 repeats) | phase12_cb_cov_fast | True | 0.00e+00 | True |
| tumor_burden | M1 | y_model identical: CatBoost | phase12_catboost_fixed | True |  | True |
| tumor_burden | M1 | y_model identical: TabPFN-V2@8 | phase12_catboost_fixed | True |  | True |
| tumors | M1 | M0 median, stored vs today (first 5 repeats) | gate_m0_plan | True | 0.00e+00 | True |
| tumors | M1 | M0 median, stored vs today (first 5 repeats) | phase12_catboost_lesions | True | 0.00e+00 | True |
| tumors | M1 | y_model identical: CatBoost | phase12_catboost_lesions | True |  | True |

