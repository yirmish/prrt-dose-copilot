# Registration H-T — TabPFN-3.5 versus nested CatBoost for lesion-level absorbed-dose prediction

**Registered 4 October 2026, 00:30 IDT (UTC+3), before the `n_estimators = 32` run (R-1) is launched.**
Companion to:
- `Radiomics_LIFEx_Results_CatBoost_2026-10-02.md` §9 (the discovery);
- `TabPFN_Modelling_Plan.md` amendment T1 (the model specification).

The SHA-256 of this file is written into the R-1 output folder before that run starts. It was not
lodged on a public registry.

> Copy note (2026-10-06): this is a transcription of the study's registration document. The
> registered SHA-256 refers to the original file, not to this copy.

---

## 0. Status — what this registration can and cannot do

**The hypothesis was generated from data that have already been analysed.**
- **Discovery estimate.** TabPFN-3.5 (n_estimators 8, run 3 October 2026) against nested CatBoost
  (run 2 October 2026), same rows, same folds, lesion level, R0 = Plan M3:
  - overall R²: ΔR² = +0.082 (95% patient-bootstrap CI +0.013 … +0.139);
  - within-patient R²: +0.117 (+0.024 … +0.203).
- **Why that estimate is not confirmatory.**
  - TabPFN-3.5 was the eighth model examined.
  - The between-model comparison was not part of amendment R.
  - The estimate is subject to selection ("winner's curse") inflation.
  - Registering it now does not make it confirmatory, and this document does not claim that it does.

**What is fixed here, before it is seen:**
1. **R-1, an internal robustness check on the existing data** (§3). It answers one question: does the
   discovery depend on the reduced `n_estimators = 8` used for speed?
2. **C-1, the confirmatory test** (§4). It runs on lesions that no model in this project has seen.

Only C-1 can confirm H-T1. R-1 can only weaken the discovery or leave it standing.

---

## 1. Hypotheses

- **H-T1 (primary).** At lesion level (`tumors`), with the frozen Plan M3 feature set and target
  log(Gy/GBq), TabPFN-3.5 achieves higher out-of-sample R² than nested CatBoost. One-sided in
  direction; tested with a two-sided 95% interval.
- **H-T2 (secondary; tested only if H-T1 is supported — fixed-sequence).** The advantage includes
  the within-patient component, defined as R² of patient-centred observed and predicted values over
  patients with ≥ 2 lesions.

The radiomic arms (R1, R2) are **not** part of H-T. Amendment R closed that question (Tier 0 in
eight model families).

---

## 2. Fixed specification (both arms)

| Item | Specification |
|---|---|
| Rows | lesion rows of V8 `tumors` joined to LIFEx as in amendment R. Discovery cohort: 307 lesions, 85 treatments, 81 patients (`base_id`) |
| Features | **R0 = Plan M3** (blocks A + B + C + D, 140 columns), defined by the frozen manifest SHA-256 `04a01a82d4b90be2844489e69d6ff74e82e6201b7cfa54557682c71b659bf7d1`; column list `m3_columns_lesion.json` (SHA-256 `a1bd184f2b7949dcac684a4bf5af5c6aad406cd635c854f8a167d8fefc65e71f`) |
| Target | log(`dos_dose_gy` / `gen_prrt_net_injected_activity_gbq`) |
| Point prediction | 0.5 quantile of each model's predictive distribution (back-transforms exactly) |
| **TabPFN-3.5** | `tabpfn` 9.0.0; `TabPFNRegressor.create_default_for_version(ModelVersion.V3_5)`; checkpoint `tabpfn-v3.5-20260909.safetensors`, SHA-256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`; **`n_estimators = 32`**; `inference_precision = float32`; CPU; package-default preprocessing; `random_state` = fold seed; no tuning of any kind |
| **CatBoost (nested)** | `CatBoostNested` (radiomics_ext `backends.py`): RMSE loss, 300 iterations, learning rate 0.05, Plain boosting, 64 borders. Fixed grid depth {4, 6} × l2 {3, 10}, selected by inner patient-grouped 3-fold CV on training rows only. Predictive quantiles = prediction + empirical quantiles of the inner OOF residuals (split-conformal) |
| Code | `radiomics_ext` package, combined SHA-256 `929c0c8c2f6224fadbefe39eacc6d4091de9fd9650087a4049df1370b064abf2` (SHA-256 of the sorted per-file SHA-256 list, Appendix A) |
| Estimand | ΔR² = R²(TabPFN-3.5) − R²(CatBoost), computed on the same lesions from the 0.5 quantiles. Uncertainty from a paired bootstrap that resamples **patients** (B = 4000) |

---

## 3. R-1 — internal robustness check on the existing data

**What is run.** The full amendment-R protocol with TabPFN-3.5 at `n_estimators = 32`:
- identical manifest, rows, fold seeds (5 × 5) and NC1 permutations (20) as the `n_estimators = 8`
  run;
- output folder `Radiomics\radiomics_ext_out_tabpfn_v3_5_n32\`.

CatBoost is not re-run; its out-of-fold predictions from 2 October are used unchanged.

**Pre-specified quantities**:
- ΔR²(32) against CatBoost (overall and within-patient), with the bootstrap above;
- ΔR²(32) − ΔR²(8), the effect of the estimator count;
- the amendment-R primary contrast (R1 − R0) and its tier at 32 estimators.

**Pre-specified reading.**

| Outcome | Condition | Consequence |
|---|---|---|
| Robust | 95% CI of ΔR²(32) excludes 0 **and** abs(ΔR²(32) − ΔR²(8)) ≤ 0.03 | discovery stands; C-1 proceeds as registered |
| Configuration-sensitive | 95% CI of ΔR²(32) excludes 0 **but** abs(ΔR²(32) − ΔR²(8)) > 0.03 | the 32-estimator value is carried forward as the discovery estimate (it is the registered configuration); C-1 sample size is recomputed from it |
| Not robust | 95% CI of ΔR²(32) includes 0 | H-T1 is reported as not supported internally; C-1 is not pursued on the strength of this discovery |

The ± 0.03 tolerance is three times the seed SD of R² measured at 8 estimators (0.009; TabPFN audit
§2.4).

The amendment-R conclusion on radiomics is considered unchanged if R1 − R0 remains Tier 0 at 32
estimators.

R-1 is reported whatever it shows.

---

## 4. C-1 — confirmatory test on independent data

**Data.** Cycle-1 ¹⁷⁷Lu-DOTATATE treatments that meet all of the following:
- they have pre-therapy ⁶⁸Ga-DOTATATE PET/CT and post-therapy SPECT/CT dosimetry;
- they were processed with the same V8 pipeline (same STP dose model, same lesion gating and the same
  5-lesion rule);
- they are **not** among the 85 treatments / 81 patients of the discovery cohort.

Two sources qualify:
- **Prospective accrual** at the department after the date of this registration.
- **An external centre's cohort.** This also tests transportability and is reported as a separate
  stratum.

**Training.** Each model is trained **once** on the full discovery cohort (307 lesions):
- CatBoost is tuned by the same inner grid on the full cohort;
- TabPFN-3.5 is configured as in §2.

Both fitted models, and the code SHA-256, are frozen and recorded **before any dose value of the new
cohort is read**.

**Evaluation.**
- R² of each model on the pooled new lesions, and ΔR² with the patient bootstrap (B = 4000) over new
  patients.
- Within-patient R² for H-T2.
- Coverage of the 80% and 95% intervals and CRPS are reported for both models, as secondary
  descriptions.

**Decision rule.**
- **H-T1 is confirmed** if the lower bound of the two-sided 95% CI of ΔR² is > 0.
- **Clinical relevance** is reported separately, against the TabPFN plan's lesion-level margin
  (ΔR² ≥ 0.07). It is not required for confirmation.
- **H-T2** is tested only if H-T1 is confirmed, by the same rule.
- **If H-T1 is not confirmed,** that is the result. No subgroup, metric or configuration is
  substituted after the fact.

**Sample size and timing.**
- The discovery CI implies SE(ΔR²) ≈ 0.032 with 81 patients.
- If the true advantage equals the discovery estimate (+0.08), 80% power at two-sided α = 0.05
  requires about **100 new patients** (~380 lesions).
- If the true advantage is half of it (+0.04), which is plausible after selection inflation, about
  **390 patients** are needed. That is beyond single-centre accrual, so C-1 is realistically a
  multicentre analysis.
- The test is run **once**, when the pre-specified accrual is reached. There are no interim looks.

---

## 5. What may and may not change after this registration

- No change to rows, features, target, either model's specification or the decision rules, except by
  a dated, logged amendment stating the reason. An amendment made after R-1 or C-1 results are known
  is labelled post hoc.
- Excluded: any tuning of TabPFN (estimator count, preprocessing, softmax temperature, thinking
  mode), and any change to CatBoost's grid.
- Allowed, but labelled exploratory: further internal analyses on the discovery cohort (20 × 5
  repeats, seed sweeps, other targets). They can neither replace R-1 nor stand in for C-1.

---

## Appendix A — per-file SHA-256 of the `radiomics_ext` package at registration

```
158e8e130d3e6dad5d5d9a197f697e192bfd8d1571d15fae40aac3ab1692ac74  __init__.py
590077de9d44ba2195bdba8e91fd415e37ce8b43ef9821fd5336478e6d1ac243  backends.py
5d4f0a02413e9399db3d856850a71ba6df52057e068532101f53f8f05d100918  bench_tabpfn.py
9a216bf18f73fc3d04b4d119b5f729c29381081b08850eae1a2191b32782a2b3  compare_models.py
99bef294362b0bf4c75cfcc2c2c812bfb4fd756d5c8a02ef35be1e8171af271d  compare_tabpfn.py
624959de32b39f1e3b35c0f91d41faadc48e09013ab22043ecf154d3297c734a  config.py
62871600a07949adbebdfceb32c819892f45c616846ab1539677005ab6cd4b1a  cv.py
b9e4df08d964b07655113f96b2d80ed9131722f894a8a2435da0791805a8c0ae  design.py
3995bfbc5b26a0439d1aaf7dad34de0272ef8167adb4d553172899a85bc025de  diagnostics.py
96a778301c1e139a8780e3b15e41c5aa68ffc7587ab00a4ca0acf5374ab7490e  features.py
96746ad0d54ab2d88d3ce33f6ecab9cc78710ff09f5b0ad4958b9c209ecf9455  io_lifex.py
6b3a015536e9456175bb9e0c2f4d1d8757b0185d77c00e1fa62a40b71427b5d8  m3.py
ca55b7f405fe6c867d1673aa759383435c4e87703a58907ed4cd3623fd31d577  plot_pred_vs_obs.py
fa40d7e57a781b407d7eb2417fb82616eb56912666b20920a98e19e3d5f6efb8  report.py
6e6fc8e22cc04371ed927bf6a703657121d1bd8dfb62640443e1e4b7ee9bfb64  rim.py
5d25cbbab8bd2b38b07bc6db15b62a7af6d975bf0c7ba621642829febe6af4e7  run_all.py
4157bea759bb0e8cec0e2a497888dd622ebe6d48978e6602b9cc86a36a1ff8ab  run_lesion_contrast.py
9745a7a696096f9fa0605e9914c9eb8a20f3b7acc65c4732f15442b9d7695334  run_smoke.py
becd8c5bf7e355c617ea5f8b89b8fb00a05dd3d092eb681ce4bc050e8dee7fc7  sample_size.py
cbe88f55a8af0c9d230138ec44ecc1d91338848ab5bdaffbd94311540e8a6618  transforms.py
```

## Appendix B — discovery evidence (out-of-fold predictions on which §0 rests)

| File | SHA-256 |
|---|---|
| `radiomics_ext_out_catboost\oof_tumors.npz` | `de08372cd041e545410d7b932d0207f93179e18c3e7a867fd9a18129d274c437` |
| `radiomics_ext_out_tabpfn_v3_5\oof_tumors.npz` | `13e418e87754b62f5838a65f82b04cf9fcfc59c4bd8946ee443572db74892144` |
| `radiomics_ext_cross_model\head_to_head_tabpfn_v3_5.csv` | the paired bootstrap results (B = 4000) |
