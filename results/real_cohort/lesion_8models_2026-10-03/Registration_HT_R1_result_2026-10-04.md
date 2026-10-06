# Registration H-T — result of check R-1

**Reported 4 October 2026.** Refers to `Registration_HT_TabPFN35_vs_CatBoost_2026-10-04.md`
(SHA-256 `004da5c16d1cc4e7dd80d4f65dc108539b94d848f84bf507cb928c3d1a393a4c`), §3. This note does
not modify the registration.

## Conduct

| Item | Value |
|---|---|
| Registration hash written to the output folder | 4 Oct 2026 00:27:10 +03:00 (`registration_reference.txt`) |
| Run start / end | 00:27:22 / 07:43:38 +03:00 (`progress.log`, `launcher.log`) |
| Configuration | TabPFN-3.5, `tabpfn` 9.0.0, checkpoint SHA-256 `ece4d67e…f0be3`, `n_estimators = 32`, float32, CPU; frozen manifest `04a01a82…` re-verified at start; code package `929c0c8c…` |
| Comparator | CatBoost-nested out-of-fold predictions of 2 Oct 2026 (`oof_tumors.npz`, SHA-256 `de08372c…c437`), unchanged |
| Deviations from §3 | none |

## Pre-specified quantities

| Quantity | Value |
|---|---|
| ΔR², TabPFN-3.5 (32) − CatBoost, lesion level, R0 | **+0.0793**, 95% CI **+0.0110 … +0.1359** |
| ΔR² at 8 estimators (discovery) | +0.0822 (+0.0130 … +0.1390) |
| ΔR²(32) − ΔR²(8) | **−0.0029** (tolerance ± 0.03) |
| Δ within-patient R² (32) vs CatBoost | +0.1175 (+0.0252 … +0.2005) |
| Amendment-R primary (R1 − R0) at 32 estimators | −0.0067 (−0.0183 … +0.0068); Tier 0 |

**Verdict: Robust.** The discovery stands, and C-1 proceeds as registered. The carried-forward
discovery estimate is the 32-estimator value: ΔR² = +0.079.

R-1 is a robustness check on the discovery cohort, not a confirmation. H-T1 remains unconfirmed
until C-1.
