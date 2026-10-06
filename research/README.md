# research/ - the code behind the real-cohort numbers

This folder is **reference code**: it produced the aggregates in `results/real_cohort/v35_2026-10-06/`
(the lesion-level 8-model tables and the leakage summary come from the study's main pipeline, which is not in this
repository; their numbers are transcribed). It
needs the study's de-identified patient-level tables, which are **not public** and are not in this
repository. It is published so that the analysis can be read, criticised and re-run by anyone with
data access under the study's approvals.

| path | what |
|---|---|
| `tabpfn_prrt_scaffold/` | the study package: data assembly with the leakage register as assertions, log(Gy/GBq) target with quantile-only back-transform, person-grouped CV, model arms with one `fit` / `predict_quantiles` interface, calibration / CRPS / PIT / threshold metrics, paired patient bootstrap |
| `hackathon_v35_run_2026-10-06/run_v35_scaffold.py` | adds the TabPFN-3.5 arm to the scaffold without editing it and runs 5 x 5 person-grouped CV, offline, weights SHA-256 checked |
| `hackathon_v35_run_2026-10-06/analyse_v35.py` | produces the tables and figures in `results/real_cohort/v35_2026-10-06/` (M1 cells) |
| `hackathon_v35_run_2026-10-06/make_synthetic_cohort.py` | Gaussian-copula generator of the synthetic cohort, with fidelity and nearest-neighbour privacy checks |

The M3 tables (`v35_M3_*.csv`) were computed from the run's per-repeat result files after
`analyse_v35.py` had run (the M3 cells finished later); see `results/real_cohort/README.md`.
