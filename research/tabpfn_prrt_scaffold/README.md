# tabpfn_prrt — code scaffold

Runnable skeleton for the plan in `TabPFN_Modelling_Plan.md`. Nothing here is a result;
it is the harness the results will come out of.

## Quickstart (Windows / PowerShell)

**Step 0 — be in the right folder.** The folder you run from must contain **both**
`tabpfn_prrt\` and `tests\`. If the zip unpacked one level deeper, go into the inner
folder.

```powershell
cd C:\path\to\project\tabpfn_prrt_scaffold
dir            # expect: tabpfn_prrt, tests, scripts, requirements.txt, README.md
```

**Step 1 — virtual environment and packages.**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Step 2 — the token.** It goes in a `.env` file, **never in the code**.

```powershell
copy .env.example .env
notepad .env                          # paste the token after TABPFN_TOKEN=
```

**Step 3 — point at the data** (same `.env`, uncomment the line):

```
PRRT_DATA_ROOT=C:\path\to\project\data\Deidentified_Export_2026-09-07
```

**Step 4 — verify, then run.**

```powershell
python scripts\check_setup.py --fit    # environment, token, data, and a real 80-row fit
python tests\smoke_test.py             # the harness itself
python -m tabpfn_prrt.power_sim        # Phase 1 (already run once; re-run to re-derive)
python -m tabpfn_prrt.run --targets tumor_burden --sizes M0,M1,M2,M3     # Phase 2
python -m tabpfn_prrt.run --all --arms TabPFN-V3,TabPFN-V2,CatBoost,RF-Akhava  # Phase 3
python -m tabpfn_prrt.run --all --rc-target --tag sens_rc                # Phase 4
python -m tabpfn_prrt.run --all --grey-zone --tag sens_greyzone
```

On Linux/macOS the same commands work with `python3 -m venv .venv`,
`source .venv/bin/activate`, and forward slashes.

## Where the token goes, and why

`.env` in the folder you run from:

```
TABPFN_TOKEN=<your token from https://ux.priorlabs.ai -> Licence / API>
```

TabPFN resolves the token in this order:

1. the `TABPFN_TOKEN` environment variable,
2. `~/.cache/tabpfn/auth_token` (written automatically after one browser login),
3. `~/.tabpfn/token` (the tabpfn-client cache).

**A `.env` file alone is not enough for stock TabPFN** — its own settings object reads
`.env` through pydantic-settings, but the token is read straight from `os.environ`, so a
`.env` entry is silently ignored. `tabpfn_prrt/env.py` closes that gap: it is imported by
`tabpfn_prrt/__init__.py` and copies `TABPFN_*`, `HF_*` and `PRRT_*` keys from `.env` into
the environment before TabPFN loads. That is the only reason the `.env` route works here.

`.env` is in `.gitignore`. Do not hardcode the token in `config.py`, and do not commit it.

A token is needed only for the **TabPFN-2.5 / 2.6 / 3** checkpoints. The **TabPFN-V2** arm
uses Apache-2.0 + attribution weights and needs none — which is exactly why it is the
reproducibility arm in the plan. To work entirely without a token:

```powershell
python -m tabpfn_prrt.run --targets tumor_burden --arms TabPFN-V2
```

No GPU is needed. At n ≤ 309 the configured default is `device="cpu"`.

## Thinking mode

`thinking_mode` is **not in the open-source `tabpfn` package** (`grep thinking_mode` over 8.5.0
returns nothing). It is API-only, TabPFN-3-Plus, via `tabpfn-client`, with a 20-fit/month quota,
and it sends your data to Prior Labs' servers. Section 1.1 of the plan sets out why it is excluded
from the CV protocol and what is admitted instead.

The *architectural* thinking rows (64 learned tokens, `num_thinking_rows=64`) are already active
in every TabPFN-2.5 / 2.6 / 3 arm here — nothing to enable.

The local, free analogue of extra test-time compute is `n_estimators`, addressable per arm:

```powershell
python -m tabpfn_prrt.run --targets tumor_burden --sizes M3 ^
    --arms TabPFN-V3@4,TabPFN-V3@8,TabPFN-V3@16,TabPFN-V3@32,TabPFN-V3@64 --tag compute_sweep
```

If CRPS and 95% coverage are flat from 16 upwards, additional inference compute is not the binding
constraint on this problem — which answers the thinking-mode question on premises, for free.

## Data layout

```
<PRRT_DATA_ROOT>/
├── Treatments_general_data_with_weight_*_deidentified.xlsx     (either filename spelling)
└── V8/{tumors,tumor_burden,kidneys,liver_healthy,spleen,bone_marrow}/<name>_dataset.csv
```

Defaults to `.\data\Deidentified_Export_2026-09-07` when `PRRT_DATA_ROOT` is unset.

Outputs land in `results/<tag>/`: `results_raw.csv` (one row per repeat),
`results_summary.csv`, `comparisons_<target>.csv` (paired bootstrap vs M0) and
`manifest_<target>_<size>.csv` — the frozen feature manifest that a human reads once,
carefully, before Phase 2.

## Modules

| file | what it is responsible for |
|---|---|
| `scripts/check_setup.py` | run this first: Python, packages, token, data layout, and a real end-to-end fit |
| `env.py` | loads `.env` into the environment so the token route actually works |
| `config.py` | every researcher degree of freedom, in one place, to be frozen in a tagged commit |
| `ids.py` | `base_id` decoding; guards against joining the Excel sheet on `pt#` instead of `pt_dicom_source` |
| `leakage.py` | the leakage register **as assertions** — a forbidden column aborts the run |
| `data.py` | assembly and the encoding repairs (grade-0 → NaN, compound mets codes, Pearson→excess kurtosis, spleen zeros, creatinine units → eGFR) |
| `targets.py` | Gy/GBq + log, and the quantile-only back-transform |
| `cv.py` | repeated grouped CV; era and scanner splits (pseudo-external validation) |
| `models.py` | M0 baseline, TabPFN, CatBoost (nested), Akhavanallaf RF |
| `metrics.py` | error · calibration/CRPS/PIT/coverage · decision curves · between/within R² · paired bootstrap |
| `power_sim.py` | the resolvability simulation that sets the superiority margin |
| `run.py` | the driver |

## Three things the scaffold refuses to do

1. **Impute.** TabPFN handles `NaN` natively, and the clinical block is missing for the
   same 22 treatments by cohort membership, not at random. Imputing it manufactures an
   era effect and hands it to the model.
2. **Back-transform a mean from log space.** Every point estimate is the 0.5 quantile;
   exponentiating a predicted mean gives a median and biases every error downwards.
3. **Fit with a forbidden column present.** `leakage.check()` raises rather than warns —
   including on `spect_scan_time_hours`, which is the variable *t* inside the dose formula
   and is admitted only in the explicitly labelled oracle arm.

## Provenance to record in the paper

`tabpfn` version, `ModelVersion`, checkpoint SHA-256, `torch` version, CV seeds, and
`inference_precision=torch.float32` for the reported runs. "We used TabPFN" is not a
reproducible statement — checkpoints change between releases.
