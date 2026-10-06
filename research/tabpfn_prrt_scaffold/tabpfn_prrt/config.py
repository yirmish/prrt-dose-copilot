"""Frozen configuration for the TabPFN PRRT dose-prediction study.

Everything a reviewer would call a researcher degree of freedom lives here, so that
it can be locked in a tagged commit before the first model is fitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths.  Override the data location with the PRRT_DATA_ROOT environment variable
# (or a line in .env) instead of editing this file.
# --------------------------------------------------------------------------------------
import os

DATA_ROOT = Path(os.environ.get("PRRT_DATA_ROOT", "data/Deidentified_Export_2026-09-07"))
V8_ROOT = DATA_ROOT / "V8"
# The treatment sheet ships under two spellings (..._2026-08-23_... and ..._20260823_...);
# data.py globs for it rather than trusting either.
TREATMENTS_XLSX_GLOB = "Treatments_general_data_with_weight*deidentified.xlsx"
TREATMENTS_XLSX = DATA_ROOT / "Treatments_general_data_with_weight_2026-08-23_deidentified.xlsx"
RESULTS_ROOT = Path(os.environ.get("PRRT_RESULTS_ROOT", "results"))

# Join key between the Excel sheet and the V8 datasets.
# NOT `pt#` -- in the Excel sheet repeat courses share `pt#` (four study numbers each
# appear twice inside the 86-treatment cohort), while `pt_dicom_source` carries the
# five-digit V8 convention. Joining on `pt#` silently merges four patients' two courses.
XLSX_JOIN_KEY = "pt_dicom_source"
V8_JOIN_KEY = "pt_id"

# --------------------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------------------
N_REPEATS = 20
N_FOLDS = 5
CV_SEEDS = tuple(range(N_REPEATS))
MODEL_SEEDS = tuple(range(10))          # for the seed-variability sensitivity analysis
N_BOOTSTRAP = 2000                      # paired bootstrap, resampled over PATIENTS

# Pre-specified superiority margins on R^2, produced by power_sim.py (Phase 1) and then
# FROZEN. Do not adjust after seeing results.
#
#   design      paired MDD   single-R2 95% CI   vs. a published R2
#   treatment   0.08-0.15    +/- 0.13 to 0.17   >= 0.26-0.34
#   lesion      0.05-0.07    +/- 0.07 to 0.08   >= 0.14-0.17
#
# Read the third column before quoting any literature comparison: at n=86 this cohort
# cannot distinguish its own result from Akhavanallaf's 0.25 or Peterson's 0.60 unless
# the gap exceeds ~0.3.
# 2026-09-20 (DEC-226): re-derived with a CLUSTERED RESIDUAL at the project's measured
# REML ICC of 0.79. The original simulation had an i.i.d. per-row residual, so it could
# not represent "effective n ~= the number of patients". Lesion-level paired MDD moves
# 0.064 -> 0.091 and the lesion-level single-R2 95% CI moves +/- 0.080 -> +/- 0.142.
# Treatment level is unaffected (one row per patient). The frozen margins below are LEFT
# AS THEY WERE -- a margin changed after seeing results is not a margin -- but every
# lesion-level result must now also be read against 0.09, and Plan 11's claim that the
# lesion design is "roughly twice as precise" is withdrawn. No conclusion flips: every
# lesion-level result is null and a larger margin only makes a null safer.
SUPERIORITY_MARGIN_R2_BY_UNIT = {"treatment": 0.15, "lesion": 0.07}
MARGIN_R2_RESIDUAL_CLUSTERED = {"treatment": 0.15, "lesion": 0.09}   # sensitivity, DEC-226
R2_CI_HALFWIDTH_CLUSTERED = {"treatment": 0.16, "lesion": 0.14}      # sensitivity, DEC-226
SUPERIORITY_MARGIN_R2 = 0.15                 # conservative default
R2_CI_HALFWIDTH_BY_UNIT = {"treatment": 0.17, "lesion": 0.08}


def margin_for(unit: str) -> float:
    return SUPERIORITY_MARGIN_R2_BY_UNIT.get(unit, SUPERIORITY_MARGIN_R2)

QUANTILE_GRID = tuple(round(q, 4) for q in [i / 100 for i in range(1, 100)])
NOMINAL_COVERAGES = (0.50, 0.80, 0.95)

# --------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetSpec:
    name: str
    folder: str
    unit: str                       # "treatment" | "lesion"
    target_col: str
    role: str                       # "primary" | "secondary" | "exploratory"
    order: int                      # Holm ordering within the secondary family
    thresholds_gy: tuple[float, ...] = ()
    extra_excluded: tuple[str, ...] = ()
    row_filter: str | None = None


TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec("tumor_burden", "tumor_burden", "treatment", "dos_dose_gy", "primary", 0,
               thresholds_gy=(20.0, 30.0, 36.5, 40.0)),
    TargetSpec("tumors", "tumors", "lesion", "dos_dose_gy", "secondary", 1,
               thresholds_gy=(20.0, 30.0, 36.2, 40.0),
               # tumor_burden unions ALL marked lesions, in-gate and out of gate, so at
               # lesion level it leaks information about rows excluded from the dataset.
               extra_excluded=("voi_tumor_burden__",)),
    TargetSpec("kidneys", "kidneys", "treatment", "dos_dose_gy", "secondary", 2,
               thresholds_gy=(5.75,)),
    TargetSpec("liver_healthy", "liver_healthy", "treatment", "dos_dose_gy", "secondary", 3),
    TargetSpec("spleen", "spleen", "treatment", "dos_dose_gy", "secondary", 4),
    TargetSpec("bone_marrow", "bone_marrow", "treatment", "dos_bm_dose_gy", "secondary", 5,
               thresholds_gy=(0.5,),
               # dos_bm_dose_gy is COMPUTED FROM the blood activity and the cal_std VOI.
               # Using either as a predictor is circular, not merely optimistic.
               extra_excluded=("gen_blood_activity_per_cc_at_scan3",
                               "gen_blood_sample_hours_after_injection")),
    TargetSpec("tumors_hepatic", "tumors", "lesion", "dos_dose_gy", "exploratory", 6,
               thresholds_gy=(30.0,),
               extra_excluded=("voi_tumor_burden__",),
               row_filter="tumor_location == 'liver'"),
)

ACTIVITY_COL = "gen_prrt_net_injected_activity_gbq"

# --------------------------------------------------------------------------------------
# Feature blocks (nested; the reported quantity is the increment of each block)
# --------------------------------------------------------------------------------------
BLOCK_A_PET_ROW = (
    "pet_volume_ml", "pet_max_suvbw", "pet_mean_suvbw", "pet_median_suvbw",
    "pet_std_suvbw", "pet_total_suvbw", "pet_tlg_suvbw_ml",
    "pet_integral_total_bqml_ml", "pet_mean_bqml", "pet_max_bqml", "pet_median_bqml",
    "pet_std_bqml", "pet_total_bqml", "pet_sphere_diam_cm",
    "pet_skewness", "pet_kurtosis_excess",   # Pearson -> excess conversion done in data.py
    "pet_regions_count",
)

# Block B is matched by prefix: voi_<roi>__<measurement>
BLOCK_B_PREFIXES = (
    "voi_whole_body__", "voi_cal_std__", "voi_bones__", "voi_kidneys__",
    "voi_liver__", "voi_spleen__", "voi_tumor_burden__",
)
BLOCK_B_EXTRA = ("gen_seg_n_lesions_not_study_gate", "gen_num_of_lesions_study_gate")

BLOCK_C_TREATMENT = (
    ACTIVITY_COL, "gen_pet_prior_prrt_0_1", "gen_lu177_drug_manufacturer",
    "gen_days_pet_to_prrt", "gen_pet_uptake_time_minutes",
    "gen_pet_ga68_inj_activity_mci", "gen_salvage_course_num",
)

BLOCK_D_BODY = (
    "pt_weight_kg", "pt_height_m", "calc_bmi", "calc_lbm_kg", "calc_sul_factor",
    "gen_age_at_prrt_years", "gen_gender_code_1m_2f",
)

BLOCK_E_CLINICAL = (
    "gen_tumor_grade_code_0unavail",          # 0 -> NaN, handled in data.py
    "gen_pet_code_origin_from_report",        # legacy counterpart is EXCLUDED (disagrees 29/86)
    "gen_mets_liver_0_1", "gen_mets_bones_0_1",
    "gen_ssa_somatostatin_analog_0_1", "gen_ssa_continued_during_prrt_0_1",
    "gen_chromogranin_a_pre_prrt_ng_ml",
    "calc_egfr",                              # derived in data.py from resolved creatinine
    "clinical_block_present",                 # MNAR indicator, reported explicitly
)

# Available before therapy, but it is an operator decision taken while looking at the
# pre-therapy PET, i.e. a human proxy for the scan's uptake level. Run with and without.
GREY_ZONE_FEATURES = ("gen_pet_lesionid_threshold_suv",)

MODEL_SIZES = {
    "M0": "baseline",                        # built per target in models.py
    "M1": ("A",),
    "M2": ("A", "B"),
    "M3": ("A", "B", "C", "D"),
    "M4": ("A", "B", "C", "D", "E"),
}

CATEGORICAL_FEATURES = (
    "gen_gender_code_1m_2f", "gen_lu177_drug_manufacturer",
    "gen_pet_code_origin_from_report", "gen_tumor_grade_code_0unavail",
    "gen_pet_camera_code_1mi_2dr", "tumor_location_3lvl",
)

# --------------------------------------------------------------------------------------
# TabPFN
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TabPFNSpec:
    version: str = "V3"                 # "V3" (default ckpt) | "V2_6" | "V2_5" | "V2"
    n_estimators: int = 32              # cheap at n<=309; buys stability
    softmax_temperature: float = 0.9
    device: str = "cpu"                 # n<=309: CPU is sufficient, GPU unnecessary
    deterministic_precision: bool = True
    random_state: int = 0


TABPFN_ARMS = (
    TabPFNSpec(version="V3"),           # headline arm
    TabPFNSpec(version="V2"),           # token-free reproducibility arm (Apache-2.0+attr weights)
)

# --------------------------------------------------------------------------------------
# Pseudo-external splits (§8.3) -- the closest available analogue to leave-one-centre-out
# --------------------------------------------------------------------------------------
ERA_SPLIT = {"early": (2018, 2019, 2020), "late": (2022, 2023, 2024)}
SCANNER_COL = "gen_pet_camera_code_1mi_2dr"
ERA_SOURCE_COL = "gen_prrt_injection_date"


@dataclass
class RunConfig:
    targets: tuple[str, ...] = field(default_factory=lambda: tuple(t.name for t in TARGETS))
    model_sizes: tuple[str, ...] = ("M0", "M1", "M2", "M3")
    include_grey_zone: bool = False
    log_target: bool = True
    normalise_by_activity: bool = True          # model Gy/GBq, report Gy
    target_col_override: str | None = None      # e.g. "dos_dose_rc_corr_gy" for sensitivity
    tag: str = "main"
    # The protocol is 20 repeats (N_REPEATS). Lowering it is legitimate ONLY for
    # configuration-selection runs that are not reported as results -- e.g. the
    # n_estimators sweep of plan 1.1, which chooses a setting rather than estimating
    # performance. Every reported number uses N_REPEATS; the value used is written into
    # run_config.json so a reader can tell the two apart.
    n_repeats: int = N_REPEATS
