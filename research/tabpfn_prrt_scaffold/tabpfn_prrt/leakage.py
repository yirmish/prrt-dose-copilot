"""The leakage register, enforced as assertions rather than as discipline.

Every rule here is traceable to a section of the dataset companion document. If a
feature matrix reaches a model with any of these columns in it, the run aborts.
"""
from __future__ import annotations

import re
import pandas as pd

# --- Hard exclusions: prefixes -------------------------------------------------------
FORBIDDEN_PREFIXES: tuple[tuple[str, str], ...] = (
    ("spect_", "measured after therapy, from the scan the label was computed from (Companion 4.6)"),
    ("dos_", "downstream of the target (Companion 4.6)"),
    ("src_", "build metadata"),
    ("qc_", "build/QC metadata"),
)

# --- Hard exclusions: exact column names ---------------------------------------------
FORBIDDEN_EXACT: dict[str, str] = {
    # acquisition-grid quantities, not comparable between Discovery MI and MI-DR
    "pet_slice_with_max": "acquisition geometry, not anatomy (Companion 2.1)",
    "pet_voxel_count": "acquisition geometry, not anatomy (Companion 2.1)",
    "pet_frame_duration_ms": "acquisition geometry, not anatomy (Companion 2.1)",
    # MIM reference-contour artefact: reference VOI = first VOI created in the session
    "pet_min_mean_ratio_refvoi": "MIM reference contour is arbitrary (Companion 8.2)",
    "pet_max_mean_ratio_refvoi": "MIM reference contour is arbitrary (Companion 8.2)",
    "pet_mean_mean_ratio_refvoi": "MIM reference contour is arbitrary (Companion 8.2)",
    "pet_median_mean_ratio_refvoi": "MIM reference contour is arbitrary (Companion 8.2)",
    "pet_std_mean_ratio_refvoi": "MIM reference contour is arbitrary; NOT a CV (Companion 8.2)",
    "pet_integral_total_mean_ratio_ml": "MIM reference contour is arbitrary (Companion 8.2)",
    "pet_tlg_mean_ratio_ml": "MIM reference contour is arbitrary (Companion 8.2)",
    # calendar time: entangled with scanner, protocol and drug source. Split var, not feature.
    "gen_treatment_num_global_id": "encodes calendar time; era confound (Companion 2.1)",
    "gen_prrt_injection_date": "encodes calendar time; era confound",
    "pet_series_date": "encodes calendar time; era confound",
    "gen_baseline_pet_date_pre_prrt_legacy": "encodes calendar time",
    "gen_xl_curated_baseline_pet_date_pre_prrt": "encodes calendar time",
    # duplicated/conflicting encodings
    "gen_primary_tumor_origin_code_legacy": "disagrees with the PET-report version in 29/86",
    "gen_study_pt_num_legacy68": "legacy identifier",
    "gen_spect_lesionid_threshold_bqml": "set on the post-therapy SPECT",
    # zero-variance / unusable
    "gen_repeat_injection_idx_in_course": "constant = 1, zero variance",
    "gen_dose_calibration_method": "populated in 14/86",
    "gen_baseline_creatinine_units_unresolved": "mixed units; use calc_egfr instead",
    "gen_pet_ki67_pct_from_report": "populated in 2/86",
    "gen_postmodel_review_codes": "populated in 2/86",
    "gen_postmodel_review_note": "populated in 2/86",
    # identifiers
    "pt_id": "identifier, not a feature",
    "pt_dicom_source": "identifier, not a feature",
    "roi_name_original": "identifier",
    "tumor_location": "epistemic variable; use tumor_location_3lvl (Companion 7.4)",
}

# --- Measurements forbidden wherever they appear, flattened or not -------------------
#
# FOUND BY READING THE PHASE-0 MANIFEST (the plan's 15.1 names this as the failure mode
# most likely to survive the assertions). The three acquisition-grid quantities are
# excluded above by EXACT name, which catches the row's own `pet_slice_with_max` and
# `pet_voxel_count` -- but block B is built by prefix from `voi_<roi>__<measurement>`,
# and the 24 flattened measurements include both of them. Seven source VOIs x 2 =
# 14 columns that encode Discovery MI versus MI-DR acquisition geometry rather than
# anatomy (Companion 2.1) were entering every treatment-level M2/M3/M4 matrix, where
# scanner is entangled with era, protocol and drug source. Matched on the measurement
# suffix so the rule holds however the column is namespaced.
FORBIDDEN_MEASUREMENTS: dict[str, str] = {
    "pet_slice_with_max": "acquisition geometry, not anatomy (Companion 2.1)",
    "pet_voxel_count": "acquisition geometry, not anatomy (Companion 2.1)",
    "pet_frame_duration_ms": "acquisition geometry, not anatomy (Companion 2.1)",
}


def _measurement_of(col: str) -> str:
    """The measurement part of a flattened `voi_<roi>__<measurement>` column name."""
    return col.split("__", 1)[1] if "__" in col else col


# --- Permitted only in the explicitly labelled oracle arm ----------------------------
ORACLE_ONLY = {
    "spect_scan_time_hours": "the variable t inside the dose formula (Companion 4.2)",
}

# --- Free-text columns: never fed to a model directly --------------------------------
#
# The bare token `report` was removed from this pattern. It matched three NUMERIC code
# columns -- gen_pet_code_origin_from_report (the PET-report origin code that the plan's
# block E is built on), gen_pet_code_predominant_organ_from_report and
# gen_pet_ki67_pct_from_report -- and aborted every run that included block E. Name
# matching alone cannot tell "derived from the report" from "is the report text", so the
# name rules below are deliberately narrow and the real net is the dtype rule in check():
# any high-cardinality non-numeric column that is not a declared categorical is free
# text whatever it is called.
FREE_TEXT = re.compile(
    r"(^notes?$|_notes?$|comments?|_text$|reconciliation_disagreements"
    r"|_report_vs_|organs_involved|_review_note$)", re.I
)

# A non-numeric column with more levels than this is prose, not a category.
MAX_CATEGORY_LEVELS = 20


class LeakageError(AssertionError):
    pass


def check(X: pd.DataFrame, *, extra_excluded: tuple[str, ...] = (),
          allow_oracle: bool = False) -> None:
    """Abort the run if any forbidden column reached the feature matrix."""
    problems: list[str] = []

    for col in X.columns:
        if col in ORACLE_ONLY:
            # The oracle escape must be checked FIRST: spect_scan_time_hours also matches
            # the `spect_` prefix rule, and the prefix rule would otherwise shadow it.
            if not allow_oracle:
                problems.append(f"{col}: {ORACLE_ONLY[col]} -- oracle arm only")
            continue
        for pref, why in FORBIDDEN_PREFIXES:
            if col.startswith(pref):
                problems.append(f"{col}: forbidden prefix {pref!r} -- {why}")
        if col in FORBIDDEN_EXACT:
            problems.append(f"{col}: {FORBIDDEN_EXACT[col]}")
        meas = _measurement_of(col)
        if meas in FORBIDDEN_MEASUREMENTS:
            problems.append(
                f"{col}: measurement {meas!r} -- {FORBIDDEN_MEASUREMENTS[meas]}")
        if FREE_TEXT.search(col):
            problems.append(f"{col}: free-text column")
        # dtype net: prose that no naming rule happened to anticipate.
        if not pd.api.types.is_numeric_dtype(X[col]):
            n_lvl = int(X[col].nunique(dropna=True))
            if n_lvl > MAX_CATEGORY_LEVELS:
                problems.append(
                    f"{col}: non-numeric with {n_lvl} levels "
                    f"(> {MAX_CATEGORY_LEVELS}) -- free text, not a category")
        for ex in extra_excluded:
            if col == ex or col.startswith(ex):
                problems.append(f"{col}: target-specific exclusion {ex!r}")

    if problems:
        raise LeakageError(
            "Leakage register violated -- refusing to fit:\n  " + "\n  ".join(sorted(set(problems)))
        )


def drop_forbidden(df: pd.DataFrame, *, extra_excluded: tuple[str, ...] = (),
                   allow_oracle: bool = False) -> pd.DataFrame:
    """Convenience: drop everything the register forbids, then verify nothing is left."""
    keep = []
    for col in df.columns:
        if any(col == e or col.startswith(e) for e in extra_excluded):
            continue
        if col in ORACLE_ONLY:
            if allow_oracle:
                keep.append(col)
            continue
        if any(col.startswith(p) for p, _ in FORBIDDEN_PREFIXES):
            continue
        if col in FORBIDDEN_EXACT:
            continue
        if FREE_TEXT.search(col):
            continue
        keep.append(col)
    out = df[keep].copy()
    check(out, extra_excluded=extra_excluded, allow_oracle=allow_oracle)
    return out


def manifest(X: pd.DataFrame, target: str, block: str) -> pd.DataFrame:
    """The frozen feature manifest a human reads once, carefully, in Phase 0."""
    return pd.DataFrame({
        "target": target,
        "block": block,
        "feature": X.columns,
        "dtype": [str(d) for d in X.dtypes],
        "non_null": X.notna().sum().to_numpy(),
        "n_unique": X.nunique(dropna=True).to_numpy(),
    })
