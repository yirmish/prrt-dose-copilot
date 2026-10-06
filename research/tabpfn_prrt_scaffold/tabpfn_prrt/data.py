"""Phase 0: assembly, encoding repairs and the frozen feature manifest.

This is where the study is won or lost. Every repair below is traceable to a numbered
section of the dataset companion document.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import leakage
from .ids import assert_join_key_is_dicom_source, base_id, cohort_summary


# =====================================================================================
# Loading
# =====================================================================================
def load_v8(target: C.TargetSpec, root: Path = C.V8_ROOT) -> pd.DataFrame:
    path = root / target.folder / f"{target.folder}_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the Deidentified_Export under {C.DATA_ROOT}."
        )
    # "NA" is a RECORDED VALUE in these files, not a missing marker (Companion 8.3).
    df = pd.read_csv(path, keep_default_na=True, na_values=[""], dtype={"pt_id": "Int64"})
    if target.row_filter:
        df = df.query(target.row_filter).reset_index(drop=True)
    return df


def find_treatments_xlsx(root: Path = C.DATA_ROOT) -> Path:
    """Locate the treatment sheet without trusting either of its two filename spellings."""
    hits = sorted(root.rglob(C.TREATMENTS_XLSX_GLOB))
    if not hits:
        raise FileNotFoundError(
            f"no file matching {C.TREATMENTS_XLSX_GLOB!r} under {root.resolve()}. "
            "Set PRRT_DATA_ROOT (env var or .env) to the export folder."
        )
    if len(hits) > 1:
        warnings.warn(f"several treatment sheets found; using {hits[0].name}")
    return hits[0]


def load_treatments_xlsx(path: Path | None = None) -> pd.DataFrame:
    """The enriched treatment sheet. Weight and height are complete here (86/86),
    unlike gen_weight_at_baseline_pet_kg in the V8 files (51/86)."""
    path = Path(path) if path is not None else find_treatments_xlsx()
    x = pd.read_excel(path)
    assert_join_key_is_dicom_source(x)
    # NOTE: `tretment_date` is deliberately NOT taken from the Excel sheet. The V8
    # datasets already carry `gen_prrt_injection_date`, so renaming it here produced a
    # column collision in the merge -> pandas suffixed both to `_x`/`_y`, the name
    # `gen_prrt_injection_date` then existed in neither, and run.py's
    #   if "gen_prrt_injection_date" in df.columns
    # guard silently skipped the entire pseudo-external validation (plan 8.3).
    keep = [c for c in ["pt_dicom_source", "pt_weight_kg", "pt_height_m",
                        "Age_at_treatment", "Age_at_PET",
                        "gender (1=Male, 2=Female)", "has_spleen",
                        "drug_manufacturer"] if c in x.columns]
    out = x[keep].dropna(subset=["pt_dicom_source"]).copy()
    out["pt_dicom_source"] = out["pt_dicom_source"].astype("Int64")
    return out.rename(columns={"pt_dicom_source": C.V8_JOIN_KEY})


# =====================================================================================
# Encoding repairs
# =====================================================================================
def repair_encodings(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Grade: 0 means UNAVAILABLE, not grade 0. A model reading it as a grade below G1
    # inverts the effect (Companion 7.3).
    if "gen_tumor_grade_code_0unavail" in d:
        g = pd.to_numeric(d["gen_tumor_grade_code_0unavail"], errors="coerce")
        d["gen_tumor_grade_code_0unavail"] = g.replace(0, np.nan).astype("category")

    # Predominant mets: compound values such as '1+4'; 0 also means unavailable.
    if "gen_predominant_organ_mets_code_compound" in d:
        s = d["gen_predominant_organ_mets_code_compound"].astype("string")
        for code, label in {1: "liver", 2: "bone", 3: "lung", 4: "nodes",
                            5: "peritoneum", 6: "brain", 7: "skin", 8: "other"}.items():
            d[f"mets_{label}_0_1"] = s.fillna("").str.split("+").apply(
                lambda parts, c=code: float(str(c) in [p.strip() for p in parts])
                if parts != [""] else np.nan)
        d = d.drop(columns=["gen_predominant_organ_mets_code_compound"])

    # Chromogranin A is a LAB VALUE that arrives as text: 46 of its 50 recorded entries
    # coerce to a number, the other four are 'na' and the left-censored '<27'. Left as an
    # object column it became a 34-level "category" -- a continuous biomarker silently
    # modelled as nominal levels, which is worse than dropping it.
    #   'na'  -> missing (Companion 8.3: the literal string is a recorded "not
    #            applicable", but it carries no concentration, so it cannot be a number)
    #   '<27' -> 13.5, i.e. LOD/2, the conventional substitution for a left-censored
    #            assay result. Recorded here because it is a substitution, not a
    #            measurement.
    if "gen_chromogranin_a_pre_prrt_ng_ml" in d:
        s = d["gen_chromogranin_a_pre_prrt_ng_ml"].astype("string").str.strip()
        lod_half = s.str.fullmatch(r"<\s*([0-9.]+)").fillna(False)
        # astype(float) BEFORE the substitution: to_numeric on an all-integer string
        # column yields nullable Int64, into which 13.5 cannot be assigned.
        v = pd.to_numeric(s, errors="coerce").astype("Float64").astype(float)
        limits = pd.to_numeric(s.str.extract(r"<\s*([0-9.]+)")[0], errors="coerce").astype(float)
        d["gen_chromogranin_a_pre_prrt_ng_ml"] = np.where(
            lod_half.to_numpy(bool), limits.to_numpy(float) / 2.0, v)

    # MIM reports PEARSON kurtosis (normal = 3); scipy/pandas report excess (normal = 0).
    # NOTE (2026-10-06, kept as run so the reported results reproduce): the delivered values
    # violate excess >= skewness^2 - 2 after this subtraction, i.e. the PET software already
    # reports EXCESS kurtosis and the -3 below is a constant offset. Tree models are invariant to
    # it; for TabPFN's preprocessing the effect is expected to be negligible but was not tested.
    # The public demo tables (data/synthetic/) undo it. See results/real_cohort/README.md.
    if "pet_kurtosis" in d:
        d["pet_kurtosis_excess"] = pd.to_numeric(d["pet_kurtosis"], errors="coerce") - 3.0
        d = d.drop(columns=["pet_kurtosis"])

    # tumor_location is an EPISTEMIC variable. Only `liver` and `bone` are positive
    # assertions; everything else asserts only 'not liver and not bone'. The `stomach`
    # label must not reach any output (Companion 7.4).
    if "tumor_location" in d:
        loc = d["tumor_location"].astype("string").str.lower().str.strip()
        # .eq()/.isin() on StringDtype propagate pd.NA, and np.where cannot evaluate it
        # ("boolean value of NA is ambiguous"). Resolve to real booleans first, then put
        # the genuinely missing rows back as NaN rather than silently calling them
        # "neither" -- absent is not the same assertion as "not liver and not bone".
        is_liver = loc.eq("liver").fillna(False).to_numpy(bool)
        is_bone = loc.isin(["bone", "ribs"]).fillna(False).to_numpy(bool)
        d["tumor_location_3lvl"] = pd.Series(
            np.where(is_liver, "liver", np.where(is_bone, "bone", "neither")),
            index=d.index).astype("category")
        unknown = loc.isna().to_numpy(bool) | loc.eq("bone-liver").fillna(False).to_numpy(bool)
        d.loc[unknown, "tumor_location_3lvl"] = np.nan   # incl. one lesion whose location could not be resolved

    # Spleen: eight "extensive" columns carry 0 rather than NaN for the 14 patients with
    # no spleen. Zero is a legitimate uptake value; asserting it is a false statement.
    if "gen_has_spleen_0_1" in d:
        no_spleen = pd.to_numeric(d["gen_has_spleen_0_1"], errors="coerce").eq(0)
        spleen_cols = [c for c in d.columns if c.startswith("voi_spleen__")]
        d.loc[no_spleen, spleen_cols] = np.nan

    # The clinical block is missing for the SAME 22 treatments (added later, never entered
    # into the source clinical database). Not MCAR. Flag it, never impute it.
    clin = [c for c in C.BLOCK_E_CLINICAL if c in d.columns and c != "clinical_block_present"]
    if clin:
        d["clinical_block_present"] = d[clin].notna().any(axis=1).astype(float)

    return d


def add_protocol_era(df: pd.DataFrame) -> pd.DataFrame:
    """The acquisition-protocol era, taken from the protocol itself, not the calendar.

    The plan (8.3) specified the era split by injection year -- early 2018-2020 versus
    late 2022-2024 -- on the understanding that this reproduces the documented 43/43
    three-time-point versus single-time-point split (Companion 2.3). In the delivered
    data it does not:

        injection year   2018  2019  2020  2021  2022  2023  2024
        treatments          6    10    12    21    15    11    11

    Calendar year gives 28 early / 37 late and discards the 21 treatments in 2021 as a
    "gap" -- a quarter of the cohort. 2021 is in fact the year the protocol changed, and
    it straddles it (15 three-time-point, 6 single-time-point).

    `gen_spect_hours_injection_to_scan_1` is populated only when a scan 1 was acquired,
    i.e. only under the three-time-point protocol (Companion 8.3 lists it as empty for
    43 treatments = the single-time-point protocol). It therefore identifies the
    acquisition era directly and reproduces the documented 43/43 split exactly.

    The label is a split variable only; it is never a feature.
    """
    d = df.copy()
    col = "gen_spect_hours_injection_to_scan_1"
    if col not in d.columns:
        warnings.warn(f"{col} absent; protocol era unavailable")
        d["protocol_era"] = pd.Series([pd.NA] * len(d), dtype="string")
        return d
    has_scan1 = pd.to_numeric(d[col], errors="coerce").notna()
    d["protocol_era"] = pd.Series(
        np.where(has_scan1, "3pts_early", "1pt_late"), index=d.index).astype("string")
    return d


def resolve_creatinine(df: pd.DataFrame) -> pd.DataFrame:
    """DEC-OPEN-027. The column mixes mg/dL and umol/L (0.5, 0.6 next to 46, 71, 112).

    Renal function is one of very few covariates with an established relationship to
    kidney absorbed dose (Svensson 2015), so leaving it unresolved discards the best
    clinical predictor in the file. Values in the ambiguous band become missing.
    """
    d = df.copy()
    col = "gen_baseline_creatinine_units_unresolved"
    if col not in d:
        d["calc_egfr"] = np.nan
        return d

    v = pd.to_numeric(d[col], errors="coerce")

    # 0.0 is a NOT-RECORDED sentinel, not a creatinine measurement: 57 of the 86
    # treatments carry it. Left as a number it passes the `v < 5` mg/dL branch, gives
    # scr = 0, and CKD-EPI raises (scr/k) to a NEGATIVE exponent -> +inf eGFR for
    # two thirds of the cohort. A plausible-looking finite feature would have been
    # worse; +inf at least fails loudly.
    n_zero = int(v.eq(0).sum())
    v = v.mask(v <= 0)
    umol = np.where(v < 5, v * 88.4, np.where(v > 20, v, np.nan))
    n_amb = int(((v >= 5) & (v <= 20)).sum())
    if n_amb:
        warnings.warn(f"{n_amb} creatinine values in the ambiguous 5-20 band -> missing")
    scr_mgdl = umol / 88.4

    age = pd.to_numeric(d.get("gen_age_at_prrt_years"), errors="coerce")
    female = pd.to_numeric(d.get("gen_gender_code_1m_2f"), errors="coerce").eq(2)
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    ratio = scr_mgdl / k
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        egfr = (142 * np.minimum(ratio, 1) ** a * np.maximum(ratio, 1) ** -1.200
                * 0.9938 ** age * np.where(female, 1.012, 1.0))
    egfr = np.where(np.isfinite(egfr), egfr, np.nan)       # never ship a non-finite feature
    d["calc_egfr"] = egfr                                  # CKD-EPI 2021, race-free
    d["calc_scr_umol_l"] = umol

    n_ok = int(np.isfinite(egfr).sum())
    if n_ok < 0.5 * len(d):
        warnings.warn(
            f"calc_egfr computable for only {n_ok}/{len(d)} rows "
            f"({n_zero} creatinine values were the 0.0 not-recorded sentinel). "
            "Renal function is NOT usable as a modelling feature in this cohort; "
            "the plan's 6.1 premise does not hold against the delivered data."
        )
    return d


def add_body_composition(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    w = pd.to_numeric(d.get("pt_weight_kg"), errors="coerce")
    h = pd.to_numeric(d.get("pt_height_m"), errors="coerce")
    sex = pd.to_numeric(d.get("gen_gender_code_1m_2f"), errors="coerce")
    d["calc_bmi"] = w / h ** 2
    bmi = d["calc_bmi"]
    d["calc_lbm_kg"] = np.where(sex.eq(2),
                                9270 * w / (8780 + 244 * bmi),
                                9270 * w / (6680 + 216 * bmi))
    d["calc_sul_factor"] = d["calc_lbm_kg"] / w
    return d


# =====================================================================================
# Feature assembly
# =====================================================================================
def block_columns(df: pd.DataFrame, block: str, target: C.TargetSpec) -> list[str]:
    if block == "A":
        cols = [c for c in C.BLOCK_A_PET_ROW if c in df.columns]
    elif block == "B":
        cols = [c for c in df.columns if c.startswith(C.BLOCK_B_PREFIXES)]
        cols += [c for c in C.BLOCK_B_EXTRA if c in df.columns]
    elif block == "C":
        cols = [c for c in C.BLOCK_C_TREATMENT if c in df.columns]
    elif block == "D":
        cols = [c for c in C.BLOCK_D_BODY if c in df.columns]
    elif block == "E":
        cols = [c for c in C.BLOCK_E_CLINICAL if c in df.columns]
        cols += [c for c in df.columns if c.startswith("mets_")]
    else:
        raise ValueError(block)
    # target-specific exclusions (e.g. voi_tumor_burden__* at lesion level), and the
    # acquisition-geometry measurements that block B would otherwise re-admit in
    # flattened form (leakage.FORBIDDEN_MEASUREMENTS).
    return [c for c in dict.fromkeys(cols)
            if not any(c == e or c.startswith(e) for e in target.extra_excluded)
            and leakage._measurement_of(c) not in leakage.FORBIDDEN_MEASUREMENTS]


def build_features(df: pd.DataFrame, blocks: tuple[str, ...], target: C.TargetSpec,
                   *, include_grey_zone: bool = False,
                   allow_oracle: bool = False) -> pd.DataFrame:
    cols: list[str] = []
    for b in blocks:
        cols += block_columns(df, b, target)
    if target.unit == "lesion" and "tumor_location_3lvl" in df.columns:
        cols.append("tumor_location_3lvl")
    if include_grey_zone:
        cols += [c for c in C.GREY_ZONE_FEATURES if c in df.columns]
    if allow_oracle and "spect_scan_time_hours" in df.columns:
        cols.append("spect_scan_time_hours")

    X = df[list(dict.fromkeys(cols))].copy()

    for c in X.columns:
        if c in C.CATEGORICAL_FEATURES:
            X[c] = X[c].astype("category")
        elif str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")

    # Drop zero-variance columns -- they cost compute and can confuse the ensemble.
    nunique = X.nunique(dropna=True)
    X = X[[c for c in X.columns if nunique[c] > 1]]

    leakage.check(X, extra_excluded=target.extra_excluded, allow_oracle=allow_oracle)
    return X


def assemble(target: C.TargetSpec, *, v8_root: Path = C.V8_ROOT,
             xlsx: Path | None = None) -> pd.DataFrame:
    df = load_v8(target, v8_root)
    try:
        tx = load_treatments_xlsx(xlsx)
        before = len(df)
        overlap = (set(tx.columns) & set(df.columns)) - {C.V8_JOIN_KEY}
        assert not overlap, (
            f"Excel columns collide with V8 columns {sorted(overlap)}: pandas would "
            "suffix both to _x/_y and the un-suffixed name would exist in neither, "
            "silently disabling anything that looks it up by name."
        )
        df = df.merge(tx, on=C.V8_JOIN_KEY, how="left", validate="many_to_one")
        assert len(df) == before, "the Excel join changed the row count -- wrong key"
    except FileNotFoundError:
        warnings.warn("treatment xlsx not found; block D will be incomplete")

    df = add_protocol_era(df)

    df = repair_encodings(df)
    df = resolve_creatinine(df)
    df = add_body_composition(df)

    df.attrs["cohort"] = cohort_summary(df, C.V8_JOIN_KEY)
    df.attrs["groups"] = df[C.V8_JOIN_KEY].map(base_id).to_numpy()
    return df
