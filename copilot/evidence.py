"""evidence.py - read the REAL-cohort aggregates shipped in results/real_cohort/.

Every number the app shows about the real cohort comes from one of these files; nothing is typed in.
Real cohort: 86 cycle-1 treatments, 82 patients, 309 lesions, one centre. 5 x 5 person-grouped
cross-validation, target log(Gy/GBq). Run of 2026-10-06: 5 of the protocol's 20 repeats - a
preliminary measurement, to be confirmed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results" / "real_cohort"
V35 = ROOT / "v35_2026-10-06"
VAL = ROOT / "validation"
LESION8 = ROOT / "lesion_8models_2026-10-03"
LESION8_FIG = LESION8 / "fig_lesion_8models.png"


def arm_label(arm: str) -> str:
    if arm.startswith("TabPFN-V3_5"):
        return "TabPFN-3.5"
    if arm.startswith("TabPFN-V2"):
        return "TabPFN-2 (open weights)"
    if arm.startswith("CatBoost"):
        return "CatBoost (tuned, quantile heads)"
    if arm.startswith("M0"):
        return "Physical baseline"
    return arm


def m1_summary() -> pd.DataFrame:
    d = pd.read_csv(V35 / "v35_summary.csv")
    d["model"] = d["arm"].map(arm_label)
    d["features"] = "M1 (17 PET features)"
    return d


def m3_summary() -> pd.DataFrame:
    d = pd.read_csv(V35 / "v35_M3_summary_from_results_raw.csv")
    d["model"] = d["arm"].map(arm_label)
    d["features"] = "M3 (~165 features)"
    return d.rename(columns={"log_cov_50": "cov_50", "log_cov_80": "cov_80", "log_cov_95": "cov_95",
                             "log_crps": "crps_log", "log_pit_ks": "pit_ks"})


def m3_deltas() -> pd.DataFrame:
    return pd.read_csv(V35 / "v35_M3_comparisons_vs_M0.csv")


def m1_deltas() -> pd.DataFrame:
    return pd.read_csv(V35 / "v35_paired_deltas.csv")


def reliability() -> pd.DataFrame:
    d = pd.read_csv(V35 / "v35_reliability_curve.csv")
    d["model"] = d["arm"].map(arm_label)
    return d


def decisive() -> pd.DataFrame:
    d = pd.read_csv(V35 / "v35_decisive_calls.csv")
    d["model"] = d["arm"].map(arm_label)
    return d


def leakage() -> pd.DataFrame:
    return pd.read_csv(VAL / "leakage_summary.csv", encoding="utf-8-sig")


# --- lesion level, 140-feature table, eight models (study amendment R, 2-4 Oct 2026) -------------
# 307 lesions / 85 treatments / 81 patients; 5x5 CV grouped by patient; six conventional models
# nested-tuned with split-conformal intervals; TabPFN-3.5 untuned with native quantiles.

def lesion8_r2() -> pd.DataFrame:
    """R2 (log scale) for eight models x six targets."""
    return pd.read_csv(LESION8 / "r2_by_model_and_target.csv")


def lesion8_head_to_head() -> pd.DataFrame:
    """TabPFN-3.5 minus each comparator: R2, within-patient R2, CRPS, with patient-bootstrap CIs."""
    return pd.read_csv(LESION8 / "lesion_head_to_head_tabpfn35.csv")


def lesion8_components() -> pd.DataFrame:
    """Between/within-patient R2, CRPS and interval coverage per model (lesion level)."""
    return pd.read_csv(LESION8 / "lesion_components_and_calibration.csv")


def lesion8_n_estimators() -> pd.DataFrame:
    """Registered robustness check: 8 vs 32 estimators."""
    return pd.read_csv(LESION8 / "n_estimators_8_vs_32.csv")


def m3_all_arms() -> pd.DataFrame:
    """M3 (~165 features), every stored arm on identical rows, folds and features."""
    d = pd.read_csv(V35 / "v35_M3_all_arms_summary.csv")
    d["model"] = d["arm"].map(arm_label)
    d = d[d["model"] != d["arm"]]                       # drop arms without a display name (spleen top-k variants)
    d["features"] = "M3 (~165 features)"
    return d.reset_index(drop=True)


def m3_decisive() -> pd.DataFrame:
    d = pd.read_csv(V35 / "v35_M3_all_arms_decisive_calls.csv")
    d["model"] = d["arm"].map(arm_label)
    return d


def m3_vs_catboost(target: str) -> dict | None:
    """TabPFN-3.5 against tuned CatBoost at M3 for one target (None where no CatBoost arm is stored).
    Post-hoc, from the stored out-of-fold quantiles; patient-clustered bootstrap."""
    r2 = pd.read_csv(V35 / "v35_M3_all_arms_paired_deltas.csv")
    r2 = r2[(r2.target == target) & r2.comparison.str.contains("CatBoost")]
    cr = pd.read_csv(V35 / "proper_scores__paired_crps_and_sharpness.csv")
    cr = cr[(cr.model_size == "M3") & (cr.target == target) & (cr.comparison == "TabPFN-3.5 minus CatBoost")]
    if not len(r2) or not len(cr):
        return None
    a, c = r2.iloc[0], cr.iloc[0]
    return {"r2_delta": float(a.delta_log_r2), "r2_lo": float(a.ci_lo), "r2_hi": float(a.ci_hi),
            "crps_delta": float(c.delta), "crps_lo": float(c.ci_lo), "crps_hi": float(c.ci_hi),
            "crps_rel": float(c.rel_change_pct)}
