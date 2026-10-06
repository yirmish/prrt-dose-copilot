"""Offline tests (no token, no weights): the physical baseline exercises every code path that the
TabPFN back-ends share; the TabPFN back-ends are exercised by scripts/smoke_test.py."""
import numpy as np
import pandas as pd
import pytest

from copilot import calibration as C
from copilot import evidence as E
from copilot.engine import (LEVELS, TARGETS, DoseModel, GBMTunedConformal, apply_whatif, feature_columns, load_context,
                            outside_context, p_exceed, split_examples)


@pytest.mark.parametrize("target", list(TARGETS))
def test_data_contract(target):
    d = load_context(target)
    assert (d["is_synthetic"] == 1).all()
    assert len(feature_columns(d)) == 17
    for c in ["gen_age_at_prrt_years", "gen_gender_code_1m_2f", "pt_height_m", "gen_days_pet_to_prrt"]:
        assert c not in d.columns                        # quasi-identifiers are not shipped
    assert (d["pt_weight_kg"] >= 40).all()
    assert (d["pet_kurtosis"] >= d["pet_skewness"] ** 2 - 2 - 1e-6).all()


@pytest.mark.parametrize("target", list(TARGETS))
def test_predict_baseline(target):
    ctx, ex = split_examples(load_context(target))
    assert set(ctx["synthetic_patient_id"]).isdisjoint(ex["synthetic_patient_id"])
    m = DoseModel(target, ctx, backend="physical-baseline").fit()
    case = ex.iloc[0].to_dict()
    r = m.predict(case, float(case["gen_prrt_net_injected_activity_gbq"]))
    assert np.all(np.diff(r["quantiles_gy"]) >= -1e-9)
    assert r["interval95_gy"][0] <= r["interval80_gy"][0] <= r["median_gy"] <= r["interval80_gy"][1] <= r["interval95_gy"][1]
    for v in r["thresholds"].values():
        assert 0.005 <= v["p_exceed"] <= 0.995


def test_add_case_does_not_mutate():
    ctx, ex = split_examples(load_context("kidneys"))
    m = DoseModel("kidneys", ctx, backend="physical-baseline").fit()
    n = len(m.context)
    m2 = m.with_added_case(ex.iloc[0].to_dict())
    assert len(m.context) == n and len(m2.context) == n + 1


def test_activity_scales_dose_linearly():
    ctx, ex = split_examples(load_context("kidneys"))
    m = DoseModel("kidneys", ctx, backend="physical-baseline").fit()
    case = ex.iloc[0].to_dict()
    a, b = m.predict(case, 4.0), m.predict(case, 8.0)
    assert b["median_gy"] == pytest.approx(2 * a["median_gy"])


def test_whatif_keeps_totals_consistent():
    case = load_context("tumors").iloc[0].to_dict()
    w = apply_whatif(case, uptake=2.0, volume=0.5)
    assert w["pet_mean_suvbw"] == pytest.approx(2 * case["pet_mean_suvbw"])
    assert w["pet_tlg_suvbw_ml"] == pytest.approx(case["pet_tlg_suvbw_ml"])          # 2 x 0.5
    assert w["pet_skewness"] == case["pet_skewness"]


def test_p_exceed_monotone():
    q = np.log(np.exp(np.linspace(0, 3, len(LEVELS))))
    ps = [p_exceed(q, t, 1.0) for t in (1.5, 3, 6, 12)]
    assert all(x >= y for x, y in zip(ps, ps[1:]))


def test_live_calibration_baseline_is_sane():
    d = load_context("tumor_burden")
    Q, y = C.oof_quantiles("tumor_burden", d, "physical-baseline", k=5)
    rep = C.report(Q, y)
    assert 0.85 <= rep["cov95"] <= 1.0 and 0.65 <= rep["cov80"] <= 0.95


def test_evidence_files_load():
    assert len(E.m1_summary()) == 24 and len(E.m3_summary()) == 12 and len(E.m3_deltas()) == 6
    assert set(E.reliability()["model"]) >= {"TabPFN-3.5", "Physical baseline", "CatBoost (tuned, quantile heads)"}
    assert len(E.leakage()) == 3


def test_gbm_quantile_backend():
    ctx, ex = split_examples(load_context("kidneys"))
    m = DoseModel("kidneys", ctx, backend="gbm-quantile").fit()
    Q = m.log_quantiles(ex.head(5))
    assert Q.shape == (5, len(LEVELS)) and np.all(np.diff(Q, axis=1) >= -1e-12)


def test_lesion8_evidence_is_consistent():
    r2 = E.lesion8_r2().set_index("target")
    comp = E.lesion8_components().set_index("model")
    h = E.lesion8_head_to_head()
    tuned = ["CatBoost", "XGBoost", "HistGBM", "LightGBM", "RandomForest", "ElasticNet"]
    # the two lesion-level tables agree, and TabPFN-3.5 is first among all eight at lesion level
    assert abs(r2.loc["tumors", "TabPFN-3.5"] - comp.loc["TabPFN-3.5", "r2_overall"]) < 1e-9
    assert r2.loc["tumors", "TabPFN-3.5"] > r2.loc["tumors", tuned + ["TabPFN-2"]].max()
    # every tuned comparator: the R2 and CRPS intervals exclude zero in TabPFN-3.5's favour
    hr = h[(h.n_estimators == 8) & h.comparator.isin(tuned)]
    assert set(hr.comparator) == set(tuned)
    assert (hr[hr.metric == "R2"].ci_lo > 0).all() and (hr[hr.metric == "CRPS"].ci_hi < 0).all()
    # registered robustness check (32 estimators) passed
    r1 = h[(h.n_estimators == 32) & (h.metric == "R2")].iloc[0]
    assert r1.comparator == "CatBoost" and r1.ci_lo > 0
    assert (E.lesion8_n_estimators().delta_32_minus_8.abs() <= 0.005).all()
    assert E.LESION8_FIG.exists()


def test_gbm_tuned_conformal_is_calibrated_and_grouped():
    d = load_context("kidneys")
    Q, y, dd = C.oof_frame("kidneys", d, "gbm-tuned-conformal", k=5)
    rep = C.report(Q, y)
    assert np.all(np.diff(Q, axis=1) >= -1e-12)
    assert 0.88 <= rep["cov95"] <= 1.0 and 0.70 <= rep["cov80"] <= 0.92   # split-conformal: near nominal
    m = DoseModel("kidneys", d, backend="gbm-tuned-conformal").fit()
    assert m._model.params in GBMTunedConformal.GRID


def test_decision_curve_monotone():
    d = load_context("tumor_burden")
    Q, y, dd = C.oof_frame("tumor_burden", d, "physical-baseline", k=5)
    dc = C.decision_curve(Q, dd, TARGETS["tumor_burden"][1])
    assert len(dc) == len(C.CUTOFFS) and dc.n_pairs.iloc[0] == len(dd) * len(TARGETS["tumor_burden"][1])
    assert (np.diff(dc.frac_decisive) <= 1e-12).all()          # a stricter cutoff never adds decisive calls
    assert C.decision_curve(Q, dd, ()).empty                    # organs without a threshold: no curve


def test_api_transient_failure_is_retried(monkeypatch):
    """A transient API failure ("worker failed") is retried, re-sending the context first."""
    import tabpfn_client
    from copilot.engine import TabPFN35
    calls = {"fit": 0, "predict": 0}

    class Flaky:
        def __init__(self, **kw): pass

        @classmethod
        def create_default_for_version(cls, v, **kw): return cls(**kw)

        def fit(self, X, y):
            calls["fit"] += 1
            self.m = float(np.mean(y))
            return self

        def predict(self, X, output_type, quantiles):
            calls["predict"] += 1
            if calls["predict"] == 1:
                raise RuntimeError("Fail to call predict with error: streamed, The worker failed")
            return [np.full(len(X), self.m + q) for q in quantiles]

    monkeypatch.setattr(tabpfn_client, "TabPFNRegressor", Flaky)
    monkeypatch.setattr(TabPFN35, "API_WAIT_S", 0.0)
    ctx, ex = split_examples(load_context("kidneys"))
    m = DoseModel("kidneys", ctx, backend="tabpfn-3.5-api").fit()
    Q = m.log_quantiles(ex.head(3))
    assert Q.shape == (3, len(LEVELS)) and calls == {"fit": 2, "predict": 2}


def test_outside_context_flags_only_extrapolation():
    ctx, ex = split_examples(load_context("kidneys"))
    case = ex.iloc[0].to_dict()
    assert outside_context(case, pd.concat([ctx, ex])) == []
    flags = outside_context(apply_whatif(case, uptake=50.0, volume=1.0), ctx)
    assert flags and all("context table" in f for f in flags)


def test_m3_all_arms_evidence():
    assert len(E.m3_all_arms()) == 20
    cb = E.m3_vs_catboost("kidneys")
    assert cb and cb["crps_hi"] < 0 and cb["r2_lo"] < 0 < cb["r2_hi"]      # better distribution, same accuracy
    assert E.m3_vs_catboost("spleen") is None                               # no CatBoost arm stored there
