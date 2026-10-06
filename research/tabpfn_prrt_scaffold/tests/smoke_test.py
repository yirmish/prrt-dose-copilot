"""Smoke test on synthetic data: everything except the TabPFN/CatBoost arms.

    python tests/smoke_test.py

Verifies that the grouping decoder, the leakage assertions, the target transform, the
CV splitter and every metric behave as intended, without needing the real cohort.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabpfn_prrt import metrics as M                                    # noqa: E402
from tabpfn_prrt.cv import leave_one_out_splits, repeated_grouped_splits  # noqa: E402
from tabpfn_prrt.ids import base_id, cohort_summary, group_vector       # noqa: E402
from tabpfn_prrt.leakage import LeakageError, check, drop_forbidden     # noqa: E402
from tabpfn_prrt.models import LinearBaseline                           # noqa: E402
from tabpfn_prrt.targets import TargetTransform                         # noqa: E402

ok = lambda msg: print(f"  ok  {msg}")


def test_ids():
    # illustrative study numbers (the convention is [base_id][2-digit treatment year])
    assert base_id(510) == 510 and base_id(51018) == 510 and base_id(51023) == 510
    assert base_id(42) == 42
    assert base_id(51) == 51 and base_id(510) == 510          # no prefix matching
    df = pd.DataFrame({"pt_id": [51018, 51023, 51418, 51423, 7, 42, 300]})
    g = group_vector(df)
    assert list(g) == [510, 510, 514, 514, 7, 42, 300]
    s = cohort_summary(df)
    assert s == {"rows": 7, "treatments": 7, "patients": 5}
    ok("patient decoding, grouping and the rows/treatments/patients summary")


def test_leakage():
    X = pd.DataFrame({"pet_mean_suvbw": [1.0], "spect_volume_ml": [2.0]})
    try:
        check(X); raise SystemExit("leakage check failed to fire")
    except LeakageError:
        pass
    for bad in ["dos_volume_ml", "pet_slice_with_max", "pet_std_mean_ratio_refvoi",
                "gen_treatment_num_global_id", "spect_scan_time_hours",
                "gen_primary_tumor_origin_code_legacy", "Yirmi's Comments", "tumor_location"]:
            try:
                check(pd.DataFrame({"pet_mean_suvbw": [1.0], bad: [1.0]}))
                raise SystemExit(f"leakage check missed {bad}")
            except LeakageError:
                pass
    keep = drop_forbidden(pd.DataFrame({"pet_mean_suvbw": [1.0], "spect_volume_ml": [2.0],
                                        "voi_tumor_burden__x": [3.0]}),
                          extra_excluded=("voi_tumor_burden__",))
    assert list(keep.columns) == ["pet_mean_suvbw"]
    check(pd.DataFrame({"pet_mean_suvbw": [1.0], "spect_scan_time_hours": [162.0]}),
          allow_oracle=True)
    ok("leakage register fires on every forbidden block, and the oracle escape works")


def test_target_transform():
    rng = np.random.default_rng(0)
    dose = rng.lognormal(3.5, 0.8, 200)
    act = rng.uniform(1.9, 8.2, 200)
    tf = TargetTransform(normalise_by_activity=True, log=True)
    y = tf.forward(dose, act)
    q = np.column_stack([y, y, y])
    back = tf.inverse_quantiles(q, act)
    assert np.allclose(back[:, 0], dose, rtol=1e-8)
    assert tf.inverse_point_is_median()
    ok("Gy/GBq + log transform round-trips exactly through the quantile back-transform")


def test_cv():
    groups = np.repeat(np.arange(82), np.repeat([1, 2, 3, 4, 5], [10, 15, 14, 8, 35]))
    y = np.random.default_rng(1).normal(size=len(groups))
    splits = list(repeated_grouped_splits(y, groups, n_repeats=3, n_folds=5))
    assert len(splits) == 15
    for sp in splits:
        assert not set(groups[sp.train]) & set(groups[sp.test])
    covered = np.zeros(len(y), bool)
    for sp in (s for s in splits if s.repeat == 0):
        covered[sp.test] = True
    assert covered.all(), "one repeat must cover every row exactly once"
    labels = pd.Series(["early"] * 43 + ["late"] * 43)
    ext = list(leave_one_out_splits(labels, np.arange(86), "era"))
    assert len(ext) == 2 and all(len(s.test) == 43 for s in ext)
    ok("repeated grouped CV never leaks a patient; era split trains on one era, tests the other")


def test_metrics():
    rng = np.random.default_rng(2)
    n = 400
    # a predictive distribution that is calibrated BY CONSTRUCTION: y | mu ~ N(mu, sd)
    mu = rng.normal(size=n)
    sd = 0.72
    y = mu + sd * rng.normal(size=n)
    from scipy.stats import norm
    levels = np.asarray([i / 100 for i in range(1, 100)], float)
    q = mu[:, None] + sd * norm.ppf(levels)[None, :]

    dm = M.distribution_metrics(y, levels, q)
    assert 0.85 <= dm["cov_95"] <= 1.0, dm["cov_95"]
    assert 0.40 <= dm["cov_50"] <= 0.62, dm["cov_50"]
    assert dm["pit_ks"] < 0.15
    assert 0.5 < dm["calib_slope"] < 1.6

    # widening the intervals must WORSEN the proper score, never improve it
    crps_ok = M.crps_from_quantiles(y, levels, q)
    q_wide = mu[:, None] + 4 * sd * norm.ppf(levels)[None, :]
    assert M.crps_from_quantiles(y, levels, q_wide) > crps_ok, "CRPS is gameable -- bug"

    p = M.prob_exceeds(levels, q, 0.0)
    assert p.min() >= 0 and p.max() <= 1
    assert abs(np.mean((y >= 0).astype(float)) - p.mean()) < 0.12

    r2 = M.r2_pooled(y, mu)
    assert 0.2 < r2 < 0.95
    nb = M.net_benefit(y, levels, q, 0.0)
    assert {"pt", "net_benefit_model", "net_benefit_all"} <= set(nb.columns)
    mc = M.management_change(p)
    assert abs(sum(mc.values()) - 1.0) < 1e-9

    groups = np.repeat(np.arange(100), 4)
    bw = M.between_within_r2(y, mu, groups)
    assert set(["r2_overall", "r2_between", "r2_within"]) <= set(bw)

    d = M.paired_bootstrap_delta(y, mu, np.zeros(n), groups, n_boot=200)
    assert d["delta"] > 0 and d["ci_lo"] > -1
    h = M.holm({"a": 0.01, "b": 0.04, "c": 0.20})
    assert h["a"] <= h["b"] <= h["c"]
    ok("error, calibration, CRPS, decision, between/within and paired-bootstrap metrics")


def test_baseline_arm():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"pet_mean_suvbw": rng.lognormal(2.5, .6, 120),
                      "pet_volume_ml": rng.lognormal(2.0, 1.0, 120),
                      "junk": rng.normal(size=120)})
    y = 0.8 * np.log(X.pet_mean_suvbw) + 0.2 * np.log(X.pet_volume_ml) + rng.normal(0, .5, 120)
    m = LinearBaseline(target_name="tumor_burden").fit(X, y.to_numpy())
    levels = np.asarray([0.025, 0.5, 0.975])
    q = m.predict_quantiles(X, levels)
    assert q.shape == (120, 3)
    assert (q[:, 0] < q[:, 1]).all() and (q[:, 1] < q[:, 2]).all()
    assert M.r2_pooled(y, q[:, 1]) > 0.4
    ok("M0 baseline fits, orders its quantiles and recovers a known signal")


if __name__ == "__main__":
    print("smoke test")
    for fn in (test_ids, test_leakage, test_target_transform, test_cv,
               test_metrics, test_baseline_arm):
        fn()
    print("\nall checks passed")
