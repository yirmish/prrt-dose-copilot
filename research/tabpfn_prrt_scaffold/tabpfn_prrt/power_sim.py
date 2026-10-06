"""Phase 1: resolvability simulation. Run this BEFORE any real model is fitted.

Review gap G12: the entire field lacks a single sample-size calculation for absorbed-dose
regression. Supplying one is a standalone methods contribution, and it is what sets
config.SUPERIORITY_MARGIN_R2.

Two DIFFERENT questions, which are routinely conflated and have very different answers:

  Q1 (marginal). How precise is a single reported R^2? This governs the confidence
      interval you must print next to "our model achieved R^2 = 0.31", and it governs
      any comparison against a NUMBER FROM ANOTHER STUDY.

  Q2 (paired). How small a difference between two arms can this cohort resolve, when
      both arms are evaluated on the same rows and the same folds? The dataset noise is
      common to both arms and cancels, so this is far smaller than Q1. This is what
      governs "did TabPFN beat the baseline".

Reporting Q1's margin for a within-study comparison is needlessly conservative;
reporting Q2's margin when quoting a literature R^2 is wrong. Both are computed here.

    python -m tabpfn_prrt.power_sim
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cv import repeated_grouped_splits
from .metrics import r2_pooled

# Observed cluster-size distribution of the lesion cohort: gate k = 1..5 for
# 10 / 15 / 14 / 8 / 39 treatments (Companion 5.3).
LESION_CLUSTERS = np.repeat([1, 2, 3, 4, 5], [10, 15, 14, 8, 39])

Z_ALPHA, Z_POWER = 1.959963985, 0.8416212336      # two-sided 0.05, 80% power
K = Z_ALPHA + Z_POWER                              # 2.80


def make_groups(design: str) -> np.ndarray:
    if design == "treatment":            # tumor_burden, kidneys, liver, marrow: n = 86
        return np.arange(86)
    if design == "treatment_spleen":     # spleen: n = 72
        return np.arange(72)
    if design == "lesion":               # tumors: 309 rows, 86 patient-courses
        return np.repeat(np.arange(len(LESION_CLUSTERS)), LESION_CLUSTERS)
    raise ValueError(design)


def simulate(groups: np.ndarray, true_r2: float, icc: float, rng,
             residual_icc: float = 0.0):
    """One synthetic dataset.

    `icc`           share of the SIGNAL that lives at patient level (near-inert, see (1)).
    `residual_icc`  share of the RESIDUAL that lives at patient level. This is the
                    parameter the project's REML estimate speaks to, and the one that
                    moves the lesion-level margin.
    """
    n, n_g = len(groups), len(np.unique(groups))
    sig = np.sqrt(icc) * rng.normal(size=n_g)[groups] + np.sqrt(1 - icc) * rng.normal(size=n)
    sig = (sig - sig.mean()) / sig.std()
    a = np.sqrt(true_r2 / (1 - true_r2))
    resid = (np.sqrt(residual_icc) * rng.normal(size=n_g)[groups]
             + np.sqrt(1 - residual_icc) * rng.normal(size=n))
    return sig, a * sig + resid


def _oof_ols(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
             n_repeats: int, n_folds: int) -> np.ndarray:
    """Out-of-fold predictions from OLS on a single feature, pooled per repeat."""
    preds = np.full((n_repeats, len(y)), np.nan)
    for sp in repeated_grouped_splits(y, groups, n_repeats=n_repeats,
                                      n_folds=n_folds, stratify=False):
        A = np.column_stack([np.ones(len(sp.train)), x[sp.train]])
        coef, *_ = np.linalg.lstsq(A, y[sp.train], rcond=None)
        preds[sp.repeat, sp.test] = np.column_stack(
            [np.ones(len(sp.test)), x[sp.test]]) @ coef
    return preds


# ---------------------------------------------------------------------------------
# 2026-09-20 (DEC-226). Two corrections, the second of which changes the margin.
#
# (1) `icc` here structures the SIGNAL only -- `simulate()` builds a clustered predictor
#     and then adds an i.i.d. per-row residual. Both arms are scored on the same y, and
#     the sampling variability of a paired R^2 difference is driven by that residual, so
#     varying `icc` does essentially nothing. Measured: lesion paired MDD 0.071 at
#     icc = 0.60 and 0.071 at 0.79; treatment 0.147 vs 0.144. The margin is robust to
#     this assumption, which is worth stating rather than leaving implicit.
#
# (2) The quantity the PROJECT measured is a different one: REML ICC ~ 0.79 of the
#     OUTCOME -- "effective n ~= the number of patients (~82)". That is a clustered
#     RESIDUAL, and the original simulation has none, so it could not represent the
#     cohort's actual structure. Adding a patient-level residual term moves the lesion
#     design substantially:
#
#       residual ICC   lesion paired MDD   lesion single-R2 95% CI
#       0.00                 0.064                +/- 0.080
#       0.40                 0.077                +/- 0.114
#       0.79                 0.091                +/- 0.142
#
#       treatment level is unaffected (0.132 -> 0.138): one row per patient, so there is
#       no within-cluster structure to have.
#
#     Consequences: the frozen lesion margin of 0.07 was derived without residual
#     clustering and is too small -- it should be ~0.09 at the measured ICC. No reported
#     conclusion flips (every lesion-level result is null, and a larger margin only makes
#     a null safer; the largest lesion delta measured was +0.037). But the lesion-level
#     confidence interval printed in Plan 11, +/- 0.07-0.08, is understated by nearly a
#     factor of two, and the plan's fourth consequence -- "`tumors` at n = 309 is roughly
#     twice as precise as `tumor_burden` at n = 86" -- does NOT survive: at residual
#     ICC 0.79 the two designs are nearly equally precise, which is exactly what
#     "effective n ~= the number of patients" means. That sentence must be withdrawn.
#
# Outputs: results/phase1/resolvability_signal_icc_2026-09-20.csv and
#          results/phase1/resolvability_residual_icc_2026-09-20.csv
# ---------------------------------------------------------------------------------
ICC_MEASURED = 0.79              # REML, this cohort -- applies to the RESIDUAL
SIGNAL_ICC_SCENARIOS = (0.60, 0.79)
RESIDUAL_ICC_SCENARIOS = (0.00, 0.40, ICC_MEASURED)


def run(designs=("treatment", "lesion"),
        true_r2_grid=(0.10, 0.25, 0.40, 0.55),
        delta_true: float = 0.05,
        icc: float = 0.6, residual_icc: float = ICC_MEASURED, n_sim: int = 400,
        n_repeats: int = 5, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Arm A sees the true signal; arm B sees an attenuated copy whose population R^2 is
    exactly `delta_true` lower. Both are scored on the same rows and the same folds."""
    rng = np.random.default_rng(seed)
    rows = []
    for design in designs:
        groups = make_groups(design)
        for r in true_r2_grid:
            if r - delta_true <= 0.01:
                continue
            c2 = delta_true / (r - delta_true)          # attenuation giving R2_B = r - delta
            est_a, est_delta = [], []
            for _ in range(n_sim):
                sig, y = simulate(groups, r, icc, rng, residual_icc=residual_icc)
                x_b = sig + np.sqrt(c2) * rng.normal(size=len(sig))
                pa = _oof_ols(sig, y, groups, n_repeats, n_folds)
                pb = _oof_ols(x_b, y, groups, n_repeats, n_folds)
                ra = np.median([r2_pooled(y, pa[i]) for i in range(n_repeats)])
                rb = np.median([r2_pooled(y, pb[i]) for i in range(n_repeats)])
                est_a.append(ra)
                est_delta.append(ra - rb)
            est_a, est_delta = np.asarray(est_a), np.asarray(est_delta)
            sd_marginal = float(est_a.std(ddof=1))
            sd_paired = float(est_delta.std(ddof=1))
            rows.append({
                "design": design, "icc": icc, "residual_icc": residual_icc,
                "n_rows": len(groups),
                "n_patients": int(len(np.unique(groups))),
                "true_r2": r,
                "mean_est_r2": float(est_a.mean()),
                "bias": float(est_a.mean() - r),
                "sd_r2_marginal": sd_marginal,
                "ci95_halfwidth_r2": float(Z_ALPHA * sd_marginal),
                "sd_delta_paired": sd_paired,
                "mdd_paired": float(K * sd_paired),          # min detectable difference, paired
                "mdd_vs_other_study": float(K * sd_marginal * np.sqrt(2)),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    out = pd.concat([run(residual_icc=v) for v in RESIDUAL_ICC_SCENARIOS],
                    ignore_index=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(sys.argv) > 1:
        out.to_csv(sys.argv[1], index=False)
        print(f"\nwrote {sys.argv[1]}")

    # The margin is set from the MEASURED residual ICC, not from an unclustered residual.
    m = out[out.residual_icc == ICC_MEASURED]
    for design in sorted(out.design.unique()):
        for v in sorted(out.residual_icc.unique()):
            t = out[(out.residual_icc == v) & (out.design == design)]
            print(f"{design:10s} residual ICC {v:.2f} -> paired MDD {t.mdd_paired.max():.3f}"
                  f" | single-R2 95% CI +/- {t.ci95_halfwidth_r2.max():.3f}")
    mdd_paired = m["mdd_paired"].max()
    halfwidth = m["ci95_halfwidth_r2"].max()
    mdd_ext = m["mdd_vs_other_study"].max()
    print(f"""
Q2  within-study, paired, same folds  -> smallest resolvable Delta R^2 : {mdd_paired:.3f}
      => set config.SUPERIORITY_MARGIN_R2 to this, then FREEZE it.

Q1  a single reported R^2 carries a 95% CI of roughly +/- {halfwidth:.3f} at this n.
      => print that interval next to every headline R^2. A comparison against a
         published R^2 from another cohort cannot resolve less than {mdd_ext:.3f}.
""")
