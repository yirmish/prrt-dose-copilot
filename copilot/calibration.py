"""calibration.py - a LIVE calibration check: grouped K-fold cross-validation on the context table.

For every held-out row the back-end returns its predictive quantiles without having seen that row
(or any row of the same synthetic patient). We then ask the question the product rests on:
does a nominal 80% interval contain the measured dose 80% of the time?

On the hosted demo this runs on the SYNTHETIC cohort. The figures for the real cohort are
precomputed aggregates (results/real_cohort/), measured the same way with person-grouped folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .engine import ACTIVITY, GROUP, LEVELS, DoseModel, log_target, p_exceed

NOMINAL = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def oof_quantiles(target: str, df: pd.DataFrame, backend: str, k: int = 5, seed: int = 0,
                  n_estimators: int = 8, progress=None) -> tuple[np.ndarray, np.ndarray]:
    Q, y, _ = oof_frame(target, df, backend, k, seed, n_estimators, progress)
    return Q, y


def oof_frame(target: str, df: pd.DataFrame, backend: str, k: int = 5, seed: int = 0,
              n_estimators: int = 8, progress=None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Out-of-fold quantiles, log targets and the matching rows (same order)."""
    y = log_target(df)
    ok = np.isfinite(y)
    df, y = df.loc[ok].reset_index(drop=True), y[ok]
    Q = np.full((len(df), len(LEVELS)), np.nan)
    folds = GroupKFold(n_splits=k, shuffle=True, random_state=seed).split(df, y, df[GROUP])
    for i, (tr, te) in enumerate(folds):
        m = DoseModel(target, df.iloc[tr].reset_index(drop=True), backend, n_estimators, seed).fit()
        Q[te] = m.log_quantiles(df.iloc[te].reset_index(drop=True))
        if progress:
            progress((i + 1) / k)
    return Q, y, df


def central_coverage(Q: np.ndarray, y: np.ndarray, nominal: float) -> float:
    lo = np.array([np.interp((1 - nominal) / 2, LEVELS, q) for q in Q])
    hi = np.array([np.interp(1 - (1 - nominal) / 2, LEVELS, q) for q in Q])
    return float(np.mean((y >= lo) & (y <= hi)))


def width_fold(Q: np.ndarray, nominal: float) -> float:
    lo = np.array([np.interp((1 - nominal) / 2, LEVELS, q) for q in Q])
    hi = np.array([np.interp(1 - (1 - nominal) / 2, LEVELS, q) for q in Q])
    return float(np.exp(np.median(hi - lo)))


def crps(Q: np.ndarray, y: np.ndarray) -> float:
    """CRPS approximated as twice the mean pinball loss over the 1..99% grid (log scale)."""
    u = y[:, None] - Q
    pin = np.maximum(LEVELS[None, :] * u, (LEVELS[None, :] - 1) * u)
    return float(2 * pin.mean())


def r2(Q: np.ndarray, y: np.ndarray) -> float:
    med = np.array([np.interp(0.5, LEVELS, q) for q in Q])
    return float(1 - ((y - med) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def report(Q: np.ndarray, y: np.ndarray) -> dict:
    return {"reliability": [(n, central_coverage(Q, y, n)) for n in NOMINAL],
            "cov50": central_coverage(Q, y, 0.5), "cov80": central_coverage(Q, y, 0.8),
            "cov95": central_coverage(Q, y, 0.95), "width95_fold": width_fold(Q, 0.95),
            "crps_log": crps(Q, y), "log_r2": r2(Q, y), "n": int(len(y))}


CUTOFFS = np.round(np.concatenate([np.arange(0.50, 0.95, 0.01), np.arange(0.95, 0.996, 0.005)]), 3)


def decision_curve(Q: np.ndarray, d: pd.DataFrame, thresholds, cutoffs=CUTOFFS) -> pd.DataFrame:
    """Risk-coverage for threshold calls. For every held-out row and clinical threshold, P(dose > t)
    is read off the out-of-fold quantiles; a call is decisive at cutoff c when P >= c or P <= 1 - c.
    Returns, per cutoff, the share of (row, threshold) pairs called decisive and the share of those
    calls that were right. A model that is better at knowing when to measure sits higher at the
    same share decisive; an overconfident one makes many decisive calls at lower accuracy."""
    if not thresholds:
        return pd.DataFrame(columns=["cutoff", "frac_decisive", "acc_decisive", "n_decisive", "n_pairs"])
    act = d[ACTIVITY].to_numpy(float)
    dose = np.exp(log_target(d)) * act
    P, T = [], []
    for t in thresholds:
        P.append([p_exceed(q, t, a) for q, a in zip(Q, act)])
        T.append(dose > t)
    P, T = np.concatenate(P), np.concatenate(T)
    rows = []
    for c in cutoffs:
        above, below = P >= c, P <= 1 - c
        dec = above | below
        right = (above & T) | (below & ~T)
        nd = int(dec.sum())
        rows.append({"cutoff": float(c), "frac_decisive": nd / len(P),
                     "acc_decisive": (int(right.sum()) / nd) if nd else np.nan,
                     "n_decisive": nd, "n_pairs": int(len(P))})
    return pd.DataFrame(rows)
