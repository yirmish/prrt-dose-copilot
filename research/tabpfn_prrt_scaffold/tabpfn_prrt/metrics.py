"""Three families of metric: error, calibration/uncertainty, decision.

Family 2 is the contribution of this study -- nothing in the published corpus reports
calibration, and exactly one paper reports a prediction interval at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# =====================================================================================
# 1. Error and discrimination
# =====================================================================================

def r2_pooled(y: np.ndarray, yhat: np.ndarray) -> float:
    """1 - SSE/SST on POOLED out-of-fold predictions.

    Deliberately not the mean of per-fold R^2: at ~17 test rows per fold that quantity
    is unstable and is not what the literature reports. State the definition in methods.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    sse = np.sum((y - yhat) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1.0 - sse / sst) if sst > 0 else np.nan


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def mrae(y, yhat, eps: float = 1e-9) -> float:
    """Mean relative absolute error -- the metric Akhavanallaf 2023/2025 report."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat) / np.maximum(np.abs(y), eps)))


def spearman(y, yhat) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(y, yhat).statistic)


def error_metrics(y, yhat, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}r2": r2_pooled(y, yhat),
        f"{prefix}mae": mae(y, yhat),
        f"{prefix}rmse": rmse(y, yhat),
        f"{prefix}mrae": mrae(y, yhat),
        f"{prefix}spearman": spearman(y, yhat),
    }


# =====================================================================================
# 2. Calibration and uncertainty
# =====================================================================================

def calibration_slope_intercept(y, yhat) -> tuple[float, float]:
    """OLS of observed on predicted.

    slope < 1: predictions too spread out. slope > 1: too compressed.
    Not reported anywhere in the published corpus (review gap G4).
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    A = np.column_stack([np.ones_like(yhat), yhat])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[1]), float(coef[0])


def pit_values(y: np.ndarray, q_levels: np.ndarray, q_pred: np.ndarray) -> np.ndarray:
    """Probability integral transform: P(Y <= y_obs) under each row's own predictive CDF.

    q_pred has shape (n_rows, n_quantiles), each row non-decreasing across quantiles.
    Uniform PIT  = calibrated. U-shaped = overconfident. Humped = underconfident.
    """
    y = np.asarray(y, float)
    q_levels = np.asarray(q_levels, float)
    q_pred = np.asarray(q_pred, float)
    out = np.empty(len(y))
    for i in range(len(y)):
        v = np.maximum.accumulate(q_pred[i])          # enforce monotonicity
        out[i] = float(np.interp(y[i], v, q_levels, left=0.0, right=1.0))
    return out


def coverage(y, q_levels, q_pred, nominal: float) -> dict[str, float]:
    """Empirical coverage of the central `nominal` interval, with a Wilson CI."""
    lo_lvl, hi_lvl = (1 - nominal) / 2, 1 - (1 - nominal) / 2
    lo = _quantile_at(q_levels, q_pred, lo_lvl)
    hi = _quantile_at(q_levels, q_pred, hi_lvl)
    y = np.asarray(y, float)
    inside = (y >= lo) & (y <= hi)
    p, n = float(inside.mean()), len(y)
    z = 1.959963985
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return {
        f"cov_{int(nominal*100)}": p,
        f"cov_{int(nominal*100)}_lo": float(max(0.0, centre - half)),
        f"cov_{int(nominal*100)}_hi": float(min(1.0, centre + half)),
        f"width_{int(nominal*100)}_median": float(np.median(hi - lo)),
    }


def _quantile_at(q_levels, q_pred, level: float) -> np.ndarray:
    q_levels = np.asarray(q_levels, float)
    q_pred = np.asarray(q_pred, float)
    j = int(np.argmin(np.abs(q_levels - level)))
    if abs(q_levels[j] - level) < 1e-9:
        return q_pred[:, j]
    return np.array([np.interp(level, q_levels, np.maximum.accumulate(row)) for row in q_pred])


def pinball(y: np.ndarray, qhat: np.ndarray, level: float) -> float:
    y, qhat = np.asarray(y, float), np.asarray(qhat, float)
    d = y - qhat
    return float(np.mean(np.maximum(level * d, (level - 1) * d)))


def crps_from_quantiles(y, q_levels, q_pred) -> float:
    """CRPS ~= 2 * mean pinball loss over a dense uniform quantile grid.

    A proper scoring rule: it rewards sharpness and calibration jointly, so it cannot be
    gamed by simply widening the intervals.
    """
    q_levels = np.asarray(q_levels, float)
    q_pred = np.asarray(q_pred, float)
    losses = [pinball(y, q_pred[:, j], float(q_levels[j])) for j in range(len(q_levels))]
    return float(2.0 * np.mean(losses))


def distribution_metrics(y, q_levels, q_pred, point=None,
                         nominals=(0.50, 0.80, 0.95)) -> dict[str, float]:
    out: dict[str, float] = {}
    if point is None:
        point = _quantile_at(q_levels, q_pred, 0.5)
    s, i = calibration_slope_intercept(y, point)
    out["calib_slope"] = s
    out["calib_intercept"] = i
    out["crps"] = crps_from_quantiles(y, q_levels, q_pred)
    pit = pit_values(y, q_levels, q_pred)
    out["pit_ks"] = _ks_uniform(pit)
    for nom in nominals:
        out.update(coverage(y, q_levels, q_pred, nom))
    return out


def _ks_uniform(pit: np.ndarray) -> float:
    """KS distance of the PIT from Uniform(0,1). 0 = perfectly calibrated."""
    from scipy.stats import kstest
    return float(kstest(pit, "uniform").statistic)


# =====================================================================================
# 3. Decision-analytic evaluation (review gap G8 -- currently empty in the field)
# =====================================================================================

def prob_exceeds(q_levels, q_pred, threshold: float) -> np.ndarray:
    """P(dose >= threshold) read off each row's own predictive CDF."""
    q_levels = np.asarray(q_levels, float)
    q_pred = np.asarray(q_pred, float)
    p = np.empty(len(q_pred))
    for i, row in enumerate(q_pred):
        v = np.maximum.accumulate(row)
        p[i] = 1.0 - float(np.interp(threshold, v, q_levels, left=0.0, right=1.0))
    return p


def threshold_metrics(y, q_levels, q_pred, threshold: float) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    label = (np.asarray(y, float) >= threshold).astype(int)
    p = prob_exceeds(q_levels, q_pred, threshold)
    out = {"thr": threshold, "prevalence": float(label.mean())}
    if 0 < label.mean() < 1:
        out["roc_auc"] = float(roc_auc_score(label, p))
        out["pr_auc"] = float(average_precision_score(label, p))
        out["brier"] = float(brier_score_loss(label, p))
    return out


def net_benefit(y, q_levels, q_pred, threshold_gy: float,
                pt_grid=np.linspace(0.05, 0.95, 19)) -> pd.DataFrame:
    """Decision curve for the binary question 'will this patient reach threshold_gy?'."""
    label = (np.asarray(y, float) >= threshold_gy).astype(int)
    p = prob_exceeds(q_levels, q_pred, threshold_gy)
    n = len(label)
    rows = []
    for pt in pt_grid:
        pred_pos = p >= pt
        tp = int(np.sum(pred_pos & (label == 1)))
        fp = int(np.sum(pred_pos & (label == 0)))
        nb = tp / n - (fp / n) * (pt / (1 - pt))
        nb_all = label.mean() - (1 - label.mean()) * (pt / (1 - pt))
        rows.append({"pt": float(pt), "net_benefit_model": float(nb),
                     "net_benefit_all": float(nb_all), "net_benefit_none": 0.0})
    return pd.DataFrame(rows)


def management_change(p_exceeds: np.ndarray, act_below: float = 0.3,
                      act_above: float = 0.7) -> dict[str, float]:
    """The single number a clinician reads: how many patients would be managed differently.

    Under the stated rule: escalate if P(reach target dose) < act_below, consider
    de-escalation / shorter course if P > act_above, otherwise no change.
    """
    p = np.asarray(p_exceeds, float)
    return {
        "frac_escalate": float(np.mean(p < act_below)),
        "frac_deescalate": float(np.mean(p > act_above)),
        "frac_unchanged": float(np.mean((p >= act_below) & (p <= act_above))),
    }


# =====================================================================================
# 4. Between- vs within-patient variance decomposition (lesion-level targets only)
# =====================================================================================

def between_within_r2(y, yhat, groups) -> dict[str, float]:
    """Split lesion-level skill into 'which patient' and 'which lesion within a patient'.

    If all the skill is between patients, lesion-level prediction is an illusion and the
    honest clinical claim is at patient level only. Stenvall 2022 found inter-patient
    r 0.71 against intra-patient r 0.45 for the raw correlation; nobody has done this
    for a model.
    """
    df = pd.DataFrame({"y": np.asarray(y, float),
                       "yhat": np.asarray(yhat, float),
                       "g": np.asarray(groups)})
    m = df.groupby("g")[["y", "yhat"]].transform("mean")
    pm = df.groupby("g")[["y", "yhat"]].mean()
    multi = df.groupby("g").size()
    multi_ids = multi[multi > 1].index

    out = {"r2_overall": r2_pooled(df.y, df.yhat),
           "r2_between": r2_pooled(pm.y, pm.yhat),
           "n_patients": int(df.g.nunique()),
           "n_patients_multilesion": int(len(multi_ids))}
    w = df[df.g.isin(multi_ids)]
    if len(w) > 2:
        wm = m.loc[w.index]
        out["r2_within"] = r2_pooled(w.y - wm.y, w.yhat - wm.yhat)
    else:
        out["r2_within"] = np.nan
    return out


# =====================================================================================
# 5. Paired comparison between arms -- bootstrap over PATIENTS, not rows
# =====================================================================================

def paired_bootstrap_delta(y, yhat_a, yhat_b, groups, n_boot: int = 2000,
                           seed: int = 0, stat=r2_pooled) -> dict[str, float]:
    """Delta(stat) between two arms, resampling patients (rows are clustered)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    ya, yb = np.asarray(yhat_a, float), np.asarray(yhat_b, float)
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx_by_group = {u: np.flatnonzero(g == u) for u in uniq}

    obs = stat(y, ya) - stat(y, yb)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[u] for u in picked])
        deltas[b] = stat(y[idx], ya[idx]) - stat(y[idx], yb[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": float(obs), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_gt_0": float(np.mean(deltas > 0))}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm adjustment within the pre-specified secondary family."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
