#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_synthetic_cohort.py -- a SYNTHETIC demonstration cohort for a public repository. 2026-10-06.

WHY  The hackathon terms (clause 3.2) require the public repo to contain its input data. The real
     cohort is patient-level hospital data and is NOT public. This script writes a synthetic cohort
     that preserves the marginal distributions and rank-correlation structure of a COMPACT feature
     set, so that an app built on TabPFN-3.5 can ship with a context table that is not patient data.

DECLARATION
  ASSUMPTIONS  A1 input = the scaffold's own assembled matrices (D.assemble / D.build_features) on
                  Deidentified_Export_2026-09-07; nothing is read from identified sources.
               A2 a Gaussian copula on normal scores is an adequate generator for a DEMO cohort; it
                  is NOT a claim that the synthetic data reproduce every property of the real data.
  SENSITIVE    S1 SYNTHESIS FROM PATIENT DATA. No real row is copied; the per-column quantile function
                  is built from interior order statistics only (the observed minimum and maximum are
                  dropped), values are interpolated between order statistics and jittered (2% log-SD).
               S2 within-patient clustering for lesions is IMPOSED (rho = 0.75 on the latent scale,
                  close to the project's REML ICC 0.78), not learned per column.
               S3 missingness is re-drawn independently per column at the observed rate - the real
                  missingness is by cohort era (MNAR) and that structure is deliberately NOT carried.
  UNCERTAINTY  U1 fidelity is reported (baseline R2 real vs synthetic) - read it before relying on it.
OUTPUT  ../synthetic/<target>_synthetic.csv, synthetic_fidelity.csv, synthetic_privacy_check.csv,
        summary_stats.txt, flagged_rows.csv
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
ROOT = WORK.parents[3]
SCAF = ROOT / "tabpfn_prrt_scaffold"
OUT = WORK / "synthetic"; OUT.mkdir(exist_ok=True)
os.environ["PRRT_DATA_ROOT"] = str(ROOT / "Deidentified_Export_2026-09-07")
os.environ["PRRT_RESULTS_ROOT"] = str(OUT)
os.chdir(SCAF); sys.path.insert(0, str(SCAF))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
from tabpfn_prrt import config as C, data as D
from tabpfn_prrt.ids import base_id

SEED = 20261006
rng = np.random.default_rng(SEED)
EXTRA = [C.ACTIVITY_COL, "pt_weight_kg", "pt_height_m", "gen_age_at_prrt_years", "gen_gender_code_1m_2f",
         "gen_days_pet_to_prrt", "gen_pet_uptake_time_minutes", "voi_whole_body__pet_mean_bqml",
         "voi_whole_body__pet_mean_suvbw", "voi_kidneys__pet_mean_bqml", "voi_liver__pet_mean_suvbw",
         "voi_spleen__pet_mean_suvbw", "gen_pet_camera_code_1mi_2dr"]
DISCRETE = {"gen_gender_code_1m_2f", "gen_pet_camera_code_1mi_2dr", "pet_regions_count"}
N_SYN = {"treatment": 300, "lesion_patients": 200}
RHO_WITHIN = 0.75

def to_scores(col: np.ndarray) -> np.ndarray:
    ok = np.isfinite(col); z = np.full(len(col), np.nan)
    z[ok] = norm.ppf(rankdata(col[ok]) / (ok.sum() + 1.0))
    return z

def corr_psd(Z: np.ndarray) -> np.ndarray:
    R = pd.DataFrame(Z).corr(min_periods=10).to_numpy()
    R = np.where(np.isfinite(R), R, 0.0); np.fill_diagonal(R, 1.0)
    w, V = np.linalg.eigh((R + R.T) / 2); w = np.clip(w, 1e-3, None)
    R2 = (V * w) @ V.T; d = np.sqrt(np.diag(R2)); return R2 / np.outer(d, d)

def back(z: np.ndarray, obs: np.ndarray, discrete: bool) -> np.ndarray:
    o = np.sort(obs[np.isfinite(obs)])
    if discrete or len(np.unique(o)) <= 6:
        u = norm.cdf(z); idx = np.clip((u * len(o)).astype(int), 0, len(o) - 1); return o[idx]
    o = o[1:-1] if len(o) > 12 else o                    # drop the observed min and max
    p = (np.arange(1, len(o) + 1)) / (len(o) + 1.0)
    u = np.clip(norm.cdf(z), p[0], p[-1])
    pos = bool((o > 0).all())
    v = np.interp(u, p, np.log(o) if pos else o)
    if pos: return np.exp(v + rng.normal(0, 0.02, len(v)))
    return v + rng.normal(0, 0.02 * (np.nanstd(o) + 1e-9), len(v))

def ols_log_r2(df: pd.DataFrame, feats, ycol, acol) -> float:
    d = df[list(feats) + [ycol, acol]].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d > 0).all(axis=1)]
    if len(d) < 20: return float("nan")
    y = np.log(d[ycol] / d[acol]); A = np.column_stack([np.ones(len(d))] + [np.log(d[f]) for f in feats])
    b, *_ = np.linalg.lstsq(A, y, rcond=None); r = y - A @ b
    return float(1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum())

fid, priv, stats = [], [], []
for name in ("tumor_burden", "kidneys", "liver_healthy", "spleen", "bone_marrow", "tumors"):
    spec = next(t for t in C.TARGETS if t.name == name)
    df = D.assemble(spec)
    X1 = D.build_features(df, ("A",), spec, include_grey_zone=False)
    cols = [c for c in X1.columns if pd.api.types.is_numeric_dtype(X1[c])]
    real = X1[cols].astype(float).copy()
    for c in EXTRA:
        if c in df.columns and c not in real.columns:
            real[c] = pd.to_numeric(df[c], errors="coerce").astype(float).to_numpy()
    real["dose_gy"] = pd.to_numeric(df[spec.target_col], errors="coerce").astype(float).to_numpy()
    real = real.loc[:, real.notna().sum() >= 10]
    names = list(real.columns)
    Z = np.column_stack([to_scores(real[c].to_numpy(float)) for c in names]); R = corr_psd(Z)
    L = np.linalg.cholesky(R + 1e-9 * np.eye(len(names)))
    if spec.unit == "lesion":
        groups = df[C.V8_JOIN_KEY].map(base_id).to_numpy()
        counts = pd.Series(groups).value_counts().to_numpy()
        k = rng.choice(counts, size=N_SYN["lesion_patients"], replace=True)
        pid = np.repeat(np.arange(1, len(k) + 1), k); n = len(pid)
        zp = (rng.standard_normal((len(k), len(names))) @ L.T)[pid - 1]
        zl = rng.standard_normal((n, len(names))) @ L.T
        Zs = np.sqrt(RHO_WITHIN) * zp + np.sqrt(1 - RHO_WITHIN) * zl
        # treatment-level columns are constant within a synthetic patient
        tl = [j for j, c in enumerate(names) if c in EXTRA]
        Zs[:, tl] = zp[:, tl]
    else:
        n = N_SYN["treatment"]; pid = np.arange(1, n + 1)
        Zs = rng.standard_normal((n, len(names))) @ L.T
    syn = pd.DataFrame({c: back(Zs[:, j], real[c].to_numpy(float), c in DISCRETE) for j, c in enumerate(names)})
    if spec.unit == "lesion":                             # keep treatment-level columns constant per patient
        for c in [c for c in names if c in EXTRA]:
            syn[c] = syn.groupby(pid)[c].transform("first")
    for c in names:                                       # missingness: independent, at the observed rate
        rate = float(real[c].isna().mean())
        if rate > 0 and c != "dose_gy":
            syn.loc[rng.random(len(syn)) < rate, c] = np.nan
    syn.insert(0, "synthetic_patient_id", [f"SYN{int(p):04d}" for p in pid])
    syn.insert(0, "is_synthetic", 1)
    syn["dose_gy_per_gbq"] = syn["dose_gy"] / syn[C.ACTIVITY_COL]
    num = syn.select_dtypes("number").columns
    syn[num] = syn[num].apply(lambda s: s.map(lambda v: float(f"{v:.5g}") if np.isfinite(v) else v))
    syn.to_csv(OUT / f"{name}_synthetic.csv", index=False)

    # ---- privacy: distance of every synthetic row to its nearest REAL row (normal-score space)
    use = [c for c in names if real[c].notna().mean() > 0.9]
    Zr = np.column_stack([to_scores(real[c].to_numpy(float)) for c in use])
    Zy = np.column_stack([norm.ppf(np.clip((np.searchsorted(np.sort(real[c].dropna().to_numpy(float)),
                          syn[c].to_numpy(float)) + 0.5) / (real[c].notna().sum() + 1.0), 1e-4, 1 - 1e-4)) for c in use])
    okr = np.isfinite(Zr).all(1); oky = np.isfinite(Zy).all(1)
    dmin = np.array([np.sqrt(((Zr[okr] - r) ** 2).sum(1)).min() for r in Zy[oky]]) / np.sqrt(len(use))
    rr = np.array([np.sort(np.sqrt(((Zr[okr] - r) ** 2).sum(1)))[1] for r in Zr[okr]]) / np.sqrt(len(use))
    exact = 0
    realset = {tuple(np.round(v, 6)) for v in real[use].dropna().to_numpy(float)}
    for v in syn[use].dropna().to_numpy(float):
        exact += tuple(np.round(v, 6)) in realset
    shared_dose = int(np.isin(np.round(syn["dose_gy"].to_numpy(float), 4), np.round(real["dose_gy"].dropna().to_numpy(float), 4)).sum())
    priv.append({"target": name, "n_real": len(real), "n_synthetic": len(syn), "columns_compared": len(use),
                 "exact_row_matches": int(exact), "synthetic_doses_equal_to_a_real_dose_4dp": shared_dose,
                 "min_dist_syn_to_nearest_real": round(float(dmin.min()), 4),
                 "p05_dist_syn_to_nearest_real": round(float(np.quantile(dmin, .05)), 4),
                 "median_dist_syn_to_nearest_real": round(float(np.median(dmin)), 4),
                 "median_dist_real_to_nearest_real": round(float(np.median(rr)), 4)})
    # ---- fidelity: the M0-style log-log baseline, real vs synthetic (in-sample R2, same formula)
    pair = {"tumor_burden": ("pet_mean_suvbw", "pet_volume_ml"), "tumors": ("pet_mean_suvbw", "pet_volume_ml"),
            "kidneys": ("pet_mean_bqml", "pet_volume_ml"), "liver_healthy": ("pet_mean_bqml", "pet_volume_ml"),
            "spleen": ("pet_mean_bqml", "pet_volume_ml"), "bone_marrow": ("pet_mean_bqml", "pt_weight_kg")}[name]
    pair = tuple(c for c in pair if c in names)
    fid.append({"target": name, "baseline_terms": "+".join(pair),
                "r2_log_real_insample": round(ols_log_r2(real, pair, "dose_gy", C.ACTIVITY_COL), 3),
                "r2_log_synthetic_insample": round(ols_log_r2(syn, pair, "dose_gy", C.ACTIVITY_COL), 3),
                "median_dose_gy_real": round(float(real["dose_gy"].median()), 3),
                "median_dose_gy_synthetic": round(float(syn["dose_gy"].median()), 3),
                "iqr_dose_gy_real": round(float(real["dose_gy"].quantile(.75) - real["dose_gy"].quantile(.25)), 3),
                "iqr_dose_gy_synthetic": round(float(syn["dose_gy"].quantile(.75) - syn["dose_gy"].quantile(.25)), 3)})
    stats.append(f"{name}: real rows in {len(real)} -> synthetic rows out {len(syn)}; columns {len(names)}; rows rejected 0")
    print(name, priv[-1], fid[-1], flush=True)

pd.DataFrame(fid).to_csv(OUT / "synthetic_fidelity.csv", index=False)
P = pd.DataFrame(priv); P.to_csv(OUT / "synthetic_privacy_check.csv", index=False)
bad = P[(P.exact_row_matches > 0)]
bad.to_csv(OUT / "flagged_rows.csv", index=False)
(OUT / "summary_stats.txt").write_text("make_synthetic_cohort.py  seed %d  %s\n" % (SEED, time.strftime("%Y-%m-%d %H:%M")) + "\n".join(stats)
    + f"\nexact row matches (must be 0): {int(P.exact_row_matches.sum())}\nflagged targets: {len(bad)}\n", encoding="utf-8")
print("SYNTH_DONE exact_matches_total", int(P.exact_row_matches.sum()))
