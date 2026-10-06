#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""proper_scores_paired.py -- paired proper-score comparison from stored out-of-fold quantiles. 2026-10-06 (evening).

NO MODEL IS FITTED. Reads the same oof_*.npz files as analyse_v35.py (the TabPFN-3.5 run, 5 repeats, and the
stored scaffold comparators cut to their first 5 repeats = identical folds, y_model checked). Needs the
non-public cohort outputs; it is published so the aggregate tables can be traced to their code.
Usage: python proper_scores_paired.py <run work folder> <output folder>

ASSUMPTIONS   q_store = (repeat, row, 99 levels) log(Gy/GBq) quantiles; groups = patient; a comparator is accepted
              only if its y_model is identical (same rule as analyse_v35.py).
SENSITIVE     none: read-only on inputs, new files only. Aggregates out, no per-row table written.
UNCERTAINTY   5 of 20 protocol repeats; POST-HOC (written after the M3 point results were seen, not pre-registered);
              the 0.9 / 0.1 decisive cut-offs are a convention. Brier at thresholds with prevalence near 0 or 1 is
              uninformative. The spleen and healthy-liver targets carry open data questions in the study.
"""
import os, sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
WORK = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
ROOT = WORK.parents[3]; RUN = WORK / "outputs" / "v35_dist_20261006"; SCAF = ROOT / "tabpfn_prrt_scaffold" / "results"
ARM = "TabPFN-V3_5@8"; NREP = 5; B = 2000
THR = {"tumor_burden": (20.0, 30.0, 36.5, 40.0), "tumors": (20.0, 30.0, 36.2, 40.0), "kidneys": (5.75,),
       "bone_marrow": (0.5,), "tumors_hepatic": (30.0,)}
rng = np.random.default_rng(20261006)
load = lambda p: {k: v for k, v in np.load(p, allow_pickle=False).items()}

def per_row(d):
    """per-row scores averaged over repeats: crps, in-interval indicators, log width, P(exceed) per threshold"""
    q = np.maximum.accumulate(d["q_store"][:NREP].astype(float), axis=2); y = d["y_model"].astype(float); lv = d["levels"].astype(float)
    e = y[None, :, None] - q
    crps = 2 * np.mean(np.maximum(lv * e, (lv - 1) * e), axis=2)                      # (rep,row)
    at = lambda L: np.array([[np.interp(L, lv, q[r, i]) for i in range(q.shape[1])] for r in range(q.shape[0])])
    out = {"crps": np.nanmean(crps, 0)}
    for n in (0.5, 0.8, 0.95):
        lo, hi = at((1 - n) / 2), at(1 - (1 - n) / 2)
        out[f"cov{int(n*100)}"] = np.nanmean(((y >= lo) & (y <= hi)).astype(float), 0); out[f"w{int(n*100)}"] = np.nanmean(hi - lo, 0)
    return out, q, lv

def pexc(d, q, lv, thr):
    t = np.log(thr / d["activity"].astype(float))
    return np.array([[1 - np.interp(t[i], q[r, i], lv, left=0.0, right=1.0) for i in range(q.shape[1])] for r in range(q.shape[0])])

def boot(g, *vecs, f):
    ug = np.unique(g); idx = {u: np.flatnonzero(g == u) for u in ug}; obs = f(*vecs); ds = []
    for _ in range(B):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug), replace=True)]); ds.append(f(*[v[s] for v in vecs]))
    lo, hi = np.nanquantile(ds, [0.025, 0.975]); return float(obs), float(lo), float(hi)

def comparators(target, size, n, y):
    found, rej = {}, 0
    for f in sorted(glob.glob(str(SCAF / "*" / f"oof_{target}_{size}_*.npz"))):
        arm = Path(f).stem.split(f"oof_{target}_")[1].split("_", 1)[1]; tag = Path(f).parent.name
        try: d = load(f)
        except Exception: rej += 1; continue
        if d["q_store"].shape[1] != n or d["q_store"].shape[0] < NREP or not np.allclose(d["y_model"].astype(float), y, atol=1e-6, equal_nan=True): rej += 1; continue
        if arm.startswith("CatBoost") or arm == "TabPFN-V2@8": found.setdefault(arm, (tag, d))
    return found, rej

cfg = json.loads((RUN / "run_config.json").read_text(encoding="utf-8"))
cov, dl, br, rc, nrej, nunits = [], [], [], [], 0, 0
for f in sorted(glob.glob(str(RUN / "M*" / "*" / f"oof_*_{ARM}.npz"))):
    size, target = Path(f).parts[-3], Path(f).parts[-2]; d35 = load(f); y = d35["y_model"].astype(float); g = d35["groups"]; n = len(y); nunits += 1
    arms = {"TabPFN-3.5": d35, "physical-baseline": load(Path(f).parent / f"oof_{target}_M0_M0.npz")}
    cmp_, r = comparators(target, size, n, y); nrej += r
    for a, (tag, d) in cmp_.items(): arms["CatBoost" if a.startswith("CatBoost") else "TabPFN-V2"] = d
    S = {a: per_row(d) for a, d in arms.items()}
    for a, (s, q, lv) in S.items():
        row = dict(model_size=size, target=target, arm=a, n_rows=n, n_persons=int(len(np.unique(g))))
        for k in ("cov50", "cov80", "cov95"):
            o, lo, hi = boot(g, s[k], f=np.mean); row.update({k: o, k + "_lo": lo, k + "_hi": hi})
        row["width95_fold"] = float(np.exp(np.median(s["w95"]))); row["crps_log"] = float(s["crps"].mean()); cov.append(row)
    s35 = S["TabPFN-3.5"][0]
    for a in arms:
        if a == "TabPFN-3.5": continue
        o, lo, hi = boot(g, s35["crps"], S[a][0]["crps"], f=lambda x, z: x.mean() - z.mean())
        dl.append(dict(model_size=size, target=target, comparison=f"TabPFN-3.5 minus {a}", metric="CRPS (log), lower is better", delta=o, ci_lo=lo, ci_hi=hi,
                       rel_change_pct=100 * o / S[a][0]["crps"].mean(), ci_excludes_zero=bool(lo > 0 or hi < 0)))
        o, lo, hi = boot(g, s35["w95"], S[a][0]["w95"], f=lambda x, z: float(np.exp(np.median(x) - np.median(z))))
        dl.append(dict(model_size=size, target=target, comparison=f"TabPFN-3.5 over {a}", metric="95% interval width ratio (fold), <1 = sharper", delta=o, ci_lo=lo, ci_hi=hi,
                       rel_change_pct=100 * (o - 1), ci_excludes_zero=bool(lo > 1 or hi < 1)))
    for thr in THR.get(target, ()):
        truth = (d35["dose_gy"].astype(float) > thr).astype(float); P = {a: pexc(arms[a], S[a][1], S[a][2], thr) for a in arms}
        bs = {a: np.nanmean((P[a] - truth[None, :]) ** 2, 0) for a in arms}
        for a in arms:
            o, lo, hi = boot(g, bs[a], f=np.mean); row = dict(model_size=size, target=target, threshold_gy=thr, prevalence=float(truth.mean()), arm=a, brier=o, brier_lo=lo, brier_hi=hi)
            if a != "TabPFN-3.5":
                o, lo, hi = boot(g, bs["TabPFN-3.5"], bs[a], f=lambda x, z: x.mean() - z.mean()); row.update(delta_35_minus_arm=o, d_lo=lo, d_hi=hi, ci_excludes_zero=bool(lo > 0 or hi < 0))
            br.append(row)
            for c in (0.6, 0.7, 0.8, 0.9, 0.95):
                up, dn = P[a] >= c, P[a] <= 1 - c; dec = up | dn; ok = (up & (truth[None, :] == 1)) | (dn & (truth[None, :] == 0))
                rc.append(dict(model_size=size, target=target, threshold_gy=thr, arm=a, cutoff=c, decisive_fraction=float(dec.mean()), decisive_n_per_repeat=float(dec.sum(1).mean()),
                               decisive_accuracy=float(ok.sum() / max(1, dec.sum())), claimed_min_accuracy=c))
pd.DataFrame(cov).round(4).to_csv(OUT / "coverage_with_ci.csv", index=False); pd.DataFrame(dl).round(4).to_csv(OUT / "paired_crps_and_sharpness.csv", index=False)
pd.DataFrame(br).round(4).to_csv(OUT / "brier_threshold_probabilities.csv", index=False); pd.DataFrame(rc).round(4).to_csv(OUT / "risk_coverage_decisive_calls.csv", index=False)
R = pd.DataFrame(rc); fl = R[(R.decisive_accuracy < R.claimed_min_accuracy - 0.05) & (R.decisive_n_per_repeat >= 10)]
fl.round(4).to_csv(OUT / "flagged_rows.csv", index=False)
(OUT / "summary_stats.txt").write_text(f"proper_scores_paired.py  units read {nunits}  arms scored {len(cov)}  comparator files rejected (shape / y_model / unreadable) {nrej}\n"
    f"rows out: coverage {len(cov)}, paired deltas {len(dl)}, brier {len(br)}, risk-coverage {len(rc)}; flagged (decisive accuracy > 5 points below the cut-off, n>=10) {len(fl)}\n"
    "No model fitted. 5 of 20 protocol repeats. Post-hoc, not pre-registered.\n", encoding="utf-8")
print("DONE", nunits, len(cov), len(dl), len(br), len(rc), len(fl))
