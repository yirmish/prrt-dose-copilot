#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyse_v35.py -- read today's TabPFN-3.5 OOF quantile stores and the stored comparators. 2026-10-06.

NO MODEL IS FITTED HERE. Everything is computed from oof_*.npz files:
  today's run      outputs/v35_dist_20261006/<size>/<target>/oof_<target>_<size>_TabPFN-V3_5@8.npz  (5 repeats)
  comparators      tabpfn_prrt_scaffold/results/*/oof_<target>_<size>_<arm>.npz                      (20 repeats)
The comparators are cut to their FIRST 5 repeats: CV seeds 0..4 give the same folds as today's run, so
the comparison is on identical folds. That identity is CHECKED (y_model equal; M0 medians equal where the
baseline shape is the same) and written to v35_gate.csv - it is not assumed.

ASSUMPTIONS   stored comparators were produced by the same scaffold on the same de-identified export.
SENSITIVE     none: read-only, new files only.
UNCERTAINTY   5 repeats; treatment-level margin 0.15, lesion-level 0.07/0.09 (scaffold config). A delta
              inside the margin is NOT a difference. Decisive-call thresholds are the scaffold's
              TargetSpec.thresholds_gy; 0.9 / 0.1 probability cut-offs are a convention, not derived.
"""
from __future__ import annotations
import os, sys, glob, json, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; WORK = HERE.parent; ROOT = WORK.parents[3]
RUN = WORK / "outputs" / os.environ.get("HK_RUN_TAG", "v35_dist_20261006")
SCAF_RES = ROOT / "tabpfn_prrt_scaffold" / "results"
OUT = WORK / "analysis"; OUT.mkdir(exist_ok=True)
ARM35 = "TabPFN-V3_5@8"; NREP = 5
NOM = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
THR = {"tumor_burden": (20.0, 30.0, 36.5, 40.0), "tumors": (20.0, 30.0, 36.2, 40.0), "kidneys": (5.75,),
       "bone_marrow": (0.5,), "tumors_hepatic": (30.0,)}
UNIT = {"tumors": "lesion", "tumors_hepatic": "lesion"}
rng = np.random.default_rng(20261006)

def load(p):
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}

def qat(q, lv, level):          # q: (..., L) monotone-enforced
    return np.apply_along_axis(lambda r: np.interp(level, lv, np.maximum.accumulate(r)), -1, q)

def metrics(d, nrep=NREP):
    q, y, lv = d["q_store"][:nrep].astype(float), d["y_model"].astype(float), d["levels"].astype(float)
    out, rel, pits = {}, {float(n): [] for n in NOM}, []
    r2s, crps, w95, w80, mae = [], [], [], [], []
    for r in range(q.shape[0]):
        ok = ~np.isnan(q[r, :, 0])
        if ok.sum() <= 10: continue
        qq = np.maximum.accumulate(q[r, ok], axis=1); yy = y[ok]
        med = qat(qq, lv, 0.5)
        r2s.append(1 - ((yy - med) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()); mae.append(np.abs(yy - med).mean())
        e = yy[:, None] - qq
        crps.append(2 * np.mean(np.maximum(lv[None, :] * e, (lv[None, :] - 1) * e)))
        for n in NOM:
            lo, hi = qat(qq, lv, (1 - n) / 2), qat(qq, lv, 1 - (1 - n) / 2)
            rel[float(n)].append(float(((yy >= lo) & (yy <= hi)).mean()))
        w95.append(np.median(qat(qq, lv, 0.975) - qat(qq, lv, 0.025))); w80.append(np.median(qat(qq, lv, 0.9) - qat(qq, lv, 0.1)))
        pits.extend(np.interp(yy[i], qq[i], lv, left=0.0, right=1.0) for i in range(len(yy)))
    pit = np.sort(np.asarray(pits)); ks = float(np.max(np.abs(pit - (np.arange(1, len(pit) + 1) / len(pit))))) if len(pit) else np.nan
    out.update(n_repeats_used=len(r2s), log_r2=np.median(r2s), log_r2_min=np.min(r2s), log_r2_max=np.max(r2s), log_mae=np.median(mae),
               crps_log=np.median(crps), cov_50=np.median(rel[0.5]), cov_80=np.median(rel[0.8]), cov_90=np.median(rel[0.9]),
               cov_95=np.median(rel[0.95]), width80_log=np.median(w80), width95_log=np.median(w95),
               width80_fold=float(np.exp(np.median(w80))), width95_fold=float(np.exp(np.median(w95))), pit_ks=ks)
    return out, {n: float(np.median(v)) for n, v in rel.items()}

def point(d, nrep=NREP):
    q, lv = d["q_store"][:nrep].astype(float), d["levels"].astype(float)
    j = int(np.argmin(np.abs(lv - 0.5))); return np.nanmedian(q[:, :, j], axis=0)

def boot_delta(y, pa, pb, g, B=2000):
    ok = np.isfinite(pa) & np.isfinite(pb); y, pa, pb, g = y[ok], pa[ok], pb[ok], g[ok]
    ug = np.unique(g); idx = {u: np.flatnonzero(g == u) for u in ug}
    r2 = lambda yy, pp: 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    obs = r2(y, pa) - r2(y, pb); ds = []
    for _ in range(B):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug), replace=True)])
        ds.append(r2(y[s], pa[s]) - r2(y[s], pb[s]))
    lo, hi = np.quantile(ds, [0.025, 0.975]); return float(obs), float(lo), float(hi)

def decisive(d, thr, nrep=NREP):
    """P(dose > thr) from the predictive distribution; a call is 'decisive' at P>=0.9 or P<=0.1."""
    q, y, lv, act, gy = d["q_store"][:nrep].astype(float), d["y_model"].astype(float), d["levels"].astype(float), d["activity"].astype(float), d["dose_gy"].astype(float)
    t = np.log(thr / act); fr, acc, n_dec, hi_ok, lo_ok = [], [], [], [], []
    for r in range(q.shape[0]):
        ok = ~np.isnan(q[r, :, 0])
        if ok.sum() <= 10: continue
        qq = np.maximum.accumulate(q[r, ok], axis=1)
        p = np.array([1 - np.interp(t[ok][i], qq[i], lv, left=0.0, right=1.0) for i in range(ok.sum())])
        truth = gy[ok] > thr; up, dn = p >= 0.9, p <= 0.1; dec = up | dn
        fr.append(dec.mean()); n_dec.append(dec.sum())
        acc.append(((up & truth) | (dn & ~truth)).sum() / max(1, dec.sum()))
    return dict(threshold_gy=thr, prevalence_above=float((gy > thr).mean()), decisive_fraction=float(np.median(fr)),
                decisive_accuracy=float(np.median(acc)), decisive_n_median=float(np.median(n_dec)))

def comparators(target, size, n):
    found = {}
    for f in sorted(glob.glob(str(SCAF_RES / "*" / f"oof_{target}_{size}_*.npz")) + glob.glob(str(SCAF_RES / "*" / f"oof_{target}_M0_M0.npz"))):
        arm = Path(f).stem.split(f"oof_{target}_")[1].split("_", 1)[1]; tag = Path(f).parent.name
        try: d = load(f)
        except Exception: continue
        if d["q_store"].shape[1] != n or d["q_store"].shape[0] < NREP: continue
        found.setdefault(arm, []).append((tag, d))
    return found

summ, relc, dec, gate, deltas = [], [], [], [], []
units = sorted(glob.glob(str(RUN / "M*" / "*" / f"oof_*_{ARM35}.npz")))
for f in units:
    size, target = Path(f).parts[-3], Path(f).parts[-2]
    d35 = load(f); n = d35["q_store"].shape[1]; y = d35["y_model"].astype(float); g = d35["groups"]
    cfg = json.loads((RUN / "run_config.json").read_text(encoding="utf-8"))
    arms = {ARM35: ("today", d35)}
    m0f = Path(f).parent / f"oof_{target}_M0_M0.npz"
    if m0f.exists(): arms[f"M0({cfg['baseline_kind_by_target'].get(target, 'plan')})"] = ("today", load(m0f))
    for arm, lst in comparators(target, size, n).items():
        for tag, d in (lst if arm == 'M0' else lst[:1]):
            same_y = bool(np.allclose(d["y_model"].astype(float), y, atol=1e-6, equal_nan=True))
            key = f"{arm}[{tag}]"
            if arm == "M0" and m0f.exists():
                dm = float(np.nanmax(np.abs(point(d) - point(load(m0f)))))
                gate.append(dict(target=target, size=size, check="M0 median, stored vs today (first 5 repeats)", stored_tag=tag, same_y_model=same_y,
                                 max_abs_diff=dm, identical=bool(dm < 5e-4), note="differs by design if the stored M0 used another baseline shape"))
                continue
            gate.append(dict(target=target, size=size, check=f"y_model identical: {arm}", stored_tag=tag, same_y_model=same_y, max_abs_diff=np.nan, identical=same_y, note=""))
            if same_y and arm != "M0": arms[key] = (tag, d)
    p35 = point(d35)
    for name, (tag, d) in arms.items():
        m, rel = metrics(d)
        summ.append(dict(target=target, unit=UNIT.get(target, "treatment"), model_size=size, arm=name, n_rows=n, n_persons=int(len(np.unique(g))), **m))
        for nn, v in rel.items(): relc.append(dict(target=target, model_size=size, arm=name, nominal=nn, empirical=v))
        for thr in THR.get(target, ()):
            dec.append(dict(target=target, model_size=size, arm=name, **decisive(d, thr)))
        if name != ARM35:
            o, lo, hi = boot_delta(y, p35, point(d), g)
            deltas.append(dict(target=target, model_size=size, comparison=f"{ARM35} minus {name}", delta_log_r2=o, ci_lo=lo, ci_hi=hi,
                               margin=0.07 if target in UNIT else 0.15, ci_excludes_zero=bool(lo > 0 or hi < 0)))
S = pd.DataFrame(summ); S.to_csv(OUT / "v35_summary.csv", index=False)
pd.DataFrame(relc).to_csv(OUT / "v35_reliability_curve.csv", index=False)
pd.DataFrame(dec).to_csv(OUT / "v35_decisive_calls.csv", index=False)
G = pd.DataFrame(gate); G.to_csv(OUT / "v35_gate.csv", index=False)
Dl = pd.DataFrame(deltas); Dl.to_csv(OUT / "v35_paired_deltas.csv", index=False)
G[~G.identical].to_csv(OUT / "flagged_rows.csv", index=False)
(OUT / "summary_stats.txt").write_text(f"analyse_v35.py {time.strftime('%Y-%m-%d %H:%M')}\nunits read (TabPFN-3.5 OOF files): {len(units)}\n"
    f"summary rows out: {len(S)}\ncomparator files rejected (y_model differs or M0 shape differs): {int((~G.identical).sum())} of {len(G)}\n", encoding="utf-8")

# ---------------- figures
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = pd.DataFrame(relc)
    for size in sorted(S.model_size.unique()):
        tg = [t for t in ("tumor_burden", "tumors", "tumors_hepatic", "kidneys", "liver_healthy", "spleen", "bone_marrow") if ((S.target == t) & (S.model_size == size)).any()]
        if not tg: continue
        fig, axes = plt.subplots(1, len(tg), figsize=(3.6 * len(tg), 3.9), squeeze=False)
        for ax, t in zip(axes[0], tg):
            sub = R[(R.target == t) & (R.model_size == size)]
            ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
            FAM = lambda a: ("TabPFN-3.5 (today)" if a == ARM35 else "TabPFN-V2 (open weights)" if a.startswith("TabPFN-V2") else
                             "CatBoost quantile heads" if a.startswith("CatBoost") else "physical baseline (2-3 terms)" if a.startswith("M0") else a.split("[")[0])
            COL = {"TabPFN-3.5 (today)": "#0b5cad", "TabPFN-V2 (open weights)": "#7f8c8d", "CatBoost quantile heads": "#d35400", "physical baseline (2-3 terms)": "#27ae60"}
            for arm, gdf in sub.groupby("arm"):
                is35 = arm == ARM35; lab = FAM(arm)
                ax.plot(gdf.nominal, gdf.empirical, marker="o", ms=5 if is35 else 3.5, lw=2.6 if is35 else 1.3,
                        color=COL.get(lab, "0.4"), alpha=1 if is35 else 0.9, label=lab, zorder=5 if is35 else 2)
            ax.set_title(t, fontsize=10); ax.set_xlabel("nominal central coverage"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.25)
        axes[0][0].set_ylabel("empirical coverage (out-of-fold, person-grouped)")
        h, l = axes[0][0].get_legend_handles_labels(); seen = {}
        for ax in axes[0]:
            for hh, ll in zip(*ax.get_legend_handles_labels()): seen.setdefault(ll, hh)
        fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=min(6, len(seen)), fontsize=8, frameon=False)
        fig.suptitle(f"Reliability of predictive intervals - feature set {size} - 5 x 5 grouped CV (first 5 protocol repeats)", fontsize=11)
        fig.tight_layout(rect=(0, 0.08, 1, 0.94)); fig.savefig(OUT / f"fig_reliability_{size}.png", dpi=150); plt.close(fig)
        # predicted median + 80% interval vs measured (Gy), repeat 0
        fig, axes = plt.subplots(1, len(tg), figsize=(3.6 * len(tg), 3.9), squeeze=False)
        for ax, t in zip(axes[0], tg):
            d = load(RUN / size / t / f"oof_{t}_{size}_{ARM35}.npz"); lv = d["levels"].astype(float)
            q = np.maximum.accumulate(d["q_store"][0].astype(float), axis=1); a = d["activity"].astype(float); gy = d["dose_gy"].astype(float)
            med, lo, hi = (np.exp(qat(q, lv, L)) * a for L in (0.5, 0.1, 0.9))
            inside = (gy >= lo) & (gy <= hi)
            ax.errorbar(gy, med, yerr=[med - lo, hi - med], fmt="none", ecolor="#9db8d6", elinewidth=0.6, alpha=0.7, zorder=1)
            ax.scatter(gy[inside], med[inside], s=9, color="#0b5cad", zorder=3, label="measured inside 80% interval")
            ax.scatter(gy[~inside], med[~inside], s=11, color="#c0392b", zorder=4, label="outside")
            lim = [np.nanmin(np.r_[gy, med]) * 0.5, np.nanmax(np.r_[gy, med]) * 2.0]; ax.plot(lim, lim, color="0.5", lw=1, ls="--")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
            import matplotlib.ticker as mt
            ax.xaxis.set_minor_formatter(mt.NullFormatter()); ax.yaxis.set_minor_formatter(mt.NullFormatter()); ax.tick_params(labelsize=8)
            ax.set_title(f"{t}  ({inside.mean():.0%} inside)", fontsize=10); ax.set_xlabel("measured dose (Gy)"); ax.grid(alpha=.25, which="both")
        axes[0][0].set_ylabel("predicted median and 80% interval (Gy)")
        fig.suptitle(f"TabPFN-3.5 out-of-fold predictive intervals vs measured dose - feature set {size} - repeat 0", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / f"fig_pred_interval_vs_measured_{size}.png", dpi=150); plt.close(fig)
except Exception as exc:
    print("figures skipped:", type(exc).__name__, exc)

# ---------------- markdown report (numbers only)
pd.set_option("display.width", 250)
def md(df, cols, fmt):
    df = df[cols].copy()
    for c, f in fmt.items():
        if c in df: df[c] = df[c].map(lambda v: "" if pd.isna(v) else f.format(v))
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy())
order = {"tumor_burden": 0, "tumors": 1, "tumors_hepatic": 2, "kidneys": 3, "liver_healthy": 4, "spleen": 5, "bone_marrow": 6}
S["o"] = S.target.map(order); S = S.sort_values(["model_size", "o", "arm"])
f3 = {c: "{:+.3f}" for c in ("log_r2", "log_r2_min", "log_r2_max", "delta_log_r2", "ci_lo", "ci_hi")}
f3.update({c: "{:.3f}" for c in ("cov_50", "cov_80", "cov_90", "cov_95", "crps_log", "pit_ks", "decisive_fraction", "decisive_accuracy", "prevalence_above")})
f3.update({c: "{:.2f}" for c in ("width80_fold", "width95_fold")})
txt = [f"# TabPFN-3.5 predictive distributions on the PRRT cohort - numbers only\n\nGenerated {time.strftime('%Y-%m-%d %H:%M')} by `analyse_v35.py` from the OOF quantile stores. "
       "**Not a protocol run** (5 of 20 repeats).\n\n"
       "Target: log(Gy/GBq). Folds: 5-fold StratifiedGroupKFold on the person, seeds 0-4. `log_r2` is the median over repeats (min, max beside it). "
       "Comparators are the stored scaffold arms cut to the same 5 repeats (identical folds - see `v35_gate.csv`). "
       "`width95_fold` = exp(median 95% interval width): the multiplicative span of the interval (e.g. 6.0 means the upper bound is 6x the lower).\n",
       "## 1. Accuracy and calibration\n", md(S, ["model_size", "target", "arm", "n_rows", "n_persons", "log_r2", "log_r2_min", "log_r2_max", "cov_50", "cov_80", "cov_95", "width80_fold", "width95_fold", "crps_log", "pit_ks"], f3),
       "\n\n## 2. Paired differences in log R2 (person-clustered bootstrap, B=2000, on identical folds)\n\nMargins: treatment level 0.15, lesion level 0.07 (0.09 with a clustered residual). A CI that includes 0, or a delta below the margin, is not a difference.\n",
       md(Dl, ["model_size", "target", "comparison", "delta_log_r2", "ci_lo", "ci_hi", "margin", "ci_excludes_zero"], f3) if len(Dl) else "(none)",
       "\n\n## 3. Decisive calls at the scaffold's clinical thresholds\n\nA call is decisive when the predictive distribution puts P(dose > threshold) >= 0.9 or <= 0.1. `decisive_accuracy` = share of decisive calls on the correct side.\n",
       md(pd.DataFrame(dec), ["model_size", "target", "arm", "threshold_gy", "prevalence_above", "decisive_fraction", "decisive_accuracy", "decisive_n_median"], f3) if dec else "(none)",
       "\n\n## 4. Identity gate\n", md(G, ["target", "size", "check", "stored_tag", "same_y_model", "max_abs_diff", "identical"], {"max_abs_diff": "{:.2e}"}) if len(G) else "(none)", "\n"]
(OUT / "V35_DISTRIBUTION_NUMBERS_2026-10-06.md").write_text("\n".join(txt), encoding="utf-8")
print(S.drop(columns="o")[["model_size", "target", "arm", "log_r2", "cov_50", "cov_80", "cov_95", "width95_fold", "crps_log", "pit_ks"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(Dl.to_string(index=False, float_format=lambda v: f"{v:.3f}") if len(Dl) else "")
print("ANALYSIS_DONE units", len(units))
