"""verify_claims.py - recompute every real-cohort number quoted in README.md from the committed CSVs.

    python scripts/verify_claims.py            # exit code 0 = every claim matches, 1 = at least one does not

Each claim is computed from a file in results/real_cohort/, formatted the way the README prints it, and
looked up in the README text. Nothing is typed in by hand
except the formatting. If a CSV changes, the README must change with it, or this script fails (and CI with it).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from copilot import evidence as E  # noqa: E402

DOCS = {"README": (ROOT / "README.md").read_text(encoding="utf-8")}
TUNED = ["CatBoost", "XGBoost", "HistGBM", "LightGBM", "RandomForest", "ElasticNet"]


def f3(x: float) -> str:
    return f"{x:.3f}".replace("-", "−")          # the README prints a true minus sign


def s3(x: float) -> str:
    return f"{x:+.3f}".replace("-", "−")


def ci(d, lo, hi) -> str:
    return f"{s3(d)} [{s3(lo)}, {s3(hi)}]"


def rng(a: float, b: float, nd: int = 2) -> str:
    return f"{a:.{nd}f}–{b:.{nd}f}"


def pct_rng(a: float, b: float) -> str:
    return f"{round(100 * a)}–{round(100 * b)}%"


def build_claims() -> list[tuple[str, str, tuple[str, ...]]]:
    c: list[tuple[str, str, tuple[str, ...]]] = []
    add = lambda label, text, where=("README",): c.append((label, text, where))  # noqa: E731

    # ---- treatment level, ~165 columns, against tuned CatBoost (post-hoc, stored OOF quantiles) --
    a3 = E.m3_all_arms()
    dz = E.m3_decisive()
    s2 = lambda x: f"{x:+.2f}".replace("-", "−")  # noqa: E731
    cb_targets = {"bone_marrow": "bone marrow", "liver_healthy": "healthy liver", "tumor_burden": "tumour burden",
                  "kidneys": "kidneys"}
    rels, cov35, covcb = [], [], []
    for t, shown in cb_targets.items():
        c35 = float(a3[(a3.target == t) & (a3.model == "TabPFN-3.5")].cov_95.iloc[0])
        ccb = float(a3[(a3.target == t) & a3.model.str.startswith("CatBoost")].cov_95.iloc[0])
        v = E.m3_vs_catboost(t)
        assert v is not None and v["crps_hi"] < 0, f"M3 {t}: CRPS CI does not exclude zero"
        assert v["r2_lo"] < 0 < v["r2_hi"], f"M3 {t}: R2 CI does not include zero"
        add(f"M3 vs CatBoost row: {shown}",
            f"| {shown} | {c35:.2f} / {ccb:.2f} | **−{abs(v['crps_rel']):.0f}%** "
            f"({f3(v['crps_delta'])} [{f3(v['crps_lo'])}, {f3(v['crps_hi'])}]) | "
            f"{s2(v['r2_delta'])} [{s2(v['r2_lo'])}, {s2(v['r2_hi'])}] |")
        rels.append(round(abs(v["crps_rel"]))); cov35.append(c35); covcb.append(ccb)
    assert all(E.m3_vs_catboost(t) is None for t in ("spleen", "tumors_hepatic")), "M3: CatBoost stored for 4 targets"
    add("M3 CRPS reduction range vs CatBoost", f"{min(rels)}–{max(rels)}% lower", ("README",))
    add("M3 cov95 TabPFN-3.5 on the 4 CatBoost targets", rng(min(cov35), max(cov35)), ("README",))
    add("M3 cov95 CatBoost", rng(min(covcb), max(covcb)), ("README",))
    kd = dz[dz.target == "kidneys"].set_index("model")
    cbk = kd.loc[[m for m in kd.index if m.startswith("CatBoost")][0]]
    add("M3 kidneys decisive, CatBoost", f"calls {cbk.decisive_fraction:.0%} of treatments decisive and is right on "
        f"{cbk.decisive_accuracy:.0%} of those calls")
    add("M3 kidneys decisive, TabPFN-3.5", f"TabPFN-3.5 calls {kd.loc['TabPFN-3.5'].decisive_fraction:.0%} and is "
        f"right on {kd.loc['TabPFN-3.5'].decisive_accuracy:.0%}")
    add("M3 kidneys decisive, physical", f"baseline calls {kd.loc['Physical baseline'].decisive_fraction:.0%} and is "
        f"right on {kd.loc['Physical baseline'].decisive_accuracy:.0%}")

    # ---- lesion level, 140 features, eight models --------------------------------------------
    comp = E.lesion8_components().set_index("model")
    h = E.lesion8_head_to_head()
    r2 = E.lesion8_r2().set_index("target")
    t35 = comp.loc["TabPFN-3.5"]
    tuned = comp.loc[TUNED]
    add("lesion R2 TabPFN-3.5", f3(t35.r2_overall), ("README",))
    add("lesion R2 best tuned (CatBoost)", f3(tuned.r2_overall.max()))
    assert tuned.r2_overall.idxmax() == "CatBoost"
    add("lesion R2 range of six tuned", rng(tuned.r2_overall.min(), tuned.r2_overall.max(), 3), ("README",))
    add("within-patient R2 TabPFN-3.5", f3(t35.r2_within_patient), ("README",))
    add("within-patient R2 range of tuned", rng(tuned.r2_within_patient.min(), tuned.r2_within_patient.max(), 3),
        ("README",))
    add("CRPS TabPFN-3.5", f3(t35.crps_log))
    add("CRPS CatBoost", f3(comp.loc["CatBoost", "crps_log"]))
    add("native 80/95 coverage", f"{t35.cov80:.2f} / {t35.cov95:.2f}")
    add("CatBoost conformal coverage", f"{comp.loc['CatBoost', 'cov80']:.3f} / {comp.loc['CatBoost', 'cov95']:.3f}")
    assert abs(r2.loc["tumors", "TabPFN-3.5"] - t35.r2_overall) < 1e-9
    names = {"CatBoost": "CatBoost (tuned)", "XGBoost": "XGBoost (tuned)", "HistGBM": "HistGradientBoosting (tuned)",
             "LightGBM": "LightGBM (tuned)", "RandomForest": "Random forest (tuned)",
             "ElasticNet": "Elastic net (tuned)", "TabPFN-2": "TabPFN-2 (open weights)"}
    h8 = h[h.n_estimators == 8].set_index(["comparator", "metric"])
    for comp_name, row_name in names.items():
        cells = [ci(*h8.loc[(comp_name, m), ["delta", "ci_lo", "ci_hi"]]) for m in ("R2", "R2_within_patient", "CRPS")]
        add(f"head-to-head row {comp_name}", f"| {row_name} | " + " | ".join(cells) + " |")
    cb = h8.loc[("CatBoost", "R2")]
    add("Δ vs CatBoost (n8)", ci(cb.delta, cb.ci_lo, cb.ci_hi), ("README",))
    r1 = h[(h.n_estimators == 32) & (h.metric == "R2") & (h.comparator == "CatBoost")].iloc[0]
    add("registered R-1 (n32)", ci(r1.delta, r1.ci_lo, r1.ci_hi), ("README",))
    # "the CI excludes 0 against all six tuned models" (R2 and CRPS)
    assert all(h8.loc[(m, "R2"), "ci_lo"] > 0 for m in TUNED), "R2 CI touches 0 for a tuned model"
    assert all(h8.loc[(m, "CRPS"), "ci_hi"] < 0 for m in TUNED), "CRPS CI touches 0 for a tuned model"
    ne = E.lesion8_n_estimators()
    assert ne.delta_32_minus_8.abs().max() <= 0.003 + 1e-9, "8 -> 32 estimators moved R2 by more than 0.003"
    # treatment level: top R2 on kidneys, liver, spleen; not on tumour burden or marrow
    others = [m for m in r2.columns if m not in ("unit", "TabPFN-3.5")]
    top = {t for t in ("tumor_burden", "kidneys", "liver_healthy", "spleen", "bone_marrow")
           if r2.loc[t, "TabPFN-3.5"] > r2.loc[t, others].max()}
    assert top == {"kidneys", "liver_healthy", "spleen"}, top

    # ---- 17 PET features, run of 2026-10-06 --------------------------------------------------
    s1 = E.m1_summary()
    tr = s1[s1.unit == "treatment"]
    for model, label in (("TabPFN-3.5", "| TabPFN-3.5 |"),
                         ("Physical baseline", "| physical baseline (2-term log-log + normal residual) |"),
                         ("CatBoost (tuned, quantile heads)", "| CatBoost, tuned, quantile heads (4 targets) |")):
        g = tr[tr.model == model]
        cells = [rng(g[k].min(), g[k].max()) for k in ("cov_50", "cov_80", "cov_95")]
        line = label + " " + " | ".join(cells) + " |"
        add(f"M1 coverage row {model}", line.replace(f"| {cells[2]} |", f"| **{cells[2]}** |")
            if model != "Physical baseline" else line)
    g = tr[tr.model == "TabPFN-3.5"]
    add("TabPFN-3.5 treatment cov95 range", rng(g.cov_95.min(), g.cov_95.max()), ("README",))
    g = tr[tr.model == "CatBoost (tuned, quantile heads)"]
    add("CatBoost treatment cov95 range", rng(g.cov_95.min(), g.cov_95.max()), ("README",))
    w = tr.pivot(index="target", columns="model", values="width95_fold")
    assert (w["TabPFN-3.5"] < w["Physical baseline"]).all() and len(w) == 5, "narrower on all 5 treatment targets"
    les = s1[s1.target == "tumors"].set_index("model")
    add("M1 lesion R2 TabPFN-3.5", f3(les.loc["TabPFN-3.5", "log_r2"]))
    add("M1 lesion R2 physical", f3(les.loc["Physical baseline", "log_r2"]))
    add("M1 lesion R2 CatBoost", f3(les.loc["CatBoost (tuned, quantile heads)", "log_r2"]))
    add("M1 lesion cov95 TabPFN-3.5", f3(les.loc["TabPFN-3.5", "cov_95"]))
    d1 = E.m1_deltas()
    x = d1[(d1.target == "tumors") & d1.comparison.str.contains("CatBoost")].iloc[0]
    add("M1 lesion Δ vs CatBoost", ci(x.delta_log_r2, x.ci_lo, x.ci_hi))

    # ---- decisive calls, tumour burden --------------------------------------------------------
    dc = E.decisive()
    dc = dc[dc.target == "tumor_burden"]
    for model, lead in (("CatBoost (tuned, quantile heads)", "calls "), ("TabPFN-3.5", "calls "),
                        ("Physical baseline", "")):
        g = dc[dc.model == model]
        add(f"decisive fraction {model}", pct_rng(g.decisive_fraction.min(), g.decisive_fraction.max()))
        add(f"decisive accuracy {model}", pct_rng(g.decisive_accuracy.min(), g.decisive_accuracy.max()))

    # ---- wide table vs physical (treatment level, ~165 features) -------------------------------
    s3_ = E.m3_summary()
    dd = E.m3_deltas().set_index("target")
    lab = {"liver_healthy": "healthy liver", "bone_marrow": "bone marrow", "spleen": "spleen",
           "tumors_hepatic": "hepatic lesions", "tumor_burden": "tumour burden", "kidneys": "kidneys"}
    for t, name in lab.items():
        g = s3_[s3_.target == t].set_index("model")
        d = dd.loc[t]
        delta = f"{d.delta:+.2f} [{d.ci_lo:+.2f}, {d.ci_hi:+.2f}]".replace("-", "−")
        if d.ci_lo > 0:
            delta = f"**{delta}**"
        line = (f"| {name} | {f3(g.loc['TabPFN-3.5', 'log_r2'])} | {f3(g.loc['Physical baseline', 'log_r2'])} | "
                f"{delta} | {g.loc['TabPFN-3.5', 'width95_fold']:.1f} / {g.loc['Physical baseline', 'width95_fold']:.1f} |")
        add(f"M3 row {t}", line.replace("−", "−"))
    t35m3 = s3_[s3_.model == "TabPFN-3.5"]
    add("M3 cov95 range", rng(t35m3.cov_95.min(), t35m3.cov_95.max()))

    # ---- leakage ------------------------------------------------------------------------------
    lk = E.leakage().set_index("dataset_id")
    for did, name in (("tumors_all", "all lesions"), ("tumors_liver", "liver lesions"), ("kidneys", "kidneys (control)")):
        add(f"leakage {did}", f"| {name} | {lk.loc[did, 'r2_log_A_median']:.2f} | {lk.loc[did, 'r2_log_B_median']:.2f} |")
    # ---- synthetic benchmark (only if it has been run and committed) -------------------------
    bench = ROOT / "results" / "synthetic_benchmark" / "benchmark_summary.csv"
    if bench.exists():
        import pandas as pd
        b = pd.read_csv(bench)
        reps = b.groupby("target").n_repeats.agg(["min", "max"]) if "n_repeats" in b else None
        complete = (b.model.str.startswith("TabPFN-3.5").groupby(b.target).any().all()
                    and b.target.nunique() == 6 and b.model.nunique() == 4
                    and reps is not None and (reps["min"] == reps["max"].max()).all()
                    and len(b) == 24)
        assert complete, ("benchmark INCOMPLETE (some TabPFN-3.5 cells missing). Rerun "
                          "scripts/benchmark_synthetic.py until it prints COMPLETE, or do not commit "
                          "results/synthetic_benchmark/ and remove the benchmark row from the README")
        piv = b.pivot(index="target", columns="model", values="crps_log")
        cov = b.pivot(index="target", columns="model", values="cov95")
        tab = [m for m in piv.columns if m.startswith("TabPFN-3.5")][0]
        treat = [t for t in piv.index if t != "tumors"]
        assert len(treat) == 5, "benchmark: expected five treatment-level targets"
        others = [m for m in piv.columns if m != tab]
        assert all(piv.loc[t, tab] <= piv.loc[t, others].min() + 0.002 for t in treat), \
            "benchmark: TabPFN-3.5 is not lowest (or tied) on CRPS at every treatment-level target"
        assert all(piv.loc[t, tab] < piv.loc[t, others].min() for t in treat if t != "kidneys"), \
            "benchmark: TabPFN-3.5 is not strictly lowest on CRPS outside kidneys"
        assert (cov.loc[treat, tab] >= 0.93 - 1e-9).all(), "benchmark: TabPFN-3.5 treatment-level cov95 < 0.93"
        qh = [m for m in cov.columns if "quantile heads" in m][0]
        assert (cov[qh] <= 0.85).all() and (cov[qh] >= 0.75).all(), "benchmark: quantile heads not 'about 0.8'"
        phys = [m for m in piv.columns if m.startswith("Physical")][0]
        assert piv.loc["tumors", phys] < piv.loc["tumors", tab], "benchmark: physical not better at lesion level"
        assert 0.90 <= cov.loc["tumors", tab] <= 0.94, "benchmark: lesion-level cov95 not about 0.92"
        print("synthetic benchmark found: its README statements hold")
    else:
        print("synthetic benchmark not committed yet: its README statements are not checked")
    return c


def main() -> int:
    try:
        claims = build_claims()
    except AssertionError as e:
        print(f"FAIL  structural claim: {e}")
        return 1
    bad = 0
    for label, text, where in claims:
        for doc in where:
            ok = text in DOCS[doc]
            bad += not ok
            if not ok:
                print(f"FAIL  {doc:10s} {label}: expected to find  {text}")
    print(f"{len(claims)} claims checked across {sum(len(w) for _, _, w in claims)} locations, "
          f"plus structural checks; {bad} mismatch(es).")
    print("CLAIMS_OK" if bad == 0 else "CLAIMS_FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
