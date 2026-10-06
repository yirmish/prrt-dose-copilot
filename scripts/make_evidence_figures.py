"""make_evidence_figures.py - figures for the README / app, made ONLY from committed aggregate CSVs.

    python scripts/make_evidence_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "real_cohort" / "lesion_8models_2026-10-03"
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#a7abb2", "#1b1f24", "#5b6673"
LABEL = {"TabPFN-3.5": "TabPFN-3.5", "TabPFN-2": "TabPFN-2", "CatBoost": "CatBoost (tuned)",
         "XGBoost": "XGBoost (tuned)", "HistGBM": "HistGBM (tuned)", "LightGBM": "LightGBM (tuned)",
         "RandomForest": "Random forest (tuned)", "ElasticNet": "Elastic net (tuned)"}


def main() -> Path:
    comp = pd.read_csv(D / "lesion_components_and_calibration.csv").set_index("model")
    h2h = pd.read_csv(D / "lesion_head_to_head_tabpfn35.csv")
    h2h = h2h[(h2h.n_estimators == 8) & (h2h.comparator != "TabPFN-2")]

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    order = comp.sort_values("r2_overall").index.tolist()
    y = range(len(order))
    for i, m in enumerate(order):
        col = BLUE if m == "TabPFN-3.5" else GREY
        for val, off, alpha in ((comp.loc[m, "r2_overall"], 0.18, 1.0), (comp.loc[m, "r2_within_patient"], -0.18, 0.55)):
            a.barh(i + off, val, height=0.34, color=col, alpha=alpha, lw=0)
            a.text(val + 0.006, i + off, f"{val:.2f}", va="center", fontsize=8, color=INK)
    a.set_yticks(list(y), [LABEL[m] for m in order], fontsize=9)
    a.set_xlim(0, 0.56)
    a.set_xlabel("R² on log(Gy/GBq), lesion level", color=MUTED)
    a.set_title("A. Lesion dose, 307 lesions / 81 patients\nsolid = overall R², light = within-patient R²",
                fontsize=10, loc="left", color=INK)

    cmp_order = (h2h[h2h.metric == "R2"].sort_values("delta").comparator.tolist())
    for j, (metric, col, off, name) in enumerate((("R2", BLUE, 0.15, "overall R²"),
                                                  ("R2_within_patient", ORANGE, -0.15, "within-patient R²"))):
        sub = h2h[h2h.metric == metric].set_index("comparator").loc[cmp_order]
        ys = [k + off for k in range(len(cmp_order))]
        b.errorbar(sub.delta, ys, xerr=[sub.delta - sub.ci_lo, sub.ci_hi - sub.delta], fmt="o", ms=6,
                   color=col, ecolor=col, elinewidth=2, capsize=0, label=name)
    b.axvline(0, color=MUTED, lw=1, ls="--")
    b.set_yticks(range(len(cmp_order)), [LABEL[c] for c in cmp_order], fontsize=9)
    b.set_xlabel("TabPFN-3.5 minus model (95% patient-bootstrap CI)", color=MUTED)
    b.set_title("B. Paired differences, identical folds\nright of the dashed line = TabPFN-3.5 better",
                fontsize=10, loc="left", color=INK)
    b.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    fig.text(0.01, 0.035, "5×5 CV grouped by patient · 140 pre-therapy features · tuned models: nested grid + "
             "split-conformal intervals · TabPFN-3.5: no tuning, native quantiles", fontsize=7.5, color=MUTED)
    fig.text(0.01, 0.008, "Discovery result on one centre's cohort; registered robustness check (8 → 32 estimators) "
             "passed; not yet confirmed on independent patients.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = D / "fig_lesion_8models.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(main())
