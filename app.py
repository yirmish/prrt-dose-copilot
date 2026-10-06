"""PRRT Dose Copilot - a dose predictor that knows when to measure.      streamlit run app.py

TabPFN-3.5 predictive distributions for 177Lu-DOTATATE absorbed dose from the pre-therapy
68Ga-DOTATATE PET. The context table is a SYNTHETIC demonstration cohort; every number about the
real cohort is a precomputed aggregate read from results/real_cohort/.
Research prototype - not a medical device, not for clinical use.
"""
from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from copilot import calibration as CAL  # noqa: E402
from copilot import evidence as EV  # noqa: E402
from copilot.engine import (ACTIVITY, BACKENDS, GROUP, TARGETS, DoseModel, TabPFN35, apply_whatif,  # noqa: E402
                            available_backends, load_context, outside_context, split_examples, summarise)

# Streamlit Cloud / HF Spaces secrets -> environment (the token is never shown or logged)
try:
    for k in ("TABPFN_TOKEN", "TABPFN35_WEIGHTS"):
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass

# In the app a visitor is waiting: answer within seconds or fall back to the physical baseline with a notice.
# (scripts/benchmark_synthetic.py keeps the engine's longer retry schedule.)
TabPFN35.API_TRIES = int(os.environ.get("TABPFN_API_TRIES", "3"))
TabPFN35.API_WAIT_S = float(os.environ.get("TABPFN_API_WAIT_S", "1.5"))

BLUE, GREY, ORANGE, GREEN, RED = "#1f5fa8", "#8a8f98", "#d9822b", "#2e9e5b", "#c0392b"
st.set_page_config(page_title="PRRT Dose Copilot", page_icon="☢️", layout="wide")

# ------------------------------------------------------------------------------------------------
# sidebar
# ------------------------------------------------------------------------------------------------
st.sidebar.title("PRRT Dose Copilot")
st.sidebar.caption("Knows when to measure.")
labels = {k: v[3] for k, v in TARGETS.items()}
target = st.sidebar.selectbox("Organ / target", list(TARGETS), format_func=labels.get)
backs = available_backends()
backend = st.sidebar.radio("Engine", backs, format_func=BACKENDS.get,
                           help="TabPFN-3.5 is the engine; the physical baseline is the comparator.")
if not any(b.startswith("tabpfn") for b in backs):
    st.sidebar.warning("No TabPFN-3.5 back-end configured (set TABPFN_TOKEN, or install the local "
                       "package). Showing the physical baseline only.")
decisive_p = st.sidebar.slider("Call a result decisive when P ≥", 0.70, 0.99, 0.90, 0.01,
                               help="A convention, not a derived quantity. 0.90 is the study's setting.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Research prototype.** Not a medical device, not for clinical use. "
                    "The context table is **synthetic** - no patient data is used or sent anywhere.")


# ------------------------------------------------------------------------------------------------
# cached models (shared, immutable) - one fit per (target, engine)
# ------------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_split(target: str):
    return split_examples(load_context(target), frac=0.2, seed=0)


@st.cache_resource(show_spinner="Putting the context table into TabPFN-3.5 (no training step)...")
def get_model(target: str, backend: str) -> DoseModel:
    ctx, _ = get_split(target)
    return DoseModel(target, ctx, backend=backend).fit()


def model_or_fallback(target: str, backend: str) -> tuple[DoseModel, str | None]:
    try:
        return get_model(target, backend), None
    except Exception as e:                                   # API down, quota, missing weights...
        return get_baseline(target), f"{BACKENDS[backend]} unavailable ({type(e).__name__}); showing the physical baseline."


@st.cache_data(show_spinner=False)
def live_cv_one(target: str, b: str) -> dict:
    Q, y, dd = CAL.oof_frame(target, load_context(target), b, k=5)
    rep = CAL.report(Q, y)
    rep["decision"] = CAL.decision_curve(Q, dd, TARGETS[target][1]).to_dict("list")
    return rep


def live_cv(target: str, backend: str) -> tuple[dict[str, dict], str | None]:
    """Same folds for every model: the selected engine, the physical baseline, gradient boosting
    with quantile heads (the usual way to get intervals from trees) and gradient boosting nested-tuned
    with split-conformal intervals (the real-cohort comparator design).
    If the TabPFN-3.5 API does not answer, the comparators are still shown and the error is reported
    (a failure is not cached, so pressing the button again retries)."""
    out, err = {}, None
    for b in dict.fromkeys([backend, "physical-baseline", "gbm-quantile", "gbm-tuned-conformal"]):
        try:
            out[b] = live_cv_one(target, b)
        except Exception as e:
            if not b.startswith("tabpfn"):
                raise
            err = (f"{BACKENDS[b]} did not answer during the live check ({type(e).__name__}); showing the "
                   "comparators only. Press the button again to retry.")
    return out, err


@st.cache_resource(show_spinner=False)
def get_baseline(target: str) -> DoseModel:
    ctx, _ = get_split(target)
    return DoseModel(target, ctx, backend="physical-baseline").fit()


def case_label(row: pd.Series, target: str) -> str:
    unit = "lesion" if TARGETS[target][2] == "lesion" else "treatment"
    return (f"{row[GROUP]} · {unit} · volume {row['pet_volume_ml']:.0f} ml · "
            f"SUVmean {row['pet_mean_suvbw']:.1f} · {row[ACTIVITY]:.1f} GBq")


SCENARIOS = np.arange(0.05, 1.0, 0.10)        # 5th, 15th, ... 95th percentile: a quantile dotplot


def interval_figure(r: dict, base: dict | None, thresholds, measured: float | None):
    fig, ax = plt.subplots(figsize=(8, 2.9 if base else 2.1))
    rows = [("TabPFN-3.5" if r["backend"].startswith("tabpfn") else "Physical baseline", r, BLUE)]
    if base is not None:
        rows.append(("Physical baseline", base, GREY))
    for i, (name, res, col) in enumerate(rows):
        y = -i
        for (lo, hi), h, a in ((res["interval95_gy"], 0.30, 0.22), (res["interval80_gy"], 0.45, 0.45),
                               (res["interval50_gy"], 0.60, 0.85)):
            ax.barh(y, hi - lo, left=lo, height=h, color=col, alpha=a, lw=0)
        ax.plot([res["median_gy"]] * 2, [y - 0.33, y + 0.33], color="black", lw=2)
        dots = np.interp(SCENARIOS, res["levels"], res["quantiles_gy"])       # ten equally likely scenarios
        ax.scatter(dots, [y] * len(dots), s=16, color="white", edgecolors=col, linewidths=1.1, zorder=3)
    for j, thr in enumerate(thresholds):
        ax.axvline(thr, color=RED, ls="--", lw=1)
        ax.text(thr, 0.50 + 0.17 * (j % 2), f"{thr:g} Gy", color=RED, fontsize=8, ha="center")
    if measured is not None:
        ax.plot([measured] * 2, [-(len(rows) - 1) - 0.45, 0.45], color=GREEN, lw=2.5)
        ax.text(measured, -(len(rows) - 1) - 0.62, "measured", color=GREEN, fontsize=8, ha="center")
    ax.set_yticks([-i for i in range(len(rows))], [n for n, _, _ in rows])
    ax.set_xscale("log")
    ax.set_xlabel("absorbed dose (Gy)\nbands: 50 / 80 / 95% intervals   line: median   dots: ten equally likely scenarios", fontsize=9)
    ax.set_ylim(-(len(rows) - 1) - 0.8, 0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def reliability_figure(curves: dict[str, list[tuple[float, float]]], colors: dict[str, str], title: str):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], color="#bbbbbb", ls="--", lw=1)
    for name, pts in curves.items():
        x, y = zip(*pts)
        ax.plot(x, y, marker="o", ms=4, lw=2 if "TabPFN-3.5" in name else 1.4,
                color=colors.get(name, "#555"), label=name)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("nominal coverage of the interval"); ax.set_ylabel("observed coverage")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8, loc="upper left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def decision_figure(curves: dict[str, dict], colors: dict[str, str], decisive_p: float):
    """Risk-coverage: share of threshold calls made decisive vs share of those calls that were right."""
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for name, c in curves.items():
        x, y, cut = np.array(c["frac_decisive"]), np.array(c["acc_decisive"], float), np.array(c["cutoff"])
        col = colors.get(name, "#555")
        ax.plot(x, y, lw=2 if "TabPFN-3.5" in name else 1.4, color=col, label=name)
        j = int(np.argmin(np.abs(cut - decisive_p)))
        ax.plot(x[j], y[j], "o", ms=7, color=col, mec="white", mew=1.5)
    ax.set_xlim(0, 1); ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("share of threshold calls made decisive"); ax.set_ylabel("share of decisive calls that were right")
    ax.set_title(f"Knowing when to measure (dot: P ≥ {decisive_p:.2f})", fontsize=10)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


MODEL_COLORS = {"TabPFN-3.5": BLUE, "Physical baseline": GREY, "CatBoost (tuned, quantile heads)": ORANGE,
                "TabPFN-2 (open weights)": "#7aa6d6"}

# ------------------------------------------------------------------------------------------------
st.title("PRRT Dose Copilot")
st.markdown("Before ¹⁷⁷Lu-DOTATATE therapy, every patient has a ⁶⁸Ga-DOTATATE PET. The absorbed dose is "
            "measured *after* therapy with SPECT/CT - too late for the first cycle. Our study found the PET "
            "predicts that dose **only partly**. So this tool does not pretend: TabPFN-3.5 returns a **calibrated "
            "dose distribution**, and for each clinical threshold it says either **decisive** or **measure**.")

tab_pred, tab_cal, tab_real, tab_val, tab_add, tab_about = st.tabs(
    ["Predict", "Is it calibrated? (live)", "Real-cohort evidence", "Honest validation",
     "Add a measured case", "About"])

ctx, examples = get_split(target)

# ------------------------------------------------------------------------------------------------
with tab_pred:
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.subheader("A new case")
        st.caption(f"{len(examples)} held-out synthetic cases the model has not seen "
                   f"(context: {len(ctx)} rows).")
        idx = st.selectbox("Pick a case", list(range(len(examples))),
                           format_func=lambda i: case_label(examples.iloc[i], target))
        base_case = examples.iloc[idx].to_dict()
        st.markdown("**What if?** (keeps the PET row internally consistent)")
        up = st.slider("Uptake ×", 0.5, 2.0, 1.0, 0.05, help="Scales every SUV / Bq/ml statistic and the totals.")
        vol = st.slider("Volume ×", 0.25, 4.0, 1.0, 0.05, help="Scales volume, totals and sphere diameter.")
        act = st.number_input("Planned activity (GBq)", 1.0, 10.0, float(round(base_case[ACTIVITY], 2)), 0.1,
                              help="Dose per GBq is what the model predicts; dose in Gy scales linearly "
                                   "with activity by construction (physics, not a learned effect).")
        reveal = st.toggle("Show the measured dose", value=True,
                           disabled=(up != 1.0 or vol != 1.0),
                           help="Only for the unedited case: it is synthetic ground truth.")
        case = apply_whatif(base_case, up, vol)

    with right:
        m, err = model_or_fallback(target, backend)
        m = st.session_state.get(f"model::{target}::{backend}") or m
        if err:
            st.error(err)
        b = get_baseline(target)
        call_s = None
        with st.spinner("Predicting..."):
            try:
                t0 = time.perf_counter()
                r = m.predict(case, act, decisive_p)
                call_s = time.perf_counter() - t0
            except Exception as e:                            # API error or quota mid-session: never a traceback
                st.error(f"{BACKENDS[m.backend]} did not answer ({type(e).__name__}). Showing the physical "
                         "baseline for this case - try again in a moment.")
                m = b
                r = m.predict(case, act, decisive_p)
        rb = b.predict(case, act, decisive_p)
        measured = None
        if reveal and up == 1.0 and vol == 1.0:
            measured = float(base_case["dose_gy"]) * act / float(base_case[ACTIVITY])
        st.subheader("Predicted absorbed dose")
        oor = outside_context(case, m.context)
        if oor:
            st.warning("**Outside the context table:** " + "; ".join(oor) + ". No case like this is in the table, "
                       "so the model is extrapolating. Treat the forecast as unreliable and measure.")
        fmt = lambda v: f"{v:.0f}" if v >= 10 else f"{v:.1f}" if v >= 1 else f"{v:.2f}"
        c1, c2, c3 = st.columns(3)
        c1.metric("Median", f"{fmt(r['median_gy'])} Gy")
        c2.metric("80% interval", f"{fmt(r['interval80_gy'][0])}–{fmt(r['interval80_gy'][1])} Gy")
        c3.metric("95% interval", f"{fmt(r['interval95_gy'][0])}–{fmt(r['interval95_gy'][1])} Gy")
        show(interval_figure(r, rb if m.backend != "physical-baseline" else None, m.thresholds, measured))
        if r["thresholds"]:
            rows = []
            for t, v in r["thresholds"].items():
                vb = rb["thresholds"][t]
                rows.append({"threshold": f"{t:g} Gy", "P(dose > threshold)": round(v["p_exceed"], 3),
                             "call": v["call"].upper(), "baseline P": round(vb["p_exceed"], 3),
                             "baseline call": vb["call"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            if any(v["call"] == "measure" for v in r["thresholds"].values()):
                st.warning("**MEASURE** - at one or more thresholds the distribution straddles the line. "
                           "The honest output is: do the post-therapy SPECT/CT.")
            else:
                st.success("**DECISIVE** at every threshold for this case, at the chosen confidence.")
        else:
            st.info("The study defines no clinical threshold for this organ - distribution only.")
        s1 = EV.m1_summary()
        row = s1[(s1.target == target) & (s1.model == "TabPFN-3.5")]
        if len(row):
            x = row.iloc[0]
            dcs = EV.decisive()
            dcs = dcs[(dcs.target == target) & (dcs.model == "TabPFN-3.5")]
            extra = "; ".join(f"at {t:g} Gy {f:.0%} of cases decisive, {a:.0%} of those right"
                              for t, f, a in zip(dcs.threshold_gy, dcs.decisive_fraction, dcs.decisive_accuracy))
            st.info(f"**How far to trust this target (real cohort, TabPFN-3.5, PET features):** "
                    f"80% intervals covered {x.cov_80:.0%} and 95% intervals {x.cov_95:.0%} of measured doses; "
                    f"log R² {x.log_r2:.2f}." + (f" Decisive calls: {extra}." if extra else ""))
        live = (f" This prediction was one live call to TabPFN-3.5 ({call_s:.1f} s)."
                if call_s is not None and m.backend.startswith("tabpfn") else "")
        st.caption(f"Engine: {BACKENDS[m.backend]}. Context: synthetic cohort, {len(m.context)} rows, "
                   f"{len(m.features)} PET features. Quantiles are back-transformed; a mean is never exponentiated."
                   + live)

# ------------------------------------------------------------------------------------------------
with tab_cal:
    st.subheader("Does an 80% interval contain the dose 80% of the time?")
    st.markdown("Calibration is the property the product rests on, so you can check it here, live: "
                "5-fold cross-validation **grouped by synthetic patient** on the context table - the engine you "
                "selected, the physical baseline, and gradient boosting two ways (quantile heads; nested-tuned with "
                "split-conformal intervals), on identical folds. "
                "A curve on the diagonal is honest; a curve below it is overconfident. On the real cohort the "
                "same check was run with person-grouped folds (right).")
    colL, colR = st.columns(2, gap="large")
    with colL:
        key = f"cv::{target}::{backend}"
        if st.button("Run the live calibration check", type="primary"):
            with st.spinner("5 folds, grouped by synthetic patient: fit on 4/5, predict quantiles for 1/5..."):
                try:
                    st.session_state[key], cv_err = live_cv(target, backend)
                    if cv_err:
                        st.error(cv_err)
                except Exception as e:
                    st.error(f"Live check failed ({type(e).__name__}). Try the physical baseline or later.")
        if key in st.session_state:
            reps = st.session_state[key]
            names = {"tabpfn-3.5-api": "TabPFN-3.5", "tabpfn-3.5-local": "TabPFN-3.5",
                     "physical-baseline": "Physical baseline", "gbm-quantile": "Gradient boosting quantile heads",
                     "gbm-tuned-conformal": "Gradient boosting tuned + conformal"}
            reps = {b: r for b, r in reps.items() if b in names}
            curves = {names[b]: r["reliability"] for b, r in reps.items()}
            colors = {**MODEL_COLORS, "Gradient boosting quantile heads": ORANGE,
                      "Gradient boosting tuned + conformal": "#8e5a2b"}
            n = next(iter(reps.values()))["n"]
            show(reliability_figure(curves, colors, f"Synthetic cohort, live, identical folds (n={n})"))
            dcur = {names[b]: r["decision"] for b, r in reps.items() if r.get("decision", {}).get("cutoff")}
            if dcur:
                show(decision_figure(dcur, colors, decisive_p))
                st.caption("Every held-out case × every clinical threshold of this target. Moving along a curve "
                           "= raising the confidence needed to skip the measurement. Higher at the same share "
                           "decisive is better; a model far to the right at lower accuracy is confidently wrong.")
            tbl = pd.DataFrame([{"model": names[b], "cover 50%": round(x["cov50"], 2), "cover 80%": round(x["cov80"], 2),
                                 "cover 95%": round(x["cov95"], 2), "95% span (×)": round(x["width95_fold"], 1),
                                 "CRPS (log)": round(x["crps_log"], 3), "log R²": round(x["log_r2"], 3)}
                                for b, x in reps.items()])
            st.dataframe(tbl, hide_index=True)
            st.caption("Synthetic data: these numbers demonstrate the mechanism, they are not the study's results. "
                       "Gradient boosting (sklearn HistGradientBoosting): *quantile heads* = one quantile-loss model "
                       "per level, fixed settings; *tuned + conformal* = settings picked by an inner patient-grouped "
                       "3-fold CV over 4 settings, intervals from the inner out-of-fold residuals (split-conformal) - "
                       "the design of the real-cohort comparators. The synthetic table is close to log-linear, which "
                       "favours the 2-term physical model; the real-cohort tab has the real comparison.")
        else:
            st.caption("About a minute: the selected engine, the physical baseline and two gradient-boosting "
                       "variants (quantile heads; tuned + conformal) on the same 5 folds. Cached afterwards.")
    with colR:
        rel = EV.reliability()
        rr = rel[(rel.target == target)]
        curves = {mname: list(zip(g.nominal, g.empirical)) for mname, g in rr.groupby("model")}
        show(reliability_figure(curves, MODEL_COLORS,
                                     "REAL cohort, person-grouped 5×5 CV (17 PET features)"))
        st.caption("Real cohort: 86 treatments / 82 patients / 309 lesions. Aggregates from "
                   "`results/real_cohort/v35_2026-10-06/v35_reliability_curve.csv`. 5 of 20 protocol repeats.")
    if TARGETS[target][2] == "lesion":
        st.warning("Lesion level: on the real cohort, with the lesion's 17 PET features (this app's feature set), "
                   "TabPFN-3.5's lesion intervals are **too narrow** (nominal 95% covers 0.89): lesions of one "
                   "patient share most of their error (ICC 0.78). With the 140-feature table, which carries "
                   "patient-level PET, they cover 0.80 / 0.94 at 80 / 95%. Read this app's lesion intervals as optimistic.")

# ------------------------------------------------------------------------------------------------
with tab_real:
    st.subheader("What the real cohort says (aggregates only)")
    st.markdown("#### 1 · Lesion dose, wide table: TabPFN-3.5 against six tuned models")
    st.markdown("307 lesions, 85 treatments, 81 patients; the 140-column pre-therapy table (the lesion's PET, "
                "the other organs' PET, treatment and body size). 5×5 CV **grouped by patient**, identical folds. "
                "Each conventional model is **nested-tuned** (inner patient-grouped CV over a 4-setting grid) and "
                "gets **split-conformal** intervals; TabPFN-3.5 is untuned and uses its native quantiles.")
    c8 = EV.lesion8_components().set_index("model")
    h8 = EV.lesion8_head_to_head()
    hR = h8[(h8.metric == "R2") & (h8.n_estimators == 8) & (h8.tuned != "no (native)")]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lesion R², TabPFN-3.5", f"{c8.loc['TabPFN-3.5', 'r2_overall']:.3f}",
              f"best tuned model {c8.drop(index=['TabPFN-3.5', 'TabPFN-2']).r2_overall.max():.3f}",
              delta_color="off", delta_arrow="off")
    cb = hR[hR.comparator == "CatBoost"].iloc[0]
    k2.metric("Δ R² vs tuned CatBoost", f"{cb.delta:+.3f}", f"95% CI [{cb.ci_lo:+.3f}, {cb.ci_hi:+.3f}]",
              delta_color="off", delta_arrow="off")
    k3.metric("Within-patient R²", f"{c8.loc['TabPFN-3.5', 'r2_within_patient']:.3f}",
              f"tuned models {c8.drop(index=['TabPFN-3.5', 'TabPFN-2']).r2_within_patient.min():.2f}–"
              f"{c8.drop(index=['TabPFN-3.5', 'TabPFN-2']).r2_within_patient.max():.2f}", delta_color="off", delta_arrow="off")
    k4.metric("Native 80 / 95% coverage", f"{c8.loc['TabPFN-3.5', 'cov80']:.2f} / {c8.loc['TabPFN-3.5', 'cov95']:.2f}",
              "no calibration step", delta_color="off", delta_arrow="off")
    if EV.LESION8_FIG.exists():
        st.image(str(EV.LESION8_FIG), width="stretch")
    tbl8 = h8[h8.n_estimators == 8].pivot_table(index="comparator", columns="metric",
                                                  values=["delta", "ci_lo", "ci_hi"], aggfunc="first")
    rows8 = []
    order8 = ["CatBoost", "XGBoost", "HistGBM", "LightGBM", "RandomForest", "ElasticNet", "TabPFN-2"]
    for comp in [c_ for c_ in order8 if c_ in tbl8.index]:
        rr_ = {"TabPFN-3.5 minus": comp + ("" if comp == "TabPFN-2" else " (tuned)")}
        for met, lab in (("R2", "Δ R²"), ("R2_within_patient", "Δ within-patient R²"), ("CRPS", "Δ CRPS (lower is better)")):
            rr_[lab] = (f"{tbl8.loc[comp, ('delta', met)]:+.3f} [{tbl8.loc[comp, ('ci_lo', met)]:+.3f}, "
                        f"{tbl8.loc[comp, ('ci_hi', met)]:+.3f}]")
        rows8.append(rr_)
    st.dataframe(pd.DataFrame(rows8), hide_index=True)
    r1 = h8[(h8.comparator == "CatBoost") & (h8.metric == "R2") & (h8.n_estimators == 32)].iloc[0]
    st.caption(f"95% CIs: patient-level paired bootstrap (B = 4000). Registered robustness check (hypothesis "
               f"registered before the re-run, 32 estimators): {r1.delta:+.3f} [{r1.ci_lo:+.3f}, {r1.ci_hi:+.3f}] "
               "→ robust. Source: `results/real_cohort/lesion_8models_2026-10-03/`.")
    with st.expander("Read before quoting: what this result is and is not"):
        st.markdown("- **Discovery plus a registered robustness check, not a confirmation.** The 8-model comparison "
                    "was not registered in advance; TabPFN-3.5 was the eighth model examined on these patients. The "
                    "confirmatory test needs about 100 new patients.\n"
                    "- **Multiplicity:** 7 comparators × 6 targets × 3 metrics, unadjusted. The lesion-level result "
                    "is uniform across all six tuned models; against TabPFN-2 nothing is resolvable.\n"
                    "- **The feature set matters.** With only the lesion's own 17 PET features, TabPFN-3.5 does "
                    "*not* beat tuned CatBoost (section 2). The advantage appears with the wide table - where tuned "
                    "trees do not improve.\n"
                    "- **Treatment level (n = 85):** TabPFN-3.5 has the top R² on kidneys, healthy liver and spleen, "
                    "not on tumour burden or bone marrow; differences there are not resolvable at this sample size.")
    st.markdown("---")
    st.markdown("#### 2 · Calibration with the 17 PET features (run of 2026-10-06)")
    st.markdown("86 cycle-1 treatments, 82 patients, 309 lesions, one centre. 5×5 cross-validation grouped by "
                "patient, target log(Gy/GBq), identical folds for every model. **5 of the protocol's 20 repeats - "
                "a preliminary measurement, to be confirmed.** Here CatBoost is nested-tuned "
                "with quantile-loss heads and *no* calibration step - the usual way to get intervals from trees.")
    s1 = EV.m1_summary()
    tbl1 = s1[s1.target == target][["model", "log_r2", "cov_50", "cov_80", "cov_95", "width95_fold", "crps_log"]]
    s3 = EV.m3_all_arms()
    tbl3 = s3[s3.target == target].sort_values("model", ascending=False)[["model", "log_r2", "cov_50", "cov_80", "cov_95", "width95_fold", "crps_log"]]
    ren = {"log_r2": "log R²", "cov_50": "cover 50%", "cov_80": "cover 80%", "cov_95": "cover 95%",
           "width95_fold": "95% span (×)", "crps_log": "CRPS (log)"}
    st.markdown("**17 PET features of the organ / lesion (M1)**")
    st.dataframe(tbl1.rename(columns=ren).round(3), hide_index=True)
    if len(tbl3):
        st.markdown("**~165 features: all organs' PET + treatment + body (M3) - every model on identical rows, folds and features**")
        st.dataframe(tbl3.rename(columns=ren).round(3), hide_index=True)
        dd = EV.m3_deltas()
        dd = dd[dd.target == target]
        if len(dd):
            x = dd.iloc[0]
            st.caption(f"M3, TabPFN-3.5 minus physical baseline, log R²: {x.delta:+.3f} "
                       f"[{x.ci_lo:+.3f}, {x.ci_hi:+.3f}] (patient-clustered bootstrap); pre-registered margin "
                       f"{x.margin:.2f}.")
        cb = EV.m3_vs_catboost(target)
        if cb:
            st.caption(f"M3, TabPFN-3.5 against tuned CatBoost (quantile heads): CRPS {cb['crps_rel']:+.0f}% "
                       f"({cb['crps_delta']:+.3f} [{cb['crps_lo']:+.3f}, {cb['crps_hi']:+.3f}]; lower is better), "
                       f"log R² {cb['r2_delta']:+.3f} [{cb['r2_lo']:+.3f}, {cb['r2_hi']:+.3f}]. Post-hoc, from the "
                       "stored out-of-fold quantiles.")
    st.markdown("---")
    st.markdown("**The point in one chart: who is confidently wrong?** Tumour burden, real cohort. "
                "Bars = share of patients the model calls *decisive*; label = how many of those calls were right.")
    dc = EV.decisive()
    dc = dc[(dc.target == "tumor_burden")]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    models = ["CatBoost (tuned, quantile heads)", "TabPFN-3.5", "Physical baseline"]
    thr = sorted(dc.threshold_gy.unique())
    w = 0.26
    for i, mname in enumerate(models):
        g = dc[dc.model == mname].set_index("threshold_gy").reindex(thr)
        xs = np.arange(len(thr)) + (i - 1) * w
        ax.bar(xs, g.decisive_fraction, width=w, color=MODEL_COLORS[mname], label=mname)
        for xv, f, acc in zip(xs, g.decisive_fraction, g.decisive_accuracy):
            ax.text(xv, f + 0.02, f"{acc:.0%}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(thr)), [f"{t:g} Gy" for t in thr]); ax.set_ylim(0, 1)
    ax.set_ylabel("decisive fraction"); ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    show(fig)
    st.markdown("Tuned CatBoost with quantile heads calls about two thirds of patients decisive and is right on 77-87% "
                "of those calls; TabPFN-3.5 calls 19-34% decisive and is right on 93-97%. The 2-term physical "
                "baseline is also honest (17-27% decisive, 88-96% right); TabPFN-3.5's intervals are narrower than "
                "the baseline's (7 of 7 targets with PET features alone; 6 of 6 with the wide table) and, with the "
                "wide table, more accurate on healthy liver and bone marrow.")
    with st.expander("What we do NOT claim"):
        st.markdown("- **With the 17 PET features alone**, an accuracy win: on tumours TabPFN-3.5 ties or trails "
                    "tuned CatBoost and the physical model at this sample size. The accuracy result is the wide-table "
                    "one in section 1, and it is a discovery result awaiting confirmation.\n"
                    "- That predicted dose is precise enough to plan therapy: a 95% tumour interval spans ~10×.\n"
                    "- That lesion intervals from the 17 PET features are calibrated (95% covers 0.89); with the "
                    "wide table they reach 0.80 / 0.94.\n"
                    "- Anything on the synthetic cohort as a study result.")

# ------------------------------------------------------------------------------------------------
with tab_val:
    st.subheader("Why the published numbers look better than ours")
    st.markdown("Lesions of one patient are alike. If the same patient's lesions sit on both sides of a "
                "train/test split, the model is graded on patients it has already seen. Same model, same data, "
                "two validation schemes:")
    lk = EV.leakage()
    name = {"tumors_liver": "Liver lesions", "tumors_all": "All lesions", "kidneys": "Kidneys (control: ~1 row per patient)"}
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ys = np.arange(len(lk))
    ax.barh(ys + 0.18, lk.r2_log_A_median, height=0.34, color=ORANGE, label="random split (lesions mixed)")
    ax.barh(ys - 0.18, lk.r2_log_B_median, height=0.34, color=BLUE, label="split by patient")
    for y_, a_, b_ in zip(ys, lk.r2_log_A_median, lk.r2_log_B_median):
        ax.text(a_ + 0.01, y_ + 0.18, f"{a_:.2f}", va="center", fontsize=8)
        ax.text(b_ + 0.01, y_ - 0.18, f"{b_:.2f}", va="center", fontsize=8)
    ax.set_yticks(ys, [name.get(x, x) for x in lk.dataset_id]); ax.set_xlim(0, 1)
    ax.set_xlabel("log R² (median of 50 repeats)"); ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    show(fig)
    st.markdown("The kidney control has about one row per patient, so there is nothing to leak - and the gap vanishes "
                "(+0.05, interval includes 0). For lesions the gap is +0.39 to +0.46. A replication of a published "
                "lesion-dose model on this cohort gave R² 0.52 with a random split and 0.31 with a patient split "
                "(paired difference 0.23, 95% CI 0.12-0.38; preliminary, B=50). The two published studies that held out "
                "patients or centres report R² 0.24 and 0.25.")
    st.caption("Source: `results/real_cohort/validation/leakage_summary.csv` (aggregates).")

# ------------------------------------------------------------------------------------------------
with tab_add:
    st.subheader("Add a measured case - no training step")
    st.markdown("TabPFN-3.5 learns in context: the table *is* the model. A new measured case is one more row; "
                "the next prediction already uses it. Additions live only in your browser session.")
    skey = f"model::{target}::{backend}"
    cur = st.session_state.get(skey) or model_or_fallback(target, backend)[0]
    st.write(f"Context rows now: **{len(cur.context)}** (shared starting table: {len(ctx)}).")
    st.markdown("Uses the case currently selected in **Predict** (with any what-if edits) and the dose you enter.")
    meas = st.number_input("Measured absorbed dose (Gy)", 0.01, 1000.0,
                           float(min(max(round(r["median_gy"], 2), 0.01), 1000.0)), 0.1,
                           key=f"meas::{target}::{idx}::{up}::{vol}::{act}")
    cA, cB = st.columns(2)
    if cA.button("Add to my context and re-predict", type="primary"):
        row = {**case, ACTIVITY: act, "dose_gy": meas, "is_synthetic": 1, GROUP: "USER-ADDED"}
        try:
            with st.spinner("Re-reading the context (seconds)..."):
                added = cur.with_added_case(row)
                new = added.predict(case, act, decisive_p)
            if added.backend == backend:          # never keep a fallback model under the TabPFN key
                st.session_state[skey] = added
            st.success(f"Context: {len(added.context)} rows. Median for this case "
                       f"{r['median_gy']:.2f} → {new['median_gy']:.2f} Gy; 80% interval "
                       f"{r['interval80_gy'][0]:.1f}–{r['interval80_gy'][1]:.1f} → "
                       f"{new['interval80_gy'][0]:.1f}–{new['interval80_gy'][1]:.1f} Gy.")
        except Exception as e:
            st.error(f"{BACKENDS[cur.backend]} did not answer ({type(e).__name__}); the case was not added. "
                     "Try again in a moment.")
    if cB.button("Reset to the shared table"):
        st.session_state.pop(skey, None)
        st.rerun()

# ------------------------------------------------------------------------------------------------
with tab_about:
    st.markdown("""
**What it is.** A research prototype built for the Prior Labs TabPFN-3.5 hackathon from a single-centre study
of ¹⁷⁷Lu-DOTATATE dosimetry (86 cycle-1 treatments, 82 patients, 309 lesions; post-therapy SPECT/CT dosimetry).

**Engine.** `TabPFNRegressor`, model version 3.5, target log(Gy/GBq), `output_type="quantiles"` on a 1-99% grid,
8 estimators, no tuning. Hosted demo: Prior Labs API (`tabpfn-client`, `fit_with_cache`). Offline: local
weights (`tabpfn` 9.0.0, checkpoint SHA-256 `ece4d67e…f0be3`).

**Data.** The context table is a synthetic cohort (Gaussian copula fitted to a compact feature set of the real,
de-identified data; no real row copied; fidelity and nearest-neighbour privacy checks in `data/synthetic/`).
Quasi-identifiers (age, sex, height, dates) are not shipped. Real-cohort results are aggregates only.

**Thresholds** are the study's: tumour 20 / 30 / 36.2-36.5 / 40 Gy, kidneys 5.75 Gy per cycle,
bone marrow 0.5 Gy. Liver and spleen have none, so none is invented.

**Not a medical device. Not for clinical use.**
""")
