"""benchmark_synthetic.py - the live calibration check, on every target, for every model, on identical folds.

    python scripts/benchmark_synthetic.py --backend api   --repeats 3     # needs TABPFN_TOKEN (synthetic data only)
    python scripts/benchmark_synthetic.py --backend local --repeats 3     # needs local TabPFN-3.5 weights
    python scripts/benchmark_synthetic.py --backend none                  # comparators only (no TabPFN)

For each target and repeat: 5-fold CV grouped by synthetic patient (seed = repeat), and for each model the
out-of-fold predictive quantiles -> coverage of the 50 / 80 / 95% intervals, 95% span, CRPS, log R2, and
the share of threshold calls made decisive at P >= 0.9 with the share of those that were right.

Models: TabPFN-3.5 (untuned, native quantiles), the 2-term physical baseline, gradient boosting with
quantile heads (fixed settings), gradient boosting nested-tuned with split-conformal intervals.

Writes results/synthetic_benchmark/{benchmark_raw.csv, benchmark_summary.csv, benchmark.md}. Every finished
cell is saved at once; rerunning the same command resumes and only fills in what is missing.
SYNTHETIC data: this demonstrates the mechanism; it is not a study result.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from copilot import calibration as C  # noqa: E402
from copilot.engine import BACKENDS, TARGETS, load_context  # noqa: E402

OUT = ROOT / "results" / "synthetic_benchmark"
COMPARATORS = ["physical-baseline", "gbm-quantile", "gbm-tuned-conformal"]
NAMES = {"tabpfn-3.5-api": "TabPFN-3.5", "tabpfn-3.5-local": "TabPFN-3.5", **{b: BACKENDS[b] for b in COMPARATORS}}


KEY = ["target", "repeat", "backend"]


def run_cell(target: str, d, b: str, rep: int, decisive: float) -> dict:
    t0 = time.time()
    Q, y, dd = C.oof_frame(target, d, b, k=5, seed=rep)
    r = C.report(Q, y)
    dc = C.decision_curve(Q, dd, TARGETS[target][1], cutoffs=[decisive])
    return {"target": target, "repeat": rep, "model": NAMES[b], "backend": b, "n": r["n"],
            "cov50": r["cov50"], "cov80": r["cov80"], "cov95": r["cov95"],
            "width95_fold": r["width95_fold"], "crps_log": r["crps_log"], "log_r2": r["log_r2"],
            "decisive_frac": dc.frac_decisive.iloc[0] if len(dc) else float("nan"),
            "decisive_acc": dc.acc_decisive.iloc[0] if len(dc) else float("nan"),
            "seconds": round(time.time() - t0, 1)}


def write_report(raw: pd.DataFrame, a) -> None:
    cols = ["cov50", "cov80", "cov95", "width95_fold", "crps_log", "log_r2", "decisive_frac", "decisive_acc"]
    order = {t: i for i, t in enumerate(TARGETS)}
    summ = (raw.groupby(["target", "model"], sort=False)[cols].mean()
            .join(raw.groupby(["target", "model"], sort=False).size().rename("n_repeats")).reset_index())
    summ = summ.sort_values("target", key=lambda s: s.map(order), kind="stable")
    summ.to_csv(OUT / "benchmark_summary.csv", index=False)
    best = summ.loc[summ.groupby("target").crps_log.idxmin(), ["target", "model"]]
    best = set(map(tuple, best.to_numpy()))
    lines = ["# Synthetic benchmark - live calibration check on every target",
             "",
             f"Generated {dt.date.today().isoformat()} by `scripts/benchmark_synthetic.py --backend {a.backend}`. "
             "5-fold CV grouped by synthetic patient, identical folds for every model; mean over the repeats "
             f"(seeds 0..{a.repeats - 1}). Decisive: P ≥ {a.decisive:.2f} or ≤ {1 - a.decisive:.2f}, pooled over the "
             "target's clinical thresholds. **Bold: lowest CRPS for the target.** **Synthetic data: this demonstrates "
             "the mechanism; it is not a study result.** The synthetic table is close to log-linear, which favours "
             "the 2-term physical model; `tumors` is lesion level, where rows of one patient are correlated.",
             "",
             "| target | model | repeats | cover 80% | cover 95% | 95% span (×) | CRPS (log) | log R² | decisive | right |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        dec = "–" if pd.isna(r.decisive_frac) else f"{r.decisive_frac:.0%}"
        acc = "–" if pd.isna(r.decisive_acc) else f"{r.decisive_acc:.0%}"
        crps = f"**{r.crps_log:.3f}**" if (r.target, r.model) in best else f"{r.crps_log:.3f}"
        lines.append(f"| {r.target} | {r.model} | {r.n_repeats} | {r.cov80:.2f} | {r.cov95:.2f} | "
                     f"{r.width95_fold:.1f} | {crps} | {r.log_r2:.3f} | {dec} | {acc} |")
    (OUT / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["api", "local", "none"], default="api")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--decisive", type=float, default=0.90)
    ap.add_argument("--fresh", action="store_true", help="ignore results already saved and start over")
    ap.add_argument("--passes", type=int, default=3, help="passes over failed cells within one run")
    ap.add_argument("--pause", type=float, default=60.0, help="seconds to wait before another pass")
    a = ap.parse_args()

    backends = ([] if a.backend == "none" else [f"tabpfn-3.5-{a.backend}"]) + COMPARATORS
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / "benchmark_raw.csv"
    raw = pd.DataFrame() if (a.fresh or not raw_path.exists()) else pd.read_csv(raw_path)
    done = set() if raw.empty else set(map(tuple, raw[KEY].astype({"repeat": int}).to_numpy().tolist()))
    if done:
        print(f"resuming: {len(done)} cell(s) already in {raw_path.name} are skipped (use --fresh to redo)")
    cells = [(t, rep, b) for t in a.targets for rep in range(a.repeats) for b in backends]
    failed: list = []
    for attempt in range(a.passes):
        todo = [c for c in cells if c not in done]
        if not todo:
            break
        if attempt:
            print(f"\npass {attempt + 1}: retrying {len(todo)} failed cell(s) after a {a.pause:.0f}s pause", flush=True)
            time.sleep(a.pause)
        failed = []
        for target, rep, b in todo:
            try:
                row = run_cell(target, load_context(target), b, rep, a.decisive)
            except Exception as e:                       # e.g. the API failed after its retries
                failed.append((target, rep, b))
                print(f"{target:13s} rep {rep}  {NAMES[b]:38s} FAILED ({type(e).__name__}: {str(e)[:90]})",
                      flush=True)
                continue
            done.add((target, rep, b))
            raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
            raw.to_csv(raw_path, index=False)              # saved after every cell: a crash loses nothing
            print(f"{target:13s} rep {rep}  {NAMES[b]:38s} cov80 {row['cov80']:.2f} cov95 {row['cov95']:.2f} "
                  f"x{row['width95_fold']:.1f} CRPS {row['crps_log']:.3f} R2 {row['log_r2']:.3f} "
                  f"({row['seconds']:.0f}s)", flush=True)
    if raw.empty:
        print("nothing to summarise")
        return
    write_report(raw, a)
    print(f"\nwrote {OUT / 'benchmark.md'}" + (f"  - INCOMPLETE: {len(failed)} cell(s) still missing; rerun the same "
                                              "command later to fill them" if failed else "  - COMPLETE"))


if __name__ == "__main__":
    main()
