#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_v35_scaffold.py -- TabPFN-3.5 through the EXISTING tabpfn_prrt scaffold. 2026-10-06.

PURPOSE (gap being closed)
    The scaffold (tabpfn_prrt_scaffold, 2026-09-14..23) measured predictive DISTRIBUTIONS
    (quantiles, coverage, CRPS, PIT) only for TabPFN-V2@8 (+ one TabPFN-V3 cell) and CatBoost.
    The V9 ladder measured TabPFN-3.5 only as a POINT predictor, on spleen and kidneys (DEC-215/216).
    Nobody has measured TabPFN-3.5's predictive distribution on this cohort. The hackathon entry
    ("knows when to measure") rests on exactly that. This script measures it, changing nothing else.

DECLARATION
  ASSUMPTIONS
    A1 The scaffold package is used UNMODIFIED (imported from tabpfn_prrt_scaffold/); data assembly,
       leakage assertions, target (log Gy/GBq), folds (StratifiedGroupKFold on base_id) are the scaffold's.
    A2 Input = Deidentified_Export_2026-09-07 (read-only), as in every scaffold run since 09-20.
    A3 tabpfn 9.0.0 exposes ModelVersion.V3_5 and accepts model_path=<.safetensors> (probed 09-16).
    A4 The 3.5 weights are the file whose SHA-256 was recorded on 09-16 (re-hashed and written here).
  SENSITIVE OPERATIONS
    S1 Monkeypatching run.make_arm IN THIS PROCESS ONLY to add the arm name "TabPFN-V3_5@N".
       No scaffold file is edited. Every other arm name goes to the original function.
    S2 n_repeats = 5, NOT the protocol's 20 (time: hackathon deadline is today). Every output is
       written with is_protocol_run = false. Seeds 0..4 are the first five protocol repeats, so the
       folds are a strict subset of the protocol's and are paired with the stored 20-repeat OOF files.
    S3 Thread count is NOT pinned to 1 (single process; speed). Recorded in run_config.json.
  UNCERTAINTIES (flagged, not resolved)
    U1 5 repeats: the repeat-to-repeat spread is under-sampled; read medians against the scaffold's
       pre-registered margins (treatment 0.15, lesion 0.07/0.09), never as a ranking.
    U2 Pre-declared settings, not tuned: n_estimators=8, seed=0, softmax_temperature=0.9 (scaffold
       default), device=cpu, float32. Same as the scaffold's "@8" arms.
  NEVER tuned toward a higher R^2. A poor cell is a result and is reported.

OUTPUTS (new files only), under outputs/<RUN_TAG>/
    <size>/<target>/ : results_raw.csv, comparisons_<target>.csv, manifest_*.csv, oof_*.npz  (scaffold format)
    folds__<target>__<size>.csv   per-fold record (DEC-204): split, row ids, feature NAMES, settings, seeds
    run_config.json, summary_stats.txt, flagged_rows.csv, results_raw_all.csv, DONE marker
"""
from __future__ import annotations
import os, sys, json, time, hashlib, traceback, warnings, platform
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
ROOT = WORK.parents[3]                      # the project root
SCAF = ROOT / "tabpfn_prrt_scaffold"
RUN_TAG = os.environ.get("HK_RUN_TAG", "v35_dist_20261006")
OUT = WORK / "outputs" / RUN_TAG
OUT.mkdir(parents=True, exist_ok=True)

# ---- environment: must be set BEFORE the scaffold / tabpfn are imported ----------------------
for k in ("TABPFN", "TABPFN_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE"):
    os.environ.pop(k, None)                  # no token visible -> no licence round-trip, no egress
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["TABPFN_NO_BROWSER"] = "1"
os.environ["PRRT_DATA_ROOT"] = str(ROOT / "Deidentified_Export_2026-09-07")
os.environ["PRRT_RESULTS_ROOT"] = str(OUT)
NTHREADS = os.environ.get("HK_THREADS", "8")
for v in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = NTHREADS
os.chdir(SCAF)                               # the scaffold resolves .env relative to cwd / package
sys.path.insert(0, str(SCAF))

import numpy as np
import pandas as pd

WEIGHTS = Path(os.environ.get("HK_WEIGHTS",
               str(Path.home() / "AppData" / "Roaming" / "tabpfn" / "tabpfn-v3.5-20260909.safetensors")))
SHA_RECORDED = "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"   # 2026-09-16


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "run.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    if not WEIGHTS.is_file():
        log(f"FATAL weights missing: {WEIGHTS}")
        return 2
    sha = sha256(WEIGHTS)
    if sha != SHA_RECORDED:
        log(f"FATAL weights SHA mismatch: {sha} != recorded {SHA_RECORDED}")
        return 2

    from tabpfn_prrt import config as C, run as R, data as D
    from tabpfn_prrt.models import TabPFNArm
    from tabpfn_prrt.cv import repeated_grouped_splits
    from tabpfn_prrt.ids import base_id
    from tabpfn_prrt.targets import TargetTransform, build_target
    import tabpfn, torch, sklearn
    try:
        from tabpfn.model_loading import ModelVersion
    except Exception:
        from tabpfn.constants import ModelVersion
    torch.set_num_threads(int(NTHREADS))

    class TabPFN35Arm(TabPFNArm):
        """TabPFN-3.5, default checkpoint, weights path pinned (avoids the per-fit filelock)."""
        def _build(self, X):
            from tabpfn import TabPFNRegressor
            return TabPFNRegressor.create_default_for_version(
                ModelVersion.V3_5,
                n_estimators=self.n_estimators,
                softmax_temperature=self.softmax_temperature,
                device=self.device,
                random_state=self.random_state,
                inference_precision=torch.float32,
                categorical_features_indices=self._cat_idx,
                model_path=str(WEIGHTS))

    _orig_make_arm = R.make_arm

    def make_arm(name, target, seed=0):
        if name.startswith("TabPFN-V3_5"):
            _, _, n_est = name.partition("@")
            return TabPFN35Arm(version="V3_5", random_state=seed, name=name,
                               n_estimators=int(n_est) if n_est else 8, device="cpu")
        return _orig_make_arm(name, target, seed=seed)
    R.make_arm = make_arm

    ARM = os.environ.get("HK_ARM", "TabPFN-V3_5@8")
    REPEATS = int(os.environ.get("HK_REPEATS", "5"))
    # queue: fast, small-feature cells first; every cell of the scaffold's headline table is covered.
    default_queue = ("M1:tumor_burden,kidneys,liver_healthy,spleen,bone_marrow,tumors_hepatic,tumors;"
                     "M3:tumor_burden,kidneys,liver_healthy,spleen,bone_marrow,tumors_hepatic")
    queue = []
    for part in os.environ.get("HK_QUEUE", default_queue).split(";"):
        size, _, names = part.partition(":")
        queue += [(size.strip(), n.strip()) for n in names.split(",") if n.strip()]
    # baseline shape: the project's own organ baseline (DEC-224) for organs, the plan's for tumours
    BASE = {"kidneys": "organ3p", "liver_healthy": "organ3p", "spleen": "organ3p",
            "bone_marrow": "organ3p"}

    cfg_dump = {
        "run_tag": RUN_TAG, "purpose": "TabPFN-3.5 predictive distribution through the scaffold",
        "arm": ARM, "n_repeats": REPEATS, "n_repeats_protocol": C.N_REPEATS,
        "is_protocol_run": False, "n_folds": C.N_FOLDS, "cv_seeds": list(C.CV_SEEDS[:REPEATS]),
        "queue": queue, "baseline_kind_by_target": {n: BASE.get(n, "plan") for _, n in queue},
        "target": "log(Gy/GBq), scaffold default (normalise_by_activity=True, log=True)",
        "tabpfn": tabpfn.__version__, "model_version": "V3_5", "weights": WEIGHTS.name,
        "weights_sha256": sha, "torch": torch.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "python": platform.python_version(), "threads": NTHREADS,
        "offline": True, "token_visible": False, "data_root": "Deidentified_Export_2026-09-07",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUT / "run_config.json").write_text(json.dumps(cfg_dump, indent=2), encoding="utf-8")
    log(f"START arm={ARM} repeats={REPEATS} threads={NTHREADS} tabpfn={tabpfn.__version__} "
        f"torch={torch.__version__} weights_sha_ok=True units={len(queue)}")

    flagged, stats = [], []
    for size, name in queue:
        unit_dir = OUT / size / name
        if (unit_dir / "UNIT_DONE").exists():
            log(f"skip (done): {size}/{name}")
            continue
        unit_dir.mkdir(parents=True, exist_ok=True)
        os.environ["PRRT_BASELINE"] = BASE.get(name, "plan")
        spec = next(t for t in C.TARGETS if t.name == name)
        cfg = C.RunConfig(model_sizes=("M0", size), tag=RUN_TAG, n_repeats=REPEATS)
        t0 = time.time()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = R.run_target(spec, cfg, (ARM,), unit_dir)
            arm_fail = [str(w.message) for w in caught if f"/{ARM}:" in str(w.message)]
            for m in arm_fail:
                flagged.append({"unit": f"{size}/{name}", "reason": "arm_failed", "detail": m[:400]})
                log(f"ARM FAILED {size}/{name}: {m[:300]}")
            res.insert(0, "baseline_kind", BASE.get(name, "plan"))
            res.insert(0, "is_protocol_run", False)
            res.to_csv(unit_dir / "results_raw.csv", index=False)

            # ---- DEC-204 per-fold record (features are a FIXED list here: no in-fold selection) ----
            df = D.assemble(spec)
            X = D.build_features(df, C.MODEL_SIZES[size], spec, include_grey_zone=False)
            tf = TargetTransform(normalise_by_activity=True, log=True)
            y_model, dose_gy, activity = build_target(df, spec.target_col, C.ACTIVITY_COL, tf)
            groups = df[C.V8_JOIN_KEY].map(base_id).to_numpy()
            npz = unit_dir / f"oof_{name}_{size}_{ARM}.npz"
            q50 = None
            if npz.exists():
                z = np.load(npz)
                j = int(np.argmin(np.abs(z["levels"] - 0.5)))
                q50 = z["q_store"][:, :, j]
            rows = []
            for sp in repeated_grouped_splits(y_model, groups, n_repeats=REPEATS,
                                              n_folds=C.N_FOLDS, seeds=C.CV_SEEDS[:REPEATS]):
                rec = {"target": name, "model_size": size, "arm": ARM, "repeat": sp.repeat,
                       "fold": sp.fold, "cv_seed": C.CV_SEEDS[sp.repeat], "model_seed": 0,
                       "n_train": len(sp.train), "n_test": len(sp.test),
                       "n_groups_test": int(len(set(groups[sp.test]))),
                       "test_row_idx": "|".join(map(str, sp.test.tolist())),
                       "test_groups_base_id": "|".join(sorted({str(g) for g in groups[sp.test]})),
                       "features_offered_n": X.shape[1], "features_selected_n": X.shape[1],
                       "features_selected": "|".join(map(str, X.columns)),
                       "in_fold_selection": "none (fixed block list)",
                       "n_estimators": 8, "softmax_temperature": 0.9, "target_transform": "log(Gy/GBq)"}
                if q50 is not None:
                    e = y_model[sp.test] - q50[sp.repeat, sp.test]
                    rec["fold_mae_log"] = float(np.nanmean(np.abs(e)))
                    rec["fold_rmse_log"] = float(np.sqrt(np.nanmean(e ** 2)))
                rows.append(rec)
            pd.DataFrame(rows).to_csv(OUT / f"folds__{name}__{size}.csv", index=False)

            ok_arm = (res["arm"] == ARM).any() and (res["model_size"] == size).any()
            if not ok_arm:
                flagged.append({"unit": f"{size}/{name}", "reason": "no_rows_for_arm", "detail": ""})
            stats.append({"unit": f"{size}/{name}", "rows_in": len(df), "rows_out": len(df),
                          "rows_rejected": 0, "features": X.shape[1],
                          "n_groups": int(len(set(groups))), "minutes": round((time.time() - t0) / 60, 1),
                          "arm_rows": int(((res["arm"] == ARM) & (res["model_size"] == size)).sum()),
                          "cohort": str(df.attrs.get("cohort", ""))})
            if ok_arm:
                (unit_dir / "UNIT_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            a = res[(res["arm"] == ARM) & (res["model_size"] == size)]
            m0 = res[res["model_size"] == "M0"]
            if len(a):
                log(f"DONE {size}/{name}  n={len(df)} p={X.shape[1]}  {stats[-1]['minutes']} min  "
                    f"log_r2 arm={a['log_r2'].median():+.3f} M0={m0['log_r2'].median():+.3f}  "
                    f"cov95={a['cov_95'].median():.3f} cov80={a['cov_80'].median():.3f} "
                    f"cov50={a['cov_50'].median():.3f}")
        except Exception as exc:
            flagged.append({"unit": f"{size}/{name}", "reason": type(exc).__name__,
                            "detail": traceback.format_exc()[-1500:]})
            log(f"UNIT FAILED {size}/{name}: {type(exc).__name__}: {exc}")

        # rewrite the QC files after EVERY unit, so an interruption leaves a truthful record
        frames = [pd.read_csv(p) for p in sorted(OUT.glob("M*/*/results_raw.csv"))]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(OUT / "results_raw_all.csv", index=False)
        pd.DataFrame(flagged, columns=["unit", "reason", "detail"]).to_csv(
            OUT / "flagged_rows.csv", index=False)
        with open(OUT / "summary_stats.txt", "w", encoding="utf-8") as fh:
            fh.write(f"run_tag {RUN_TAG}  arm {ARM}  repeats {REPEATS} (protocol 20) -> is_protocol_run=False\n")
            fh.write(f"units requested {len(queue)}  completed {len(stats)}  flagged {len(flagged)}\n")
            fh.write("rows in == rows out for every unit: this script filters nothing; the scaffold's\n"
                     "own row rules (data.assemble) are the only ones applied and are printed per unit.\n\n")
            fh.write(pd.DataFrame(stats).to_string(index=False) if stats else "no unit completed")
            fh.write("\n")
    (OUT / "QUEUE_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("QUEUE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
