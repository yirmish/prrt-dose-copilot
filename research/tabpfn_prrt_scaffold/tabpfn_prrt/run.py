"""Driver: 7 targets x model sizes x arms, on identical folds.

    python -m tabpfn_prrt.run --targets tumor_burden --arms M0,TabPFN-V3
    python -m tabpfn_prrt.run --all
"""
from __future__ import annotations

import argparse
import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import metrics as M
from .cv import Split, era_of, leave_one_out_splits, repeated_grouped_splits
from .ids import base_id
from .models import (CatBoostArm, LinearBaseline, RFAkhavanallaf, TabPFNArm,
                     TopKFilterArm)
from .targets import TargetTransform, build_target

LEVELS = np.asarray(C.QUANTILE_GRID, float)


def _tabpfn_provenance() -> dict[str, str]:
    """Version provenance. "We used TabPFN" is not a reproducible statement (plan 13)."""
    out: dict[str, str] = {}
    try:
        import hashlib
        import tabpfn
        import torch as _torch
        out["tabpfn"] = tabpfn.__version__
        out["torch"] = _torch.__version__
        ckpt = Path(C.os.environ.get("TABPFN_MODEL_CACHE_DIR", "")) / "tabpfn-v2-regressor.ckpt"
        if ckpt.exists():
            h = hashlib.sha256()
            with open(ckpt, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out["v2_checkpoint_sha256"] = h.hexdigest()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def make_arm(name: str, target: C.TargetSpec, seed: int = 0):
    """Arm names: "M0" | "CatBoost" | "RF-Akhava" | "TabPFN-<version>[@<n_estimators>]".

    The @N suffix is the LOCAL, free, in-protocol analogue of paying for extra test-time
    compute: each extra estimator is another preprocessing permutation through the same
    weights. Sweep it (TabPFN-V3@4,TabPFN-V3@16,TabPFN-V3@64) before considering the
    API-only thinking mode -- if more compute does not move CRPS here, it will not move
    it there either.
    """
    if name == "M0":
        return LinearBaseline(target_name=target.name)
    if name.startswith("TabPFN"):
        spec = name.split("-", 1)[1]
        version, _, n_est = spec.partition("@")
        return TabPFNArm(version=version, random_state=seed,
                         n_estimators=int(n_est) if n_est else C.TabPFNSpec().n_estimators,
                         device=C.TabPFNSpec().device)
    if name.startswith("TopK"):
        # EXPLORATORY, NOT IN THE FROZEN PLAN (declared, DEC-225). In-fold univariate
        # filter: keep the K features with the largest |Spearman| against the training
        # fold's y, then fit the inner arm on those alone. Fitted inside the training
        # fold only, so it is honest -- but it is a researcher degree of freedom the plan
        # does not contain, and every run using it must be tagged is_protocol_run: false.
        # Purpose: test whether the scaffold's weak spleen result (TabPFN 0.197 on 165
        # unselected features) is a dimensionality artefact, given that the project's own
        # V9 pipeline reaches 0.383 on the same 72 rows with K ~ 3-5 selected in-fold.
        k_str, _, inner_name = name[4:].partition("-")
        return TopKFilterArm(inner_factory=lambda: make_arm(inner_name, target, seed=seed),
                             k=int(k_str), name=name)
    if name == "CatBoost":
        return CatBoostArm(random_state=seed)
    if name == "RF-Akhava":
        return RFAkhavanallaf(random_state=seed)
    raise ValueError(name)


# CatBoost fits one quantile head per level, and CatBoostArm.predict_quantiles maps them
# onto the 1-99% reporting grid with np.interp -- which CLAMPS outside the fitted range.
# With the original seven levels (0.05 ... 0.95) the reported "95%" interval, read at
# q0.025 / q0.975, was silently the fitted q0.05 / q0.95 interval: a NOMINAL 90% interval
# reported and compared against TabPFN's true 95% one, and a CRPS integrated over flat,
# artificial tails. Verified by executing this code path on a standard normal:
#   reported 95% interval [-1.640, +1.640]  vs  true [-1.960, +1.960].
# cov_50 and cov_80 were unaffected -- 0.25/0.75 and 0.10/0.90 were fitted levels.
# The fitted set now spans the reporting grid's endpoints. 11 heads per fold, not 7.
# Fixed 2026-09-20 (DEC-221).
CATBOOST_LEVELS = np.array([0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99])


def _fit_predict(arm, Xtr, ytr, Xte) -> np.ndarray:
    if isinstance(arm, CatBoostArm):
        arm.fit(Xtr, ytr, levels=CATBOOST_LEVELS)
    else:
        arm.fit(Xtr, ytr)
    return arm.predict_quantiles(Xte, LEVELS)


def evaluate(df: pd.DataFrame, X: pd.DataFrame, target: C.TargetSpec,
             arm_name: str, cfg: C.RunConfig, splits: list[Split],
             seed: int = 0) -> tuple[pd.DataFrame, dict]:
    tf = TargetTransform(normalise_by_activity=cfg.normalise_by_activity, log=cfg.log_target)
    tcol = cfg.target_col_override or target.target_col
    y_model, dose_gy, activity = build_target(df, tcol, C.ACTIVITY_COL, tf)
    groups = df[C.V8_JOIN_KEY].map(base_id).to_numpy()

    n_rep = max(s.repeat for s in splits) + 1
    # float32: the quantile store is (repeats x rows x 99) and is only ever used for
    # metric computation, where float32 is far below the noise floor of the estimates.
    q_store = np.full((n_rep, len(df), len(LEVELS)), np.nan, dtype=np.float32)

    for sp in splits:
        arm = make_arm(arm_name, target, seed=seed)
        q = _fit_predict(arm, X.iloc[sp.train], y_model[sp.train], X.iloc[sp.test])
        q_store[sp.repeat, sp.test, :] = q
        # Each fold builds a FRESH TabPFN instance holding its own copy of the
        # checkpoint's weights. Without an explicit release the per-fold models
        # accumulate across the 100 fits of a repeated-CV run and exhaust memory --
        # observed as a hard kernel termination ~45 min into a 4-arm sweep on a machine
        # with ~2.4 GB free. Dropping the reference and collecting bounds the run at one
        # live model.
        if hasattr(arm, "_model"):
            arm._model = None
        del arm
        gc.collect()

    rows = []
    for r in range(n_rep):
        ok = ~np.isnan(q_store[r, :, 0])
        if ok.sum() < 10:
            continue
        q_gy = tf.inverse_quantiles(q_store[r][ok], activity[ok])
        point_gy = M._quantile_at(LEVELS, q_gy, 0.5)         # median: log-safe by construction
        rec = {"target": target.name, "arm": arm_name, "repeat": r, "seed": seed,
               "kind": splits[0].kind, **M.error_metrics(dose_gy[ok], point_gy),
               **M.distribution_metrics(dose_gy[ok], LEVELS, q_gy,
                                        point=point_gy, nominals=C.NOMINAL_COVERAGES)}
        # log-scale error, for a stable secondary view of a heavy-tailed target
        med_log = M._quantile_at(LEVELS, q_store[r][ok], 0.5)
        rec.update(M.error_metrics(y_model[ok], med_log, prefix="log_"))

        # MODEL-SCALE distribution metrics. CRPS on the Gy scale is not usable on a
        # lesion-level target: it integrates the full 1-99% grid, and the back-transform
        # exp(q)*A turns the outermost bar of TabPFN's predictive distribution into an
        # astronomical dose. Measured on `tumors`, Gy-scale CRPS was corrupted in 16 of
        # 20 repeats, ranging 15 to 1.9e23, while R2, coverage and the 95% width -- which
        # never touch a quantile beyond 97.5% -- were unaffected. Taking a median over
        # repeats does not rescue it; the median itself was 2.8e10.
        #
        # CRPS in log(Gy/GBq) is a proper scoring rule for the distribution the model
        # actually fits, needs no back-transform, and is immune to this. It is the value
        # to report for heavy-tailed targets; the Gy-scale one stays for comparability
        # with the treatment-level targets, where it is well behaved.
        dm_log = M.distribution_metrics(y_model[ok], LEVELS, q_store[r][ok],
                                        point=med_log, nominals=C.NOMINAL_COVERAGES)
        rec.update({f"log_{k}": v for k, v in dm_log.items()})
        rec["crps_gy_finite"] = bool(np.isfinite(rec.get("crps", np.nan))
                                     and abs(rec.get("crps", np.inf)) < 1e6)
        if target.unit == "lesion":
            rec.update(M.between_within_r2(dose_gy[ok], point_gy, groups[ok]))
        for thr in target.thresholds_gy:
            tm = M.threshold_metrics(dose_gy[ok], LEVELS, q_gy, thr)
            rec.update({f"thr{thr:g}_{k}": v for k, v in tm.items() if k != "thr"})
            rec.update({f"thr{thr:g}_{k}": v for k, v in
                        M.management_change(M.prob_exceeds(LEVELS, q_gy, thr)).items()})
        rows.append(rec)

    oof = {"q_store": q_store, "dose_gy": dose_gy, "activity": activity,
           "groups": groups, "y_model": y_model}
    return pd.DataFrame(rows), oof


def run_target(target: C.TargetSpec, cfg: C.RunConfig, arms: tuple[str, ...],
               out_dir: Path) -> pd.DataFrame:
    df = D.assemble(target)
    print(f"\n=== {target.name}: {df.attrs['cohort']}")
    all_rows, oof_by_arm = [], {}

    for size in cfg.model_sizes:
        blocks = C.MODEL_SIZES[size]
        if size == "M0":
            X = D.build_features(df, ("A", "B", "C", "D", "E"), target,
                                 include_grey_zone=cfg.include_grey_zone)
        else:
            X = D.build_features(df, blocks, target, include_grey_zone=cfg.include_grey_zone)

        D.leakage.manifest(X, target.name, size).to_csv(
            out_dir / f"manifest_{target.name}_{size}.csv", index=False)

        y_for_split, _, _ = build_target(
            df, cfg.target_col_override or target.target_col, C.ACTIVITY_COL,
            TargetTransform(cfg.normalise_by_activity, cfg.log_target))
        groups = df[C.V8_JOIN_KEY].map(base_id).to_numpy()
        splits = list(repeated_grouped_splits(y_for_split, groups,
                                              n_repeats=cfg.n_repeats, n_folds=C.N_FOLDS,
                                              seeds=C.CV_SEEDS[:cfg.n_repeats]))

        for arm_name in (("M0",) if size == "M0" else arms):
            if size != "M0" and arm_name == "M0":
                continue
            try:
                res, oof = evaluate(df, X, target, arm_name, cfg, splits)
            except Exception as exc:                      # one arm failing must not kill the run
                warnings.warn(f"{target.name}/{size}/{arm_name}: {type(exc).__name__}: {exc}")
                continue
            res.insert(2, "model_size", size)
            all_rows.append(res)
            oof_by_arm[(size, arm_name)] = oof
            # Persist the out-of-fold quantile store. Without it the PIT histograms and
            # reliability plots the plan calls the study's contribution (9.2) are not
            # recoverable, and neither is an eval-only leave-one-patient-out jackknife of
            # any comparison. Sizes are small: 20 x rows x 99 float32.
            # Added 2026-09-20 (DEC-223).
            np.savez_compressed(
                out_dir / f"oof_{target.name}_{size}_{arm_name.replace('/', '_')}.npz",
                q_store=oof["q_store"], dose_gy=oof["dose_gy"], activity=oof["activity"],
                groups=oof["groups"], y_model=oof["y_model"], levels=LEVELS)
            print(f"  {size:3s} {arm_name:12s} "
                  f"R2={res.r2.median():+.3f} [{res.r2.quantile(.05):+.3f},{res.r2.quantile(.95):+.3f}]  "
                  f"cov95={res.cov_95.median():.2f}  calib={res.calib_slope.median():.2f}")

        # pseudo-external validation: the closest analogue to leave-one-centre-out.
        #
        # Each held-out stratum is evaluated in its OWN evaluate() call. Passing both
        # directions in one list silently pooled them: every leave-one-out Split carries
        # repeat=0, so evaluate() allocated a single repeat row and both directions wrote
        # into disjoint test positions of the same slice, yielding one merged metric.
        # The plan (8.3) asks for both directions, in both directions -- train-early /
        # test-late is a different question from train-late / test-early.
        if size == "M3":
            era_labels = (df["protocol_era"] if "protocol_era" in df.columns
                          else pd.Series([pd.NA] * len(df), dtype="string"))
            for label_name, labels in (("era", era_labels),
                                       ("scanner", df.get(C.SCANNER_COL, pd.Series(dtype=object)).astype("string"))):
                # M0 belongs in the pseudo-external comparison. It was never in `arms`,
                # so every era / scanner table in Phases 2-5 reports a model collapsing
                # out of distribution with NO comparator -- and the study's own thesis is
                # that two features do as well as anything. Whether the baseline collapses
                # too is the informative half of that analysis.
                # Fixed 2026-09-20 (DEC-222).
                ext_arms = ("M0",) + tuple(a for a in arms if a != "M0")
                for sp in leave_one_out_splits(labels, groups, label_name):
                    for arm_name in ext_arms:
                        try:
                            res, _ = evaluate(df, X, target, arm_name, cfg, [sp])
                            res.insert(2, "model_size", f"{size}/{sp.kind}")
                            res["n_train"] = len(sp.train)
                            res["n_test"] = len(sp.test)
                            all_rows.append(res)
                        except Exception as exc:
                            warnings.warn(f"{label_name} split {sp.kind} failed: {exc}")

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    # paired comparisons against M0, bootstrapped over PATIENTS
    comps = []
    base_key = ("M0", "M0")
    if base_key in oof_by_arm:
        b = oof_by_arm[base_key]
        for (size, arm_name), a in oof_by_arm.items():
            if (size, arm_name) == base_key:
                continue
            pa = np.nanmedian(a["q_store"][:, :, len(LEVELS) // 2], axis=0)
            pb = np.nanmedian(b["q_store"][:, :, len(LEVELS) // 2], axis=0)
            ok = np.isfinite(pa) & np.isfinite(pb)
            d = M.paired_bootstrap_delta(a["y_model"][ok], pa[ok], pb[ok], a["groups"][ok],
                                         n_boot=C.N_BOOTSTRAP)
            margin = C.margin_for(target.unit)
            d.update({"target": target.name, "model_size": size, "arm": arm_name,
                      "margin": margin,
                      "beats_baseline": bool(d["ci_lo"] > 0 and d["delta"] >= margin)})
            comps.append(d)
    if comps:
        pd.DataFrame(comps).to_csv(out_dir / f"comparisons_{target.name}.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="tumor_burden")
    ap.add_argument("--arms", default="TabPFN-V3,CatBoost,RF-Akhava")
    ap.add_argument("--sizes", default="M0,M1,M2,M3")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--rc-target", action="store_true",
                    help="sensitivity analysis: dos_dose_rc_corr_gy instead of dos_dose_gy")
    ap.add_argument("--grey-zone", action="store_true",
                    help="include gen_pet_lesionid_threshold_suv")
    # The plan models Gy/GBq and reports Gy (3.2): predicting Gy directly spends R^2 on
    # re-learning the administered activity, which is known before therapy. The PROJECT's
    # V9 ladder does NOT normalise -- its `r2_row_log` is on log(Gy). That is one of the
    # four remaining differences between the two pipelines on the spleen target, and it
    # is the only one that can be isolated with a single flag. Added 2026-09-20 to make
    # the two comparable on the same target definition, NOT to change the plan's endpoint.
    ap.add_argument("--no-activity-norm", action="store_true",
                    help="model log(Gy) instead of log(Gy/GBq) -- matches the V9 target "
                         "definition; an addition for cross-pipeline comparison only")
    ap.add_argument("--repeats", type=int, default=C.N_REPEATS,
                    help="CV repeats. Leave at the protocol value for any reported "
                         "result; lower it only for configuration-selection runs.")
    a = ap.parse_args()

    names = [t.name for t in C.TARGETS] if a.all else a.targets.split(",")
    cfg = C.RunConfig(model_sizes=tuple(a.sizes.split(",")),
                      include_grey_zone=a.grey_zone,
                      normalise_by_activity=not a.no_activity_norm,
                      target_col_override="dos_dose_rc_corr_gy" if a.rc_target else None,
                      tag=a.tag, n_repeats=a.repeats)
    arms = tuple(a.arms.split(","))

    out_dir = C.RESULTS_ROOT / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(
        {"targets": names, "arms": arms, "sizes": cfg.model_sizes,
         "n_repeats": cfg.n_repeats, "n_repeats_protocol": C.N_REPEATS,
         "is_protocol_run": cfg.n_repeats == C.N_REPEATS,
         "n_folds": C.N_FOLDS, "margin_r2": C.SUPERIORITY_MARGIN_R2,
         "rc_target": bool(a.rc_target), "grey_zone": bool(a.grey_zone),
         "normalise_by_activity": cfg.normalise_by_activity,
         "baseline_kind": C.os.environ.get("PRRT_BASELINE", "plan"),
         "tabpfn_version": _tabpfn_provenance()}, indent=2))

    frames = []
    for name in names:
        spec = next(t for t in C.TARGETS if t.name == name)
        frames.append(run_target(spec, cfg, arms, out_dir))
    res = pd.concat([f for f in frames if len(f)], ignore_index=True)
    res.to_csv(out_dir / "results_raw.csv", index=False)

    summary = (res.groupby(["target", "model_size", "arm"])
                  .agg(r2_med=("r2", "median"),
                       r2_p05=("r2", lambda s: s.quantile(.05)),
                       r2_p95=("r2", lambda s: s.quantile(.95)),
                       mae=("mae", "median"), mrae=("mrae", "median"),
                       crps=("crps", "median"),
                       calib_slope=("calib_slope", "median"),
                       cov95=("cov_95", "median"),
                       width95=("width_95_median", "median"))
                  .reset_index())
    summary.to_csv(out_dir / "results_summary.csv", index=False)
    print(f"\nwrote {out_dir}/results_summary.csv")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
