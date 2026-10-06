"""engine.py - the prediction engine of PRRT Dose Copilot (no UI code; testable on its own).

Context table -> one CSV per target (the SYNTHETIC demonstration cohort in data/synthetic/)
Model         -> TabPFN-3.5 regressor on log(Gy/GBq); quantiles out, never a back-transformed mean
Output        -> dose distribution in Gy for a planned activity, P(dose > threshold), decisive / measure

Three interchangeable back-ends, one interface (`fit`, `log_quantiles`):
  * "tabpfn-3.5-api"   TabPFN-3.5 through the Prior Labs API (tabpfn-client); needs TABPFN_TOKEN.
                       Used by the hosted demo - it only ever sees the synthetic cohort.
  * "tabpfn-3.5-local" TabPFN-3.5 with local weights (tabpfn package); offline.
  * "physical-baseline" 2-3 term log-log regression with a normal residual - the comparator.

Research prototype. Not a medical device. Not for clinical use.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ACTIVITY = "gen_prrt_net_injected_activity_gbq"
GROUP = "synthetic_patient_id"
LEVELS = np.round(np.arange(0.01, 1.0, 0.01), 2)               # 1% .. 99%
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"

TARGETS = {
    # name: (csv, clinical thresholds in Gy defined by the study, unit of a row, display label)
    "tumor_burden":  ("tumor_burden.csv",  (20.0, 30.0, 36.5, 40.0), "treatment", "Tumour burden (all lesions)"),
    "tumors":        ("tumors.csv",        (20.0, 30.0, 36.2, 40.0), "lesion",    "Single lesion"),
    "kidneys":       ("kidneys.csv",       (5.75,),                  "treatment", "Kidneys"),
    "bone_marrow":   ("bone_marrow.csv",   (0.5,),                   "treatment", "Bone marrow"),
    "liver_healthy": ("liver_healthy.csv", (),                       "treatment", "Healthy liver"),
    "spleen":        ("spleen.csv",        (),                       "treatment", "Spleen"),
}
# The physical comparator: dose per GBq ~ uptake^a * size^b (log-log OLS), the study's M0.
BASELINE = {"tumor_burden": ("pet_mean_suvbw", "pet_volume_ml"), "tumors": ("pet_mean_suvbw", "pet_volume_ml"),
            "kidneys": ("pet_mean_bqml", "pet_volume_ml"), "liver_healthy": ("pet_mean_bqml", "pet_volume_ml"),
            "spleen": ("pet_mean_bqml", "pet_volume_ml"), "bone_marrow": ("pet_mean_bqml", "pt_weight_kg")}

BACKENDS = {
    "tabpfn-3.5-api": "TabPFN-3.5 (Prior Labs API)",
    "tabpfn-3.5-local": "TabPFN-3.5 (local weights)",
    "physical-baseline": "Physical baseline (2-term log-log)",
    "gbm-quantile": "Gradient boosting, quantile heads",
    "gbm-tuned-conformal": "Gradient boosting, tuned + conformal",
}

# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------
def load_context(target: str, data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / TARGETS[target][0])


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The 17 per-ROI PET features ("M1" in the study) - the feature set the real-cohort
    calibration figures were measured on."""
    return [c for c in df.columns if c.startswith("pet_") and pd.api.types.is_numeric_dtype(df[c])]


def log_target(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["dose_gy"].astype(float) / df[ACTIVITY].astype(float)).to_numpy()


def split_examples(df: pd.DataFrame, frac: float = 0.2, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole synthetic patients as 'new cases' the model has not seen."""
    ids = df[GROUP].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    held = set(rng.choice(ids, size=max(1, int(round(frac * len(ids)))), replace=False))
    mask = df[GROUP].isin(held)
    return df.loc[~mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)


# What-if edits that keep a PET row internally consistent (totals = mean x volume, etc.)
_UPTAKE = ["pet_max_suvbw", "pet_mean_suvbw", "pet_median_suvbw", "pet_std_suvbw",
           "pet_mean_bqml", "pet_max_bqml", "pet_median_bqml", "pet_std_bqml"]
_TOTALS = ["pet_total_suvbw", "pet_tlg_suvbw_ml", "pet_integral_total_bqml_ml", "pet_total_bqml"]


def apply_whatif(case: dict, uptake: float = 1.0, volume: float = 1.0) -> dict:
    out = dict(case)
    for c in _UPTAKE:
        if c in out and pd.notna(out[c]):
            out[c] = out[c] * uptake
    for c in _TOTALS:
        if c in out and pd.notna(out[c]):
            out[c] = out[c] * uptake * volume
    if pd.notna(out.get("pet_volume_ml", np.nan)):
        out["pet_volume_ml"] = out["pet_volume_ml"] * volume
    if pd.notna(out.get("pet_sphere_diam_cm", np.nan)):
        out["pet_sphere_diam_cm"] = out["pet_sphere_diam_cm"] * volume ** (1 / 3)
    return out


# ------------------------------------------------------------------------------------------------
# back-ends
# ------------------------------------------------------------------------------------------------
class PhysicalBaseline:
    name = "physical-baseline"

    def __init__(self, target: str):
        self.terms = BASELINE[target]

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "PhysicalBaseline":
        d = df[list(self.terms)].astype(float).copy()
        d["_y"] = y
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        d = d[(d[list(self.terms)] > 0).all(axis=1)]
        A = np.column_stack([np.ones(len(d))] + [np.log(d[t]) for t in self.terms])
        self.coef, *_ = np.linalg.lstsq(A, d["_y"].to_numpy(), rcond=None)
        r = d["_y"].to_numpy() - A @ self.coef
        self.sigma = float(np.sqrt((r ** 2).sum() / max(1, len(d) - A.shape[1])))
        self.median_log = {t: float(np.log(d[t]).median()) for t in self.terms}
        return self

    def log_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for t in self.terms:
            v = X[t].astype(float).to_numpy() if t in X else np.full(len(X), np.nan)
            lv = np.where(np.isfinite(v) & (v > 0), np.log(np.where(v > 0, v, 1.0)), self.median_log[t])
            cols.append(lv)
        mu = self.coef[0] + np.column_stack(cols) @ self.coef[1:]
        z = np.array([NormalDist().inv_cdf(float(l)) for l in LEVELS])
        return mu[:, None] + self.sigma * z[None, :]


class TabPFN35:
    """TabPFN-3.5, either through the API (tabpfn-client) or with local weights (tabpfn)."""

    def __init__(self, mode: str = "api", n_estimators: int = 8, seed: int = 0, model_path: str | None = None):
        self.mode, self.n_estimators, self.seed = mode, n_estimators, seed
        self.model_path = model_path or os.environ.get("TABPFN35_WEIGHTS")
        self.name = f"tabpfn-3.5-{mode}"

    def _make(self):
        if self.mode == "api":
            from tabpfn_client import TabPFNRegressor
            token = os.environ.get("TABPFN_TOKEN")
            if token:
                from tabpfn_client import set_access_token
                set_access_token(token)
            # TABPFN_FIT_MODE=default sends no fit mode (the server's own default, no server-side cache)
            mode = os.environ.get("TABPFN_FIT_MODE", "fit_with_cache")
            kw = {} if mode in ("default", "none", "") else {"fit_mode": mode}
            return TabPFNRegressor.create_default_for_version(
                "v3.5", n_estimators=self.n_estimators, random_state=self.seed, **kw)
        import torch
        from tabpfn import TabPFNRegressor
        try:
            from tabpfn.model_loading import ModelVersion
        except Exception:                                         # older layout
            from tabpfn.constants import ModelVersion
        kw = dict(n_estimators=self.n_estimators, softmax_temperature=0.9, device="cpu",
                  random_state=self.seed, inference_precision=torch.float32)
        if self.model_path:
            kw["model_path"] = self.model_path                    # pin the file: offline, no per-fit lock
        return TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, **kw)

    # The hosted API can fail transiently ("The worker failed to process this request"). API calls are
    # retried with back-off; a failed prediction re-sends the context before the next attempt.
    API_TRIES, API_WAIT_S = 4, 3.0

    def _retry(self, call, on_fail=None):
        if self.mode != "api":
            return call()
        import time
        for i in range(self.API_TRIES):
            try:
                return call()
            except (ValueError, TypeError, KeyError):
                raise                                             # our bug, not the service's
            except Exception:
                if i == self.API_TRIES - 1:
                    raise
                time.sleep(self.API_WAIT_S * 2 ** i)
                if on_fail is not None:
                    on_fail()

    def _fit_once(self):
        self.model = self._make()
        self.model.fit(self._X, self._y)                          # NaN is fine; no imputation, no scaling

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TabPFN35":
        self.columns = list(X.columns)
        self._X, self._y = X.astype(float), np.asarray(y, float)
        self._retry(self._fit_once)
        return self

    def log_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        Xq = X[self.columns].astype(float)
        qs = self._retry(lambda: self.model.predict(Xq, output_type="quantiles",
                                                    quantiles=[float(l) for l in LEVELS]),
                         on_fail=lambda: self._retry(self._fit_once))
        Q = np.column_stack([np.asarray(a, dtype=float).ravel() for a in qs])   # (n_rows, n_levels)
        return np.maximum.accumulate(Q, axis=1)                    # enforce monotone quantiles


class GBMQuantile:
    """The usual way to get intervals from gradient boosting: one quantile-loss model per level
    (sklearn HistGradientBoosting, fixed settings, not tuned). The study's real-cohort comparator
    was the CatBoost equivalent. Levels outside the fitted range are held flat."""
    name = "gbm-quantile"
    FIT_LEVELS = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GBMQuantile":
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.columns = list(X.columns)
        self.models = [HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=150,
                                                     learning_rate=0.05, max_leaf_nodes=15,
                                                     min_samples_leaf=10, random_state=self.seed
                                                     ).fit(X.astype(float), y)
                       for q in self.FIT_LEVELS]
        return self

    def log_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        P = np.column_stack([m.predict(X[self.columns].astype(float)) for m in self.models])
        P = np.sort(P, axis=1)                                   # quantile heads can cross
        return np.array([np.interp(LEVELS, self.FIT_LEVELS, row) for row in P])


class GBMTunedConformal:
    """The real-cohort comparator design (study amendment R), live: gradient boosting whose settings are
    chosen by an inner 3-fold CV grouped by patient over a 4-setting grid, with split-conformal
    intervals - the point prediction plus the empirical quantiles of the chosen setting's inner
    out-of-fold residuals, widened by the finite-sample factor (n + 1) / n. Nothing here sees the
    outer test fold."""
    name = "gbm-tuned-conformal"
    needs_groups = True
    GRID = tuple({"max_leaf_nodes": m, "l2_regularization": l} for m in (8, 31) for l in (0.0, 1.0))

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _new(self, params):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=self.seed,
                                             early_stopping=False, **params)

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups=None) -> "GBMTunedConformal":
        from sklearn.model_selection import GroupKFold
        self.columns = list(X.columns)
        Xf, y = X.astype(float).to_numpy(), np.asarray(y, float)
        g = np.asarray(groups) if groups is not None else np.arange(len(y))
        inner = list(GroupKFold(n_splits=3, shuffle=True, random_state=self.seed).split(Xf, y, g))
        best = None
        for params in self.GRID:
            oof = np.full(len(y), np.nan)
            for tr, va in inner:
                oof[va] = self._new(params).fit(Xf[tr], y[tr]).predict(Xf[va])
            mse = float(np.mean((y - oof) ** 2))
            if best is None or mse < best[0]:
                best = (mse, params, y - oof)
        _, self.params, resid = best
        n = len(resid)
        adj = np.clip(0.5 + (LEVELS - 0.5) * (n + 1) / n, 0.0, 1.0)      # finite-sample widening
        self.offsets = np.quantile(resid, adj)
        self.model = self._new(self.params).fit(Xf, y)
        return self

    def log_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        mu = self.model.predict(X[self.columns].astype(float).to_numpy())
        return mu[:, None] + self.offsets[None, :]


def make_backend(name: str, target: str, n_estimators: int = 8, seed: int = 0):
    if name == "physical-baseline":
        return PhysicalBaseline(target)
    if name == "gbm-quantile":
        return GBMQuantile(seed)
    if name == "gbm-tuned-conformal":
        return GBMTunedConformal(seed)
    if name == "tabpfn-3.5-api":
        return TabPFN35("api", n_estimators, seed)
    if name == "tabpfn-3.5-local":
        return TabPFN35("local", n_estimators, seed)
    raise ValueError(name)


def available_backends() -> list[str]:
    out = []
    if os.environ.get("TABPFN_TOKEN"):
        try:
            import tabpfn_client  # noqa: F401
            out.append("tabpfn-3.5-api")
        except Exception:
            pass
    try:
        import importlib.util
        if importlib.util.find_spec("tabpfn") is not None and importlib.util.find_spec("torch") is not None:
            out.append("tabpfn-3.5-local")
    except Exception:
        pass
    return out + ["physical-baseline"]


# ------------------------------------------------------------------------------------------------
# the model a user talks to
# ------------------------------------------------------------------------------------------------
@dataclass
class DoseModel:
    target: str
    context: pd.DataFrame
    backend: str = "tabpfn-3.5-api"
    n_estimators: int = 8
    seed: int = 0
    features: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.features = self.features or feature_columns(self.context)
        self.thresholds = TARGETS[self.target][1]
        self._model = None

    def fit(self) -> "DoseModel":
        y = log_target(self.context)
        ok = np.isfinite(y)
        X = self.context.loc[ok, self.features].reset_index(drop=True)
        self._model = make_backend(self.backend, self.target, self.n_estimators, self.seed)
        if self.backend == "physical-baseline":
            self._model.fit(self.context.loc[ok].reset_index(drop=True), y[ok])
        elif getattr(self._model, "needs_groups", False):
            self._model.fit(X, y[ok], groups=self.context.loc[ok, GROUP].to_numpy())
        else:
            self._model.fit(X, y[ok])
        return self

    def with_added_case(self, row: dict) -> "DoseModel":
        """'No training': a newly measured case is one more context row. Returns a NEW model
        (the shared one is never mutated, so one visitor's additions never reach another)."""
        ctx = pd.concat([self.context, pd.DataFrame([row])], ignore_index=True)
        return DoseModel(self.target, ctx, self.backend, self.n_estimators, self.seed, list(self.features)).fit()

    def log_quantiles(self, rows: pd.DataFrame) -> np.ndarray:
        if self.backend == "physical-baseline":
            return self._model.log_quantiles(rows)
        X = rows.reindex(columns=self.features)
        return self._model.log_quantiles(X)

    def predict(self, case: dict, activity_gbq: float, decisive_p: float = 0.9) -> dict:
        q_log = self.log_quantiles(pd.DataFrame([case]))[0]
        return summarise(q_log, activity_gbq, self.thresholds, decisive_p, self.backend, self.target)


RANGE_CHECKED = ("pet_mean_suvbw", "pet_mean_bqml", "pet_volume_ml")


def outside_context(case: dict, context: pd.DataFrame, cols: tuple[str, ...] = RANGE_CHECKED) -> list[str]:
    """Inputs beyond anything in the context table: there the model extrapolates and the honest call
    is 'measure'. Returns one readable line per offending input (empty list = inside)."""
    out = []
    for c in cols:
        v = case.get(c)
        if c not in context or v is None or not np.isfinite(float(v)):
            continue
        lo, hi = float(context[c].min()), float(context[c].max())
        if not lo <= float(v) <= hi:
            out.append(f"{c} = {float(v):.3g} (context table: {lo:.3g} to {hi:.3g})")
    return out


def p_exceed(q_log: np.ndarray, threshold_gy: float, activity_gbq: float) -> float:
    """P(dose > threshold) read off the predictive quantile function. The grid stops at 1% / 99%,
    so the answer is clipped to [0.005, 0.995]: the app never reports certainty."""
    cdf = float(np.interp(np.log(threshold_gy / activity_gbq), q_log, LEVELS, left=0.0, right=1.0))
    return float(np.clip(1.0 - cdf, 0.005, 0.995))


def call_for(p: float, decisive_p: float) -> str:
    return "decisive: above" if p >= decisive_p else "decisive: below" if p <= 1 - decisive_p else "measure"


def summarise(q_log, activity_gbq, thresholds, decisive_p=0.9, backend="", target="") -> dict:
    q_gy = np.exp(q_log) * activity_gbq                       # back-transform QUANTILES, never a mean
    at = lambda level: float(np.interp(level, LEVELS, q_gy))
    out = {"backend": backend, "target": target, "activity_gbq": activity_gbq,
           "median_gy": at(0.5), "interval50_gy": (at(0.25), at(0.75)), "interval80_gy": (at(0.10), at(0.90)),
           "interval95_gy": (at(0.025), at(0.975)), "levels": LEVELS, "quantiles_gy": q_gy, "thresholds": {}}
    for thr in thresholds:
        p = p_exceed(q_log, thr, activity_gbq)
        out["thresholds"][thr] = {"p_exceed": p, "call": call_for(p, decisive_p)}
    return out
