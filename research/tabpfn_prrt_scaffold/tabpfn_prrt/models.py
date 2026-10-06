"""Model arms. Every arm exposes the same interface: fit, then predict quantiles.

Arms:
  M0-lin        pre-specified simple baseline (the comparator gap G5 requires)
  TabPFN-V3     headline arm, no tuning
  TabPFN-V2     token-free reproducibility arm (Apache-2.0 + attribution weights)
  CatBoost      strongest conventional competitor, nested tuning
  RF-Akhava     replication of Akhavanallaf 2023 (3-feature random forest)
  Oracle-t      TabPFN + spect_scan_time_hours -- an UPPER BOUND, never a result
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class QuantileModel(Protocol):
    name: str
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "QuantileModel": ...
    def predict_quantiles(self, X: pd.DataFrame, levels: np.ndarray) -> np.ndarray: ...


# =====================================================================================
# TabPFN
# =====================================================================================
_MODEL_VERSIONS = {"V2": "V2", "V2_5": "V2_5", "V2_6": "V2_6", "V3": None}


@dataclass
class TabPFNArm:
    version: str = "V3"
    n_estimators: int = 32
    softmax_temperature: float = 0.9
    device: str = "cpu"
    random_state: int = 0
    deterministic_precision: bool = True
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"TabPFN-{self.version}"
        self._model = None
        self._cat_idx: list[int] | None = None

    def _build(self, X: pd.DataFrame):
        import torch
        from tabpfn import TabPFNRegressor

        kwargs = dict(
            n_estimators=self.n_estimators,
            softmax_temperature=self.softmax_temperature,
            device=self.device,
            random_state=self.random_state,
            # Mixed-precision autocast is faster and slightly non-deterministic. At
            # n <= 309 the speed is irrelevant and the determinism is not.
            inference_precision=(torch.float32 if self.deterministic_precision else "auto"),
            categorical_features_indices=self._cat_idx,
        )
        if _MODEL_VERSIONS.get(self.version):
            from tabpfn.constants import ModelVersion
            return TabPFNRegressor.create_default_for_version(
                getattr(ModelVersion, self.version), **kwargs)
        return TabPFNRegressor(**kwargs)

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        # Re-normalise the token variables HERE, not only at package import.
        #
        # The credential store re-injects `TABPFN` into the process environment for each
        # cell execution, so normalising once at import is not durable: a kernel that
        # imported cleanly still hits
        #   SettingsError: error parsing value for field "tabpfn" from source
        #                  "EnvSettingsSource"
        # on a later fit, because pydantic re-reads os.environ when TabPFN's settings are
        # constructed. Observed twice -- once silently emptying two target runs (Phase 0
        # F10), and once after a daemon restart mid-session, caught that time by the
        # driver's no-rows assertion. Making it point-of-use costs one dict lookup.
        from . import env as _env
        _env.normalise_token_vars()

        # Documented guidance: do NOT scale or one-hot encode; pass raw values and mark
        # categoricals. Missing values are handled natively -- do not impute.
        self._cat_idx = [i for i, c in enumerate(X.columns)
                         if str(X[c].dtype) in ("category", "object", "bool")]
        self._model = self._build(X)
        self._model.fit(X, np.asarray(y, float))
        return self

    def predict_quantiles(self, X: pd.DataFrame, levels: np.ndarray) -> np.ndarray:
        # One batched call: predict() recomputes the training set each time, so calling
        # it per row is ~n times slower for identical output.
        qs = self._model.predict(X, output_type="quantiles",
                                 quantiles=[float(q) for q in levels])
        return np.column_stack([np.asarray(a, float) for a in qs])

    def provenance(self) -> dict[str, str]:
        import tabpfn
        return {"arm": self.name, "tabpfn_version": tabpfn.__version__,
                "model_version": self.version, "n_estimators": str(self.n_estimators)}


# =====================================================================================
# M0: the pre-specified simple baseline
# =====================================================================================
BASELINE_FEATURES = {
    # tumours: the relationship the whole corpus rests on -- SUVmean and lesion volume
    "tumors": ("pet_mean_suvbw", "pet_volume_ml"),
    "tumors_hepatic": ("pet_mean_suvbw", "pet_volume_ml"),
    "tumor_burden": ("pet_mean_suvbw", "pet_volume_ml"),
    # kidneys: renal PET uptake + body size.
    #
    # DEVIATION FROM THE PRE-SPECIFIED PLAN (6.1 / 10), recorded rather than silent.
    # The plan specified renal PET uptake + eGFR + body size. eGFR is not computable in
    # this cohort: gen_baseline_creatinine_units_unresolved is the 0.0 not-recorded
    # sentinel in 57/86 treatments and blank in a further 22, leaving 7 real creatinine
    # values (3 mg/dL, 4 umol/L). A baseline whose third term is median-imputed for
    # 79/86 rows is not the pre-specified baseline; it is a two-feature baseline with an
    # added constant. The term is therefore dropped from M0 and the plan's claim that
    # resolving the units "is worth the hour" does not survive contact with the data.
    # calc_egfr remains available in block E (M4) at its true 7/86 coverage.
    "kidneys": ("pet_mean_bqml", "pt_weight_kg"),
    "liver_healthy": ("pet_mean_suvbw", "pet_volume_ml"),
    "spleen": ("pet_mean_suvbw", "pet_volume_ml"),
    # marrow: cal_std PET uptake + body size. Blood activity is an INPUT to the label.
    "bone_marrow": ("pet_mean_bqml", "pt_weight_kg"),
}
LOG_FEATURES = {"pet_mean_suvbw", "pet_volume_ml", "pet_mean_bqml", "pet_max_suvbw"}

# --- Baseline shape, selectable and declared -----------------------------------------
#
# Three shapes now exist, and which one is used decides two of this study's conclusions.
#
#   "plan"      the frozen plan's M0 (above). For liver and spleen it applies the TUMOUR
#               relationship to a normal organ, where its two features correlate with the
#               outcome at Spearman 0.16 and 0.04.
#   "corrected" Phase 5 s2's addition: organ uptake (Bq/ml) + body weight, for liver and
#               spleen. Declared there as an addition, not a substitution.
#   "organ3p"   the shape the PROJECT's own V9 ladder uses for organs
#               (`3p_physical_ols`): log organ uptake + log organ volume + the organ's
#               self / whole-body uptake ratio.
#
# Why "organ3p" had to be added (2026-09-20, DEC-224). On kidneys the V9 3p baseline
# reaches r2_row_log = 0.2665, which is ABOVE every kidney arm this scaffold produced
# (best: TabPFN M1 = 0.2619) and far above this scaffold's kidney M0 = 0.1427. The
# reported "+0.125, promising" kidney gain is therefore a property of a weak baseline,
# not of the model -- exactly the error Phase 5 caught on liver and spleen, left unfixed
# on kidneys. The same shape must also be run on liver before the liver result, the
# study's only pre-specified success, is reported.
#
# Select with the PRRT_BASELINE environment variable. Any value other than "plan" makes
# the run an addition to the frozen plan and must be declared in the phase report.
BASELINE_CORRECTED = {
    "liver_healthy": ("pet_mean_bqml", "pt_weight_kg"),
    "spleen": ("pet_mean_bqml", "pt_weight_kg"),
}
BASELINE_ORGAN_3P = {
    "kidneys": ("pet_mean_bqml", "pet_volume_ml", "ratio_self_wb_bqml"),
    "liver_healthy": ("pet_mean_bqml", "pet_volume_ml", "ratio_self_wb_bqml"),
    "spleen": ("pet_mean_bqml", "pet_volume_ml", "ratio_self_wb_bqml"),
    "bone_marrow": ("pet_mean_bqml", "pet_volume_ml", "ratio_self_wb_bqml"),
}
# Derived baseline terms, computed from columns already in the feature matrix. The
# whole-body VOI is block B and is admissible at treatment level under the leakage
# register; it is pre-therapeutic PET like every other term here.
DERIVED_BASELINE_TERMS = {
    "ratio_self_wb_bqml": ("pet_mean_bqml", "voi_whole_body__pet_mean_bqml"),
}
LOG_FEATURES |= {"ratio_self_wb_bqml"}


def baseline_features_for(target_name: str) -> tuple[str, ...]:
    """The M0 feature set for a target under the currently selected baseline shape."""
    import os
    kind = os.environ.get("PRRT_BASELINE", "plan").strip().lower()
    if kind == "corrected" and target_name in BASELINE_CORRECTED:
        return BASELINE_CORRECTED[target_name]
    if kind == "organ3p" and target_name in BASELINE_ORGAN_3P:
        return BASELINE_ORGAN_3P[target_name]
    return BASELINE_FEATURES[target_name]


@dataclass
class LinearBaseline:
    target_name: str
    name: str = "M0-lin"

    def __post_init__(self):
        self._cols: tuple[str, ...] = ()
        self._coef = None
        self._sigma = 1.0
        self._med = None

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        X = _with_derived_terms(X, self._cols)
        Z = X[list(self._cols)].astype(float).copy()
        for c in self._cols:
            if c in LOG_FEATURES:
                Z[c] = np.log(np.maximum(Z[c].to_numpy(float), 1e-6))
        Z = Z.fillna(pd.Series(self._med, index=self._cols))
        return np.column_stack([np.ones(len(Z)), Z.to_numpy(float)])

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        wanted = baseline_features_for(self.target_name)
        X = _with_derived_terms(X, wanted)
        self._cols = tuple(c for c in wanted if c in X.columns)
        missing = tuple(c for c in wanted if c not in X.columns)
        if missing:
            # Silently dropping a baseline term changes what the comparison means.
            raise KeyError(f"baseline term(s) {missing} missing for {self.target_name}")
        raw = X[list(self._cols)].astype(float)
        for c in self._cols:
            if c in LOG_FEATURES:
                raw[c] = np.log(np.maximum(raw[c].to_numpy(float), 1e-6))
        # median imputation fitted INSIDE the training fold only
        self._med = raw.median(numeric_only=True).to_dict()
        A = self._design(X)
        yy = np.asarray(y, float)
        self._coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        resid = yy - A @ self._coef
        dof = max(1, len(yy) - A.shape[1])
        self._sigma = float(np.sqrt(np.sum(resid ** 2) / dof))
        return self

    def predict_quantiles(self, X: pd.DataFrame, levels: np.ndarray) -> np.ndarray:
        from scipy.stats import norm
        mu = self._design(X) @ self._coef
        return mu[:, None] + self._sigma * norm.ppf(np.asarray(levels, float))[None, :]


# =====================================================================================
# CatBoost, nested tuning -- the strongest conventional competitor
# =====================================================================================
@dataclass
class CatBoostArm:
    name: str = "CatBoost"
    nested: bool = True
    random_state: int = 0
    grid: tuple[dict, ...] = (
        {"depth": 3, "learning_rate": 0.05, "iterations": 400},
        {"depth": 4, "learning_rate": 0.05, "iterations": 400},
        {"depth": 6, "learning_rate": 0.03, "iterations": 800},
    )

    def __post_init__(self):
        self._models: dict[float, object] = {}
        self._best: dict | None = None
        self._cols: list[str] = []

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = X.copy()
        for c in Z.columns:
            if str(Z[c].dtype) in ("object", "category", "bool"):
                Z[c] = Z[c].astype("string").fillna("__NA__")
        return Z

    def fit(self, X: pd.DataFrame, y: np.ndarray, levels: np.ndarray | None = None):
        from catboost import CatBoostRegressor
        from sklearn.model_selection import KFold
        Z = self._prep(X)
        self._cols = list(Z.columns)
        cat = [c for c in Z.columns if str(Z[c].dtype) == "string"]
        yy = np.asarray(y, float)

        # INNER loop only -- never sees the outer test fold.
        if self.nested and len(yy) >= 30:
            best, best_err = None, np.inf
            for params in self.grid:
                errs = []
                for tr, va in KFold(3, shuffle=True, random_state=self.random_state).split(Z):
                    m = CatBoostRegressor(**params, loss_function="RMSE", verbose=0,
                                          random_seed=self.random_state, cat_features=cat)
                    m.fit(Z.iloc[tr], yy[tr])
                    errs.append(float(np.mean((yy[va] - m.predict(Z.iloc[va])) ** 2)))
                if np.mean(errs) < best_err:
                    best, best_err = params, float(np.mean(errs))
            self._best = best
        else:
            self._best = self.grid[0]

        levels = np.asarray(levels if levels is not None else [0.05, 0.25, 0.5, 0.75, 0.95], float)
        self._models = {}
        for q in levels:
            m = CatBoostRegressor(**self._best, loss_function=f"Quantile:alpha={float(q)}",
                                  verbose=0, random_seed=self.random_state, cat_features=cat)
            m.fit(Z, yy)
            self._models[float(q)] = m
        return self

    def predict_quantiles(self, X: pd.DataFrame, levels: np.ndarray) -> np.ndarray:
        Z = self._prep(X)[self._cols]
        have = np.array(sorted(self._models))
        preds = np.column_stack([self._models[q].predict(Z) for q in have])
        preds = np.maximum.accumulate(preds, axis=1)      # enforce monotone quantiles
        return np.column_stack([
            np.array([np.interp(l, have, row) for row in preds]) for l in np.asarray(levels, float)
        ])


def _with_derived_terms(X: pd.DataFrame, wanted) -> pd.DataFrame:
    """Add any derived baseline term that `wanted` asks for and `X` does not carry."""
    need = [c for c in wanted if c in DERIVED_BASELINE_TERMS and c not in X.columns]
    if not need:
        return X
    X = X.copy()
    for c in need:
        num, den = DERIVED_BASELINE_TERMS[c]
        if num not in X.columns or den not in X.columns:
            continue
        d = pd.to_numeric(X[den], errors="coerce").to_numpy(float)
        n = pd.to_numeric(X[num], errors="coerce").to_numpy(float)
        X[c] = np.where(np.isfinite(d) & (d > 0), n / np.where(d > 0, d, np.nan), np.nan)
    return X


# =====================================================================================
# In-fold univariate filter -- EXPLORATORY, not in the frozen plan (DEC-225)
# =====================================================================================
@dataclass
class TopKFilterArm:
    """Keep the K features with the largest |Spearman| against y, fitted in-fold only.

    This is a researcher degree of freedom the plan does not contain, and any run using
    it is `is_protocol_run: false`. It exists to answer one specific question: the
    project's V9 ladder reaches r2_row_log 0.383 (TabPFN-3) and 0.427 (HistGBM) on the
    72-row spleen target with K ~ 3-5 features selected inside each fold, while this
    scaffold's TabPFN reaches 0.197 on 165 unselected features and was therefore reported
    as a retraction of the spleen result. If the gap closes under a K-feature filter, the
    retraction is a dimensionality artefact of this harness, not a fact about the target.

    Selection uses the TRAINING fold's y only. Constant and all-NaN columns are dropped
    before ranking; non-numeric columns are kept out of the filter and never selected.
    """
    inner_factory: object = None
    k: int = 5
    name: str = "TopK"

    def __post_init__(self):
        self._inner = None
        self._cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        from scipy.stats import spearmanr
        yy = np.asarray(y, float)
        scores = []
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                continue
            v = pd.to_numeric(X[c], errors="coerce").to_numpy(float)
            ok = np.isfinite(v)
            if ok.sum() < max(10, 0.5 * len(v)) or np.nanstd(v[ok]) == 0:
                continue
            try:
                r = spearmanr(v[ok], yy[ok]).statistic
            except Exception:
                continue
            if np.isfinite(r):
                scores.append((abs(float(r)), c))
        scores.sort(reverse=True)
        self._cols = [c for _, c in scores[: self.k]]
        if not self._cols:
            raise ValueError("TopK filter selected no features")
        self._inner = self.inner_factory()
        self._inner.fit(X[self._cols], yy)
        return self

    def predict_quantiles(self, X: pd.DataFrame, levels: np.ndarray) -> np.ndarray:
        return self._inner.predict_quantiles(X[self._cols], levels)


# =====================================================================================
# Akhavanallaf 2023 replication
# =====================================================================================
@dataclass
class RFAkhavanallaf:
    name: str = "RF-Akhava"
    features: tuple[str, ...] = ("pet_mean_suvbw", "pet_max_suvbw", "pet_volume_ml")
    random_state: int = 0

    def __post_init__(self):
        self._m = None
        self._cols: list[str] = []
        self._med = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        from sklearn.ensemble import RandomForestRegressor
        self._cols = [c for c in self.features if c in X.columns]
        Z = X[self._cols].astype(float)
        self._med = Z.median()
        self._m = RandomForestRegressor(n_estimators=1000, min_samples_leaf=2,
                                        random_state=self.random_state, n_jobs=-1)
        self._m.fit(Z.fillna(self._med), np.asarray(y, float))
        return self

    def predict_quantiles(self, X: pd.DataFrame, levels: np.ndarray) -> np.ndarray:
        Z = X[self._cols].astype(float).fillna(self._med)
        per_tree = np.column_stack([t.predict(Z.to_numpy()) for t in self._m.estimators_])
        return np.column_stack([np.quantile(per_tree, float(l), axis=1)
                                for l in np.asarray(levels, float)])
