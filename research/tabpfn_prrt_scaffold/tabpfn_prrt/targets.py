"""Target construction and the back-transform rule.

Two rules, both easy to get wrong and both load-bearing:

1. Model Gy/GBq, report Gy. Net activity spans 1.93-8.16 GBq including four patients at
   a quarter of the standard activity; predicting Gy directly means spending R^2 on
   re-learning a quantity already known at prediction time.
   Caveat stated in the methods: the STP formula raises the concentration term to
   a1 in [0.85, 0.99], so Dose/A ~ A^(a1-1) is not exactly activity-invariant. Net
   activity therefore stays in the feature set to absorb the residual.

2. Fit on log, and back-transform QUANTILES, never means. exp() of a predicted mean in
   log space is a median, not a mean, and biases every error downwards. Exponentiation
   is monotone, so quantiles -- and therefore every interval and coverage statistic --
   transfer exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetTransform:
    normalise_by_activity: bool = True
    log: bool = True
    floor: float = 1e-4

    def forward(self, dose_gy: np.ndarray, activity_gbq: np.ndarray) -> np.ndarray:
        y = np.asarray(dose_gy, float)
        if self.normalise_by_activity:
            a = np.asarray(activity_gbq, float)
            if np.any(~np.isfinite(a)) or np.any(a <= 0):
                raise ValueError("net injected activity must be positive and complete")
            y = y / a
        return np.log(np.maximum(y, self.floor)) if self.log else y

    def inverse_quantiles(self, q_pred: np.ndarray, activity_gbq: np.ndarray) -> np.ndarray:
        """Back-transform a (n_rows, n_quantiles) array to Gy. Quantiles only."""
        q = np.asarray(q_pred, float)
        if self.log:
            q = np.exp(q)
        if self.normalise_by_activity:
            q = q * np.asarray(activity_gbq, float)[:, None]
        return q

    def inverse_point_is_median(self) -> bool:
        """True when a back-transformed point prediction is a median, not a mean.

        The scaffold therefore always takes the point estimate as the 0.5 quantile
        rather than from output_type='mean'.
        """
        return self.log


def build_target(df: pd.DataFrame, target_col: str, activity_col: str,
                 transform: TargetTransform) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_model_space, dose_gy, activity_gbq) with non-finite rows dropped upstream."""
    dose = pd.to_numeric(df[target_col], errors="coerce").to_numpy(float)
    act = pd.to_numeric(df[activity_col], errors="coerce").to_numpy(float)
    ok = np.isfinite(dose) & np.isfinite(act) & (act > 0) & (dose > 0)
    if not ok.all():
        raise ValueError(
            f"{int((~ok).sum())} rows have a non-positive or missing dose/activity; "
            "handle them explicitly in data.py rather than silently here"
        )
    return transform.forward(dose, act), dose, act
