"""Splitting. Everything groups on `base_id`; nothing here ever splits on rows alone."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


@dataclass
class Split:
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray
    kind: str = "cv"


def repeated_grouped_splits(y: np.ndarray, groups: np.ndarray, *,
                            n_repeats: int = 20, n_folds: int = 5,
                            seeds: tuple[int, ...] | None = None,
                            stratify: bool = True) -> Iterator[Split]:
    """20 x 5 repeated grouped CV.

    Folds are stratified on target quintiles where the grouping permits, so that a fold
    cannot happen to contain every high-dose case. Both courses of a repeat patient are
    always in the same fold -- that is what grouping on base_id buys.
    """
    seeds = seeds or tuple(range(n_repeats))
    y = np.asarray(y, float)
    groups = np.asarray(groups)

    if stratify:
        try:
            strata = pd.qcut(y, q=min(5, max(2, len(np.unique(y)) // 4)),
                             labels=False, duplicates="drop")
            strata = np.asarray(strata)
        except Exception:
            strata = np.zeros(len(y), dtype=int)
    else:
        strata = np.zeros(len(y), dtype=int)

    for r, seed in enumerate(seeds[:n_repeats]):
        if stratify:
            splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            it = splitter.split(np.zeros((len(y), 1)), strata, groups)
        else:
            try:
                splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                it = splitter.split(np.zeros((len(y), 1)), y, groups)
            except TypeError:                      # sklearn < 1.6: shuffle by permuting groups
                rng = np.random.default_rng(seed)
                uniq = np.unique(groups)
                relabel = dict(zip(uniq, rng.permutation(len(uniq))))
                shuffled = np.array([relabel[g] for g in groups])
                it = GroupKFold(n_splits=n_folds).split(
                    np.zeros((len(y), 1)), y, shuffled)
        for f, (tr, te) in enumerate(it):
            assert not set(groups[tr]) & set(groups[te]), "patient leaked across the split"
            yield Split(repeat=r, fold=f, train=np.asarray(tr), test=np.asarray(te))


def era_of(injection_date: pd.Series) -> pd.Series:
    """early = 2018-2020 (three-time-point protocol), late = 2022-2024 (single time point)."""
    yr = pd.to_datetime(injection_date, errors="coerce").dt.year
    return pd.Series(np.where(yr <= 2020, "early", np.where(yr >= 2022, "late", "gap")),
                     index=injection_date.index)


def leave_one_out_splits(labels: pd.Series, groups: np.ndarray,
                         kind: str) -> Iterator[Split]:
    """Pseudo-external validation: train on one stratum, test on the other.

    This is the closest available analogue to Akhavanallaf 2025's leave-one-centre-out
    inside single-centre data. Era and scanner are entangled at n=86 (Companion 2.1) --
    the two splits bound the effect, they do not decompose it.
    """
    labels = pd.Series(labels).reset_index(drop=True)
    for i, held in enumerate(sorted(x for x in labels.dropna().unique() if x != "gap")):
        te = np.flatnonzero((labels == held).to_numpy())
        tr = np.flatnonzero((labels != held).to_numpy() & labels.notna().to_numpy())
        if len(te) < 10 or len(tr) < 20:
            continue
        overlap = set(np.asarray(groups)[tr]) & set(np.asarray(groups)[te])
        if overlap:
            # a repeat patient straddling two eras: send both courses to the test side
            keep = ~np.isin(np.asarray(groups)[tr], list(overlap))
            tr = tr[keep]
        yield Split(repeat=0, fold=i, train=tr, test=te, kind=f"{kind}:{held}")
