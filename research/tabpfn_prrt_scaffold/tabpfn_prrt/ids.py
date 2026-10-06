"""Patient identity. Every split in this study groups on `base_id`, never on `pt_id`.

Four patients in the 86-treatment cohort received two courses years apart, each course
carrying its own `pt_id`. Without decoding, their two courses land on opposite sides of
a split and the model is scored on remembering a patient it has already seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The study's four repeat-course pairs {base_id: (pt_id_course1, pt_id_course2)} are configured
# locally and deliberately not published (study number + treatment years).
KNOWN_REPEAT_PAIRS: dict[int, tuple[int, int]] = {}


def base_id(pt_id) -> int:
    """Decode a V8 `pt_id` into the patient it belongs to.

    2-3 digits -> the patient number itself.
    5 digits   -> first three digits are the patient, last two the treatment year.

    Never use string prefix matching here: str(51018).startswith('51') is True, but
    51 and 510 are different patients.
    """
    s = str(int(pt_id))
    if len(s) == 5 and 10 <= int(s[3:]) <= 29:
        return int(s[:3])
    return int(s)


def group_vector(df: pd.DataFrame, id_col: str = "pt_id") -> np.ndarray:
    """Grouping vector for GroupKFold. Asserts the four known repeat patients resolve."""
    if id_col not in df.columns:
        raise KeyError(f"{id_col!r} not in frame; columns start with {list(df.columns)[:8]}")
    groups = df[id_col].map(base_id).to_numpy()

    present = set(int(v) for v in df[id_col].dropna().unique())
    for base, (a, b) in KNOWN_REPEAT_PAIRS.items():
        if a in present and b in present:
            mapped = {base_id(a), base_id(b)}
            assert mapped == {base}, f"repeat pair {a}/{b} did not decode to {base}: {mapped}"
    return groups


def assert_join_key_is_dicom_source(xlsx: pd.DataFrame) -> None:
    """Guard against joining the Excel sheet on `pt#` instead of `pt_dicom_source`.

    Inside the 86-treatment cohort `pt#` repeats for four patients; `pt_dicom_source`
    does not. A duplicated `pt#` is the signature of the wrong key.
    """
    if "pt#" in xlsx.columns:
        dup = xlsx["pt#"].dropna()
        n_dup = int(dup.duplicated().sum())
        if n_dup:
            msg = (f"`pt#` has {n_dup} duplicate values -- it is NOT a treatment key. "
                   f"Join on `pt_dicom_source`.")
            assert "pt_dicom_source" in xlsx.columns, msg
    if "pt_dicom_source" in xlsx.columns:
        d = xlsx["pt_dicom_source"].dropna()
        assert not d.duplicated().any(), "pt_dicom_source is not unique; investigate before joining"


def cohort_summary(df: pd.DataFrame, id_col: str = "pt_id") -> dict[str, int]:
    """Every results table reports rows, treatments and patients -- not just rows."""
    g = df[id_col].map(base_id)
    return {
        "rows": int(len(df)),
        "treatments": int(df[id_col].nunique()),
        "patients": int(g.nunique()),
    }
