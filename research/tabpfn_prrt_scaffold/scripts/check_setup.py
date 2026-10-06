"""Run this first. It answers 'is this machine ready?' without touching the cohort.

    python scripts/check_setup.py            # environment, token, data layout
    python scripts/check_setup.py --fit      # also downloads the checkpoint and fits
                                             # 80 synthetic rows end to end

Every check prints PASS / WARN / FAIL and, when it fails, the one thing to do next.
The token is never printed -- only its length and last four characters.
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, str]] = []


def say(status: str, what: str, detail: str = "") -> None:
    RESULTS.append((status, what))
    mark = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[status]
    print(f"[{mark}] {what}")
    if detail:
        for line in detail.splitlines():
            print(f"       {line}")


# ---------------------------------------------------------------------------------
def check_python() -> None:
    v = sys.version_info
    detail = f"{sys.version.split()[0]} on {platform.system()} {platform.machine()}"
    if (v.major, v.minor) >= (3, 10):
        say("PASS", "Python 3.10+", detail)
    else:
        say("FAIL", "Python 3.10+", detail + "\nTabPFN needs 3.10-3.14. Install a newer Python.")


def check_layout() -> None:
    pkg = ROOT / "tabpfn_prrt" / "__init__.py"
    tests = ROOT / "tests" / "smoke_test.py"
    if pkg.is_file() and tests.is_file():
        say("PASS", "folder layout", f"running from {ROOT}")
    else:
        say("FAIL", "folder layout",
            f"{ROOT} should contain BOTH tabpfn_prrt\\ and tests\\.\n"
            "If the zip unpacked one level deeper, cd into the inner tabpfn_prrt folder.")


def check_imports(need_tabpfn: bool) -> None:
    core = ["numpy", "pandas", "sklearn", "scipy", "openpyxl"]
    optional = {"tabpfn": "the TabPFN arms", "torch": "TabPFN's backend",
                "catboost": "the CatBoost comparator", "matplotlib": "figures"}
    missing = [m for m in core if not _importable(m)]
    if missing:
        say("FAIL", "core packages", "missing: " + ", ".join(missing)
            + "\n  pip install -r requirements.txt")
    else:
        say("PASS", "core packages", ", ".join(f"{m}={_ver(m)}" for m in core))

    for mod, why in optional.items():
        if _importable(mod):
            say("PASS", f"{mod} ({why})", f"version {_ver(mod)}")
        elif mod in ("tabpfn", "torch") and need_tabpfn:
            say("FAIL", f"{mod} ({why})", "  pip install -r requirements.txt")
        else:
            say("WARN", f"{mod} ({why})", "not installed; that arm will be skipped")


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _ver(name: str) -> str:
    try:
        return getattr(importlib.import_module(name), "__version__", "?")
    except Exception:
        return "?"


def check_token() -> None:
    from tabpfn_prrt.env import load_dotenv, token_status
    loaded = load_dotenv()
    if loaded:
        say("PASS", ".env loaded", "keys set: " + ", ".join(loaded))
    else:
        say("WARN", ".env", "none found (fine if the token is already in the environment)")

    status = token_status()
    if status.startswith("no token"):
        say("WARN", "TabPFN token", status + "\n"
            "Either let the browser open once, or put TABPFN_TOKEN in .env.")
    else:
        say("PASS", "TabPFN token", status)


def check_data() -> None:
    from tabpfn_prrt import config as C
    root = C.DATA_ROOT
    if not root.exists():
        say("FAIL", "data folder", f"{root.resolve()} does not exist.\n"
            "Set PRRT_DATA_ROOT in .env to the Deidentified_Export folder.")
        return
    say("PASS", "data folder", str(root.resolve()))

    present, absent = [], []
    for spec in C.TARGETS:
        p = C.V8_ROOT / spec.folder / f"{spec.folder}_dataset.csv"
        (present if p.exists() else absent).append(spec.folder)
    present = sorted(set(present))
    if present:
        say("PASS", "V8 datasets", ", ".join(present))
    if sorted(set(absent)) and not present:
        say("FAIL", "V8 datasets", "none found under " + str(C.V8_ROOT.resolve()))
    elif sorted(set(absent)):
        say("WARN", "V8 datasets", "missing: " + ", ".join(sorted(set(absent))))

    try:
        from tabpfn_prrt.data import find_treatments_xlsx
        say("PASS", "treatment spreadsheet", str(find_treatments_xlsx()))
    except Exception as exc:
        say("WARN", "treatment spreadsheet", f"{exc}\nBlock D (body size) will be incomplete.")


def check_fit(version: str) -> None:
    """Prove the whole chain works on 80 synthetic rows: checkpoint, licence, quantiles."""
    import numpy as np
    import pandas as pd
    from tabpfn_prrt.models import TabPFNArm

    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({
        "suv": rng.lognormal(2.5, 0.6, n),
        "vol": rng.lognormal(2.0, 1.0, n),
        "site": pd.Categorical(rng.choice(["liver", "bone", "neither"], n)),
    })
    X.loc[X.sample(8, random_state=1).index, "vol"] = np.nan     # NaN passes through
    y = 0.8 * np.log(X.suv) + 0.2 * np.log(X.vol.fillna(X.vol.median())) \
        + rng.normal(0, 0.4, n)

    levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    try:
        arm = TabPFNArm(version=version, n_estimators=4, device="cpu")
        q = arm.fit(X.iloc[:60], y.to_numpy()[:60]).predict_quantiles(X.iloc[60:], levels)
    except Exception as exc:
        say("FAIL", f"TabPFN-{version} end-to-end fit", f"{type(exc).__name__}: {exc}")
        return

    monotone = bool(np.all(np.diff(q, axis=1) >= -1e-8))
    inside = float(np.mean((y.to_numpy()[60:] >= q[:, 0]) & (y.to_numpy()[60:] <= q[:, 4])))
    if monotone:
        say("PASS", f"TabPFN-{version} end-to-end fit",
            f"quantiles monotone; 90% interval covered {inside:.0%} of 20 held-out rows; "
            f"{arm.provenance()}")
    else:
        say("FAIL", f"TabPFN-{version} end-to-end fit", "quantiles are not monotone")


# ---------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true",
                    help="also download the checkpoint and fit 80 synthetic rows")
    ap.add_argument("--version", default="V2",
                    help="checkpoint for --fit: V2 (no token) | V2_5 | V2_6 | V3")
    a = ap.parse_args()

    print(f"setup check -- {ROOT}\n")
    check_python()
    check_layout()
    check_imports(need_tabpfn=a.fit)
    check_token()
    check_data()
    if a.fit:
        print()
        check_fit(a.version)

    n_fail = sum(1 for s, _ in RESULTS if s == "FAIL")
    n_warn = sum(1 for s, _ in RESULTS if s == "WARN")
    print(f"\n{len(RESULTS) - n_fail - n_warn} pass, {n_warn} warn, {n_fail} fail")
    if n_fail:
        print("Fix the FAIL lines above, then re-run this script.")
        return 1
    print("Ready. Next:  python tests\\smoke_test.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
