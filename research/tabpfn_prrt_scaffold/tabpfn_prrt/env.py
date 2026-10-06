"""Load a local `.env` into os.environ.

Why this file exists. TabPFN's own settings object reads `.env` through
pydantic-settings, but the auth token is NOT one of its fields: `get_cached_token()`
reads `os.environ["TABPFN_TOKEN"]` directly. So a `.env` containing TABPFN_TOKEN is
silently ignored by TabPFN itself. This loader closes that gap, so that putting the
token in `.env` works the way everyone expects.

Resolution order used by TabPFN once this has run:
    1. TABPFN_TOKEN environment variable   <- what `.env` populates here
    2. ~/.cache/tabpfn/auth_token          <- written after a browser login
    3. ~/.tabpfn/token                     <- tabpfn-client's cache

Never commit `.env`. It is in .gitignore, and the token never belongs in source.
"""
from __future__ import annotations

import os
from pathlib import Path

_PREFIXES = ("TABPFN_", "HF_", "PRRT_")


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> list[str]:
    """Read KEY=VALUE lines from `.env` and set them in os.environ.

    Searched, in order: an explicit `path`, then `.env` in the working directory,
    then `.env` beside the package. Returns the names of the keys set (never values).
    """
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates += [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]

    for env_path in candidates:
        if not env_path.is_file():
            continue
        loaded: list[str] = []
        for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key.startswith(_PREFIXES):
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
        return loaded
    return []


RUNNING_OFFLINE_NOTE = """
Running with no network egress
------------------------------
The V2 checkpoint is cached (tabpfn_models/tabpfn-v2-regressor.ckpt, 42.3 MB) and V2 will
fit with no network at all -- but ONLY if no token is visible to it. With TABPFN_TOKEN set,
TabPFN attempts licence validation against api.priorlabs.ai and, with egress down, the fit
blocks instead of failing fast. Measured: 21.7 s to fit offline with the token removed;
three probes stopped after 5-25 min with it present.

So for an offline run, pop the bare credential at the START of the cell -- which also makes
normalise_token_vars() below a no-op for the rest of that cell:

    for k in ("TABPFN", "TABPFN_TOKEN"):
        os.environ.pop(k, None)
    os.environ["HF_HUB_OFFLINE"] = "1"

V3 / V3.5 cannot run offline at all: they are gated and require licence verification.
Use V2 when egress is down, and note the arm in the results.

One further trap, unrelated to the network: an execution that had to be FORCE-CLEARED
leaves this environment's kernel wedged -- every later cell in it produces no output,
including cells that never touch TabPFN. That looks exactly like another hang. A fresh
kernel clears it; check with a trivial `print(..., flush=True)` before concluding the
network is at fault.
"""


def normalise_token_vars() -> list[str]:
    """Move bare secret-manager variable names onto the names the libraries read.

    A credential store that exports its entries under their own names injects a
    variable called exactly `TABPFN`. That breaks `import tabpfn` outright, before any
    model is touched: `tabpfn.settings.Settings` is a pydantic-settings model with a
    complex field named `tabpfn`, so pydantic tries to JSON-parse the variable's value,
    a 53-character opaque token, and raises

        SettingsError: error parsing value for field "tabpfn" from source
        "EnvSettingsSource"
          -> JSONDecodeError: Expecting value: line 1 column 1 (char 0)

    The traceback points at pydantic and names no token, so the cause is easy to
    misread as a broken install. It also only appears in a kernel that starts *after*
    the credential is configured, which makes it look intermittent.

    The same applies to a bare `HUGGINGFACE` variable, which is not the name
    huggingface_hub reads (`HF_TOKEN`) and is needed for the gated V3 repositories.

    Returns the names of the variables set (never values).
    """
    moved: list[str] = []
    for bare, targets in (("TABPFN", ("TABPFN_TOKEN",)),
                          ("HUGGINGFACE", ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))):
        value = os.environ.pop(bare, "").strip()
        if not value:
            continue
        for target in targets:
            if not os.environ.get(target, "").strip():
                os.environ[target] = value
                moved.append(target)
    return moved


def token_status() -> str:
    """Human-readable, value-free description of where the token is coming from."""
    tok = os.environ.get("TABPFN_TOKEN", "").strip()
    if tok:
        return f"TABPFN_TOKEN set in environment (len={len(tok)}, ends ...{tok[-4:]})"
    for p in (Path.home() / ".cache" / "tabpfn" / "auth_token",
              Path.home() / ".tabpfn" / "token"):
        if p.is_file() and p.read_text().strip():
            return f"cached token file: {p}"
    return ("no token found -- TabPFN-3 will open a browser for licence acceptance on "
            "first use, or use the TabPFN-V2 arm, whose weights need no token")


# Loaded on import so that `python -m tabpfn_prrt.run` picks up `.env` without ceremony.
# normalise_token_vars() must run BEFORE anything imports tabpfn -- see its docstring.
normalise_token_vars()
load_dotenv()
