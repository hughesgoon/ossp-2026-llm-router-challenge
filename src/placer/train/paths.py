from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

REPO = Path(os.environ.get("OSSP_REPO", PACKAGE_ROOT))
ARTIFACTS = Path(os.environ.get("OSSP_ARTIFACTS", PACKAGE_ROOT / "artifacts"))
DATA = Path(os.environ.get("OSSP_DATA", REPO / "data"))


def ensure_artifacts() -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS
