"""Where the measured data lives -- one definition, so every script agrees.

The data root is the unpacked data bundle: results/ with the MegaMoE stage
measurements, the routing distributions, and the AIC systems overlay. It is
never inferred from the checkout, because the measurements are not in git.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV = "AFD_MEGAMOE_ROOT"


def data_root() -> Path:
    value = os.environ.get(ENV)
    if not value:
        raise SystemExit(
            f"{ENV} is not set. Point it at the unpacked data bundle -- the "
            "directory that contains results/ and the routing JSONs. See "
            ".claude/skills/aic-afd-megamoe/SKILL.md."
        )
    root = Path(value).expanduser().resolve()
    if not (root / "results").is_dir():
        raise SystemExit(f"{ENV}={root} has no results/ directory -- is that the bundle root?")
    return root
