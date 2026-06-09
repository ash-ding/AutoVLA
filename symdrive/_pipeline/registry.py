"""
Dispatch table: maps a ``cot_style`` config field to the corresponding
``(prompt_module, verifier_module)`` pair.

Folder convention: each design lives under ``symdrive/<style>/`` with two files:
  - ``prompt.py``    — exposes ``get_symbolic_cot_prompt``, ``ego_state_to_qualitative``,
                       ``action_string_to_symbolic``
  - ``verifier.py``  — exposes ``SymbolicParser``, ``SymbolicValidator``,
                       ``compute_symbolic_complexity`` (rlib1.x), and/or ``verify`` (rlib2.x)

Python package names cannot contain ``.`` so the on-disk folder uses ``_`` (e.g.
``rlib1_0``), but the ``cot_style`` config string keeps the dotted human form
(e.g. ``rlib1.0``).

Version discovery is automatic: any subdirectory of ``symdrive/`` whose name
matches ``rlib<digits>(_<digits>)+`` and contains both ``prompt.py`` and
``verifier.py`` is registered. Supports arbitrary-depth sub-version numbers —
``rlib1_0``, ``rlib1_0_2``, ``rlib10_3_5_7`` all qualify. Drop a new version
folder under ``symdrive/`` with the two required modules and it's
immediately usable.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import Tuple


# Matches a versioned rlib directory: `rlib` + digit group + one or more
# `_digits` groups. Same regex used by the .claude/hooks/check_rlib_readme.sh
# hook so the two stay in sync.
_VERSION_DIR_RE = re.compile(r"^rlib\d+(?:_\d+)+$")


@lru_cache(maxsize=1)
def _discover_styles() -> frozenset[str]:
    """Scan ``symdrive/`` for version subdirs and return their dotted names.

    A folder qualifies iff:
      1. its name matches ``rlib<digits>(_<digits>)+``
      2. it contains both ``prompt.py`` and ``verifier.py``

    Cached: filesystem layout is fixed for the lifetime of the process.
    """
    here = os.path.dirname(os.path.abspath(__file__))   # symdrive/_pipeline/
    parent = os.path.dirname(here)                       # symdrive/
    found: set[str] = set()
    for name in os.listdir(parent):
        path = os.path.join(parent, name)
        if not (os.path.isdir(path) and _VERSION_DIR_RE.match(name)):
            continue
        # Require both module files so a registry hit doesn't ImportError later.
        if not (os.path.isfile(os.path.join(path, "prompt.py"))
                and os.path.isfile(os.path.join(path, "verifier.py"))):
            continue
        found.add(name.replace('_', '.'))
    return frozenset(found)


def _style_to_pkg(style: str) -> str:
    """``rlib1.0`` → ``symdrive.rlib1_0``."""
    return f"symdrive.{style.replace('.', '_')}"


def get_cot_style(style: str) -> Tuple[ModuleType, ModuleType]:
    """Return ``(prompt_module, verifier_module)`` for the given style name.

    Raises ``ValueError`` if the style is unknown.
    """
    styles = _discover_styles()
    if style not in styles:
        raise ValueError(
            f"Unknown cot_style {style!r}; valid options: {sorted(styles)}"
        )
    pkg = _style_to_pkg(style)
    return import_module(f"{pkg}.prompt"), import_module(f"{pkg}.verifier")


def valid_styles() -> set[str]:
    """Snapshot of currently-registered styles (for argparse choices, error messages)."""
    return set(_discover_styles())
