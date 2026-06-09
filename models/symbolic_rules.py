"""
Backwards-compat shim. The canonical home for symbolic rule verification is now
[symdrive/rlib1.0/verifier.py](../symdrive/rlib1.0/verifier.py).

This shim preserves `from models.symbolic_rules import ...` imports.
"""

from symdrive.rlib1_0.verifier import *  # noqa: F401, F403
from symdrive.rlib1_0.verifier import (  # noqa: F401 — re-export commonly imported names
    SymbolicSchema,
    SymbolicParser,
    SymbolicValidator,
    compute_symbolic_complexity,
)
