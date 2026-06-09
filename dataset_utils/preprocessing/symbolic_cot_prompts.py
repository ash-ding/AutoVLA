"""
Backwards-compat shim. The canonical home for the rlib1.0 symbolic CoT prompt is
now [symdrive/rlib1.0/prompt.py](../../symdrive/rlib1.0/prompt.py).

This shim preserves `from dataset_utils.preprocessing.symbolic_cot_prompts import ...`
imports.
"""

from symdrive.rlib1_0.prompt import *  # noqa: F401, F403
from symdrive.rlib1_0.prompt import (  # noqa: F401 — commonly imported names
    action_string_to_symbolic,
    ego_state_to_qualitative,
    get_symbolic_cot_prompt,
)
