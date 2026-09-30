#!/usr/bin/env python3
"""MiMo adapter for the frozen Q200v2 runner.

The published kit hard-requires Qwen kwargs (enable_thinking, thinking,
reasoning_effort=low). MiMo's template has no reasoning_effort. PROCEDURES.md
says that case gets a thin adapter: override the validator, keep the graders.
This file does not modify the hash-pinned kit.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping


KIT = Path("/home/r0b0tdgx/projects/r0b0bench/subsets/q200v2/scripts/run_quality_set.py")
ALLOWED = {"enable_thinking": False, "thinking": False}


def validate_mimo_think_off(chat_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(chat_kwargs)
    if values != ALLOWED:
        raise ValueError(
            "MiMo Q200 adapter accepts only "
            '{"enable_thinking": false, "thinking": false}; got '
            + repr(values)
        )
    return values


def main() -> int:
    sys.path.insert(0, str(KIT.parent))
    spec = importlib.util.spec_from_file_location("q200_run_quality_set", KIT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {KIT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_native_thinking = validate_mimo_think_off
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
