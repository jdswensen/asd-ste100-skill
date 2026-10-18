#!/usr/bin/env python3
"""Thin shim. The real implementation lives in the importable ste_lint module
at the repo root. This file preserves the historical scripts/ste-lint.py
invocation path used by SKILL.md and README.md.
"""
import os
import sys

# Put the repo root on sys.path so this runs whether invoked as
# ./scripts/ste-lint.py or `python scripts/ste-lint.py`, from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ste_lint import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
