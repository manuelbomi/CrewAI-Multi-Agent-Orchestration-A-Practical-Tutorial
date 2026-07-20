#!/usr/bin/env python
"""Run every tutorial module in order. Useful as a smoke test and as a
single command to see the whole tutorial execute.

Usage:
    python run_all.py
"""
import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

MODULES = [
    "m1_agents_tasks_crews",
    "m2_sequential_process",
    "m3_custom_tools",
    "m4_hierarchical_process",
    "m5_grounded_agent",
]

if __name__ == "__main__":
    for name in MODULES:
        runpy.run_module(name, run_name="__main__")
    print("\nAll tutorial modules ran successfully.")
