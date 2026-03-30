#!/usr/bin/env python3
"""Backward-compatible wrapper: mutation_combo_greedy.py --driver ddilog."""

import os
import sys

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    os.execv(
        sys.executable,
        [
            sys.executable,
            os.path.join(root, "mutation_combo_greedy.py"),
            "--driver",
            "ddilog",
        ]
        + sys.argv[1:],
    )
