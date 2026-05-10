"""Wrapper around scripts/compare_results.py."""

import os
import re
import subprocess
import sys


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compare(baseline_csv: str, candidate_csv: str, min_digits: float) -> dict:
    """Compare two result CSVs and return pass/fail + numeric details.

    Returns:
        {pass: bool, min_precise_digits: float|None, log: str, exit_code: int}
    """
    root = _repo_root()
    compare_py = os.path.join(root, "scripts", "compare_results.py")
    result = subprocess.run(
        [
            sys.executable,
            compare_py,
            baseline_csv,
            candidate_csv,
            "--min-digits",
            str(min_digits),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    log = result.stdout or ""

    # compare_results.py emits a single aggregate line: "... min_precise_digits=X ..."
    m = re.search(r"(?<![_a-z])min_precise_digits=([0-9.eE+-]+)", log)
    min_precise_digits = float(m.group(1)) if m else None

    return {
        "pass": result.returncode == 0,
        "min_precise_digits": min_precise_digits,
        "log": log[-4000:],
        "exit_code": result.returncode,
    }
