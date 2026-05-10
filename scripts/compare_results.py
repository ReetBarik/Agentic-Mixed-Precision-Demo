#!/usr/bin/env python3
"""
Compare a candidate results CSV against a baseline CSV (hex-encoded float64 scalars).

CSV layout:
  Line 1: header — "id" plus one column per scalar value being verified.
          Column names are informational; every value column is interpreted as a
          float64 IEEE bit pattern. Complex outputs are split into two columns by
          the producing driver (typically `<name>__re`, `<name>__im`).
  Line 2: run metadata, must start with '#'. Space-separated key=value tokens, e.g.:
          # target_id=ddilog seed=1 batch_size=64 x_min=-4.0 x_max=4.0
  Following lines: one sample per row.

Baseline and candidate must carry the same run metadata (target, seed, batch, input
domain keys when present); otherwise comparison is refused.

Precise digits follow the logic of qcdloop error_analysis plot_precision_analysis.py
calculate_precise_digits (FP64-style), with max 16 digits and 1e-16 relative floor.

Pass: per-column min digits across samples >= threshold for EVERY value column.
Rows with NaN/Inf in baseline or candidate for a given column are skipped for that
column; if any column ends up with no finite samples, exit with error.
"""

import argparse
import csv
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple


META_KEYS_REQUIRED = frozenset({"target_id", "seed", "batch_size"})
# If any optional key appears in either file, both must define it and match.
META_KEYS_OPTIONAL_DOMAIN = frozenset({"x_min", "x_max"})


def hex_to_float(hex_str: str) -> float:
    """Interpret hex payload as IEEE-754; <=8 hex digits -> float32 else float64 (csv_parser logic)."""
    s = hex_str.strip()
    if s.lower().startswith("0x"):
        s = s[2:]
    if not s:
        return float("nan")
    hex_digits = len(s)
    hex_value = int(s, 16)
    if hex_digits <= 8:
        packed = struct.pack("I", hex_value & 0xFFFFFFFF)
        return struct.unpack("f", packed)[0]
    packed = struct.pack("Q", hex_value & 0xFFFFFFFFFFFFFFFF)
    return struct.unpack("d", packed)[0]


def calculate_precise_digits(true_value: float, absolute_error: float) -> float:
    """FP64-style cap at 16 decimal digits; floor scale 1e-16 (see plot_precision_analysis.py)."""
    max_precise = 16

    if (
        math.isnan(true_value)
        or math.isnan(absolute_error)
        or math.isinf(true_value)
        or math.isinf(absolute_error)
    ):
        return 0.0

    if absolute_error > abs(true_value):
        return 0.0

    if true_value == 0.0 and absolute_error == 0.0:
        return float(max_precise)

    if absolute_error == 0.0:
        return float(max_precise)

    if true_value == 0.0:
        return 0.0

    min_representable_error = abs(true_value) * 1e-16
    if absolute_error < min_representable_error:
        return float(max_precise)

    return -math.log10(abs(absolute_error) / abs(true_value))


def parse_meta_line(line: str) -> Dict[str, str]:
    if not line.startswith("#"):
        raise ValueError(f"expected metadata line starting with '#', got: {line[:80]!r}")
    raw = line[1:].strip()
    out: Dict[str, str] = {}
    for token in raw.split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def assert_comparable_meta(base: Dict[str, str], cand: Dict[str, str], path_a: str, path_b: str) -> None:
    for k in META_KEYS_REQUIRED:
        if k not in base:
            raise ValueError(f"{path_a}: metadata missing required key {k!r}")
        if k not in cand:
            raise ValueError(f"{path_b}: metadata missing required key {k!r}")
        if base[k] != cand[k]:
            raise ValueError(
                f"refusing compare: metadata mismatch for {k!r}: "
                f"baseline {base[k]!r} vs candidate {cand[k]!r} "
                f"(different runs for this target must not be compared)"
            )

    optional_present = (
        META_KEYS_OPTIONAL_DOMAIN
        & set(base.keys())
        | META_KEYS_OPTIONAL_DOMAIN & set(cand.keys())
    )
    for k in sorted(optional_present):
        if k not in base:
            raise ValueError(
                f"{path_a}: missing optional domain key {k!r} present in other file — incomparable"
            )
        if k not in cand:
            raise ValueError(
                f"{path_b}: missing optional domain key {k!r} present in other file — incomparable"
            )
        if base[k] != cand[k]:
            raise ValueError(
                f"refusing compare: metadata mismatch for {k!r}: "
                f"baseline {base[k]!r} vs candidate {cand[k]!r}"
            )


def load_csv(path: Path) -> Tuple[List[str], Dict[str, str], List[List[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path}: need header, metadata line, and at least one data row")

    header = next(csv.reader([lines[0]]))
    header = [h.strip() for h in header]
    meta = parse_meta_line(lines[1])

    data_rows: List[List[str]] = []
    for line in lines[2:]:
        if not line.strip():
            continue
        row = next(csv.reader([line]))
        row = [c.strip() for c in row]
        if not any(row):
            continue
        data_rows.append(row)

    return header, meta, data_rows


def validate_and_pair(
    base_rows: List[List[str]],
    cand_rows: List[List[str]],
    expected_cols: int,
    batch_size: int,
) -> None:
    if len(base_rows) != batch_size:
        raise ValueError(f"baseline row count {len(base_rows)} != batch_size metadata {batch_size}")
    if len(cand_rows) != batch_size:
        raise ValueError(f"candidate row count {len(cand_rows)} != batch_size metadata {batch_size}")
    if len(base_rows) != len(cand_rows):
        raise ValueError("baseline and candidate data row counts differ")

    for i, (br, cr) in enumerate(zip(base_rows, cand_rows)):
        if len(br) != expected_cols:
            raise ValueError(f"row {i} (baseline): expected {expected_cols} columns, got {len(br)}")
        if len(cr) != expected_cols:
            raise ValueError(f"row {i} (candidate): expected {expected_cols} columns, got {len(cr)}")
        if br[0] != cr[0]:
            raise ValueError(f"row {i}: id mismatch {br[0]!r} vs {cr[0]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate candidate CSV against baseline (hex scalars).")
    ap.add_argument("baseline", type=Path, help="Baseline CSV path")
    ap.add_argument("candidate", type=Path, help="Candidate CSV path")
    ap.add_argument(
        "--min-digits",
        type=float,
        default=10.0,
        help="Minimum precise digits required (default: 10)",
    )
    args = ap.parse_args()

    try:
        h0, m0, r0 = load_csv(args.baseline)
        h1, m1, r1 = load_csv(args.candidate)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if h0 != h1:
        print(
            f"error: header mismatch: baseline {h0!r} vs candidate {h1!r}",
            file=sys.stderr,
        )
        return 2

    if len(h0) < 2 or h0[0] != "id":
        print(f"error: header must be 'id,<col1>,...'; got {h0!r}", file=sys.stderr)
        return 2

    value_cols = h0[1:]
    expected_cols = len(h0)

    try:
        assert_comparable_meta(m0, m1, str(args.baseline), str(args.candidate))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    batch_size = int(m0["batch_size"])
    try:
        validate_and_pair(r0, r1, expected_cols, batch_size)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Per-column digit lists
    per_col_digits: List[List[float]] = [[] for _ in value_cols]

    for br, cr in zip(r0, r1):
        for col_idx in range(len(value_cols)):
            tb = hex_to_float(br[col_idx + 1])
            tc = hex_to_float(cr[col_idx + 1])
            if math.isnan(tb) or math.isnan(tc) or math.isinf(tb) or math.isinf(tc):
                continue
            per_col_digits[col_idx].append(calculate_precise_digits(tb, abs(tc - tb)))

    # Every column must have at least one finite sample
    for name, digits in zip(value_cols, per_col_digits):
        if not digits:
            print(
                f"error: column {name!r} has no finite samples after skipping NaN/Inf",
                file=sys.stderr,
            )
            return 2

    per_col_min = [min(d) for d in per_col_digits]
    aggregate_min = min(per_col_min)
    th = args.min_digits
    ok = aggregate_min >= th

    cols_summary = " ".join(
        "{0}:{1:.6g}".format(name, mn) for name, mn in zip(value_cols, per_col_min)
    )
    print(
        "samples_used={n} columns=[{cols}] min_precise_digits={agg:.6g} threshold={th}".format(
            n=batch_size,
            cols=cols_summary,
            agg=aggregate_min,
            th=th,
        )
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
