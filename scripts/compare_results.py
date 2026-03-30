#!/usr/bin/env python3
"""
Compare a candidate results CSV against a baseline CSV (hex-encoded scalars).

CSV layout:
  Line 1: header — either "id,real hex" (real output) or "id,real hex,imag hex" (complex).
  Line 2: run metadata, must start with '#'. Space-separated key=value tokens, e.g.:
          # target_id=ddilog seed=1 batch_size=64 x_min=-4.0 x_max=4.0
  Following lines: one sample per row.

Baseline and candidate must carry the same run metadata (target, seed, batch, input domain
keys when present); otherwise comparison is refused.

Precise digits follow the logic of qcdloop error_analysis plot_precision_analysis.py
calculate_precise_digits (FP64-style), with max 16 digits and 1e-16 relative floor.

Pass (complex): min_i digits(real) >= threshold AND min_i digits(imag) >= threshold.
Pass (real): min_i digits(real) >= threshold.
Rows with NaN in baseline or candidate for a component are skipped for that component;
if no finite rows remain, exit with error.
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


def classify_header(header: List[str]) -> str:
    if len(header) == 2:
        return "real"
    if len(header) == 3:
        return "complex"
    raise ValueError(f"expected 2 or 3 columns, got {len(header)}: {header}")


def validate_and_pair(
    base_rows: List[List[str]],
    cand_rows: List[List[str]],
    kind: str,
    batch_size: int,
) -> None:
    if len(base_rows) != batch_size:
        raise ValueError(f"baseline row count {len(base_rows)} != batch_size metadata {batch_size}")
    if len(cand_rows) != batch_size:
        raise ValueError(f"candidate row count {len(cand_rows)} != batch_size metadata {batch_size}")
    if len(base_rows) != len(cand_rows):
        raise ValueError("baseline and candidate data row counts differ")

    for i, (br, cr) in enumerate(zip(base_rows, cand_rows)):
        if len(br) != len(cr):
            raise ValueError(f"row {i}: column count mismatch")
        if br[0] != cr[0]:
            raise ValueError(f"row {i}: id mismatch {br[0]!r} vs {cr[0]!r}")
        if kind == "real" and len(br) != 2:
            raise ValueError(f"row {i}: expected 2 columns for real mode")
        if kind == "complex" and len(br) != 3:
            raise ValueError(f"row {i}: expected 3 columns for complex mode")


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

    try:
        assert_comparable_meta(m0, m1, str(args.baseline), str(args.candidate))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    kind = classify_header(h0)
    batch_size = int(m0["batch_size"])
    try:
        validate_and_pair(r0, r1, kind, batch_size)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    digits_real: List[float] = []
    digits_imag: List[float] = []

    for i, (br, cr) in enumerate(zip(r0, r1)):
        if kind == "real":
            tb = hex_to_float(br[1])
            tc = hex_to_float(cr[1])
            if math.isnan(tb) or math.isnan(tc) or math.isinf(tb) or math.isinf(tc):
                continue
            err = abs(tc - tb)
            digits_real.append(calculate_precise_digits(tb, err))
        else:
            tb_r = hex_to_float(br[1])
            tb_i = hex_to_float(br[2])
            tc_r = hex_to_float(cr[1])
            tc_i = hex_to_float(cr[2])
            skip_r = (
                math.isnan(tb_r)
                or math.isnan(tc_r)
                or math.isinf(tb_r)
                or math.isinf(tc_r)
            )
            skip_i = (
                math.isnan(tb_i)
                or math.isnan(tc_i)
                or math.isinf(tb_i)
                or math.isinf(tc_i)
            )
            if not skip_r:
                digits_real.append(calculate_precise_digits(tb_r, abs(tc_r - tb_r)))
            if not skip_i:
                digits_imag.append(calculate_precise_digits(tb_i, abs(tc_i - tb_i)))

    th = args.min_digits
    if kind == "real":
        if not digits_real:
            print("error: no finite rows left after skipping NaN/Inf", file=sys.stderr)
            return 2
        mnr = min(digits_real)
        ok = mnr >= th
        print(f"mode=real samples_used={len(digits_real)} min_precise_digits={mnr:.6g} threshold={th}")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    if not digits_real or not digits_imag:
        print(
            "error: need at least one finite sample for both real and imag after skipping NaN/Inf",
            file=sys.stderr,
        )
        return 2
    mnr = min(digits_real)
    mni = min(digits_imag)
    ok = mnr >= th and mni >= th
    print(
        f"mode=complex samples_used_real={len(digits_real)} samples_used_imag={len(digits_imag)} "
        f"min_precise_digits_real={mnr:.6g} min_precise_digits_imag={mni:.6g} threshold={th}"
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
