#!/usr/bin/env python3
"""Static cost model: what the qcdloop routing is worth on an NVIDIA GB300.

GB300 (Blackwell Ultra) cut FP64 vector throughput hard -- ~1.2-1.3 TFLOPS against
~75-80 TFLOPS FP32 vector, a ~62x ratio.  That ratio is the entire reason emulating
a wider type out of FP32 limbs can pay for itself, and this module works out
whether it actually does, per integral.

METHOD
    cost of one source-level op  =  FP ops counted statically from the vendored
                                    emulation primitive that implements it
    time                         =  sum over ops of (count x cost / throughput)

Op COUNTS are dynamic -- the ``ops`` counters in the characterization report are
incremented per traced journal record (agents/shared/stability_reducer.py), so they
are execution counts, not source counts.  Op COSTS are static, counted here from
third_party/include/{ff,dd,qf}_math.hpp.

Calibration: the counter independently reproduces ff add = 11 and dd multiply = 31,
the textbook Dekker/TwoSum values.  If a header refresh moves those, the counter has
drifted and the numbers below are not trustworthy -- ``--check`` prints them.

WHAT THIS MODEL IS NOT
    A pure instruction-throughput roofline.  No memory traffic, no occupancy, no
    branch divergence.  ff doubles and qf quadruples live register count, which
    plausibly cuts occupancy on complex-heavy kernels; that cost is absent here and
    hurts the qf integrals (B12/B16) worst.  Treat outputs as a ranking and an order
    of magnitude, not a prediction of wall clock.

THE ISO-ACCURACY BASELINE
    Speedup is quoted against the cheapest rung that clears the same tolerance gate,
    not against the app as shipped.  B12 and B16 score 3.69 and 6.57 digits at double
    against a 7.0-digit requirement -- they FAIL at double, so dividing by an
    all-double runtime would compare against a run you would have to throw away.
    Their baseline is dd, which is what this project actually routed them to before
    the qf rung existed.  Every other integral is measured against double.

    Strictly this is iso-TOLERANCE, not iso-accuracy: qf and dd both deliver ~15.8
    digits against a 7.0 requirement, so both overshoot.  The label follows the
    convention (iso- = holding one quantity fixed while measuring another) but the
    quantity actually held fixed is "passes the same gate".
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_INC = _REPO / "third_party" / "include"
_QCD = _REPO / "runs" / "qcdloop"

CHARACTERIZATION = _QCD / "report_smoke.json"
RUN_REPORT = _QCD / "strategy" / "20260813_185337_237840a2" / "report.json"

# --------------------------------------------------------------------- hardware
# Midpoints of the quoted GB300 vector ranges.  TFLOPS are quoted assuming FMA
# (2 flop/instruction), so instruction throughput is TFLOPS/2.  The /2 cancels in
# every ratio taken below, but leaving it explicit keeps the units honest: the cost
# tables are in INSTRUCTIONS, and emulation sequences are mostly non-FMA add/sub.
FP32_TFLOPS, FP64_TFLOPS = 77.5, 1.25
FP32_IPS, FP64_IPS = FP32_TFLOPS / 2, FP64_TFLOPS / 2

# Scalar libm / hardware seed costs, in instructions of the native width.  The
# emulated transcendentals all start from a scalar seed and Newton-polish it.
# FP32 gets SFU help on NVIDIA (MUFU.RSQ/LG2/EX2 + polish); FP64 has no SFU path,
# so every FP64 transcendental is a software polynomial and costs far more.
_SEED32 = {"sqrt": 8, "log": 15, "exp": 15, "atan2": 40, "fabs": 0, "abs": 0,
           "copysign": 1, "ldexpf": 2, "ldexp": 2, "floor": 1, "printf": 0}
_SEED64 = {"sqrt": 20, "log": 50, "exp": 50, "atan2": 90, "fabs": 0, "abs": 0,
           "copysign": 1, "ldexpf": 2, "ldexp": 2, "floor": 1, "printf": 0}

# Native FP64 cost of one source-level op, for the double baseline.  add/sub/mul are
# single instructions; div/sqrt/transcendentals are multi-instruction sequences.
# These are ESTIMATES, not counted -- the largest uncounted assumption in the model.
# Sensitivity: swinging log 20->100 and atan2 35->180 moves the whole-app
# iso-accuracy speedup only across 3.94x -> 3.23x, so the ranking is stable.
DOUBLE_COST = {"add": 1, "sub": 1, "mul": 1, "neg": 0.5, "abs": 0.5,
               "div": 20, "sqrt": 20, "log": 50, "atan2": 90}

# Cost of an emulated primitive, in instructions of its LIMB width.  ff and qf are
# built from FP32 limbs, dd from FP64 limbs -- so dd costs divide by FP64 throughput
# and ff/qf by FP32.  Mixing those units is the easiest way to get this model wrong.
_FAM = {
    "ff": dict(file="ff_math.hpp", seeds=_SEED32, unit="fp32",
               nq_exp=4, eps_exp=1e-15, nq_sc=4, eps_sc=1e-15,
               log_newton=2, angle_newton=3, sq=4, dbl=4),
    "dd": dict(file="dd_math.hpp", seeds=_SEED64, unit="fp64",
               nq_exp=6, eps_exp=1e-32, nq_sc=5, eps_sc=1e-32,
               log_newton=3, angle_newton=3, sq=6, dbl=4),
    "qf": dict(file="qf_math.hpp", seeds=_SEED32, unit="fp32",
               nq_exp=6, eps_exp=1e-28, nq_sc=5, eps_sc=1e-28,
               log_newton=3, angle_newton=3, sq=6, dbl=5),
}

# Characterization op name -> the primitive that implements it.
OPMAP = {"add": "add", "sub": "subtract", "mul": "multiply", "div": "divide",
         "neg": "negate", "abs": "abs", "sqrt": "sqrt", "log": "log",
         "atan2": "angle"}

_FP_TYPES = {"float", "double", "FloatFloat", "DoubleDouble", "QuadFloat",
             "ffloat", "ddouble", "qfloat"}

# div/sqrt inside an emulated primitive are themselves multi-instruction.
_HW = {"add": 1, "mul": 1, "neg": 1, "div": 12, "other": 1}


# ------------------------------------------------------------------ preprocess
def _strip_comments(src: str) -> str:
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", " ", src, flags=re.S))


def _resolve_cuda_arch(src: str) -> str:
    """Keep the __CUDA_ARCH__ (device) branch -- we are costing a GPU, not the host."""
    out, stack = [], []
    for ln in src.split("\n"):
        m = re.match(r"#\s*(ifndef|ifdef|if|else|elif|endif)\b(.*)", ln.strip())
        if m:
            kind, rest = m.group(1), m.group(2)
            if kind in ("ifndef", "ifdef", "if"):
                cuda = "__CUDA_ARCH__" in rest
                if kind == "ifndef" and cuda:
                    keep = False
                elif kind == "ifdef" and cuda:
                    keep = True
                elif kind == "if" and cuda:
                    keep = "defined" in rest and "!" not in rest
                else:
                    keep = True          # unrelated guard: keep both sides
                stack.append([keep, cuda])
            elif kind == "else" and stack and stack[-1][1]:
                stack[-1][0] = not stack[-1][0]
            elif kind == "endif" and stack:
                stack.pop()
            continue
        if all(f[0] for f in stack):
            out.append(ln)
    return "\n".join(out)


_DEF_RE = re.compile(
    r"(?:static\s+)?KOKKOS_INLINE_FUNCTION\s+"
    r"(?P<ret>[A-Za-z_][A-Za-z0-9_:<>,\s\*&]*?)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"\((?P<args>[^;{}]*?)\)\s*(?:const\s*)?\{")

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*(?:[eE][-+]?\d+)?[fF]?|\S")


def _functions(src: str):
    """Yield (name, args, body) with BRACE MATCHING -- a regex to '}' would stop at
    the first nested close and silently truncate every non-trivial primitive."""
    for m in _DEF_RE.finditer(src):
        start = src.index("{", m.end() - 1)
        depth, i = 0, start
        while i < len(src):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        yield m.group("name"), m.group("args"), src[start + 1:i]


def _split_loops(body: str):
    """Return (setup_src, [loop_body, ...]) for top-level for/while loops.

    Load-bearing: the transcendentals are Taylor/Newton loops, and scaling the WHOLE
    body by a trip count would also multiply the one-time argument reduction.
    """
    loops, out, i = [], [], 0
    while i < len(body):
        m = re.compile(r"\b(for|while)\s*\(").search(body, i)
        if not m:
            out.append(body[i:])
            break
        out.append(body[i:m.start()])
        j, d = body.index("(", m.start()), 0
        while j < len(body):
            if body[j] == "(":
                d += 1
            elif body[j] == ")":
                d -= 1
                if d == 0:
                    break
            j += 1
        j += 1
        while j < len(body) and body[j] in " \t\n":
            j += 1
        if j < len(body) and body[j] == "{":
            d, k = 0, j
            while k < len(body):
                if body[k] == "{":
                    d += 1
                elif body[k] == "}":
                    d -= 1
                    if d == 0:
                        break
                k += 1
            loops.append(body[j + 1:k])
            i = k + 1
        else:
            k = body.index(";", j) if ";" in body[j:] else len(body) - 1
            loops.append(body[j:k + 1])
            i = k + 1
    return "".join(out), loops


def _fp_names(args: str, body: str) -> set:
    names = set()
    for d in re.finditer(r"\b(" + "|".join(_FP_TYPES) + r")\s*[&*]?\s*"
                         r"([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)",
                         args + ";" + body):
        names.update(n.strip() for n in d.group(2).split(","))
    return names


def _is_fp(tok: str, fp: set) -> bool:
    return (tok in fp or tok in ("hi", "lo")
            or bool(re.match(r"^\d+\.\d*", tok))
            or (tok.endswith(("f", "F")) and "." in tok))


def _count(src: str, fp: set, known: set):
    """Count FP ops, calls to sibling primitives, and scalar seeds in a fragment.

    Type-aware so loop counters and exponent arithmetic stay out of the FLOP count.
    A ``Kokkos::``-qualified call is a SCALAR seed, not a sibling: ff_math has both
    an emulated ``angle`` and a wrapper ``atan2`` that calls it, so resolving the
    bare name inside ``angle``'s own ``Kokkos::atan2`` seed creates an
    angle -> atan2 -> angle cycle that diverges to ~1e17.
    """
    toks = _TOKEN_RE.findall(src)
    ops, calls, seeds = Counter(), Counter(), Counter()
    for i, t in enumerate(toks):
        prv = toks[i - 1] if i else ""
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        nx2 = toks[i + 2] if i + 2 < len(toks) else ""
        if nxt == "(" and re.match(r"^[A-Za-z_]", t):
            if prv == ":":
                seeds[t] += 1
            elif t in known:
                calls[t] += 1
            elif t in _SEED32:
                seeds[t] += 1
            continue
        if t in "+-*/":
            if prv in ("", "(", ",", "=", "+", "-", "*", "/", "<", ">", "?", ":",
                       "{", "}", ";", "return", "&&", "||", "["):
                if t == "-":
                    ops["neg"] += 1
                continue
            right = (_is_fp(nxt, fp) or nxt == "(" or nx2 in ("hi", "lo")
                     or nxt in known or nxt in _SEED32)
            if not (_is_fp(prv, fp) or right):
                continue
            ops["add" if t in "+-" else ("mul" if t == "*" else "div")] += 1
    return ops, calls, seeds


# ------------------------------------------------------------- trip counts
def _exp_iters(nq: float, eps: float) -> int:
    """Taylor terms for exp after argument reduction: |s| <= ln2/2 * 2^-nq."""
    x, t, n = math.log(2) / 2 * 2.0 ** -nq, 1.0, 0
    while t > eps and n < 200:
        n += 1
        t *= x / n
    return n


def _sincos_iters(nq: float, eps: float) -> int:
    """Taylor terms for sincos after reduction: |r| <= pi * 2^-nq."""
    r, k = math.pi * 2.0 ** -nq, 0
    t = r
    while t > eps and k < 200:
        k += 1
        t *= r * r / ((2 * k) * (2 * k + 1))
    return k


def primitive_costs(fam: str) -> dict:
    """Inclusive cost of every primitive in one family, in limb-width instructions."""
    cfg = _FAM[fam]
    src = _resolve_cuda_arch(_strip_comments((_INC / cfg["file"]).read_text()))
    defs, names = {}, set()
    for n, a, b in _functions(src):
        if "Experimental" in a:          # thin alias wrappers, not the implementation
            continue
        names.add(n)
        defs.setdefault(n, (a, b))

    trips = {"exp": [_exp_iters(cfg["nq_exp"], cfg["eps_exp"]), cfg["sq"]],
             "sincos": [_sincos_iters(cfg["nq_sc"], cfg["eps_sc"]), cfg["dbl"]],
             "log": [cfg["log_newton"]], "angle": [cfg["angle_newton"]]}

    parts = {}
    for n, (a, b) in defs.items():
        fp = _fp_names(a, b) | {"hi", "lo", "x"}
        setup, loops = _split_loops(b)
        tr = trips.get(n, [1] * len(loops))
        segs = [(1.0, _count(setup, fp, names))]
        for idx, lb in enumerate(loops):
            segs.append((float(tr[idx] if idx < len(tr) else 1), _count(lb, fp, names)))
        parts[n] = segs

    seeds = cfg["seeds"]
    cost = {n: 0.0 for n in names}
    for _ in range(80):                  # fixpoint over the (acyclic) call graph
        new = {}
        for n in names:
            tot = 0.0
            for trip, (ops, calls, sds) in parts[n]:
                s = sum(_HW.get(k, 1) * v for k, v in ops.items())
                s += sum(seeds.get(k, 5) * v for k, v in sds.items())
                for c, k in calls.items():
                    # a primitive calling its own name is the scalar libm seed
                    s += (seeds.get(n, 10) if c == n else cost.get(c, 0.0)) * k
                tot += trip * s
            new[n] = tot
        if all(abs(new[n] - cost[n]) < 1e-6 for n in names):
            return new
        cost = new
    return cost


# ---------------------------------------------------------------- op counts
def op_counts_per_integral() -> dict:
    doc = json.loads(CHARACTERIZATION.read_text())
    per = defaultdict(Counter)
    for name, body in doc["integrals"].items():
        regions = body["regions"]
        regions = list(regions.values()) if isinstance(regions, dict) else regions
        for r in regions:
            for k, v in (r.get("ops") or {}).items():
                per[name][k] += v
    return per


def _time(ops: Counter, rung: str, costs: dict) -> float:
    if rung == "double":
        return sum(c * DOUBLE_COST.get(op, 1) / FP64_IPS for op, c in ops.items())
    ips = FP32_IPS if _FAM[rung]["unit"] == "fp32" else FP64_IPS
    return sum(c * costs[rung][OPMAP[op]] / ips for op, c in ops.items())


def per_integral_speedup() -> dict:
    """{integral: {rung, ops, iso, iso_base, vs_double}} on GB300.

    ``iso``       speedup against the cheapest rung clearing the tolerance gate
    ``vs_double`` speedup against the same integral run at double, always
    """
    costs = {f: primitive_costs(f) for f in _FAM}
    per = op_counts_per_integral()
    routing = json.loads(RUN_REPORT.read_text())["tu_routing"]

    out = {}
    for name, rung in routing.items():
        ops = per[name]
        t_now = _time(ops, rung, costs)
        t_dbl = _time(ops, "double", costs)
        # Only the qf integrals needed more than double; they are the sole case where
        # the iso-accuracy baseline is not the double run.
        base = "dd" if rung == "qf" else "double"
        t_base = _time(ops, base, costs) if base != "double" else t_dbl
        out[name] = {"rung": rung, "ops": sum(ops.values()),
                     "iso": t_base / t_now, "iso_base": base,
                     "vs_double": t_dbl / t_now,
                     "t_now": t_now, "t_base": t_base}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="print the calibration values and the cost table")
    args = ap.parse_args()

    costs = {f: primitive_costs(f) for f in _FAM}
    if args.check:
        print(f"calibration  ff add = {costs['ff']['add']:.0f} (expect 11)   "
              f"dd multiply = {costs['dd']['multiply']:.0f} (expect 31)\n")
        print(f"  {'op':6s} {'double FP64':>12s} {'ff FP32':>9s} "
              f"{'qf FP32':>9s} {'dd FP64':>9s}")
        for op in ("add", "sub", "mul", "div", "neg", "abs", "sqrt", "log", "atan2"):
            p = OPMAP[op]
            print(f"  {op:6s} {DOUBLE_COST[op]:12.0f} {costs['ff'][p]:9.0f} "
                  f"{costs['qf'][p]:9.0f} {costs['dd'][p]:9.0f}")
        print()

    res = per_integral_speedup()
    order = sorted(res, key=lambda n: (re.match(r"^([A-Za-z]+)", n).group(1),
                                       int(re.search(r"(\d+)$", n).group(1))))
    print(f"{'integral':9s} {'rung':7s} {'ops':>10s} {'iso base':>9s} "
          f"{'iso':>8s} {'vs double':>10s}")
    tb = tn = 0.0
    for n in order:
        r = res[n]
        tb += r["t_base"]
        tn += r["t_now"]
        print(f"{n:9s} {r['rung']:7s} {r['ops']:10,d} {r['iso_base']:>9s} "
              f"{r['iso']:7.2f}x {r['vs_double']:9.2f}x")
    print(f"\nwhole app, iso-accuracy: {tb / tn:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
