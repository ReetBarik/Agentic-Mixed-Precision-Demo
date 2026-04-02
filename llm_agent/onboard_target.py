#!/usr/bin/env python3
"""Spec-driven onboarding + validation with ephemeral generated drivers."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from llm_agent.argo_client import ArgoClient


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _write_text(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _run(cmd, cwd, env=None):
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        return p.returncode, p.stdout or ""
    except FileNotFoundError as exc:
        return 127, "error: command not found: {0}".format(exc.filename)


def _run_with_prepare(root, cmd, cwd, env=None):
    prepare = os.path.join(root, "scripts", "prepare.sh")
    if not os.path.isfile(prepare):
        return 127, "error: missing prepare script: {0}".format(prepare)
    kokkos_setup = os.path.join(root, "kokkos", "setup.sh")
    quoted_cmd = " ".join(shlex.quote(x) for x in cmd)
    prefix = "source {0}".format(shlex.quote(prepare))
    if os.path.isfile(kokkos_setup):
        prefix += " && source {0}".format(shlex.quote(kokkos_setup))
    shell_cmd = "{0} && {1}".format(prefix, quoted_cmd)
    return _run(["bash", "-lc", shell_cmd], cwd=cwd, env=env)


def _mk_step(step, ok, errors=None, warnings=None, artifacts=None, details=None):
    return {
        "step": step,
        "ok": bool(ok),
        "errors": errors or [],
        "warnings": warnings or [],
        "artifacts": artifacts or {},
        "details": details or {},
    }


def _run_project_build_pipeline(root):
    prepare = os.path.join(root, "scripts", "prepare.sh")
    build_with_kokkos = os.path.join(root, "scripts", "build_with_Kokkos.sh")
    compile_sh = os.path.join(root, "scripts", "compile.sh")
    kokkos_setup = os.path.join(root, "kokkos", "setup.sh")
    cache = os.path.join(root, "build", "CMakeCache.txt")

    if not os.path.isfile(prepare):
        return (
            127,
            "error: missing prepare script: {0}".format(prepare),
            "missing",
        )
    can_repeat = os.path.isfile(kokkos_setup) and os.path.isfile(cache)
    if can_repeat:
        if not os.path.isfile(compile_sh):
            return (
                127,
                "error: missing compile script: {0}".format(compile_sh),
                "missing",
            )
        shell_cmd = "source {0} && source {1} {2}".format(
            shlex.quote(prepare),
            shlex.quote(compile_sh),
            shlex.quote(root),
        )
        rc, out = _run(["bash", "-lc", shell_cmd], cwd=root)
        if rc == 0:
            return rc, out, "repeat"
        if not os.path.isfile(build_with_kokkos):
            return rc, out, "repeat"
        shell_cmd_bootstrap = "source {0} && source {1} {2}".format(
            shlex.quote(prepare),
            shlex.quote(build_with_kokkos),
            shlex.quote(root),
        )
        rc_boot, out_boot = _run(["bash", "-lc", shell_cmd_bootstrap], cwd=root)
        if rc_boot == 0:
            merged = (
                "repeat_failed_then_bootstrap_succeeded\n"
                + out[-4000:]
                + "\n--- bootstrap ---\n"
                + out_boot[-4000:]
            )
            return 0, merged, "bootstrap_fallback"
        merged = (
            "repeat_failed_then_bootstrap_failed\n"
            + out[-4000:]
            + "\n--- bootstrap ---\n"
            + out_boot[-4000:]
        )
        return rc_boot, merged, "bootstrap_fallback"

    if not os.path.isfile(build_with_kokkos):
        return (
            127,
            "error: missing bootstrap script: {0}".format(build_with_kokkos),
            "missing",
        )
    shell_cmd = "source {0} && source {1} {2}".format(
        shlex.quote(prepare),
        shlex.quote(build_with_kokkos),
        shlex.quote(root),
    )
    rc, out = _run(["bash", "-lc", shell_cmd], cwd=root)
    return rc, out, "bootstrap"


def _validate_target_id(target_id):
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*", target_id))


def _parse_min_precise_digits(text):
    m = re.search(r"min_precise_digits=([0-9.eE+-]+)", text or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extract_cpp(text):
    s = (text or "").strip()
    m = re.search(r"```(?:cpp|c\+\+|c)?\s*(.*?)```", s, flags=re.S | re.I)
    if m:
        return m.group(1).strip() + "\n"
    return s + ("\n" if s and not s.endswith("\n") else "")


def _enforce_csv_contract(driver_src):
    """
    Enforce CSV contract used by compare_results.py:
      line 1 emitted in runtime file must be header
      line 2 emitted in runtime file must be metadata starting '#'
    """
    lines = driver_src.splitlines()
    header_idx = -1
    meta_start = -1
    meta_end = -1

    for i, ln in enumerate(lines):
        s = ln.strip()
        if "out << \"id,real hex" in s or "out << \"id,real hex,imag hex" in s:
            header_idx = i
        if "out << \"# target_id=" in s and meta_start < 0:
            meta_start = i
            j = i
            while j < len(lines):
                if "; " in lines[j] or lines[j].rstrip().endswith(";"):
                    meta_end = j
                    break
                j += 1
            if meta_end < 0:
                meta_end = i

    if header_idx < 0 or meta_start < 0:
        return driver_src, []

    warnings = []
    if meta_start < header_idx:
        # Move metadata block immediately after header line.
        meta_block = lines[meta_start : meta_end + 1]
        kept = lines[:meta_start] + lines[meta_end + 1 :]
        # header index shifts left if metadata was before it.
        new_header_idx = header_idx - len(meta_block)
        fixed = kept[: new_header_idx + 1] + meta_block + kept[new_header_idx + 1 :]
        warnings.append("W_DRIVER_CONTRACT_FIXED: moved metadata line after header")
        return "\n".join(fixed) + "\n", warnings

    return driver_src, warnings


def _render_driver_source(spec):
    target_id = spec["id"]
    output_mode = spec["output_mode"]
    inputs = spec.get("inputs") or [
        {
            "name": "x",
            "ctype": "TMass",
            "distribution": "uniform_real",
            "min": spec["x_min"],
            "max": spec["x_max"],
        }
    ]
    call_expr = (spec.get("call") or {}).get("expression", "").strip()
    if not call_expr:
        fn = spec["function_symbol"]
        call_expr = "ql::{0}<TOutput, TMass, TScale>({x})".format(fn)

    view_decl_lines = []
    mirror_decl_lines = []
    dist_decl_lines = []
    fill_lines = []
    copy_lines = []
    meta_pairs = []
    call_eval = call_expr

    for inp in inputs:
        name = inp["name"]
        ctype = inp.get("ctype", "TMass")
        lo = float(inp.get("min", -4.0))
        hi = float(inp.get("max", 4.0))
        call_eval = call_eval.replace("{" + name + "}", "{0}_d(i)".format(name))

        view_decl_lines.append(
            '        Kokkos::View<{0}*> {1}_d("{1}", batch_size);'.format(ctype, name)
        )
        mirror_decl_lines.append(
            "        auto {0}_h = Kokkos::create_mirror_view({0}_d);".format(name)
        )
        dist_decl_lines.append(
            "        std::uniform_real_distribution<double> dist_{0}({1}, {2});".format(
                name, lo, hi
            )
        )
        fill_lines.append(
            "            {0}_h(i) = static_cast<{1}>(dist_{0}(rng));".format(name, ctype)
        )
        copy_lines.append("        Kokkos::deep_copy({0}_d, {0}_h);".format(name))
        meta_pairs.append("{0}_min={1} {0}_max={2}".format(name, lo, hi))

    write_line = (
        "            ql::printDoubleBits(y_h(i), out);\n"
        if output_mode == "real"
        else (
            "            ql::printDoubleBits(y_h(i).real(), out);\n"
            "            out << ',';\n"
            "            ql::printDoubleBits(y_h(i).imag(), out);\n"
        )
    )
    header = "id,real hex" if output_mode == "real" else "id,real hex,imag hex"
    y_view_decl = (
        "Kokkos::View<TMass*> y_d(\"y\", batch_size);"
        if output_mode == "real"
        else "Kokkos::View<TOutput*> y_d(\"y\", batch_size);"
    )
    meta_suffix = (" " + " ".join(meta_pairs)) if meta_pairs else ""

    return """#include <Kokkos_Core.hpp>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "kokkosUtils.h"

// AUTO-GENERATED EPHEMERAL DRIVER FROM TARGET SPEC.
using TOutput = Kokkos::complex<double>;
using TMass = double;
using TScale = double;

int main(int argc, char* argv[]) {{
    Kokkos::initialize(argc, argv);
    {{
        if (argc < 2) {{
            std::cerr << "usage: " << argv[0] << " <batch_size> [output.csv] [seed]\\n";
            Kokkos::finalize();
            return 1;
        }}

        const std::size_t batch_size = static_cast<std::size_t>(std::stoull(argv[1]));
        const std::string out_path = (argc >= 3) ? argv[2] : "{target_id}_out.csv";
        std::uint64_t seed = 1;
        if (argc >= 4) {{
            seed = static_cast<std::uint64_t>(std::stoull(argv[3]));
        }}

{input_views}
        {y_view_decl}

{input_mirrors}
        std::mt19937_64 rng(seed);
{input_dists}
        for (std::size_t i = 0; i < batch_size; ++i) {{
{input_fill}
        }}
{input_copy}

        Kokkos::parallel_for(
            "{target_id}_batch",
            Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, batch_size),
            KOKKOS_LAMBDA(std::size_t i) {{
                y_d(i) = {call_eval};
            }});
        Kokkos::fence();

        auto y_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y_d);

        std::ofstream out(out_path);
        if (!out) {{
            std::cerr << "failed to open " << out_path << '\\n';
            Kokkos::finalize();
            return 1;
        }}
        out << "{header}\\n";
        out << "# target_id={target_id} seed=" << seed << " batch_size=" << batch_size
            << "{meta_suffix}\\n";
        for (std::size_t i = 0; i < batch_size; ++i) {{
            out << i << ',';
{write_line}            out << '\\n';
        }}
    }}
    Kokkos::finalize();
    return 0;
}}
""".format(
        target_id=target_id,
        header=header,
        input_views="\n".join(view_decl_lines),
        input_mirrors="\n".join(mirror_decl_lines),
        input_dists="\n".join(dist_decl_lines),
        input_fill="\n".join(fill_lines),
        input_copy="\n".join(copy_lines),
        y_view_decl=y_view_decl,
        call_eval=call_eval,
        meta_suffix=meta_suffix,
        write_line=write_line,
    )


def _render_llm_driver_messages(spec, header_text, previous_code="", feedback=""):
    system = (
        "You generate C++ Kokkos drivers.\n"
        "Return ONLY complete C++ source code (no markdown, no explanations).\n"
        "The code must compile with C++17 and include kokkosUtils.h.\n"
        "Use CLI: <batch_size> [output.csv] [seed].\n"
    )
    user = (
        "Target id: {id}\n"
        "Header path: {header_path}\n"
        "Function symbol: {function_symbol}\n"
        "Input x range: [{x_min}, {x_max}]\n"
        "Output mode: {output_mode}\n"
        "locals_for_downcast: {locals}\n\n"
        "Driver requirements:\n"
        "- Include Kokkos_Core.hpp and kokkosUtils.h\n"
        "- Use TOutput=Kokkos::complex<double>, TMass=double, TScale=double\n"
        "- In real mode, use Kokkos::View<TMass*> y_d and write 'id,real hex'\n"
        "- In complex mode, use Kokkos::View<TOutput*> y_d and write 'id,real hex,imag hex'\n"
        "- Use ql::printDoubleBits(...) for CSV hex output\n"
        "- CSV output order MUST be: line1 header, line2 metadata starting '#'\n"
        "- Emit metadata line: # target_id=<id> seed=<seed> batch_size=<batch> x_min=<x_min> x_max=<x_max>\n"
        "- Keep style close to driver/ddilog_driver.cc\n\n"
        "{prev_block}"
        "{feedback_block}"
        "Header content:\n"
        "===== BEGIN HEADER =====\n"
        "{header}\n"
        "===== END HEADER =====\n"
    ).format(
        id=spec["id"],
        header_path=spec["header_path"],
        function_symbol=spec["function_symbol"],
        x_min=spec["x_min"],
        x_max=spec["x_max"],
        output_mode=spec["output_mode"],
        locals=", ".join(spec.get("locals_for_downcast", [])) or "(none)",
        prev_block=(
            "Previous draft to revise:\n{0}\n\n".format(previous_code)
            if previous_code
            else ""
        ),
        feedback_block=(
            "Compiler feedback to fix:\n{0}\n\n".format(feedback) if feedback else ""
        ),
        header=header_text,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _render_ephemeral_cmakelists(root, target_id, driver_src_abs):
    return """cmake_minimum_required(VERSION 3.16)
project(EphemeralOnboardValidate LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Kokkos REQUIRED)
add_executable({target} "{driver_src}")
target_include_directories({target} PRIVATE "{src_dir}")
target_link_libraries({target} PRIVATE Kokkos::kokkos)
""".format(
        target=target_id + "_driver",
        driver_src=driver_src_abs,
        src_dir=os.path.join(root, "src"),
    )


def _load_spec(args, root):
    if args.spec_file:
        payload = _load_json(args.spec_file)
        targets = payload.get("targets") if isinstance(payload, dict) else None
        if isinstance(targets, list):
            for t in targets:
                if t.get("id") == args.target_id:
                    return t
            return None
        if isinstance(payload, dict) and payload.get("id") == args.target_id:
            return payload
        return None
    return {
        "id": args.target_id,
        "header_path": args.header_path,
        "function_symbol": args.function_symbol,
        "locals_for_downcast": [x.strip() for x in args.locals.split(",") if x.strip()],
        "x_min": args.x_min,
        "x_max": args.x_max,
        "output_mode": args.output_mode,
    }


def _normalize_spec(raw):
    if raw is None:
        return None, ["E_SPEC_NOT_FOUND: target id not found in spec"]
    spec = dict(raw)
    errs = []
    spec.setdefault("header_path", "src/kokkosUtils.h")
    spec.setdefault("x_min", -4.0)
    spec.setdefault("x_max", 4.0)
    spec.setdefault("output_mode", "real")
    spec.setdefault("locals_for_downcast", [])
    if not spec.get("id"):
        errs.append("E_SPEC_ID_REQUIRED: missing id")
    if not spec.get("function_symbol"):
        errs.append("E_SPEC_FUNCTION_REQUIRED: missing function_symbol")
    if spec.get("output_mode") not in ("real", "complex"):
        errs.append("E_SPEC_OUTPUT_MODE: output_mode must be real|complex")
    return spec, errs


def main():
    ap = argparse.ArgumentParser(
        description="Spec-driven onboarding/validation with ephemeral generated drivers."
    )
    ap.add_argument("--target-id", required=True, help="Target id to load/validate")
    ap.add_argument(
        "--spec-file",
        default="",
        help="Optional JSON spec path. Accepts single object or {targets:[...]}",
    )
    ap.add_argument("--function-symbol", default="", help="Used if --spec-file is omitted")
    ap.add_argument("--header-path", default="src/kokkosUtils.h", help="Used without --spec-file")
    ap.add_argument("--locals", default="", help="Comma-separated locals_for_downcast (optional)")
    ap.add_argument("--x-min", type=float, default=-4.0, help="Used without --spec-file")
    ap.add_argument("--x-max", type=float, default=4.0, help="Used without --spec-file")
    ap.add_argument(
        "--output-mode",
        default="real",
        choices=["real", "complex"],
        help="CSV output format mode",
    )
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-digits", type=float, default=10.0)
    ap.add_argument(
        "--generator",
        default="deterministic",
        choices=["deterministic", "llm"],
        help="Driver generator mode",
    )
    ap.add_argument("--base-url", default=None, help="Argo proxy base URL for LLM generation")
    ap.add_argument("--user", default=None, help="Argo username override")
    ap.add_argument("--model", default="claudeopus46")
    ap.add_argument("--fallback-model", default="gpt4turbo")
    ap.add_argument(
        "--llm-driver-max-iterations",
        type=int,
        default=3,
        help="Max LLM draft-repair iterations when generator=llm",
    )
    ap.add_argument("--apply", action="store_true", help="Generate, compile, baseline, smoke-compare")
    ap.add_argument(
        "--allow-existing",
        action="store_true",
        help="Treat existing generated artifacts as warnings",
    )
    ap.add_argument(
        "--regen-baseline",
        action="store_true",
        help="Regenerate baseline even if baseline file already exists",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Optional output dir (default experiments/<target-id>/generated)",
    )
    args = ap.parse_args()

    root = _repo_root()
    out_dir = args.output_dir or os.path.join(root, "experiments", args.target_id, "generated")
    driver_dir = os.path.join(root, "experiments", args.target_id, "generated_driver")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(driver_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(
        out_dir,
        "onboarding_{0}_{1}.json".format("apply" if args.apply else "preflight_only", ts),
    )

    steps = []
    raw = _load_spec(args, root)
    spec, spec_errs = _normalize_spec(raw)
    steps.append(
        _mk_step(
            "load_spec",
            ok=(len(spec_errs) == 0),
            errors=spec_errs,
            artifacts={"spec_file": args.spec_file or "(inline args)"},
        )
    )
    if spec is None:
        summary = {
            "ok": False,
            "target_id": args.target_id,
            "mode": "preflight_only",
            "next_recommended_action": "fix_spec_and_retry",
            "steps": steps,
        }
        _write_json(report_path, summary)
        print(json.dumps({"report_file": report_path, **summary}, indent=2))
        return 2

    header_abs = os.path.join(root, spec["header_path"])
    driver_src_abs = os.path.join(driver_dir, "{0}_driver.cc".format(spec["id"]))
    baseline_csv = os.path.join(
        out_dir, "{0}_baseline_{1}_{2}.csv".format(spec["id"], args.batch, args.seed)
    )
    run_csv = os.path.join(
        out_dir, "{0}_run_{1}_{2}_{3}.csv".format(spec["id"], args.batch, args.seed, ts)
    )

    steps.append(
        _mk_step(
            "validate_target_id",
            ok=_validate_target_id(spec["id"]),
            errors=(
                []
                if _validate_target_id(spec["id"])
                else ["E_TARGET_ID_FORMAT: id must match [a-z][a-z0-9_]*"]
            ),
            artifacts={"target_id": spec["id"]},
        )
    )
    header_ok = os.path.isfile(header_abs)
    steps.append(
        _mk_step(
            "validate_header_path",
            ok=header_ok,
            errors=(
                []
                if header_ok
                else ["E_HEADER_NOT_FOUND: header path missing: {0}".format(spec["header_path"])]
            ),
            artifacts={"header_path": spec["header_path"]},
        )
    )
    symbol_found = False
    if header_ok:
        symbol_found = re.search(r"\b{0}\b".format(re.escape(spec["function_symbol"])), _read_text(header_abs)) is not None
    steps.append(
        _mk_step(
            "validate_function_symbol",
            ok=symbol_found,
            errors=(
                []
                if symbol_found
                else [
                    "E_FUNCTION_SYMBOL_NOT_FOUND: '{0}' not found in {1}".format(
                        spec["function_symbol"], spec["header_path"]
                    )
                ]
            ),
            artifacts={"function_symbol": spec["function_symbol"]},
        )
    )

    warnings = []
    if not spec.get("locals_for_downcast"):
        warnings.append("W_LOCALS_EMPTY: locals_for_downcast is empty")
    steps.append(
        _mk_step(
            "validate_locals_for_downcast",
            ok=True,
            warnings=warnings,
            artifacts={"locals_for_downcast": spec.get("locals_for_downcast", [])},
        )
    )

    driver_exists = os.path.exists(driver_src_abs)
    baseline_exists = os.path.exists(baseline_csv)
    path_errors = []
    path_warnings = []
    if driver_exists and not args.allow_existing:
        path_errors.append("E_DRIVER_EXISTS: {0}".format(driver_src_abs))
    if driver_exists and args.allow_existing:
        path_warnings.append("W_DRIVER_EXISTS: {0}".format(driver_src_abs))
    if baseline_exists:
        path_warnings.append("W_BASELINE_EXISTS: {0}".format(baseline_csv))
    steps.append(
        _mk_step(
            "plan_paths",
            ok=(len(path_errors) == 0),
            errors=path_errors,
            warnings=path_warnings,
            artifacts={
                "driver_source": driver_src_abs,
                "baseline_csv": baseline_csv,
                "run_csv": run_csv,
            },
        )
    )

    preflight_ok = all(s["ok"] for s in steps)

    if args.apply and preflight_ok:
        if not driver_exists or args.allow_existing:
            if args.generator == "deterministic":
                driver_src = _render_driver_source(spec)
                driver_src, fix_warnings = _enforce_csv_contract(driver_src)
                _write_text(driver_src_abs, driver_src)
                steps.append(
                    _mk_step(
                        "generate_driver",
                        ok=True,
                        artifacts={"driver_source": driver_src_abs, "generator": "deterministic"},
                        warnings=(
                            ["W_DRIVER_OVERWRITTEN: existing generated driver refreshed"]
                            if driver_exists
                            else []
                        )
                        + fix_warnings,
                    )
                )
            else:
                try:
                    client = ArgoClient(
                        base_url=args.base_url,
                        user=args.user,
                        model=args.model,
                        fallback_model=args.fallback_model,
                    )
                    draft = ""
                    msgs = _render_llm_driver_messages(
                        spec=spec,
                        header_text=_read_text(header_abs),
                        previous_code="",
                        feedback="",
                    )
                    resp = client.chat(messages=msgs)
                    draft = _extract_cpp(client.extract_text(resp))
                    draft, fix_warnings = _enforce_csv_contract(draft)
                    _write_text(driver_src_abs, draft)
                    steps.append(
                        _mk_step(
                            "generate_driver",
                            ok=True,
                            artifacts={"driver_source": driver_src_abs, "generator": "llm"},
                            warnings=(
                                ["W_DRIVER_OVERWRITTEN: existing generated driver refreshed"]
                                if driver_exists
                                else []
                            )
                            + fix_warnings,
                            details={"llm_iteration": 1},
                        )
                    )
                except Exception as exc:
                    steps.append(
                        _mk_step(
                            "generate_driver",
                            ok=False,
                            errors=["E_LLM_DRIVER_GENERATION_FAILED: {0}".format(str(exc)[:500])],
                            artifacts={"driver_source": driver_src_abs, "generator": "llm"},
                        )
                    )
        else:
            steps.append(
                _mk_step(
                    "generate_driver",
                    ok=False,
                    errors=["E_DRIVER_EXISTS: {0}".format(driver_src_abs)],
                )
            )

    overall_ok = all(s["ok"] for s in steps)

    if args.apply and overall_ok:
        rc_pipe, out_pipe, mode_pipe = _run_project_build_pipeline(root)
        steps.append(
            _mk_step(
                "project_build_pipeline",
                ok=(rc_pipe == 0),
                errors=([] if rc_pipe == 0 else ["E_PROJECT_BUILD_PIPELINE_FAILED"]),
                artifacts={
                    "mode": mode_pipe,
                    "prepare_script": os.path.join(root, "scripts", "prepare.sh"),
                    "bootstrap_script": os.path.join(root, "scripts", "build_with_Kokkos.sh"),
                    "compile_script": os.path.join(root, "scripts", "compile.sh"),
                },
                details={"log_tail": out_pipe[-6000:]},
            )
        )
        if not steps[-1]["ok"]:
            overall_ok = False

    if args.apply and overall_ok:
        tmp = tempfile.mkdtemp(prefix="onboard-validate-")
        try:
            cmakelists = _render_ephemeral_cmakelists(root, spec["id"], driver_src_abs)
            cmakelists_path = os.path.join(tmp, "CMakeLists.txt")
            _write_text(cmakelists_path, cmakelists)
            build_dir = os.path.join(tmp, "build")
            os.makedirs(build_dir, exist_ok=True)

            rc_cfg, out_cfg = _run_with_prepare(
                root, ["cmake", "-S", tmp, "-B", build_dir], cwd=root
            )
            steps.append(
                _mk_step(
                    "compile_configure",
                    ok=(rc_cfg == 0),
                    errors=([] if rc_cfg == 0 else ["E_CONFIGURE_FAILED"]),
                    artifacts={"build_dir": build_dir},
                    details={"log_tail": out_cfg[-4000:]},
                )
            )
            if rc_cfg == 0:
                exe_path = os.path.join(build_dir, spec["id"] + "_driver")
                llm_attempt = 1
                rc_bld = 1
                out_bld = ""
                max_attempts = (
                    max(1, args.llm_driver_max_iterations)
                    if args.generator == "llm"
                    else 1
                )
                while llm_attempt <= max_attempts:
                    rc_bld, out_bld = _run_with_prepare(
                        root,
                        ["cmake", "--build", build_dir, "--target", spec["id"] + "_driver"],
                        cwd=root,
                    )
                    if rc_bld == 0 and os.path.isfile(exe_path):
                        break
                    if args.generator != "llm" or llm_attempt >= max_attempts:
                        break
                    # Ask LLM to repair driver from compile diagnostics.
                    try:
                        client = ArgoClient(
                            base_url=args.base_url,
                            user=args.user,
                            model=args.model,
                            fallback_model=args.fallback_model,
                        )
                        revised_msgs = _render_llm_driver_messages(
                            spec=spec,
                            header_text=_read_text(header_abs),
                            previous_code=_read_text(driver_src_abs),
                            feedback=out_bld[-4000:],
                        )
                        revised_resp = client.chat(messages=revised_msgs)
                        revised_src = _extract_cpp(client.extract_text(revised_resp))
                        revised_src, _ = _enforce_csv_contract(revised_src)
                        _write_text(driver_src_abs, revised_src)
                        llm_attempt += 1
                    except Exception:
                        break
                steps.append(
                    _mk_step(
                        "compile_driver",
                        ok=(rc_bld == 0 and os.path.isfile(exe_path)),
                        errors=([] if rc_bld == 0 and os.path.isfile(exe_path) else ["E_BUILD_FAILED"]),
                        artifacts={"driver_executable": exe_path},
                        details={
                            "log_tail": out_bld[-4000:],
                            "generator": args.generator,
                            "attempts_used": llm_attempt,
                        },
                    )
                )

                if rc_bld == 0 and os.path.isfile(exe_path):
                    if (not baseline_exists) or args.regen_baseline:
                        rc_base, out_base = _run(
                            [exe_path, str(args.batch), baseline_csv, str(args.seed)],
                            cwd=root,
                        )
                        steps.append(
                            _mk_step(
                                "generate_baseline",
                                ok=(rc_base == 0 and os.path.isfile(baseline_csv)),
                                errors=(
                                    []
                                    if rc_base == 0 and os.path.isfile(baseline_csv)
                                    else ["E_BASELINE_GEN_FAILED"]
                                ),
                                artifacts={"baseline_csv": baseline_csv},
                                details={"log_tail": out_base[-4000:]},
                            )
                        )
                    else:
                        steps.append(
                            _mk_step(
                                "generate_baseline",
                                ok=True,
                                warnings=["W_BASELINE_REUSED: existing baseline kept"],
                                artifacts={"baseline_csv": baseline_csv},
                            )
                        )

                    rc_run, out_run = _run([exe_path, str(args.batch), run_csv, str(args.seed)], cwd=root)
                    steps.append(
                        _mk_step(
                            "run_candidate",
                            ok=(rc_run == 0 and os.path.isfile(run_csv)),
                            errors=(
                                []
                                if rc_run == 0 and os.path.isfile(run_csv)
                                else ["E_CANDIDATE_RUN_FAILED"]
                            ),
                            artifacts={"run_csv": run_csv},
                            details={"log_tail": out_run[-4000:]},
                        )
                    )

                    if rc_run == 0 and os.path.isfile(run_csv):
                        compare_py = os.path.join(root, "scripts", "compare_results.py")
                        rc_cmp, out_cmp = _run(
                            [
                                sys.executable,
                                compare_py,
                                baseline_csv,
                                run_csv,
                                "--min-digits",
                                str(args.min_digits),
                            ],
                            cwd=root,
                        )
                        steps.append(
                            _mk_step(
                                "smoke_compare",
                                ok=(rc_cmp == 0),
                                errors=([] if rc_cmp == 0 else ["E_COMPARE_FAILED"]),
                                artifacts={
                                    "baseline_csv": baseline_csv,
                                    "run_csv": run_csv,
                                },
                                details={
                                    "compare_exit": rc_cmp,
                                    "min_precise_digits": _parse_min_precise_digits(out_cmp),
                                    "log_tail": out_cmp[-4000:],
                                },
                            )
                        )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    overall_ok = all(s["ok"] for s in steps)
    summary = {
        "ok": overall_ok,
        "target_id": spec["id"],
        "mode": "apply" if args.apply else "preflight_only",
        "next_recommended_action": (
            "target_ready_for_llm_runs"
            if overall_ok and args.apply
            else ("safe_to_apply_validation" if overall_ok else "fix_preflight_or_apply_errors_then_retry")
        ),
        "spec": {
            "id": spec["id"],
            "header_path": spec["header_path"],
            "function_symbol": spec["function_symbol"],
            "locals_for_downcast": spec.get("locals_for_downcast", []),
            "x_min": spec["x_min"],
            "x_max": spec["x_max"],
            "output_mode": spec["output_mode"],
        },
        "steps": steps,
    }
    _write_json(report_path, summary)
    print(json.dumps({"report_file": report_path, **summary}, indent=2))
    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())
