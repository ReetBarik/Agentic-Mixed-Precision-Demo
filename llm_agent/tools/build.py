"""Ephemeral cmake build + driver run tool (temp-dir isolation, no git worktrees)."""

import os
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import List


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_with_prepare(root: str, cmd: list, cwd: str) -> tuple:
    """Source prepare.sh (and kokkos/setup.sh if present) then run cmd."""
    prepare = os.path.join(root, "scripts", "prepare.sh")
    kokkos_setup = os.path.join(root, "kokkos", "setup.sh")
    prefix = "source {0}".format(shlex.quote(prepare))
    if os.path.isfile(kokkos_setup):
        prefix += " && source {0}".format(shlex.quote(kokkos_setup))
    shell_cmd = prefix + " && " + " ".join(shlex.quote(x) for x in cmd)
    p = subprocess.run(
        ["bash", "-lc", shell_cmd],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return p.returncode, p.stdout or ""


def validate_target_id(target_id: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*", target_id))


_HEADER_LINE_RE = re.compile(r'out\s*<<\s*"id,')


def enforce_csv_contract(driver_src: str) -> tuple:
    """Ensure header line appears before metadata line in the generated driver."""
    lines = driver_src.splitlines()
    header_idx = -1
    meta_start = -1
    meta_end = -1
    for i, ln in enumerate(lines):
        s = ln.strip()
        if header_idx < 0 and _HEADER_LINE_RE.search(s):
            header_idx = i
        if 'out << "# target_id=' in s and meta_start < 0:
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
        meta_block = lines[meta_start : meta_end + 1]
        kept = lines[:meta_start] + lines[meta_end + 1 :]
        new_header_idx = header_idx - len(meta_block)
        fixed = kept[: new_header_idx + 1] + meta_block + kept[new_header_idx + 1 :]
        warnings.append("W_DRIVER_CONTRACT_FIXED: moved metadata line after header")
        return "\n".join(fixed) + "\n", warnings
    return driver_src, warnings


_COMPLEX_PREFIXES = ("Kokkos::complex<", "std::complex<")
_REAL_SCALAR_TYPES = frozenset({"float", "double", "long double"})


def _resolve_type(ctype: str, template_types: dict) -> str:
    """Resolve a template alias to its concrete C++ type, recursively if needed."""
    seen = set()
    cur = ctype.strip()
    while cur in template_types and cur not in seen:
        seen.add(cur)
        cur = template_types[cur].strip()
    return cur


def _classify_floating(ctype: str, template_types: dict) -> str:
    """Return 'real', 'complex', or 'other' for a value-type ctype.

    Resolves through concrete_template_types so aliases (e.g. TOutput = Kokkos::complex<double>)
    classify correctly. Anything that isn't a floating real or complex (e.g. int, bool, struct)
    is 'other' — computed by the driver but not serialized to CSV.
    """
    resolved = _resolve_type(ctype, template_types)
    if any(resolved.startswith(p) for p in _COMPLEX_PREFIXES):
        return "complex"
    if resolved in _REAL_SCALAR_TYPES:
        return "real"
    return "other"


def render_driver_source(spec: dict) -> str:
    """Generate an ephemeral C++ driver from a target spec.

    Generic over input/output shape: the spec's `inputs`, `outputs`, `return_type`, and
    `call.expression` (with `{name}` placeholders for both inputs and outputs) drive every
    part of the generated driver. Value-returning functions get a `_y_d` view and the call
    is emitted as `_y_d(i) = call_expr`. Void-returning functions emit `call_expr;` and
    rely on output-by-reference parameters to deliver results. CSV columns are produced
    for the return value (if floating) and every floating output param (real → 1 hex col,
    complex → 2 hex cols). Non-floating outputs (e.g. integer flags) are computed and
    written back to host memory but skipped from the CSV.
    """
    target_id = spec["id"]
    return_type = spec.get("return_type")
    if not return_type:
        raise ValueError("spec missing return_type")
    header_path = spec.get("header_path")
    if not header_path:
        raise ValueError("spec missing header_path")
    header_fname = os.path.basename(header_path)

    inputs = spec.get("inputs")
    if not inputs:
        raise ValueError("spec missing inputs")

    # call.expression uses {name} as placeholder tokens, not Python format placeholders
    call_expr = (spec.get("call") or {}).get("expression", "").strip()
    if not call_expr:
        raise ValueError("spec missing call.expression")

    template_types = spec.get("concrete_template_types") or {}
    outputs = spec.get("outputs") or []

    input_view_lines = []
    input_mirror_lines = []
    input_dist_lines = []
    input_fill_lines = []
    input_copy_lines = []
    meta_pairs = []
    call_eval = call_expr

    for inp in inputs:
        name = inp["name"]
        ctype = inp.get("ctype")
        if not ctype:
            raise ValueError("input {0!r} missing ctype".format(name))
        lo = float(inp.get("min", -4.0))
        hi = float(inp.get("max", 4.0))
        call_eval = call_eval.replace("{" + name + "}", "{0}_d(i)".format(name))
        input_view_lines.append(
            '        Kokkos::View<{0}*> {1}_d("{1}", batch_size);'.format(ctype, name)
        )
        input_mirror_lines.append(
            "        auto {0}_h = Kokkos::create_mirror_view({0}_d);".format(name)
        )
        input_dist_lines.append(
            "        std::uniform_real_distribution<double> dist_{0}({1}, {2});".format(
                name, lo, hi
            )
        )
        input_fill_lines.append(
            "            {0}_h(i) = static_cast<{1}>(dist_{0}(rng));".format(name, ctype)
        )
        input_copy_lines.append("        Kokkos::deep_copy({0}_d, {0}_h);".format(name))
        meta_pairs.append("{0}_min={1} {0}_max={2}".format(name, lo, hi))

    # Output-by-reference params: declare a device View per output, substitute the
    # placeholder in the call, and copy back to host after the kernel runs. Floating
    # outputs additionally get a CSV column; non-floating outputs are computed but
    # excluded from numerical verification.
    output_view_lines = []
    output_copy_back_lines = []
    csv_columns = []  # list of (col_name, host_expr)
    for out in outputs:
        name = out["name"]
        ctype = out.get("ctype")
        if not ctype:
            raise ValueError("output {0!r} missing ctype".format(name))
        call_eval = call_eval.replace("{" + name + "}", "{0}_d(i)".format(name))
        output_view_lines.append(
            '        Kokkos::View<{0}*> {1}_d("{1}", batch_size);'.format(ctype, name)
        )
        output_copy_back_lines.append(
            "        auto {0}_h = Kokkos::create_mirror_view_and_copy("
            "Kokkos::HostSpace(), {0}_d);".format(name)
        )
        kind = _classify_floating(ctype, template_types)
        if kind == "real":
            csv_columns.append((name, "{0}_h(i)".format(name)))
        elif kind == "complex":
            csv_columns.append((name + "__re", "{0}_h(i).real()".format(name)))
            csv_columns.append((name + "__im", "{0}_h(i).imag()".format(name)))
        # 'other' → not serialized

    # Return-value handling: a value-returning function gets its own _y_d view that
    # the kernel writes into; void-returning functions just invoke the call as a
    # statement. The leading underscore on _y avoids clashing with any user-named
    # output called 'y'.
    is_void = return_type.strip() == "void"
    if is_void:
        y_view_block = ""
        kernel_body = "                {0};".format(call_eval)
        y_copy_back = ""
    else:
        y_view_block = (
            '        Kokkos::View<{0}*> _y_d("_y", batch_size);'.format(return_type)
        )
        kernel_body = "                _y_d(i) = {0};".format(call_eval)
        y_copy_back = (
            "        auto _y_h = Kokkos::create_mirror_view_and_copy("
            "Kokkos::HostSpace(), _y_d);"
        )
        kind = _classify_floating(return_type, template_types)
        if kind == "real":
            csv_columns.append(("_y", "_y_h(i)"))
        elif kind == "complex":
            csv_columns.append(("_y__re", "_y_h(i).real()"))
            csv_columns.append(("_y__im", "_y_h(i).imag()"))
        else:
            raise ValueError(
                "return_type {0!r} is neither floating real nor complex; "
                "cannot verify numerically".format(return_type)
            )

    if not csv_columns:
        raise ValueError(
            "function has no floating-point outputs to verify against "
            "(return_type={0!r}, outputs={1!r})".format(
                return_type, [o["name"] for o in outputs]
            )
        )

    header = "id," + ",".join(c[0] for c in csv_columns)
    write_lines = []
    for idx, (_col_name, host_expr) in enumerate(csv_columns):
        write_lines.append(
            "            print_double_bits({0}, out);".format(host_expr)
        )
        if idx < len(csv_columns) - 1:
            write_lines.append("            out << ',';")
    write_block = "\n".join(write_lines)

    meta_suffix = (" " + " ".join(meta_pairs)) if meta_pairs else ""
    using_decls = "\n".join(
        "using {0} = {1};".format(name, value) for name, value in template_types.items()
    )

    return """#include <Kokkos_Core.hpp>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "{header_fname}"

namespace {{
inline void print_double_bits(double x, std::ostream& os) {{
    union {{ double d; std::uint64_t u; }} conv;
    conv.d = x;
    char buf[32];
    if (std::snprintf(buf, sizeof(buf), "0x%016" PRIx64, conv.u) > 0)
        os << buf;
}}
}}

// AUTO-GENERATED EPHEMERAL DRIVER FROM TARGET SPEC.
{using_decls}

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
{output_views}
{y_view_block}

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
{kernel_body}
            }});
        Kokkos::fence();

{output_copies}
{y_copy_back}

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
{write_block}
            out << '\\n';
        }}
    }}
    Kokkos::finalize();
    return 0;
}}
""".format(
        target_id=target_id,
        header_fname=header_fname,
        header=header,
        using_decls=using_decls,
        input_views="\n".join(input_view_lines),
        output_views="\n".join(output_view_lines),
        y_view_block=y_view_block,
        input_mirrors="\n".join(input_mirror_lines),
        input_dists="\n".join(input_dist_lines),
        input_fill="\n".join(input_fill_lines),
        input_copy="\n".join(input_copy_lines),
        output_copies="\n".join(output_copy_back_lines),
        y_copy_back=y_copy_back,
        kernel_body=kernel_body,
        meta_suffix=meta_suffix,
        write_block=write_block,
    )


def render_ephemeral_cmakelists(target_id: str, driver_src_abs: str, src_dir: str) -> str:
    """CMakeLists.txt for building a single ephemeral driver against Kokkos."""
    return """cmake_minimum_required(VERSION 3.16)
project(EphemeralDriver LANGUAGES CXX)
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
        src_dir=src_dir,
    )


def apply_patches(impl_source: str, patches: List[dict]) -> str:
    """Apply a list of PatchProposal dicts to source text via exact line replacement."""
    for p in patches:
        old = p["old_line"].rstrip("\n")
        new = p["new_line"].rstrip("\n")
        lines = impl_source.splitlines()
        idx = next((i for i, line in enumerate(lines) if line == old), -1)
        if idx < 0:
            raise ValueError("old_line not found in source: {0!r}".format(old))
        lines[idx] = new
        impl_source = "\n".join(lines) + "\n"
    return impl_source


def build_and_run(
    root: str,
    spec: dict,
    impl_source: str,
    batch: int,
    seed: int,
    out_csv: str,
) -> dict:
    """Build an ephemeral driver from spec + impl_source, run it, write CSV to out_csv.

    Steps:
      1. Copy src/ headers to a temp dir, replace the target header with impl_source.
      2. Generate driver source from spec.
      3. Write an ephemeral CMakeLists.txt.
      4. cmake configure + build (via prepare.sh for module loading).
      5. Run the driver executable, writing output to out_csv.

    Returns:
        {ok: bool, csv_path: str|None, error: str|None, logs: dict}
    """
    target_id = spec["id"]
    header_rel = spec.get("header_path")
    if not header_rel:
        return {
            "ok": False,
            "csv_path": None,
            "error": "spec missing header_path",
            "logs": {},
        }
    header_fname = os.path.basename(header_rel)

    driver_src, _ = enforce_csv_contract(render_driver_source(spec))

    tmp = tempfile.mkdtemp(prefix="llm-build-")
    try:
        # --- src/ with patched header ---
        src_dir = os.path.join(tmp, "src")
        os.makedirs(src_dir)
        repo_src = os.path.join(root, "src")
        for fname in os.listdir(repo_src):
            if fname.endswith((".h", ".hpp")):
                shutil.copy2(os.path.join(repo_src, fname), src_dir)
        with open(os.path.join(src_dir, header_fname), "w", encoding="utf-8") as f:
            f.write(impl_source)

        # --- driver source ---
        driver_path = os.path.join(tmp, "{0}_driver.cc".format(target_id))
        with open(driver_path, "w", encoding="utf-8") as f:
            f.write(driver_src)

        # --- ephemeral CMakeLists.txt ---
        cmake_txt = render_ephemeral_cmakelists(target_id, driver_path, src_dir)
        with open(os.path.join(tmp, "CMakeLists.txt"), "w", encoding="utf-8") as f:
            f.write(cmake_txt)

        build_dir = os.path.join(tmp, "build")
        os.makedirs(build_dir)

        # cmake configure
        cfg_rc, cfg_out = _run_with_prepare(
            root, ["cmake", "-S", tmp, "-B", build_dir], cwd=root
        )
        if cfg_rc != 0:
            return {
                "ok": False,
                "csv_path": None,
                "error": "cmake configure failed",
                "logs": {"configure": cfg_out[-4000:]},
            }

        # cmake build
        exe_path = os.path.join(build_dir, "{0}_driver".format(target_id))
        bld_rc, bld_out = _run_with_prepare(
            root,
            ["cmake", "--build", build_dir, "--target", "{0}_driver".format(target_id)],
            cwd=root,
        )
        if bld_rc != 0 or not os.path.isfile(exe_path):
            return {
                "ok": False,
                "csv_path": None,
                "error": "build failed",
                "logs": {"build": bld_out[-4000:]},
            }

        # Run driver
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        run_result = subprocess.run(
            [exe_path, str(batch), out_csv, str(seed)],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        if run_result.returncode != 0 or not os.path.isfile(out_csv):
            return {
                "ok": False,
                "csv_path": None,
                "error": "driver run failed",
                "logs": {"run": (run_result.stdout or "")[-4000:]},
            }

        return {"ok": True, "csv_path": out_csv, "error": None, "logs": {}}

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compile_driver(
    root: str,
    driver_src: str,
    cmake_src: str,
    src_include_dir: str,
) -> dict:
    """Compile an LLM-generated driver in a temp dir.

    The LLM provides both driver_src and cmake_src; this function injects an
    additional target_include_directories line so the driver can find the target
    function's header regardless of framework.

    Returns:
        {ok, exe_path (caller must delete), error, logs}
        On failure, exe_path is None and error/logs describe the problem.
    """
    tmp = tempfile.mkdtemp(prefix="llm-compile-")
    try:
        driver_path = os.path.join(tmp, "driver.cc")
        with open(driver_path, "w", encoding="utf-8") as f:
            f.write(driver_src)

        # Extract the CMake target name so we can inject the right
        # target_include_directories and find the built exe by name.
        import re as _re
        _m = _re.search(r'add_executable\s*\(\s*(\S+)', cmake_src)
        cmake_target = _m.group(1) if _m else "driver"

        include_line = '\ntarget_include_directories({target} PRIVATE "{inc}")\n'.format(
            target=cmake_target,
            inc=src_include_dir.replace("\\", "/"),
        )
        if _m:
            # Insert after the matched add_executable(...) block
            cmake_src_patched = _re.sub(
                r"(add_executable\s*\([^\)]+\))",
                lambda mo: mo.group(0) + include_line,
                cmake_src,
                count=1,
            )
        else:
            cmake_src_patched = cmake_src + include_line

        cmake_path = os.path.join(tmp, "CMakeLists.txt")
        with open(cmake_path, "w", encoding="utf-8") as f:
            f.write(cmake_src_patched)

        build_dir = os.path.join(tmp, "build")
        os.makedirs(build_dir)

        cfg_rc, cfg_out = _run_with_prepare(
            root, ["cmake", "-S", tmp, "-B", build_dir], cwd=root
        )
        if cfg_rc != 0:
            return {
                "ok": False,
                "exe_path": None,
                "error": "cmake configure failed:\n{0}".format(cfg_out[-3000:]),
                "logs": {"configure": cfg_out},
            }

        bld_rc, bld_out = _run_with_prepare(
            root, ["cmake", "--build", build_dir, "--target", cmake_target], cwd=root
        )
        # Find the built executable by the known target name
        exe_path = os.path.join(build_dir, cmake_target)
        if not os.path.isfile(exe_path):
            exe_path = None

        if bld_rc != 0 or exe_path is None:
            return {
                "ok": False,
                "exe_path": None,
                "error": "build failed (target={0}):\n{1}".format(cmake_target, bld_out[-3000:]),
                "logs": {"build": bld_out},
            }

        # Copy exe out of tmp dir so the caller can use it after tmp cleanup
        import stat as _stat
        fd, stable_path = tempfile.mkstemp(prefix="llm-driver-exe-")
        os.close(fd)
        shutil.copy2(exe_path, stable_path)
        os.chmod(stable_path, os.stat(stable_path).st_mode | _stat.S_IEXEC | _stat.S_IXGRP)

        return {"ok": True, "exe_path": stable_path, "error": None, "logs": {}}

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_driver(exe_path: str, out_csv: str, batch: int, seed: int) -> dict:
    """Run a compiled driver binary, writing CSV output to out_csv.

    Passes batch and seed as positional argv[1] and argv[2].
    Cleans up the exe_path temp file after running.

    Returns:
        {ok, error}
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    try:
        result = subprocess.run(
            [exe_path, str(batch), out_csv, str(seed)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        if result.returncode != 0 or not os.path.isfile(out_csv):
            return {
                "ok": False,
                "error": "driver run failed (rc={0}): {1}".format(
                    result.returncode, (result.stdout or "")[-2000:]
                ),
            }
        return {"ok": True, "error": None}
    finally:
        try:
            os.unlink(exe_path)
        except OSError:
            pass
