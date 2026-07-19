"""The four P3 dispatch paths end-to-end through the Patcher (mocked gate)."""

import subprocess
from pathlib import Path

from agents.patcher import result as R
from agents.patcher.agent import make_patcher_fn

from tests.patcher.conftest import git, intent, make_shim_integrator, ok_gate


def _branch_commit_count(root, start):
    return git(root, "rev-list", "--count", f"{start}..HEAD").stdout.strip()


def _blob_exists(root, sha, path):
    return subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{sha}:{path}"],
                          capture_output=True).returncode == 0


# -- 1. regional-integrator -------------------------------------------------

def test_regional_ff_commits_shim(repo, make_ctx):
    root, start = repo
    fn = make_patcher_fn(gate_fn=ok_gate,
                         integrators={"ff": make_shim_integrator(root)})
    resp = fn(intent("double-to-ff", flavor="speedup"), make_ctx(root, start))
    assert resp["status"] == R.OK
    assert resp["candidate_sha"] and resp["parent_sha"] == start
    assert resp["llm_tokens"] == 42
    assert _branch_commit_count(root, start) == "1"
    # the shim is part of the committed tree
    assert _blob_exists(root, resp["candidate_sha"], "region_ff.h")


def test_regional_dd_uses_dd_integrator(repo, make_ctx):
    root, start = repo
    fn = make_patcher_fn(gate_fn=ok_gate,
                         integrators={"dd": make_shim_integrator(root, shim_name="region_dd.h")})
    resp = fn(intent("double-to-dd"), make_ctx(root, start))
    assert resp["status"] == R.OK
    assert resp["artifacts"]["shim_paths"] and "region_dd.h" in resp["artifacts"]["shim_paths"][0]


# -- 1b. Wave-2: template-typed float demotion routes to the float integrator ---

def test_regional_double_to_float_via_regional_uses_float_integrator(repo, make_ctx):
    # A template-typed region: Strategy tags `double-to-float` via="regional", so
    # the Patcher must route it to the FLOAT integrator (a regional shim), NOT the
    # plain-edit path.  Post-Wave-1 this kind went to plain-edit and died
    # patch_inapplicable on template code; now it generates a float shim.
    root, start = repo
    fn = make_patcher_fn(gate_fn=ok_gate,
                         integrators={"float": make_shim_integrator(root, shim_name="region_float.h")})
    resp = fn(intent("double-to-float", flavor="speedup", via="regional"),
              make_ctx(root, start))
    assert resp["status"] == R.OK
    assert _blob_exists(root, resp["candidate_sha"], "region_float.h")
    assert "float" in resp["artifacts"]["shim_paths"][0]


def test_regional_double_to_float_passes_float_scalar_to_integrator(repo, make_ctx):
    # The integrator is invoked with scalar_type="float" (not ffloat/ddouble).
    root, start = repo
    seen = {}

    def _integ(**kw):
        seen.update(kw)
        shim = Path(root) / "region_float.h"
        shim.write_text(f"// {kw['scalar_type']} shim\n")
        return __import__("agents.integrator_base.region", fromlist=["RegionIntegrationResult"]) \
            .RegionIntegrationResult(status="ok", shim_paths=[str(shim)],
                                     boundary_patch=None, llm_tokens=7)

    fn = make_patcher_fn(gate_fn=ok_gate, integrators={"float": _integ})
    resp = fn(intent("double-to-float", flavor="speedup", via="regional"),
              make_ctx(root, start))
    assert resp["status"] == R.OK
    assert seen["scalar_type"] == "float"
    assert seen["caller_type"] == "double"


def test_double_to_float_plain_still_uses_plain_edit(repo, make_ctx):
    # Control: without via="regional" (a non-templated region), `double-to-float`
    # keeps the historical plain-type-edit path — Wave 2 must not regress it.
    root, start = repo
    fn = make_patcher_fn(gate_fn=ok_gate)   # no float integrator wired
    resp = fn(intent("double-to-float", line_start=4, line_end=6, flavor="speedup",
                     variables=["result"]),
              make_ctx(root, start))
    assert resp["status"] == R.OK
    committed = git(root, "show", f"{resp['candidate_sha']}:region.h").stdout
    assert "inline float compute(float a, float b)" in committed


def test_dispatch_path_routing_for_float():
    # Unit-level: the routing table distinguishes plain vs regional float.
    from agents.patcher import dispatch
    assert dispatch.dispatch_path("double-to-float", "regional") == dispatch.PATH_REGIONAL
    assert dispatch.dispatch_path("ff-to-float", "regional") == dispatch.PATH_REGIONAL
    assert dispatch.dispatch_path("double-to-float", "plain") == dispatch.PATH_PLAIN_EDIT
    assert dispatch.dispatch_path("ff-to-float", "plain") == dispatch.PATH_REVERT
    # non-float kinds ignore `via`
    assert dispatch.dispatch_path("double-to-ff", "regional") == dispatch.PATH_REGIONAL
    assert dispatch.dispatch_path("double-to-dd", "plain") == dispatch.PATH_REGIONAL


# -- 2. plain-type-edit -----------------------------------------------------

def test_plain_edit_double_to_float(repo, make_ctx):
    root, start = repo
    fn = make_patcher_fn(gate_fn=ok_gate)
    # region.h line 4: "    inline double compute(double a, float b) {"
    resp = fn(intent("double-to-float", line_start=4, line_end=6, flavor="speedup",
                     variables=["result"]),
              make_ctx(root, start))
    assert resp["status"] == R.OK
    committed = git(root, "show", f"{resp['candidate_sha']}:region.h").stdout
    assert "inline float compute(float a, float b)" in committed
    assert "float_traits" in committed          # identifier survived


def test_plain_edit_no_bare_double_is_inapplicable(repo, make_ctx):
    # Line 8 ("struct float_traits { int x; }") carries no bare `double` token, so
    # the `double-to-float` plain edit cannot apply — this is a template-typed-style
    # inapplicable rung, NOT a malformed intent, so the status is patch_inapplicable
    # (benign), not the strategy_bug patch_apply_failed.
    root, start = repo
    fn = make_patcher_fn(gate_fn=ok_gate)
    resp = fn(intent("double-to-float", line_start=8, line_end=8, flavor="speedup",
                     variables=["x"]),
              make_ctx(root, start))
    assert resp["status"] == R.PATCH_INAPPLICABLE
    assert "no bare `double`" in resp["error"]["detail"]
    # nothing committed; the tree is reset back to the parent
    assert _branch_commit_count(root, start) == "0"


# -- 3. git-revert ----------------------------------------------------------

def _install_ff(root):
    """Simulate a prior ff-install commit the revert path will look up."""
    (root / "region_ff.h").write_text("// ff shim\n")
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t.t", "commit", "--no-gpg-sign",
        "-q", "-m", "[iter_0] double-to-ff region.h:4-6")
    return git(root, "rev-parse", "HEAD").stdout.strip()


def test_git_revert_ff_to_double(repo, make_ctx):
    root, start = repo
    ff_head = _install_ff(root)
    fn = make_patcher_fn(gate_fn=ok_gate)
    resp = fn(intent("ff-to-double"), {**make_ctx(root, start), "parent_sha": ff_head})
    assert resp["status"] == R.OK
    # after revert the shim is gone from the candidate tree
    assert not _blob_exists(root, resp["candidate_sha"], "region_ff.h")


def test_git_revert_missing_commit_is_apply_failed(repo, make_ctx):
    root, start = repo
    fn = make_patcher_fn(gate_fn=ok_gate)
    resp = fn(intent("dd-to-double"), make_ctx(root, start))
    assert resp["status"] == R.PATCH_APPLY_FAILED
    assert "no introducing" in resp["error"]["detail"]


# -- 4. llm-rewrite ---------------------------------------------------------

def test_llm_rewrite_kahan(repo, make_ctx):
    root, start = repo
    canned = "    inline double compute(double a, float b) {\n" \
             "        double result = std::fma(a, (double)b, 0.0);  // kahan\n" \
             "        return result;"

    def llm_call(system, user, attempt):
        assert "Kahan" in user or "compensated" in user
        return canned

    fn = make_patcher_fn(gate_fn=ok_gate, llm_call=llm_call)
    resp = fn(intent("reformulate-kahan", line_start=4, line_end=6),
              make_ctx(root, start))
    assert resp["status"] == R.OK
    assert "kahan" in git(root, "show", f"{resp['candidate_sha']}:region.h").stdout


def test_llm_rewrite_identity_passes_identity_to_prompt(repo, make_ctx):
    root, start = repo
    seen = {}

    def llm_call(system, user, attempt):
        seen["user"] = user
        return "        double result = std::log1p(a);"

    fn = make_patcher_fn(gate_fn=ok_gate, llm_call=llm_call)
    resp = fn(intent("reformulate-identity", identity="log1p", line_start=5, line_end=5),
              make_ctx(root, start))
    assert resp["status"] == R.OK
    assert "log1p" in seen["user"]
