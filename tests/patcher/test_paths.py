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
