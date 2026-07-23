"""Multi-intent per pass — shared prefix uses the same intermediate variant, and
byte-identical over-generation across paths generates both variants (no dedup)."""

from __future__ import annotations

import json

from agents.patcher import fanout
from tests.patcher.fanout.conftest import requires_libclang


def _manifest(tree):
    txt = (tree / "app.h").read_text()
    line = [l for l in txt.split("\n") if fanout._MANIFEST_PREFIX.strip() in l][0]
    payload = line.split(fanout._MANIFEST_PREFIX.strip())[1].strip()
    return {v["variant_name"]: v for v in json.loads(payload)["variants"]}


@requires_libclang
def test_shared_prefix_variant_accumulates(synth_tree, synth_graph):
    g = synth_graph
    # intent 1: region in f (path entry->h->g->f, among others) -> g_h_B1 reroutes f
    ff = g.defs["f"][0]
    fanout.fan_out_region(
        file="app.h", line_start=ff.line_start + 5, line_end=ff.line_start + 5,
        reads=["a"], writes=[], integral="B1", graph=g, tree_root=synth_tree,
        scalar_type="Ext", two_limb=False, shim_include=None)
    # intent 2: region in g (path entry->h->g) -> g_h_B1 gets its OWN promotion
    gg = g.defs["g"][0]
    fanout.fan_out_region(
        file="app.h", line_start=gg.line_start + 2, line_end=gg.line_start + 2,
        reads=["x"], writes=[], integral="B1", graph=g, tree_root=synth_tree,
        scalar_type="Ext", two_limb=False, shim_include=None)

    m = _manifest(synth_tree)
    assert "g_h_B1" in m
    # the SAME g_h_B1 carries both intent-1's reroute (f -> f_g_h_B1) and intent-2's
    # region promotion — the shared prefix converged onto one intermediate variant.
    assert m["g_h_B1"]["reroutes"].get("f") == "f_g_h_B1"
    assert len(m["g_h_B1"]["promotes"]) == 1


@requires_libclang
def test_byte_identical_over_generation(synth_tree, synth_graph):
    g = synth_graph
    ff = g.defs["f"][0]
    res = fanout.fan_out_region(
        file="app.h", line_start=ff.line_start + 5, line_end=ff.line_start + 5,
        reads=["a"], writes=[], integral="B1", graph=g, tree_root=synth_tree,
        scalar_type="Ext", two_limb=False, shim_include=None)
    # two paths to f (via g under h, and via g under h2) produce f-variants whose
    # bodies are byte-identical modulo the function name — both are generated.
    assert {"f_g_h_B1", "f_g_h2_B1"} <= set(res.declared_variants)

    m = _manifest(synth_tree)
    a, b = m["f_g_h_B1"], m["f_g_h2_B1"]
    # same original, same promotion, no reroutes (f is the bottom) -> identical recipe
    assert a["orig_name"] == b["orig_name"] == "f"
    assert a["promotes"] == b["promotes"]
    assert a["reroutes"] == b["reroutes"] == {}

    # and the rendered bodies differ only by the variant name
    va = fanout.render_variant(fanout.VariantSpec.from_json(a))
    vb = fanout.render_variant(fanout.VariantSpec.from_json(b))
    assert va.replace("f_g_h_B1", "X") == vb.replace("f_g_h2_B1", "X")
