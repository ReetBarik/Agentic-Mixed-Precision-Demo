"""Fast, parallel merge of stability-reducer shard reports.

Produces exactly what ``finalize_report(merge_reports(all_shards))`` would, but
in ~minutes instead of hours for large (100k-sample) runs. The design attacks
every bottleneck that a single-process merge hits:

* **Source-var filter** -- ``finalize_report`` keeps only ``is_source_var``
  variables (~9%); the rest are sample-scoped intermediates it discards, so we
  drop them up front (identical output, ~11x less data).
* **Partition by integral** -- the report's integrals are independent, so each
  is merged + finalized + serialized in its own worker; no process ever builds
  the full ~23M-variable structure.
* **prov_vars set-merge** -- region provenance can be huge (10^6 entries); we
  union into a set once instead of ``sorted(set|set)`` per shard (O(N), not
  O(N^2)).
* **orjson + gc.disable** -- fast (de)serialization and no cyclic-GC rescans of
  the large object graphs being built.

Falls back to the stdlib ``json`` module if ``orjson`` is not installed.
"""
from __future__ import annotations

import gc
import glob
import json as _json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import stability_reducer as sr

try:
    import orjson

    def _loads(b: bytes):
        return orjson.loads(b)

    def _dumps(obj) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS)
except ImportError:  # pragma: no cover - portability fallback
    orjson = None

    def _loads(b: bytes):
        return _json.loads(b)

    def _dumps(obj) -> bytes:
        return _json.dumps(obj).encode()


def _map_shards(wid: int, paths: list[str], mapdir: str) -> dict:
    """Parse a group of shards, keep source vars, emit per-integral fragments."""
    gc.disable()
    acc: dict[str, dict] = {}
    samples: dict[str, int] = {}
    no_id = 0
    for p in paths:
        d = _loads(Path(p).read_bytes())
        no_id += d.get("no_id_records", 0)
        for name, c in d.get("samples_seen", {}).items():
            samples[name] = samples.get(name, 0) + c
        for name, idata in d.get("integrals", {}).items():
            a = acc.setdefault(name, {"regions": {}, "variables": {}, "cascade_chains": {}})
            for loc, reg in idata.get("regions", {}).items():
                sr._merge_region(a["regions"].setdefault(loc, sr._new_region_json()), reg)
            for vid, var in idata.get("variables", {}).items():
                if var.get("is_source_var"):
                    sr._merge_variable(
                        a["variables"].setdefault(vid, sr._new_variable_json()), var)
            # cascade chains are keyed by (stable) chain_id; union across shards,
            # never merged — mirrors merge_reports (Strategy owns line overlap).
            a["cascade_chains"].update(idata.get("cascade_chains", {}))
    for name, a in acc.items():
        Path(f"{mapdir}/frag_{name}__{wid}.json").write_bytes(_dumps(a))
    return {"samples": samples, "no_id": no_id}


def _reduce_integral(name: str, n_samples: int, mapdir: str,
                     cfg: sr.ReducerConfig) -> tuple[str, str]:
    """Merge one integral's fragments, finalize it, serialize to a blob file."""
    gc.disable()
    a = {"regions": {}, "variables": {}, "cascade_chains": {}}
    prov: dict[str, set] = {}  # union region provenance once, not per-shard
    for f in glob.glob(f"{mapdir}/frag_{name}__*.json"):
        frag = _loads(Path(f).read_bytes())
        for loc, reg in frag["regions"].items():
            pv = reg.get("prov_vars")
            if pv:
                prov.setdefault(loc, set()).update(pv)
                reg["prov_vars"] = ()  # keep _merge_region's prov re-sort trivial
            sr._merge_region(a["regions"].setdefault(loc, sr._new_region_json()), reg)
        for vid, var in frag["variables"].items():
            sr._merge_variable(a["variables"].setdefault(vid, sr._new_variable_json()), var)
        a["cascade_chains"].update(frag.get("cascade_chains", {}))
    for loc, d in a["regions"].items():
        if loc in prov:
            d["prov_vars"] = sorted(prov[loc])

    # mirror finalize_report for this single integral
    regions = {loc: sr._classify_region(reg, cfg) for loc, reg in a["regions"].items()}
    variables = {vid: sr._classify_variable(var, cfg)
                 for vid, var in a["variables"].items() if var.get("is_source_var")}
    class_counts: dict[str, int] = {}
    for r in regions.values():
        class_counts[r["signal_class"]] = class_counts.get(r["signal_class"], 0) + 1
    out = {
        "samples": n_samples,
        "class_counts": class_counts,
        "top_regions_by_rel_err": [
            {"location": loc, **regions[loc]}
            for loc in sorted(regions, key=lambda l: (-regions[l]["max_rel_err"], l))
        ][:10],
        "regions": regions,
        "variables": variables,
        # mirror finalize_report: chain_id-keyed dict -> deterministic list
        "cascade_chains": [a["cascade_chains"][cid]
                           for cid in sorted(a["cascade_chains"])],
    }
    outp = f"{mapdir}/out_{name}.json"
    Path(outp).write_bytes(_dumps(out))
    return name, outp


def merge_shard_files(paths, out_path, workers: int | None = None,
                      tmp_dir: str | None = None,
                      cfg: sr.ReducerConfig | None = None) -> dict:
    """Merge shard report files into a consolidated report at ``out_path``.

    Returns ``samples_seen`` so callers can assert every integral saw the
    expected sample count. Semantically identical to
    ``finalize_report(merge_reports(<all shards>))``.
    """
    paths = sorted(paths)
    if not paths:
        raise ValueError("no shard files to merge")
    cfg = cfg or sr.ReducerConfig()
    W = workers or min(32, os.cpu_count() or 8)
    mapdir = tmp_dir or (str(Path(out_path).parent / ".fast_merge_tmp"))
    if os.path.isdir(mapdir):
        shutil.rmtree(mapdir, ignore_errors=True)
    os.makedirs(mapdir, exist_ok=True)
    t0 = time.monotonic()
    try:
        groups = [paths[i::W] for i in range(W)]
        groups = [g for g in groups if g]

        samples_all: dict[str, int] = {}
        no_id = 0
        with ProcessPoolExecutor(max_workers=W) as ex:
            futs = [ex.submit(_map_shards, k, g, mapdir) for k, g in enumerate(groups)]
            for f in futs:
                r = f.result()
                for name, c in r["samples"].items():
                    samples_all[name] = samples_all.get(name, 0) + c
                no_id += r["no_id"]

        names = sorted(samples_all)
        out_paths: dict[str, str] = {}
        with ProcessPoolExecutor(max_workers=min(len(names), W)) as ex:
            futs = [ex.submit(_reduce_integral, n, samples_all[n], mapdir, cfg)
                    for n in names]
            for f in futs:
                name, outp = f.result()
                out_paths[name] = outp

        # stitch pre-serialized per-integral blobs -- never rebuild the full dict
        with open(out_path, "wb") as fo:
            fo.write(b'{"schema_version":' + _dumps(sr.SCHEMA_VERSION))
            fo.write(b',"kind":"stability_report"')
            fo.write(b',"samples_seen":' + _dumps(samples_all))
            fo.write(b',"no_id_records":' + _dumps(no_id))
            fo.write(b',"integrals":{')
            for i, name in enumerate(names):
                if i:
                    fo.write(b',')
                fo.write(_dumps(name) + b':')
                fo.write(Path(out_paths[name]).read_bytes())
            fo.write(b'}}')
        return samples_all
    finally:
        shutil.rmtree(mapdir, ignore_errors=True)
        gc.enable()
