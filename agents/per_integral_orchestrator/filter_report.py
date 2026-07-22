"""Filter a consolidated v2 characterization report down to one integral.

The characterizer's ``stability_report`` is already top-level keyed by integral::

    {"schema_version": 2, "kind": "stability_report",
     "samples_seen": {"B1": 5000, ...},
     "no_id_records": ...,
     "integrals": {"B1": {"regions": {...}, "cascade_chains": [...], ...},
                   "B4": {...}, ...}}

So filtering to a single integral is structurally trivial: keep
``integrals[name]`` plus the top-level metadata, and narrow ``samples_seen`` to
that one integral.  The subtlety is *fidelity* — Phase A (commit ``5b6b82c``)
stamped every region and chain record with an explicit ``integral`` tag, and this
filter **asserts** every retained record carries the target tag, so a mis-keyed
report fails loudly instead of leaking another integral's signal into the pass.

Cascade chains are kept **whole**: a chain is a self-contained multi-line span
that lives entirely inside one integral (``chain_id`` is namespaced per integral —
see the Phase B probe), so keeping ``integrals[name]["cascade_chains"]`` verbatim
preserves chain integrity for Strategy's ``load_chains`` loader with no truncation.
"""

from __future__ import annotations

import json
from pathlib import Path


def _region_tags_ok(regions: dict, integral: str) -> list[str]:
    """Return region keys whose ``integral`` tag disagrees with ``integral``.

    A missing tag is tolerated (older/partial records default to the container's
    integral); only an explicit *disagreeing* tag is a fidelity violation.
    """
    bad = []
    for rkey, region in (regions or {}).items():
        tag = region.get("integral")
        if tag is not None and tag != integral:
            bad.append(f"region {rkey!r} tagged {tag!r}")
    return bad


def _chain_tags_ok(chains: list, integral: str) -> list[str]:
    bad = []
    for i, chain in enumerate(chains or []):
        tag = chain.get("integral")
        if tag is not None and tag != integral:
            bad.append(f"chain #{i} ({chain.get('chain_id')!r}) tagged {tag!r}")
    return bad


def filter_report(report_path: str | Path, integral: str,
                  out_path: str | Path) -> dict:
    """Write a single-integral view of ``report_path`` to ``out_path``.

    Keeps ``integrals[integral]`` and the top-level ``schema_version`` / ``kind``
    / ``no_id_records``; narrows ``samples_seen`` to the one integral.  Raises
    ``KeyError`` if the integral is absent and ``ValueError`` if any retained
    record's explicit ``integral`` tag disagrees with ``integral`` (fidelity
    guard).  Returns a small meta summary ``{integral, n_regions, n_chains,
    schema_version}``.
    """
    data = json.loads(Path(report_path).read_text())
    integrals = data.get("integrals", {})
    if integral not in integrals:
        raise KeyError(
            f"integral {integral!r} not in report {report_path} "
            f"(have {sorted(integrals)})")

    idata = integrals[integral]
    regions = idata.get("regions", {}) or {}
    chains = idata.get("cascade_chains", []) or []

    violations = (_region_tags_ok(regions, integral)
                  + _chain_tags_ok(chains, integral))
    if violations:
        raise ValueError(
            f"report {report_path} has {len(violations)} record(s) whose "
            f"integral tag != {integral!r} (filter fidelity violation): "
            + "; ".join(violations[:5])
            + (" ..." if len(violations) > 5 else ""))

    samples_seen = data.get("samples_seen", {})
    filtered = {
        "schema_version": data.get("schema_version"),
        "kind": data.get("kind"),
        "samples_seen": ({integral: samples_seen[integral]}
                         if integral in samples_seen else {}),
        "no_id_records": data.get("no_id_records"),
        "integrals": {integral: idata},
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Compact (no indent): a single-integral slice of the 820 MB report is still
    # tens of MB (large `variables` / `prov_vars` arrays kept for faithfulness);
    # pretty-printing would inflate it ~40% for a file no human reads by eye.
    out_path.write_text(json.dumps(filtered, separators=(",", ":")))

    return {
        "integral": integral,
        "n_regions": len(regions),
        "n_chains": len(chains),
        "schema_version": data.get("schema_version"),
    }
