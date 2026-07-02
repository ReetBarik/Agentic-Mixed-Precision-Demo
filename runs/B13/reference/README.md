# reference/

Semantic ground truth for the tracked interop shim.

## kokkosMaths_dd.h

Snapshot from `ReetBarik/qcdloop@ddfun_enabled` — the working
double-double precision-swap implementation. Every overload in
`../src/ql_tracked_interop.hpp` mirrors an entry here in name,
signature shape, and semantics.

When debugging a missing or misbehaving tracked overload, cross-check
against the corresponding DD overload in this file. The DD branch is
production-tested end-to-end (all Bnm groups, boxGPU tests pass), so
its semantics are authoritative.

**Do not build against this file.** It's here as documentation only.

To refresh:

```sh
gh api /repos/ReetBarik/qcdloop/contents/src/qcdloop/kokkosMaths_dd.h?ref=ddfun_enabled \
  --jq '.content' | base64 -d > runs/B13/reference/kokkosMaths_dd.h
```
