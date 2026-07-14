# Vendored qcdloop headers

Snapshot from `ReetBarik/qcdloop@master`, commit `8de2089` (2026-06-08).

**Do not edit these files.** They are a verbatim upstream snapshot so
that the B13 spike faithfully represents qcdloop's production behavior.
Any additional ql:: definitions needed to compile B13 against tracked
types live in `../src/ql_tracked_interop.hpp` — kept out of these
headers on purpose to make the tracked-interop surface auditable.

To refresh:

```sh
cd $(git rev-parse --show-toplevel)
for f in boxGPU.h kokkosMaths.h kokkosMaths_wrapper.h kokkosUtils.h timer.h \
         box/B0m.h box/B1m.h box/B2m.h box/B3m.h box/B4m.h box/box_common.h; do
  gh api "/repos/ReetBarik/qcdloop/contents/src/qcdloop/$f?ref=master" \
    --jq '.content' | base64 -d > runs/B13/qcdloop_headers/$f
done
```

Then update the commit sha above and in `../README.md`.
