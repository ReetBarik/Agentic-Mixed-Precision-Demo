# Vendored qcdloop headers

Two-source snapshot representing the qcdloop library the Agentic pipeline
is optimizing:

| file(s) | source | purpose |
|---|---|---|
| `boxGPU.h`, `kokkosMaths.h`, `kokkosMaths_wrapper.h`, `kokkosUtils.h`, `timer.h`, `box/*.h` | `ReetBarik/qcdloop@master`, commit `8de2089` (2026-06-08) | double-precision qcdloop primary — the library under test |
| `kokkosMaths_dd.h` | `ReetBarik/qcdloop@ddfun_enabled`, commit `2229ec4` (path: `src/qcdloop/kokkosMaths_dd.h`) | qcdloop's own dd-precision `Constants<T>` (43-term Chebyshev `_C`, 25-term Bernoulli `_B`, dd `_pi()` via `ddfun::dd_pi()`, dd-appropriate tolerances). Consumed by the Agentic pipeline as source input when synthesizing dd support for leaf-callee clones (`Lnrat_B10`, `ddilog_B10`, etc.). Analogous to how the mainline `scarrazza/qcdloop:tools.cc` publishes both 19-double and 43-quadmath `_C` tables side by side. |

`kokkosMaths_dd.h` carries a one-line namespace shim at the top
(`namespace ql { namespace ddfun = ::quad::ddfun; }`) so the fork's
authorship — written against `ql::ddfun` — resolves against this repo's
vendored `quad::ddfun` primitives at `third_party/include/`. The header
body is otherwise verbatim from `qcdloop@ddfun_enabled`.

**Edit policy.** Keep both files as verbatim mirrors of their respective
upstream sources (modulo the documented namespace shim). Refresh via the
scripts below rather than local edits; any pipeline-side dd support the
tables don't cover (e.g. Class-1 wrappers like `ql::kAbs`/`ql::kLog` at
dd) is synthesized by the pipeline, not written here.

## Refresh

Double-precision headers (from `qcdloop@master`):

```sh
cd $(git rev-parse --show-toplevel)
for f in boxGPU.h kokkosMaths.h kokkosMaths_wrapper.h kokkosUtils.h timer.h \
         box/B0m.h box/B1m.h box/B2m.h box/B3m.h box/B4m.h box/box_common.h; do
  gh api "/repos/ReetBarik/qcdloop/contents/src/qcdloop/$f?ref=master" \
    --jq '.content' | base64 -d > runs/qcdloop_headers_full/$f
done
```

dd-precision `Constants<T>` (from `qcdloop@ddfun_enabled`):

```sh
gh api "/repos/ReetBarik/qcdloop/contents/src/qcdloop/kokkosMaths_dd.h?ref=ddfun_enabled" \
  --jq '.content' | base64 -d > runs/qcdloop_headers_full/kokkosMaths_dd.h
# Then re-apply the namespace shim at the top of the file:
#   namespace ql { namespace ddfun = ::quad::ddfun; }
```

After either refresh, bump the commit sha(s) in the table above and in
`../README.md`.
