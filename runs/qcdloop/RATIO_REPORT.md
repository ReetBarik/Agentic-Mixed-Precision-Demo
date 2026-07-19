# Op-count ratio report — Wave-1+2 10k walk

- **run:** `20260719_185132_033cbe69` · langgraph-agents @ `3fd5ad3` · CALIBRATION_v2
- **workload:** 21 integrals × 10k samples (characterization report `report_10k.json`)
- **final precision distribution:** double 266 · dd 134 · float 86 · ff 2 (488 regions)
- **op-count source:** `integrals.<B>.regions` (per-`file:line` dynamic op counts); `top_regions_by_rel_err` and `cascade_chains` excluded to avoid double-counting

## TL;DR

Baseline (vanilla all-double) is **100.00% double / 0.00% float** arithmetic throughput. After the walk, throughput is **98.95% double flops / 1.05% float flops** (primitives-only, source-exact) — or **99.87% / 0.13%** if the data-dependent dd transcendental (log/atan2) expansions are included.

**Interpretation:** in raw arithmetic-throughput terms the walk's footprint is dominated by the **dd correctness expansion**, not by the float speedup. Of the 194,205,502 logical ops, **65.7% moved to dd** (double→double-double, ~11–44× flops each) and only **7.8% moved to float/ff** (speedup). Double-double is expensive, so total flops *inflate* 14.4× (primitives) to 68.0× (with transcendentals) — the float slice is real but small in throughput share. The conclusion is robust to the transcendental estimate: float share stays ≈0.1–1.2% under any plausible dd log/atan2 cost.

## Multipliers (WI1)

Flops per one logical op. Primitives are exact static counts from the header bodies (`dd_math.hpp` / `ff_math.hpp`); **no FMA** (Dekker TwoProduct splitting for mul, long-division renormalization for div). Transcendentals are expanded through the header definitions but contain data-dependent loops → **documented estimates**.

| op-kind | native double | native float | dd (double flops) | ff (float flops) | source |
|---|---|---|---|---|---|
| add | 1 | 1 | 11 | 11 | TwoSum |
| sub | 1 | 1 | 11 | 11 | TwoSum |
| mul | 1 | 1 | 32 | 32 | Dekker TwoProduct (no FMA) |
| div | 1 | 1 | 42 | 42 | long-division renorm |
| sqrt | 1 | 1 | 44 | 44 | Newton + ddmuldd |
| neg | 1 | 1 | 2 | 2 | sign flip hi,lo |
| abs | 1 | 1 | 2 | 2 | branch |
| log | 20 | 20 | 4100 *(est)* | 2350 *(est)* | **est** 3×Newton(exp) |
| atan2 | 30 | 30 | 4300 *(est)* | 2500 *(est)* | **est** 3×Newton(sincos) |

- **FMA path:** not used. `ddmul`/`ffmul` use Dekker splitting (splitter 2²⁷+1 / 2¹³+1); `dddiv`/`ffdiv` use the long-division renormalization sequence (2 hardware divides).
- **sqrt** counts one native `sqrt` instruction as 1 flop plus the dd/ff refinement.
- **Transcendentals estimated, not extracted:** dd `log` ≈ 20 (native seed) + 3 Newton steps × [dd `exp` ≈1300 + 64]; dd `atan2` (`ddang`) ≈ setup(233) + 3 Newton × [dd `sincos` ≈1300 + 64]. Native libm log≈20 / atan2≈30 flops. ff log/atan2 shown for completeness but **unused** (no ff region carries a transcendental).

## Op-count matrix — raw logical ops (WI2, pre-multiplier)

Rows = op-kind, columns = final precision. Cells = dynamic op count over the workload. Grand total is the conservation anchor (every op counted exactly once; retyping moves ops between columns but never creates or destroys them).

| op-kind | double | dd | float | ff | row total |
|---|--:|--:|--:|--:|--:|
| abs | 3,441,746 | 260,990 | 71,498 | 0 | 3,774,234 |
| add | 8,644,612 | 30,070,321 | 3,703,255 | 130,002 | 42,548,190 |
| atan2 | 5,098 | 2,236,016 | 4 | 0 | 2,241,118 |
| div | 6,413,901 | 5,156,906 | 1,860,012 | 30,000 | 13,460,819 |
| log | 1,338,995 | 2,511,715 | 30,052 | 0 | 3,880,762 |
| mul | 17,485,843 | 51,121,681 | 5,632,976 | 230,004 | 74,470,504 |
| neg | 3,468,689 | 7,059,450 | 360,000 | 0 | 10,888,139 |
| sqrt | 755,032 | 3,233,766 | 247,501 | 80,002 | 4,316,301 |
| sub | 9,987,816 | 25,856,849 | 2,760,770 | 20,000 | 38,625,435 |
| **col total** | 51,541,732 | 127,507,694 | 14,666,068 | 490,008 | **194,205,502** |
| **share** | 26.54% | 65.66% | 7.55% | 0.25% | 100% |

**Conservation:** row totals sum to 194,205,502 = the baseline double op count (vanilla app runs every one of these at double). ✓

## Reduced matrix — after multipliers (WI2)

Cells expressed in **flops** after applying the WI1 multipliers. Baseline column = every op at native double. `double` and `float` after-columns are native (1 flop/op for primitives); `dd`/`ff` are expanded. Column sums give the app-wide totals.

### Primitives only (source-exact, excludes log/atan2)

| op-kind | baseline dbl | after dbl (native+dd) | after float (native+ff) |
|---|--:|--:|--:|
| abs | 3,774,234 | 3,963,726 | 71,498 |
| add | 42,548,190 | 339,418,143 | 5,133,277 |
| div | 13,460,819 | 223,003,953 | 3,120,012 |
| mul | 74,470,504 | 1,653,379,635 | 12,993,104 |
| neg | 10,888,139 | 17,587,589 | 360,000 |
| sqrt | 4,316,301 | 143,040,736 | 3,767,589 |
| sub | 38,625,435 | 294,413,155 | 2,980,770 |
| **total** | **188,083,622** | **2,674,806,937** | **28,426,250** |

- after-walk throughput: **98.95% double / 1.05% float**  (total 2,703,233,187 flops, 14.37× baseline)

### Including transcendentals (dd log/atan2 estimated — flagged)

| op-kind | baseline dbl | after dbl (native+dd) | after float (native+ff) |
|---|--:|--:|--:|
| abs | 3,774,234 | 3,963,726 | 71,498 |
| add | 42,548,190 | 339,418,143 | 5,133,277 |
| atan2 *(est)* | 67,233,540 | 9,615,021,740 | 120 |
| div | 13,460,819 | 223,003,953 | 3,120,012 |
| log *(est)* | 77,615,240 | 10,324,811,400 | 601,040 |
| mul | 74,470,504 | 1,653,379,635 | 12,993,104 |
| neg | 10,888,139 | 17,587,589 | 360,000 |
| sqrt | 4,316,301 | 143,040,736 | 3,767,589 |
| sub | 38,625,435 | 294,413,155 | 2,980,770 |
| **total** | **332,932,402** | **22,614,640,077** | **29,027,410** |

- after-walk throughput: **99.87% double / 0.13% float**  (total 22,643,667,487 flops, 68.01× baseline)

## Delta view — baseline → after

Baseline is 100% double / 0% float by construction. The shift is driven by two opposite forces: **dd expansion** inflates the double column (correctness), while **float/ff** conversions add a float column (speedup). Top op-kinds by their contribution to each:

**Double-flop inflation from dd (top contributors):**

| op-kind | dd ops | × mult | dd double flops added |
|---|--:|--:|--:|
| log *(est)* | 2,511,715 | 4100 | 10,298,031,500 |
| atan2 *(est)* | 2,236,016 | 4300 | 9,614,868,800 |
| mul | 51,121,681 | 32 | 1,635,893,792 |
| add | 30,070,321 | 11 | 330,773,531 |
| sub | 25,856,849 | 11 | 284,425,339 |

**Float-flop creation (top contributors):**

| op-kind | after float flops | native float | ff expand |
|---|--:|--:|--:|
| mul | 12,993,104 | 5,632,976 | 7,360,128 |
| add | 5,133,277 | 3,703,255 | 1,430,022 |
| sqrt | 3,767,589 | 247,501 | 3,520,088 |
| div | 3,120,012 | 1,860,012 | 1,260,000 |
| sub | 2,980,770 | 2,760,770 | 220,000 |

## Per-precision contribution — where the flops come from

- **Float column (29,027,410 flops):** 15,237,172 (52.5%) from **native float** (the 86 float lines, 1:1 op count) + 13,790,238 (47.5%) from **ff expansion** (the 2 ff lines, ~11–44×).
- **Double column, after (22,614,640,077 flops):** 77,130,479 (0.3%) **native double** (unconverted lines) + 22,537,509,598 (99.7%) **dd expansion** — the dd expansion is where essentially all of the throughput went.
- **Primitives-only double column (2,674,806,937 flops):** 50,197,639 native + 2,624,609,298 dd expansion (98.1% dd).

## Caveats

- **Workload-conditional.** Op counts are for the 21-integral × 10k-sample characterization workload; a different sample count or integral mix reweights the rows.
- **Theoretical arithmetic throughput, NOT measured wall speedup.** The app is one monolithic TU and is build-bound; this report is a flop-accounting exercise, not a timing.
- **dd/ff multipliers are algorithm-specific.** They reflect these headers' choices: Dekker TwoProduct (no FMA) for mul, long-division renormalization for div, Newton+ddmuldd for sqrt. A FMA-based dd library would roughly halve the mul multiplier.
- **Transcendental estimates are flagged.** dd `log`/`atan2` (≈4100/4300) carry data-dependent loop counts; they are the *only* estimated multipliers and they dominate the "full" number. The primitives-only ratio is the robust figure — and the qualitative conclusion (float share ≈0.1–1.2%, dd dominates) holds under any plausible transcendental cost.
- **Distinct-line reconciliation:** matched 222/222 precision-assignment lines to report regions — dd 134 · float 86 · ff 2, exactly the CALIBRATION_v2 distribution (all assigned lines resolved).
- **Non-localizable dd lines counted at dd.** 6 dd-assigned lines (`kokkosUtils.h:254/666/673/748/754`, `B4m.h:209`) are flagged `non_localizable` by the characterizer (inlined utility-header lines reached from many callers) yet carry large op counts — `kokkosUtils.h:254` alone is ~14M ops across BIN0–4. Because the walk edited those source lines to dd (propagated via `required_by`, plus `B4m.h:209` = direct accept iter_82), their ops are counted at dd, not double. Treating them as double instead would shrink the dd column but not change the qualitative result.
- **cascade_chains excluded.** The ~527k cascade-chain entries re-attribute ops already counted in `regions`; including them would multiply-count the same arithmetic. The `regions` keys (one per distinct `file:line` per integral, plus one empty-key catch-all) are taken to partition each integral's op set; under that assumption their op counts sum to the 194,205,502 conservation total used throughout.

_Generated by `runs/qcdloop/ratio_report.py` · scan 28.8s · 1314 keyed + 21 empty-key region instances (75 flagged non-localizable)._
