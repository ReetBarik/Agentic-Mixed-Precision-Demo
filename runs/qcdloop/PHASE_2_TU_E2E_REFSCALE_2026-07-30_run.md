# Strategy run `20260730_053618_84184a19` (whole-TU-only)

- **status:** success
- **mode:** tu_only (no Patcher LLM, no region/chain walk)
- **tolerance:** 7.0 precise digits
- **duration:** 105.19 s
- **starting SHA:** `None`
- **final branch:** `strategy/20260730_053618_84184a19`
- **iterations:** 42 (see `/home/rbarik/Agentic-Mixed-Precision-Demo/runs/qcdloop/strategy/20260730_053618_84184a19/iterations.jsonl`)

## Routing table

`base` = raw-double p100; `dd`/`float`/`ff` = candidate p100 (min over samples/components); `—` = not attempted.

| integral | base | dd | float | ff | route |
|---|---|---|---|---|---|
| B1 | 11.808 | —n | — | 9.264A | **ff** |
| B10 | 10.093 | —n | — | 7.891A | **ff** |
| B11 | 9.460 | —n | — | 7.769A | **ff** |
| B12 | 3.691 | 14.331A | — | 2.406r | **dd** |
| B13 | 8.578 | —n | — | 7.269A | **ff** |
| B14 | 13.038 | —n | — | 10.821A | **ff** |
| B15 | 0.000 | 0.000A | — | 0.000r | **dd** |
| B16 | 6.564 | 12.551A | — | 5.099r | **dd** |
| B2 | 12.142 | —n | — | 10.045A | **ff** |
| B3 | 12.271 | —n | — | 9.502A | **ff** |
| B4 | 10.250 | —n | — | 8.423A | **ff** |
| B5 | 11.585 | —n | — | 9.045A | **ff** |
| B6 | 12.269 | —n | — | 10.105A | **ff** |
| B7 | 11.626 | —n | — | 10.182A | **ff** |
| B8 | 10.139 | —n | — | 8.593A | **ff** |
| B9 | 11.530 | —n | — | 8.642A | **ff** |
| BIN0 | 0.000 | 0.000r | — | 0.000r | **double** |
| BIN1 | 8.068 | —n | — | 0.000r | **double** |
| BIN2 | 9.383 | —n | — | 0.000r | **double** |
| BIN3 | 9.195 | —n | — | 7.487A | **ff** |
| BIN4 | 9.038 | —n | — | 0.000r | **double** |

## Precision distribution

| precision | integrals |
|---|---|
| float | 0 |
| ff | 14 |
| double | 4 |
| dd | 3 |
| **total** | 21 |

## Two-phase walk

| phase | measures | accepts |
|---|---|---|
| correctness (dd) | 21 | 3 |
| speedup (float→ff) | 21 | 14 |
