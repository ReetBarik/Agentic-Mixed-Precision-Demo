# Pipeline walkthrough slides

Ground-up walkthrough of the qcdloop mixed-precision pipeline (PIPELINE_v1
as-is). Numerics-literate audience. One SVG per slide, 1280×720 (16:9),
importable directly into PowerPoint 2016+ (Insert → Pictures → This Device).

## Slide order

1. `01_toy_app.svg` — 6-region toy kernel: source with tracked scopes,
   region DAG, per-region signal table.
2. `02_tracked_module.svg` — Tracked datatype foundation: what it records
   per op, cond-number proxy formulas, role in the pipeline.
3. `03_characterizer_report.svg` — Characterizer sweep + reduction: journal
   → region-keyed report, with forward-error cascade formula.

More slides pending (correctness walk, speedup walk,
patcher+integrators+validator, optional big-picture summary).

## Viewing on GitHub

GitHub renders SVGs natively when the file is opened directly (not from
the tree view). Click through the file link to preview.
