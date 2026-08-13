# Pipeline walkthrough slides

Ground-up walkthrough of the qcdloop mixed-precision pipeline (PIPELINE_v1
as-is). Numerics-literate audience. One SVG per slide, 1280×720 (16:9),
importable directly into PowerPoint 2016+ (Insert → Pictures → This Device).

The deck is a presentation artifact, not a spec: it follows one worked
6-region example end to end, and simplifies wherever a simplification makes
the picture clearer than the code would.

## Slide order

1. `01_toy_app.svg` — **Toy app: 6-region kernel.** Source with tracked
   scopes, region DAG, per-region signal table. R1…R6 are the running
   example for every slide that follows.
2. `02_tracked_module.svg` — **Tracked datatype.** What it records per op,
   the cond-number proxy formulas, and its role in the pipeline.
3. `03_characterizer_report.svg` — **Characterizer: sweep → journal →
   report.** Sampled sweep at plain double, journal keyed back to region
   scopes, forward-error cascade formula.
4. `04_correctness_walk.svg` — **Strategy walk, correctness phase.** Queue
   1 of 2: the regions that miss tolerance at double, each up-bumped to the
   cheapest rung that clears. R4 double → qf accepts; R2 double → qf is
   pruned by the range guard before any build, then double → dd accepts.
5. `05_speedup_walk.svg` — **Strategy walk, speedup phase.** Queue 2 of 2:
   the regions that already meet tolerance at double, heaviest first, each
   single-stepping down until a rung rejects. R6 → float, R5 rejects its
   first step and settles at double, R1 → float, R3 → ff.
6. `06_patcher_validator.svg` — **Patcher & Validator, inside one walk
   step.** Zoom into R4's double → qf attempt: shim generation, the retry
   loop, whole-sample-space validation, verdict, state commit.
7. `07_big_picture.svg` — **Pipeline, end-to-end data flow.** All modules
   with named artifacts on each edge; pink borders mark the two LLM
   touchpoints.
8. `07b_rewrite_catalog.svg` — **Future work: algorithmic rewrite
   catalog.** Same loop, but the Patcher attempts a compensated-arithmetic
   rewrite first and treats the precision containers as fallbacks.

## Two things worth knowing before presenting

**The qf rung.** The ladder is `float → ff → double → qf → dd`
(`agents/strategy/models.py:37`). qf is quad-float — four FP32 limbs,
~96-bit significand, ~29 digits — so it sits above double in *precision*
while still living in the FP32 *range*. Range is therefore not monotone
along the ladder, and a range-flagged region has every FP32-family rung
(`float`, `ff`, `qf`) dropped from its up-targets before any build is
attempted. That is exactly what happens to R2 on slide 04: the pruned
attempt gets a grey verdict, with no build and no validation.

qf also has no region-level integrator (`REGION_REALIZABLE` omits it), so
in the real pipeline it is reachable only via a whole-TU flip. The deck
shows it as an ordinary region-level rung; that is one of the deliberate
simplifications.

The worked example ends with all five rungs occupied — float ×2 (R1, R6),
ff ×1 (R3), double ×1 (R5), qf ×1 (R4), dd ×1 (R2). Showing the mixed-ness
is the point of the deck.

**Validation covers the whole sample space.** There is no random battery
and no tail battery. The Validator replays *all N* characterization samples
(seed 12345, OpenMP-parallel across cores), scores per-output-component
precise digits against the dd oracle, and applies an absolute floor of
tol = 7 digits plus a regression guard against the current baseline. dd
remains the oracle at every rung — a qf candidate is scored against dd,
never against itself.

A ninth slide, `07a_iterative_loop.svg`, used to sit between 07 and 07b and
illustrated an iterative resample/re-validate loop. It was deleted: with
validation already covering the whole sample space, there is no such loop.

## Rendering and checking

XML well-formedness plus a raster render:

```sh
python3 -c "import xml.dom.minidom; xml.dom.minidom.parse('docs/slides/06_patcher_validator.svg'); print('XML OK')"
convert -background white docs/slides/06_patcher_validator.svg /tmp/s06.png
```

Always eyeball the PNG afterwards. The recurring defect in this deck is
text overflowing a panel edge or the 1280px canvas — SVG has no text
metrics at author time, so nothing catches it but looking. Rough budgets
for the shared classes: `.sub-sm` (11px sans) ≈ 5.4px/char, `.catalog-item`
and `.edge-lbl` (11px mono) ≈ 6.6px/char.

`convert` picks up the font aliases in `~/.config/fontconfig/fonts.conf`;
without them the `ui-sans-serif` / `ui-monospace` stacks fall back to
something much wider and the render will look broken in ways the slide
isn't.

## Viewing on GitHub

GitHub renders SVGs natively when the file is opened directly (not from
the tree view). Click through the file link to preview.
