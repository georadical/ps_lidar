# Sweep Classification for Radiata Pine — Authoritative Literature Review

**Date**: 2026-05-28
**Purpose**: Ground the sweep-classification thresholds used by
`src/core/stem_description.py::classify_sweep_zones` in authoritative
forestry literature rather than only the Interpine HQP quickcard
(`HQP Quickcard LiDAR pt 1.pdf`, this folder).

This document is the source of truth for **why** the PS_LiDAR pipeline
treats some sweep thresholds as rigid (the amplitudes) and others as
soft / operational (the section lengths).

---

## 1. Authoritative anchor: MPI log-grade tolerances

**Source**: NZ Ministry for Primary Industries (MPI), *"NZ log grades
for radiata pine"*, last reviewed 27-01-2026.
URL: https://www.mpi.govt.nz/forestry/forest-industry-and-workforce/forestry-wood-processing-data/new-zealand-log-grades-for-radiata-pine

MPI publishes sweep tolerances **as fractions of SED** (Small End
Diameter) per log grade. These are the only sweep numbers published by
the NZ government and apply at grade gates:

| MPI grade                              | Sweep tolerance |
|----------------------------------------|-----------------|
| Pruned (P40/P1, P30/P2, P35)           | SED/4           |
| Pruned Peeler (PP)                     | SED/8           |
| Structural (S40/S1, S30/S2, S20/S3)    | SED/4           |
| Industrial (KI), Export pulp (KIS)     | SED/3           |

**MPI does NOT publish the codes `8 / L / S / 3 / 1 / X`** and does
**NOT** define minimum section lengths per code. The amplitudes that
the PS_LiDAR classifier uses (SED/8, SED/5, SED/3, SED/1) are
anchored to MPI through these grade-tolerance ratios.

---

## 2. The 8/L/S/3/1/X codes are Interpine's operational convention

**Source**: Interpine, *"Better predicting yield of long logs vs short
logs of the same grade quality"*, 26-Nov-2013.
URL: https://interpine.nz/better-predicting-yield-of-long-logs-vs-short-logs-of-the-same-grade-quality/

Interpine introduced the shorts-vs-longs distinction explicitly as an
**operational tool for segregation**, not as a forestry standard. From
the article:

- **Shorts** ≤ 6.1 m: gentle sweep, may wobble in > 1 direction over
  longer lengths.
- **Longs** up to 12 m: gentle sweep, single consistent direction.

The article's own framing is "choose the option that best segregates"
— i.e., the 6.1 m boundary between `L` (longs) and `S` (shorts) is the
**operational length below which a stem still produces useful shorter
logs but cannot be relied on for long-log products**. The codes in the
quickcard (`L` ≥ 6 m, `S` ≥ 4 m, `3` over 4 m of moderate sweep) are
the field implementation of that segregation. They are intentionally
flexible.

**Practical consequence for PS_LiDAR**: an automated classifier should
treat the length thresholds as soft. A `8` zone of 4-5 m is still a
valid "gun-barrel straight" segment; an `S` zone of 4.5 m is still
short-log-only; an `L` zone of 5.5 m is acceptable when the situation
warrants. Rigidly demanding 6.0 m for `L` and 4.0 m for `S` would
generate false-negative classifications.

---

## 3. What is NOT in the authoritative literature

The review checked the standard sources for explicit publication of
the `8 / L / S / 3 / 1 / X` codes or their minimum lengths:

- **NZFOA / Pine Products NZ**: *"Grading Guidelines for Radiata Pine"*
  exists (https://www.pineproducts.co.nz/resources/file/picker/62fc4058642d5.pdf)
  but content of the PDF was not parseable via fetch tools. Existence
  confirmed; specific length-per-code thresholds unverified. Worth a
  direct read by a team member with access.
- **NZS 3631:1988** covers sawn timber, not standing logs. No sweep
  codes.
- **AS / NZS standards**: no published "8/L/S/3/1/X" codes found.
- **PlotSafe (Silmetra) manual**: referenced by Interpine
  (https://interpine.nz/15-plotsafe-forest-inventory-procedures-manual-available/)
  but the download is gated. Likely replicates the quickcard codes
  since the field tally feeds PlotSafe — but cannot be verified
  without direct access to the manual.
- **Scion / Forest Research NZ**:
  - Park (1980), *NZJFS* 10(2):419-438 — foundational NZ log-grading
    paper; full text was not accessed.
  - Cown et al. (1984), *NZJFS* 14(1):109-123, *"Timber recovery from
    pruned Pinus radiata butt logs at Mangatu"* — quantifies sweep
    effect on recovery (straight 57.9%, moderate 52.0%, severe 46.1%)
    but does not codify operational classes.
  - Marshall & Murphy (2004), *NZJFS* 34(2) — stem scanning systems
    for bucking; sweep is a model input but no operator codes defined.
- **Goulding bucking work** (Scion, 1990s, AVIS bucking-optimisation)
  treats sweep as a continuous variable fed to an optimiser, not as
  numerical codes by section length.
- **Australian sources**: Ivković et al. (2007), *Australian Forestry*
  70(3):173-184, *"Modelling the effects of stem sweep, branch size
  and wood stiffness of radiata pine on structural timber production"*
  quantifies the yield loss (~7 % per 0.1 increase in sweep:diameter
  ratio) but does not publish categorical codes. ForestrySA Log
  Standard Specification Manual exists (PDF not parsed) but is South
  Australian, not NZ.

---

## 4. Identified literature gap

No peer-reviewed study was found on **inter-operator variability in
sweep classification for radiata pine**. Adjacent literature (e.g.
Murphy 2009, *IJFE* 20(2)) covers manual measurement error for length
and diameter but not for sweep codes. This is a genuine open question
in the field — the kind of dataset PS_LiDAR could eventually
contribute to once it is calibrated against operator tallies on the
same plots.

---

## 5. Implications for the PS_LiDAR sweep classifier

| Parameter             | Authority | Treatment in pipeline |
|-----------------------|-----------|-----------------------|
| Amplitude thresholds  | MPI       | **Rigid.** Do not adjust without a counter-citation. |
| `L` length minimum    | Interpine 2013 (soft) | Default 5.0 m, parametrisable. Quickcard says 6 m; we accept 5-7 m. |
| `S` length minimum    | Interpine 2013 (soft) | Default 3.0 m, parametrisable. Quickcard says 4 m; we accept 3-5 m. |
| `3` length minimum    | Quickcard (soft)      | Default 3.0 m. |
| `8` length minimum    | Quickcard (soft)      | Default 3.0 m. Operationally "gun-barrel straight" implies a long stretch but does not have a hard 6 m floor. |
| `1` length minimum    | Quickcard (soft)      | Default 2.0 m. |
| `X` length            | Quickcard (defining) | 0.3-1.0 m. The X code is defined precisely as a short severe section, so it is exempt from any length floor. |

These soft minimums are encoded as a single `min_zone_length_m`
default that the classifier (step F1.2 of the Phase 1B follow-up plan)
applies after the upgrade-rule pass to absorb noise-length zones into
their higher-severity neighbour. The defaults are tunable per plot if
operator calibration warrants.

## 6. Symmetric noise-floor absorption rule — justification

The zonal classifier (`classify_sweep_zones`) enforces minimum zone
lengths **symmetrically** (F1.3, supersedes the F1.2 asymmetric rule):

> **Any non-X zone shorter than its code's operative minimum is absorbed
> into the higher-severity neighbour (worst-wins). Boundary zones — a
> single neighbour — are absorbed too. `X` is exempt; its 0.3-1 m
> short-severe definition is intrinsic.**

### From asymmetric (F1.2) to symmetric (F1.3) — what changed and why

The earlier F1.2 rule was asymmetric: only *better*-than-both-neighbours
zones were absorbed (the upgrade direction), under the binding-constraint
principle that "defects don't average out". A literal reading of the
Interpine HQP quickcard's UPGRADE RULES box supports this:

> "Only Upgrade Branch or Sweep Class (to **Smaller Branch Class or
> Better Sweep Code**) if > 3 m Section Between Zones of Lower Quality."
>
> — *HQP Quickcard LiDAR pt 1.pdf*, UPGRADE RULES box.

The rule explicitly gates only *upgrades*; it says nothing about
downgrades. F1.2 read that as "a worse zone keeps its code at any
length".

The first run on real-plot data (T460298B, 51 trees, 223 sweep zones)
revealed the literal-asymmetric reading produces too much sub-operative
noise: **131 / 223 zones (59 %)** were below their code minimum, almost
all of them worse-than-neighbour zones preserved by the asymmetric rule:

```
Code   Violations / total   Median zone length
S      52 / 69              1.6 m
8      58 / 99              3.5 m
3      20 / 38              2.6 m
1       1 / 10              4.9 m
```

A field operator producing a PlotSafe stem tally would never call out a
0.4-1 m S zone or a 0.6 m "3" zone — they sit below the operational
resolution of the manual cruising protocol. The Interpine upgrade rule's
**> 3 m threshold itself implicitly assumes operative lengths**: zones
below 3 m are outside the framework rather than zones whose downgrade
status the rule chooses to ignore.

The F1.3 rule reinterprets the principle:

> "Defects don't average out" applies **only to defects at operative
> length** (≥ the code's minimum). Sub-operative zones are polyline
> noise — at the polyline sampling resolution (0.2 m per node from the
> 0.2 m section step), a 0.4-2 m zone is 2-10 nodes, well below the
> stability threshold for amplitude classification. They are absorbed
> into the surrounding code rather than preserved as defects.

Real defects (≥ code minimum) are still preserved by construction — a
3 m "3" or a 4 m "1" sits at its minimum and is not subject to absorption.
Recovery-loss literature (Cown et al. 1984; Ivković et al. 2007) still
matters at operative lengths but the data the operator can act on starts
at those lengths anyway.

### Absorption direction — worst-wins (conservative)

When a short zone has two neighbours, the **higher-severity** code wins:
the short zone is replaced with the worse neighbour's code. Ties (same
severity on both sides) resolve to the preceding neighbour for stability.
At a boundary (one neighbour) the short zone absorbs into that single
neighbour regardless of severity.

This is the *conservative* direction: if the algorithm cannot resolve
a sub-operative zone, it errs toward the worse code, never artificially
upgrading. The cascade behaviour (absorbed zones can shift their former
neighbour's situation) is allowed to iterate to a fixed point.

### Worked examples

Severity order (best → worst): `8 < L = S < 3 < 1 < X`.
Per-code minimum lengths: `8: 4 m, L: 4 m, S: 4 m, 3: 3 m, 1: 2 m`;
`X` exempt.

**Example A — short better zone absorbed (boundary):**
Input zones `[1 (0-3 m), 8 (3-4 m), 1 (4-9 m)]`.
- The `8` is 1 m. Worst-neighbour wins → `1`.
- → `[1 (0-9 m)]`.
- *Forestry reading*: a 1 m straight blip in a sweepy stem does not
  yield a merchantable straight log; the section is operationally `1`.

**Example B — short worse zone absorbed (the F1.3 change vs F1.2):**
Input zones `[8 (0-5 m), 3 (5-7 m), 8 (7-12 m)]`.
- The `3` is 2 m, below its 3 m minimum. Both neighbours `8` (sev 0).
- Worst-neighbour wins → `8`. Absorbed.
- → `[8 (0-12 m)]`.
- *Forestry reading*: a 2 m moderate-sweep defect sits below the
  operative resolution of the tally; the operator would call out the
  whole 12 m as a clean `8`.
- Covered by
  `tests/test_stem_description.py::TestClassifySweepZones::
  test_short_worse_zone_absorbed_symmetrically`.

**Example C — short boundary zone absorbed:**
Input zones `[8 (0-1 m), L (1-7 m), 8 (7-8 m)]`.
- Both `8` caps are 1 m, single-neighbour boundary, < 4 m min.
- Single neighbour L (sev 1). Absorbed.
- → `[L (0-8 m)]`.
- *Forestry reading*: the boundary `8` caps are chord-endpoint
  artifacts of the global-chord amplitude metric, not real clean
  sections. Filtering them out matches the operator's call.
- Covered by `test_short_zone_at_boundary_absorbed`.

**Example D — reclassification within the same amplitude class
(unchanged from F1.2):**
A SED/5 zone of 4 m (ratio in (1/8, 1/5]).
- Not an absorption — same amplitude class.
- `L` requires ≥ 5 m (long-log capable). 4 m < 5 m → relabelled `S`.
- Just renamed to the code that matches its length.
- Covered by `test_short_sed5_zone_reclassified_to_S`.

**Example E — exempt severe section:**
Input zones `[8 (0-5 m), X (5-5.5 m), 8 (5.5-10 m)]`.
- `X` is exempt from the length floor.
- → preserved exactly: `[8 (0-5 m), X (5-5.5 m), 8 (5.5-10 m)]`.
- *Forestry reading*: the `X` code is **defined** for 0.3-1 m severe
  sections; the operator calls these out specifically because they
  prevent a log from being placed on a truck.
- Covered by `test_short_X_zone_preserved`.

### Future normative-traceability hook

A derived `MPI_grade` column on each 6 m log window of the stem
(mapping the worst amplitude observed in that window to `P1` / `PP` /
`S1` / `KI` per MPI's SED-fraction tolerances) would give a directly
traceable inventory output independent of the Interpine code
conventions. Out of scope for F1.2 but planned for after F2.

---

## 7. Direction pattern (L / S / W / K) — a shared, mostly-unimplemented discriminator

The quickcard separates the gentle-sweep codes **not by amplitude and
length alone, but also by direction pattern**:

| Code | Amplitude | Length / window | Direction | Outcome |
|------|-----------|-----------------|-----------|---------|
| `L`  | SED/5     | ≥ 6 m           | **consistent (single) direction** | OK for logs > 6.1 m |
| `S`  | SED/5     | ~ 4 m           | **back and forth** (reversals) | short logs < 6.1 m |
| `W`  | **> 5 cm absolute** (not an SED fraction) | 4 m window | **back and forth**, larger | "generally pulp quality" |
| `K`  | —         | single segment  | **sharp** single direction change | — |

`L` is the *only* consistent-direction sweep. `S`, `W`, `K` all hinge
on direction reversals / sharp direction change. Note `W` is measured
in **absolute centimetres (> 5 cm)**, unlike the SED-fraction geometry
codes — confirmed from the quickcard W panel ("Wobble / Movement Back
and Forth / > 5 cm", 4 m window).

### Implementation status

**F1.5a (DONE)** — direction-aware L/S split. A new primitive
`_polyline_direction_metrics` in `src/core/stem_description.py`
returns `n_bows` (direction reversals on the lateral-offset profile
with a 1 cm prominence deadband), `max_abs_offset_m`, and
`max_turn_deg`. `classify_sweep_zones` step 5 now decides L vs S by
`n_bows` (`≤ 1` → L consistent; `≥ 2` → S back-and-forth) — length is
no longer the L/S criterion. The metric is computed **once globally
per centerline** because a back-and-forth pattern manifests as
multiple SED/5 zones with axis crossings between bows; slicing
per-zone would always read each zone as monotonic. **S has no maximum
length** under the Interpine convention — a 10 m back-and-forth stays
`S`. F1.4 noise-floor minimums still apply in step 6.

**F1.5b (DONE)** — `W` and `K` detection wired in:
- `W` upgrade in step 5: a back-and-forth verdict (`n_bows ≥ 2`)
  with `max_abs_offset_m > 5 cm` (absolute, not SED-fraction) becomes
  `W` instead of `S`. Reported in the `Sw` column at the same level
  as `8 / L / S / 3 / 1 / X`.
- `K` detection in step 3.5: per-segment XY turn-angle scan; nodes
  whose adjacent segments meet at > 15° (default) get their codes
  overridden to `K` within ± 0.25 m → ~ 0.5 m K zones after coalesce
  (matches the quickcard's "Max 0.5 m" reference).
- Severity ordering (Jorge's quickcard mapping): sawmill quality
  `8 < L = S < 3` above the "Generally Pulp Quality" line; pulp
  quality `W = K < 1 < X` below.
- Min zone lengths: `W: 2 m` (defect-flag short min — not the
  quickcard's 4 m observational window). `K` and `X` exempt from
  the length floor (intrinsic short defect flags, always treated as
  stable in `_is_stable`).

### Known limitations (F1.5a + F1.5b)

The global `n_bows` per centerline assumes one direction character
per tree. A real tree with a consistent lower-trunk lean AND an
upper-trunk wobble would be labelled `S` (or `W`) everywhere. In
plantation radiata pine this single-character assumption holds in
practice; revisit if real-data shows mixed patterns.

The sub-operational alternating `S/3/S/3/S` cluster on tree 11 of
T460298B (2 m of back-and-forth fragmented into S and 3 sub-zones)
is the textbook `W` case but is **not** captured yet — its sub-zones
are individually below their own mins and F1.4 absorbs the whole
cluster into the surrounding `8`. Capturing it as `W` would require
a region-merging pass that groups adjacent sweep zones into a single
defect zone before applying the W test (F1.5c, future).

---

## Sources (live links, in priority order)

1. [MPI — NZ log grades for radiata pine](https://www.mpi.govt.nz/forestry/forest-industry-and-workforce/forestry-wood-processing-data/new-zealand-log-grades-for-radiata-pine) — **authoritative**.
2. [Interpine — Long vs Short logs of same grade quality (2013)](https://interpine.nz/better-predicting-yield-of-long-logs-vs-short-logs-of-the-same-grade-quality/) — **canonical operational rationale for `L`/`S`**.
3. [Pine Products NZ — Grading Guidelines for Radiata Pine (PDF)](https://www.pineproducts.co.nz/resources/file/picker/62fc4058642d5.pdf) — existence confirmed, content unverified.
4. [Park (1980), NZJFS 10(2):419-438](https://www.scionresearch.com/__data/assets/pdf_file/0019/59221/NZJFS1021980PARK419_438.pdf)
5. [Cown et al. (1984), NZJFS 14(1):109-123](https://www.scionresearch.com/__data/assets/pdf_file/0019/30916/NZJFS1411984COWN109_123.pdf)
6. [Marshall & Murphy (2004), NZJFS 34(2)](https://www.scionresearch.com/__data/assets/pdf_file/0020/5375/03_Marshall_Murphy.pdf)
7. [Ivković et al. (2007), Australian Forestry 70(3)](https://www.tandfonline.com/doi/abs/10.1080/00049158.2007.10675018)
8. [Interpine — PlotSafe Forest Inventory Procedures Manual (gated)](https://interpine.nz/15-plotsafe-forest-inventory-procedures-manual-available/)

---

## Cross-references inside the PS_LiDAR repo

- **`external_references/interpine/HQP Quickcard LiDAR pt 1.pdf`** — the
  field cheat-sheet whose thresholds are reviewed here.
- **`src/core/stem_description.py::classify_sweep_zones`** — the
  consumer of these thresholds. F1.1 implements the per-node global-
  chord algorithm with `8 / L / S / 3 / 1 / X` mapping. F1.2 (planned)
  adds the soft per-code length floor described in section 5.
- **`src/core/stem_description.py::build_stem_description_rows`** —
  emits one `Sw` row per detected zone in the Interpine CSV schema.
