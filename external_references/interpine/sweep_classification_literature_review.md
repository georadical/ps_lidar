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

## 6. Asymmetric absorption rule — justification

The zonal classifier (`classify_sweep_zones`) enforces minimum zone
lengths **asymmetrically**:

> **Only a zone that is BETTER than both its neighbours AND shorter
> than its code's minimum length is absorbed (into the worse
> neighbour). A WORSE zone keeps its code at any length — local defects
> are never averaged out.**

### Why asymmetric — the source

The Interpine HQP quickcard's **UPGRADE RULES** box states verbatim:

> "Only Upgrade Branch or Sweep Class (to **Smaller Branch Class or
> Better Sweep Code**) if > 3 m Section Between Zones of Lower Quality."
>
> — *HQP Quickcard LiDAR pt 1.pdf*, UPGRADE RULES box.

The rule is explicitly **one-directional**: the length gate applies
only when *upgrading* (claiming a better/smaller code). It says nothing
about downgrading — because a defect does not need a minimum length to
count. This is the standard log-grading **binding-constraint
principle**: a log is graded by its *worst* feature within the grading
length, so a short stretch of worse sweep degrades the section, while a
short stretch of better sweep cannot rescue it.

The recovery literature supports preserving short defects: sweep
materially reduces sawn-timber recovery (straight 57.9 %, moderate
52.0 %, severe 46.1 % — Cown et al. 1984, *NZJFS* 14(1):109-123;
~7 % loss per 0.1 increase in sweep:diameter ratio — Ivković et al.
2007, *Australian Forestry* 70(3):173-184). Averaging a short moderate
or severe section into a surrounding "clean" code would overstate grade
and risk placing defective wood in a high-value log.

### Worked examples

Severity order (best → worst): `8 < L = S < 3 < 1 < X`.
Per-code minimum lengths: `8: 4 m, L: 4 m, S: 4 m, 3: 3 m, 1: 2 m`;
`X` exempt.

**Example A — better zone absorbed (upgrade direction, gated):**
Input zones `[1 (0-3 m), 8 (3-4 m), 1 (4-9 m)]`.
- The `8` is 1 m, **better** than both `1` neighbours.
- To *claim* an `8` (gun-barrel straight) requires ≥ 4 m. 1 m < 4 m.
- → absorbed into the worse neighbour → `[1 (0-9 m)]`.
- *Forestry reading*: a 1 m straight blip inside a sweepy stem does not
  yield a merchantable straight log; the section is operationally `1`.

**Example B — worse zone preserved (downgrade direction, NOT gated):**
Input zones `[8 (0-4 m), 3 (4-6 m), 8 (6-10 m)]`.
- The `3` is 2 m, **worse** than both `8` neighbours.
- The upgrade rule does not apply (this is not an upgrade).
- → the 2 m `3` **stays**, even though 2 m < its 3 m minimum →
  `[8 (0-4 m), 3 (4-6 m), 8 (6-10 m)]`.
- *Forestry reading*: a 2 m moderate-sweep defect is real; hiding it
  would overstate quality and could route bad wood into a high grade.
- Covered by `tests/test_stem_description.py::TestClassifySweepZones::
  test_short_worse_zone_preserved`.

**Example C — reclassification within the same amplitude class:**
A SED/5 zone of 4 m (ratio in (1/8, 1/5]).
- Not an upgrade or downgrade — same amplitude class.
- `L` requires ≥ 5 m (long-log capable). 4 m < 5 m → relabelled `S`.
- Not absorbed into a neighbour; just renamed to the code that matches
  its length (`S` = short-log gentle sweep).
- Covered by `test_short_sed5_zone_reclassified_to_S`.

### Edge case — boundary zones

A short zone at the very base or top of the stem has only one neighbour,
so the "between two worse zones" condition cannot be satisfied; it is
never absorbed. This is deliberate — there is no second side to confirm
the surrounding quality. Covered implicitly by
`test_min_length_absorbs_short_better_zone_between_worse` (the boundary
`8` zones survive while the central one is absorbed).

### Future normative-traceability hook

A derived `MPI_grade` column on each 6 m log window of the stem
(mapping the worst amplitude observed in that window to `P1` / `PP` /
`S1` / `KI` per MPI's SED-fraction tolerances) would give a directly
traceable inventory output independent of the Interpine code
conventions. Out of scope for F1.2 but planned for after F2.

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
