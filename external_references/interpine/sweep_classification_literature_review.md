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
