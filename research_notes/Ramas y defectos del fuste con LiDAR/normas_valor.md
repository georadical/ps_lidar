# Branches, knots and stem defects: how log specifications, grading rules and breeding programmes define them, and what they are worth (Australian softwood priority)

Scope note: these notes cover only branches, knots and stem defects other than sweep. The sweep standards are already in `../Rectitud del fuste con LiDAR/normas_definiciones.md` and `trozado_industria.md`, and the sweep breeding scores are in `mejora_genetica_fenotipado.md`. Research date: Nov 2026. Several PDFs were downloaded and their text extracted locally with `pdftotext`. Quotes marked "verbatim" were checked against that extracted text.

Tags used in these notes:
- [PR] peer-reviewed
- [GREY] FWPA, government or industry report
- [VENDOR] vendor or consultant material
- [SNIPPET] only a search summary or abstract was seen

---

## 1. Numeric limits on branches, knots, spike knots, forks, nodal swelling, fluting, ovality, damage and rot in log specifications

### Takeaway
I found one fully readable public Australian softwood specification, ForestrySA SLSD002 (2016) with its export chart (2015). It sets explicit numeric knot limits:
- sawlog: knots ≤100 mm, one knot up to 150 mm, and no knot larger than SED/2
- industrial sawlog: up to 200/300 mm on large logs
- preservation: knots ≤25 mm and no spike knots
- export A/K: roundness ≤1.25

The PlotSafe RAD05 dictionary (2007) sorts branches into grade bands: ≤7 cm small-knot, ≤10 cm medium-knot, ≤12 cm large-knot, ≤25 cm (or SED/2) rough sawlog only, and >25 cm pulp. It maps every feature code to the grades it restricts. Forks, crutch zones, major damage and N5+ nodal swelling restrict a section to pulp or rough sawlog. Rot restricts it to waste. Fluting and ovality mainly exclude peeler grades.

I found no public HQPlantations, OneFortyOne or FCNSW log specification. AS 2858 and AS 4785 knot limits are paywalled, and I found no reliable public summary of their numbers.

### Cited Findings

**PlotSafe manual (CNI Regional YTGEN User Group, Feb 2007; local copy read in full for these sections)**

Source: `external_references/interpine/PLOTSAFE Overlapping Feature Cruising Forest Inventory Procedures - Feb 2007.pdf`, sections 4.1.4.1, 4.1.4.3, 4.1.5 and 5.1.2. [VENDOR/industry manual]

General conventions:
- §4.1.4: there are four overlapping domains: Br, Sw, F and I. "Stem sections of lower quality should never be upgraded to a higher quality code." Above a stem diameter of 150 mm, the quality code does not need to change further.
- §4.1.4.1: branches are assessed outside bark, at right angles to the stem. The default code is ">1 to 7cm". Spike branches are treated as features, not branches. §5.1.2 adds: "All knots (except spikes) are measured at right angles to stem."

RAD05 Br domain (code, description, grade meaning, MARVL equivalents):

| Code | Description | Grade meaning | MARVL codes |
|---|---|---|---|
| 0 | Pruned or clear | Clear grades | B, C, D |
| 1 (E) | Epicormics or stem cones <1 cm | Clear or small-knot grades | K, N, G |
| 7 (default) | Branches ≤7 cm | Small-knot grades | K, N, G |
| 10 | Branches ≤10 cm | Medium-knot grades | Q, S |
| 12 | Branches ≤12 cm | Large-knot grades | L, J |
| 25 | Branches up to SED/2, max 25 cm | Rough sawlogs only | H |
| 99 | Branches >25 cm | Pulp and waste only | P, Y |

The HQP Br codes 0/4/7/1/2/99 in the user's background appear to follow the RAD05-A / RAD06 variants, which add a 4 cm class. See the Interpine dictionary note cited in `trozado_industria.md`.

F domain (§4.1.4.3 and §5.1.2), verbatim descriptions with grade restrictions:

| Code | Description | Grade restriction |
|---|---|---|
| D | Damage / Scar Major (>10% diameter depth) | Pulp and waste only |
| S7+ | Spike knot 7–10 cm in length and <30° | Not allowed in peeler; restricted in some sawlog grades |
| S10+ | Spike knot 10–16 cm | Same as S7+ |
| S16+ | Spike knot 16–25 cm | Rough sawlog, pulp and waste only |
| S25+ | Spike knot 25+ cm | Pulp and waste only |
| N5+ | Nodal greater than 5 cm | Rough sawlog, pulp and waste only |
| O1.2+ | Ovality > 1.2x | Not allowed in peeler grades ("not B or K") |
| F5+ | Fluting greater than 5 cm | Not allowed in peeler grades |
| F10+ | Fluting greater than 10 cm | Not allowed in sawlog grades |
| B10+ | Butt flare 10+ | Sometimes restricted to rough sawlog, pulp and waste |
| C | Crutch or reduction zone greater than 0.5 m | Pulp and waste only (MARVL P, Y, <) |
| R | Rot | Waste only |
| NONE (default) | No feature | Unrestricted |

Feature definitions in §4.1.4.3:
- **Spike knot:** a branch at less than 30° to the stem. Its size is "the length of the hard knot cut along the stem (longest face)", from the intersection with the stem centre to the top of the branch entry. Footnote 8 says this is "a little conservative" because common log grades often use <20°, and some grades use a ratio of branch size to hard-knot length.
- **Nodal swelling:** the maximum distance from the top of the swelling to a line parallel to the stem taper above it.
- **Fluting:** the maximum depth outside bark, in cm, from a straight line bridging the flute. The fluting class should be tapered down along its length, e.g. F10+ to F5+.
- **Ovality:** the ratio of the longest to the shortest diameter through the centre. The two axes need not be at 90°. Calipers are recommended for audits.
- **Damage:** measured by depth. Includes holes, pulled branches and thinning scars. Any sign of rotten wood or insects means the section is coded R.
- **Butt flare:** measured from the edge of the flare to the projected taper.

Forks (§4.1.5):
- A fork is recorded when two or more of the resulting stems are >100 mm in diameter. Smaller ones are treated as broken tops.
- Fork height is the base of the crutch zone, not the point where the stems separate.
- A structural "<" crutch zone of 0.5 m is assumed above each fork. A longer crutch is coded as the F feature "C" for its full length.

Other tree-level codes (§5.1.2): a live windblown tree (Y) is restricted to pulp and waste. Severe resin (ResinCode 3) is "Restricted to low value clear, and or sawlogs grades".

**ForestrySA Log Standard Specification, Mt Lofty Ranges (SLSD002, issued 15/01/2016, P. radiata)** — [ForestrySA SLSD002](https://www.forestrysa.com.au/?p=2214) [GREY, full PDF text extracted]

MLR Sawlog (SAW), SED ≥150 mm, 4.85 or 6.05 m:
- Trimming: "Branches shall be trimmed to within 35mm of the log surface with bark on, although up to two branches to a maximum of 75mm are permitted."
- Knot size, verbatim: "Individual knots ≤100mm diameter measured across the longer axis are permitted. Up to one knot per log of up to 150mm diameter is permitted. No knot may exceed half of the small end diameter of the log."
- Spike knots: "Occasional spike knots ≤50mm per length are permitted."
- Shape: abrupt diameter change >40% is not permitted.
- Butt flare: ≤100 mm.
- End splits: ≤150 mm.
- Decay: "Is unacceptable." Externally visible or extensive dryside (a scar from earlier thinning) is unacceptable.
- "Occasional" is defined as "the exceptional circumstance and not the rule".

MLR Industrial Sawlog (ISW), 2.5 or 3.7 m:
- Knot size, verbatim: "For logs <350mm Small End Diameter individual knots ≤100mm diameter … and one knot up to 150mm … For logs>350mm Small End Diameter individual knots ≤200mm diameter … and two knots up to 300mm diameter are permitted. No knot may exceed half the diameter of the log."
- Spike knots: "Occasional spike knots ≤100mm per length are permitted."
- Decay is unacceptable.

Preservation (posts, rails and strainers, SED 75–200 mm):
- Flush trimmed. Branch stubs ≤15 mm.
- "Individual knots ≤25mm diameter … are acceptable."
- Spike knots and cracked ends are unacceptable.
- Abrupt diameter change >20% is not permitted, and excessive butt flare is not permitted.
- Decay and dryside are unacceptable.

The specification's defect photo glossary covers butt fluting, decay, double heart, dry side, end splits, felling scar, nodal swelling, slovens, spike knots and ramicorns, knot size, sweep and trimming. Two definitions:
- Double heart: "Arise from double leaders, and in some cases, large spike knots."
- Butt fluting: "a misshaped log, usually occurring in butt logs, but may occur higher up a tree."

**ForestrySA MLR Export Product Specifications (chart, last review 17/12/2015)** — [ForestrySA export spec](https://www.forestrysa.com.au/?p=2221) [GREY]

The chart was extracted from a table layout, so column alignment is my reading and should be checked against the PDF.
- AAX, ASW and KDS (A and K export sawlogs, SED 220–420+ mm, 3.88 or 5.88 m) appear to share these limits:
  - knots: "Sed/3, <120" (mm)
  - spike knots: "1 60mm/4m, <3/m >50mm"
  - sweep: Sed/4
  - wobble: Sed/6
  - nodal swelling: <50 mm where SED <300, <70 mm where SED >300
  - surface defects: <30 mm
- SPS (SED 170–260): knots Sed/3 and <120, spike knots "No", sweep Sed/2.
- KIG (industrial export, SED 320+): knots Sed/2 and <240, spike knots "1 100mm/4m", sweep Sed/2, nodal swelling "Ok".
- P10 and P12 export pulp: branches and spikes "Must be cut flush with main stem". Sweep Sed/1.5 (SED >200) or Sed/1 (SED <200). Nodal swelling "Ok if debarking possible".
- General clauses:
  - "Log Roundness. Longest face ≤1.25 times the shortest face for A, K, KIG & ≤ 1.3 times for KMS."
  - "Dryside / Decay / Insect – Not Permitted."
  - Scar face "Not acceptable for pulp … ≤ 50mm for A & K".
  - "Abrupt diameter changes by more than 20% are not permitted."

**New Zealand MPI / Scion log grades**
- Domestic grades (snippet of an MPI-derived table):
  - P1 and P2 pruned: max knot 0
  - S1, S2 and S3: max knot 60 mm
  - L1, L2 and L3: max knot 140 mm
  - minimum SED: 400 mm (S1, L1), 300 mm (S2, L2), 200 mm (S3, L3)
  - pulp: no knot limit

  An older Scion paper gives the same largest-branch limits (6 cm for S grades, 14 cm for L1). [SNIPPET; undated ANU copy; host refused connection] — [ANU Fenner "NZ log grades"](https://fennerschool-associated.anu.edu.au/marketreport/NZ_log_grades.pdf); [MPI log grades page (fetch returned empty)](https://www.mpi.govt.nz/forestry/forest-industry-and-workforce/forestry-wood-processing-data/new-zealand-log-grades-for-radiata-pine); [Haslett 1991 NZJFS 21(2/3)](https://www.scionresearch.com/__data/assets/pdf_file/0005/59873/NZJFS212_31991HASLETT228_234.pdf)
- MPI cautions that the Scion grades "have not been universally adopted" and that specifications vary between suppliers. [SNIPPET] — [MPI log grades page](https://www.mpi.govt.nz/forestry/forest-industry-and-workforce/forestry-wood-processing-data/new-zealand-log-grades-for-radiata-pine)
- Export wharf-gate grades (AgriHQ, May 2023), max knot size:
  - P: 0
  - A: 10–15 cm
  - K and KS: 15 cm
  - KI: 25 cm
  - KIS: no limit

  Minimum SED is 30–40 cm for A and 22 cm for K. [SNIPPET; subscription report] — [AgriHQ Log Price Report 2023-05](https://agrihq.co.nz/s/AgriHQ-Log-Price-Report-2023-05.pdf)
- The NEFD yield-table "Unpruned" log type uses max branch 14 cm. This is already in `normas_definiciones.md`. — [Canopy NEFD](https://canopy.govt.nz/forestry-data-research/national-exotic-forest-description)

**Branch index (BIX) definitions**
- FAO glossary: "Average diameter of the largest branch in each of the four quadrants of a standard log length." [SNIPPET] — [FAO i3274e13](https://fao.org/4/i3274e/i3274e13.pdf)
- STBA breeding objective BIX: "maximum diameter of branches in four quadrants of a log in cm". The present Australian radiata mean is 5.5 cm. — [FWPA PN01.1904 (2005)](https://fwpa.com.au/wp-content/uploads/2005/08/PN01.1904.pdf) [GREY, text extracted]
- A NZ study found the mean of the three largest branches per quadrant correlated slightly better with framing grade than the standard BIX (largest branch per quadrant, Inglis & Cleland 1982). BIX was strongly correlated with tree diameter (r² = 0.80 in one Canterbury study). [SNIPPET] — [Grace et al. NZJFS 29(3) 1999](https://www.scionresearch.com/__data/assets/pdf_file/0007/59443/NZJFS293_1999_391_408GRACE.pdf); [Watt et al. NZJFS 30(3) 2000 (403)](https://www.scionresearch.com/__data/assets/pdf_file/0006/59388/NZJFS3032000_315_331_WATT.pdf)

**Sawn-timber knot rules (AS 2858 / AS 4785, MGP)**
- AS 2858:2023 is current and supersedes the 2008 edition. Its visual grades are Structural No. 1–5 plus a stud grade, governed by strength-limiting characteristics such as knots, through a knot-area-ratio (KAR) definition. The numeric limits are in the paywalled standard. — [Standards Australia AS 2858:2023](https://store.standards.org.au/product/as-2858-2023)
- WoodSolutions states only that "Different limits on KAR apply for knots in the margin" versus the centre. Strength ratios are given as a % of clear wood, e.g. 75% for No. 1 and 30% for No. 5. No numeric KAR limits are published there. — [WoodSolutions, Visual stress-grading](https://www.woodsolutions.com.au/visual-stress-grading)
- In WA MGP practice, "there is a visual override of the MSG for knots in most MGP production systems". — [FWPA PNC059-0809 (Blakemore, Cown et al. 2010)](https://fwpa.com.au/wp-content/uploads/2010/02/PNC059-0809_Research_Report_West_Softwood.pdf) [GREY, text extracted]

### Inferences
- The specifications agree on a hierarchy that the classifier can use directly:
  - pulp or waste regardless of branching: rot, forks with their crutch zone, major damage (>10% of diameter deep), and very large branches (>25 cm, or more than SED/2)
  - rough or industrial sawlog only: large spikes (≥16 cm hard-knot length), nodal swelling >5 cm, and branches up to SED/2
  - excluded from peeler grades only: fluting 5–10 cm, ovality >1.2 and small spikes
- The ForestrySA limits (100 mm sawlog knot, 1 knot ≤150 mm, ≤SED/2) are coarser than the RAD05 small/medium/large-knot classes. For a mill paying for SAW versus ISW, the branch decision effectively happens at about 10–15 cm and at SED/2. Finer classes (4/7/10/12 cm) matter mostly for appearance, export and peeler grades, and in breeding.
- Ovality has two thresholds in practice: 1.2 in PlotSafe and 1.25–1.3 in the ForestrySA export roundness rule.

### Gaps
- No public HQPlantations log specification (Queensland sawlog, peeler or preservation grades) was found. Forestry Corporation NSW, OneFortyOne and HVP specifications were also not found publicly. HQP may share its cutting strategies with the user directly.
- The AS 2858 / AS 4785 numeric KAR or knot-width limits are paywalled. No reliable free summary was found.
- The current MPI log-grade page could not be read (empty fetch). The NZ numbers above come from a secondary, undated table.
- The ForestrySA SLSD002 review date was 2017. A newer version may exist.

---

## 2. Economic and recovery impact of branch size and stem defects compared with sweep

### Takeaway
The strongest Australian quantitative evidence is the STBA radiata bio-economic model, using about 2005 prices (PN01.1904). A 10% reduction in branch index (BIX) is worth more than a 10% reduction in sweep:
- grower: +6.9% versus +2.9% NPV
- integrated enterprise: +7.9% versus +4.5% NPV

Per unit, 1 cm of BIX is worth about 4–5 times 1 mm/m of sweep for the grower. Ivković et al. (2007) agree: a 10% branch-size reduction cut degraded sawlog volume by 68%, while a 10% sweep reduction cut sawlog degrade by 17%.

For Queensland slash pine, the FWPA resource study concluded that, because density is already high, "knots and resin defects … create the bulk of the variability in in-grade recovery". Resin defects, which are linked to branches and wind, cost about $4–5 M per year in Queensland. Forks and structural defects are handled categorically, as pulp-only zones, not as graded value effects.

### Cited Findings
- **STBA bio-economic model, Australian radiata** — [FWPA PN01.1904 summary (Wu, Ivković et al. 2005)](https://fwpa.com.au/wp-content/uploads/2005/08/PN01.1904.pdf) [GREY, text extracted]

  Effect of a 10% trait improvement (Table 1):

  | Production system | Base NPV | MAI +10% | SWE −10% | BIX −10% | MoE +10% |
  |---|---|---|---|---|---|
  | Grower | $2,104/ha | +31.2% | +2.9% ($61/ha) | +6.9% ($145/ha) | 0 |
  | Sawmill | $41/m³ | −1.0% | +3.1% | +2.8% | +28.8% |
  | Integrated | $4,539/ha | +20.6% | +4.5% ($204/ha) | +7.9% ($362/ha) | +23.9% |

  - For the integrated system, a 10% reduction in SWE or BIX raises net present incomes by $590 or $1,091/ha respectively.
  - Unit economic weights:
    - grower: H_G = 291·MAI − 58·SWE − 248·BIX ($/ha)
    - sawmill: H_M = −0.18·MAI − 1.19·SWE − 1.97·BIX + 10.5·MoE ($/m³)
    - integrated: H_I = 416·MAI − 194·SWE − 620·BIX + 977·MoE ($/ha)
    - units: SWE in mm/m, BIX in cm
  - Monte Carlo weights for growers: SWE −49.0 ± 7.7 (30-year rotation) and −34.5 ± 5.8 (35-year). Branch size −266.4 ± 33.0 and −183.7 ± 24.4 NPV$/ha per unit.
- **Ivković, Wu, Spencer & McRae (2007), Australian Forestry 70(3):173–184.** Per the abstract:
  - a 10% reduction in branch size "decreased the volume of degraded sawlog by 68% and increased structural timber recovery by 0.6–1.6%"
  - a 10% sweep reduction cut sawlog degrade by 17.1% and added about 0.5% green recovery

  [PR][SNIPPET abstract] — [Scilit](https://www.scilit.net/publications/5f28defbc2ffa86062b641085760ff1b); [BibBase](https://bibbase.org/network/publication/ivkovi-wu-spencer-mcrae-modellingtheeffectsofstemsweepbranchsizeandwoodstiffnessofradiatapineonstructuraltimberproduction-2007)
- **Queensland slash pine (Harding, FWPA PN06.3016, Sept 2008)** — [PN06.3016](https://era.dpi.qld.gov.au/id/eprint/3144/2/PN06.3016_Slash_pine_Resource_Evaluation_2009%5B1%5D-sec.pdf) [GREY, text extracted]
  - Mechanism, verbatim: "the inherently high wood density of slash pine mature wood is above the density thresh-hold needed to consistently produce in-grade structural timber. Therefore it is defects in the wood such as knots and resin defects that create the bulk of the variability in in-grade recovery."
  - Tree height explained only 25% of variation in in-grade recovery. Knot-area-ratio effects from branch architecture "have a somewhat random impact … as they depend on where saw cuts fall". Sweep and taper affect where cuts fall and therefore whether KAR is maximised.
  - Resin shake is "the major cause of lost recovery and sawn product rejection for solid wood processors in Queensland". Lost recovery and grade fall-down from resin defects was valued at "$4 and $5 million /annum in Queensland (Harding et al., 2007)". Early studies linked resin defects with branches. More recent work suggests wind torsion stress peaks at about 2–4 m above ground.
  - Green-mill downgrade reasons were recorded as "knot, hole, wane, pith, other". Sample trees deliberately excluded "Bendy trees and any trees with significant ramicorn branches or double leaders", so the study does not quantify fork or ramicorn losses.
- **Value of the structural / non-structural split in Queensland:** structural board (MGP10 and above) is worth about $350/m³ versus about $80/m³ for non-structural. [SNIPPET] — [FWPA news, "What's the best use for that timber"](https://fwpa.com.au/news/whats-the-best-use-for-that-timber)
- **WA radiata and pinaster (Blakemore, Cown et al., FWPA PNC059-0809, 2010)** — [PNC059-0809](https://fwpa.com.au/wp-content/uploads/2010/02/PNC059-0809_Research_Report_West_Softwood.pdf) [GREY, text extracted]
  - Board MoE (average and low) was "negatively affected by branch size classification, P. radiata more strongly so than P. pinaster". Part of this runs through density, since smaller-branched radiata trees had higher density.
  - Recommendation: genetic improvement "to minimise the size of branching and increase internodal distances, would be of benefit for improving recovery of full length structural graded (MGP) material".
  - Visual branching (BIX, number of nodes per log) was "useful for predicting board stiffness".
  - In regrading tests, board KAR gave the best improvement in in-grade properties. Nominal MGP12 boards with KAR >40% and MGP10 boards with KAR >30% were downgraded one grade. Average log branch size was the next best variable. Sample sizes were small.
- **Southern pines resource characterisation (Bailleres, Lee et al., FWPA PNC361-1415, April 2019).** Its 91 growth plots (79 Qld, 12 NSW) recorded per tree "height to and type of limiting defect" and ovality at 1.3 m with calipers, alongside acoustics. This shows that Queensland inventory practice treats a single "limiting defect" height per tree. I did not find a frequency table of defect types in the extracted text. — [PNC361-1415](https://era.dpi.qld.gov.au/7939/1/Final_Report_Southern_Pines_PNC361-1416.pdf) [GREY]
- **NZ benchmarks for other defects** [SNIPPET]:
  - Eliminating all stems with sweep on one NZ site would give a 4% value recovery gain.
  - 0.7–2.0 resin pockets per m² reduced lumber value from pruned logs by 7–34%.

  — [NZJFS search results, Scion archive (Park; Cown)](https://www.scionresearch.com/__data/assets/pdf_file/0013/60007/NZJFS1911989PARK41_53.pdf)
- **Forks and structural defects** are handled categorically in cruising and bucking. PlotSafe "<" and "C" crutch zones are pulp or waste only. Swedish practice forbids sawlogs over fork or spike zones (already in `normas_definiciones.md`). I found no Australian study giving the % value loss per fork.

### Inferences
- Per tree, the most value-relevant feature after diameter and stiffness is **largest branch size by log section** (BIX or max branch). Its value effect is about 1.8–2.4 times that of sweep for an equal 10% change. A LiDAR classifier should therefore prioritise Br (max branch diameter per section, especially in the first 2–3 logs) right after the sweep domain.
- **Forks (with their crutch zone), broken tops and rot** are binary pulp or waste switches. They are rarer than branch-size issues but each one kills a sawlog over the affected section. Detecting the fork height (crutch base) is therefore high-value even if the fork is only coarsely localised.
- In Queensland, resin defects and wind-related damage are major value drivers, but they are internal and not LiDAR-visible. Branches and ramicorns are proxies at best.
- Fluting, ovality and small spikes mostly affect peeler eligibility. Their priority depends on whether HQP strategies include peeler (veneer) grades for the stand.

### Gaps
- No Queensland (PEE, PCH or PHYB) study quantifying % structural-grade loss by knot size was found in the extracted text. The FWPA "Southern pines" and PN06.3016 reports discuss knots qualitatively.
- The full text of Ivković et al. (2007) was not accessed. Only the abstract numbers are available.
- Benyon's ANU thesis, "The relative economic importance of stem straightness, branch diameter and stem diameter in P. radiata breeding", is likely highly relevant but could not be opened (PDF >10 MB). — [ANU repository](https://openresearch-repository.anu.edu.au/bitstream/1885/143322/2/b1748067x_Benyon_R_G.pdf)
- The economic weights date from about 2005 prices. Current (2026) weights from TBA are not published on its web page.

---

## 3. How branch traits are scored in tree breeding (Queensland hybrid pine, STBA/TBA radiata)

### Takeaway
Australian radiata breeding (STBA, now Tree Breeding Australia) uses objective BIX in cm (maximum branch diameter in four quadrants of a log) as the breeding-objective trait. Its early selection criteria are relative 1–6 scores for branch size and branch angle, plus a count of branch clusters between 1 and 6 m. TBA publishes a BRANCH EBV in cm against a 5.5 cm baseline.

Queensland hybrid pine trials (PEE × PCH) record branch diameter in mm, branch angle on a 1–9 scale, and presence/absence of ramicorns and double leaders at about age 4. Reported heritabilities for branching traits are moderate, about 0.10–0.49. Branch diameter, whorl number and whorl spacing are LiDAR-measurable proxies. Angle and spike-knot status are harder.

### Cited Findings
- **STBA trait definitions**, verbatim: early selection includes "branch size (BRS – measured by score 1-6, 6 is the best with 5% smallest branches in a trial, this is different from definition for breeding objective trait BIX), branch angle (BRA – … 6 is the best with 5% flattest branches in a trial) and branch cluster frequency (BRC – number of branch clusters (whorls) between 1-6 meter above ground along the stem)".
  - Expected gains from 1-in-10 selection: BIX −0.31 to −0.63 cm, with a present mean of 5.5 cm.
  - Source: [FWPA PN01.1904](https://fwpa.com.au/wp-content/uploads/2005/08/PN01.1904.pdf)
- **TBA current EBVs:** "Branch size in centimetres from baseline production of 5.5 cm" (lower is better; an EBV of −0.91 gives 4.59 cm). These sit alongside SWEEP (mm/m, baseline 10.5), STIFFNESS (GPa, baseline 11.3) and MAI (22.6 m³/ha/yr) in an NPV index at 8%. The page does not give trait weights, measurement height or age, or forking, angle and cluster traits. — [Tree Breeding Australia, radiata breeding values (accessed Nov 2026)](https://treebreeding.com/breeding-values/radiata-pine-breeding-values)
- **STBA second-generation trials:** branch angle and branch size scored 1–6. The genetic correlation of straightness with branch size score was rA = +0.27 ± 0.10 (straighter trees had smaller branches). See `mejora_genetica_fenotipado.md`. — [FWPA PNC050-0304 (2009)](https://era.dpi.qld.gov.au/id/eprint/1897/1/PNC050-0304_Juvenile_Wood_Initiative_Research_Report_0.pdf)
- **Queensland hybrid pine:**
  - Series IV clonal trials (about 2,000 clones, 7 Qld sites): straightness 1–6, double leaders, ramicorns, branch angle (1–9) and branch diameter (mm) assessed at age 4.
  - The clone-level correlation of straightness with branch diameter was −0.33, with ramicorns −0.36 and with double leaders −0.28.
  - Exp. 749 TBS F1 families: double leaders and ramicorns scored as presence/absence.
  - Source: [FWPA PNC057-0809 (2010)](https://era.dpi.qld.gov.au/id/eprint/2573/1/PNC057-0809.pdf) (details in `mejora_genetica_fenotipado.md`)
- **Hybrid QTL study** (Shepherd, Cross, Dieters & Henry 2002, Ann. For. Sci.): found moderate additive QTL for branch diameter, branches per whorl, whorl spacing and whorl regularity in a PEE × PCH family. No marker associations were found for branch angle. The paper cites literature narrow-sense h² of 0.10–0.49 for branching traits, with branch angle "amongst the more heritable". [PR][SNIPPET] — [Qld DAF eRA 10092](https://era.dpi.qld.gov.au/id/eprint/10092/1/Branch%20architecture%20QTL%20for%20Pinus%20elliottii%20var.%20elliottii%20x%20Pinus%20caribaea%20var.%20hondurensis%20hybrids.pdf)
- **Rationale for the hybrid:** it combines "slash pine's straightness and wood density with Caribbean pine's better branching" (finer branches). [GREY][SNIPPET] — [Southern pines (Qld DPI eRA)](https://era.dpi.qld.gov.au/id/eprint/5800/1/Southern%20pines_for%20eRA.pdf)
- **Slash pine (SE Queensland):** Dieters et al. included crown defects such as ramicorn branches and basket whorls among assessed traits, with moderate h² for growth and straightness. [SNIPPET] — [core.ac.uk 15092071](https://core.ac.uk/download/15092071.pdf)
- **WA operational branch score**, per 5 m log for the first 4 logs, on 30 stems per plot. It is a visual BIX expressed in cm. — [FWPA PNC059-0809](https://fwpa.com.au/wp-content/uploads/2010/02/PNC059-0809_Research_Report_West_Softwood.pdf)

  | Score | Branch size | Nominal mean |
  |---|---|---|
  | 0 | No branches | — |
  | 1 | ≤3 cm | 1.5 cm |
  | 2 | 3–6 cm | 4.5 cm |
  | 3 | 6–9 cm | 7.5 cm |
  | 4 | ≥9 cm | — |

- **NZ:** branch cluster frequency is scored 1–9 (1 = uninodal, 9 = extremely multinodal). See `mejora_genetica_fenotipado.md`. — [Li et al. BMC Genomics 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6935163/)

### Inferences
- Several traits map directly onto LiDAR measurements:
  - BIX or max branch diameter per log, which is the economic trait in cm
  - whorl or cluster count between 1 and 6 m, and internode length
  - presence of double leaders or forks
  - large ramicorns, which are steep and thick branches
- Branch angle is relevant twice: as a breeding trait, and as the spike-knot criterion (<30° in PlotSafe, <20° in many grades). It requires branch-level segmentation, which is harder than diameter.
- Because relative 1–6 scores are trial-normalised, an absolute LiDAR metric in cm or mm is more transferable. It can be converted to within-trial ranks if needed.

### Gaps
- No published heritability specifically for branch diameter or angle in Queensland PEE × PCH F1 trials was retrieved. PNC057 tables may contain them but were not re-read for branch traits.
- No document linked Queensland hybrid branch scores to HQP log-grade outcomes.

---

## 4. Which features are most frequent in Queensland pine plantations

### Takeaway
I found no published frequency table of stem features (forks, fluting, nodal swelling, spikes) for Queensland slash, Caribbean or hybrid pine. Available sources only indicate the following:
- double leaders and ramicorns are routinely recorded in hybrid trials and negatively correlated with straightness
- resin defects are the dominant processing loss
- wind and torsion stress is concentrated about 2–4 m above ground
- Queensland research inventories record a single "height to and type of limiting defect" per tree

### Cited Findings
- Double leaders and ramicorns are assessed in all PEE × PCH trial series and correlate negatively with straightness (−0.28 and −0.36 at clone level). — [FWPA PNC057-0809](https://era.dpi.qld.gov.au/id/eprint/2573/1/PNC057-0809.pdf)
- Resin shake is the major cause of sawn-product rejection in Queensland (about $4–5 M per year). It is associated with branches and possibly wind torsion at 2–4 m. — [FWPA PN06.3016](https://era.dpi.qld.gov.au/id/eprint/3144/2/PN06.3016_Slash_pine_Resource_Evaluation_2009%5B1%5D-sec.pdf)
- Caribbean pine has poorer straightness and wind-firmness than slash pine. Wind-firmness and foxtail codes were taken from field comments in the hybrid clonal trials. — [FWPA PNC057-0809](https://era.dpi.qld.gov.au/id/eprint/2573/1/PNC057-0809.pdf) (see `trozado_industria.md` and `mejora_genetica_fenotipado.md`)
- PNC361 growth plots recorded "height to and type of limiting defect" and ovality at 1.3 m in 91 plots (Qld and NSW). — [FWPA PNC361-1415](https://era.dpi.qld.gov.au/7939/1/Final_Report_Southern_Pines_PNC361-1416.pdf)
- From a NZ nursery study: persistent malformation followed apical abortion in only 1–2% of cases, but followed 10–15% of bifurcations, especially those near the ground. [SNIPPET] — [NZJFS (Scion archive)](https://www.scionresearch.com/__data/assets/pdf_file/0006/59829/NZJFS222_31992BURDON138_159.pdf)

### Inferences
- In the absence of frequency data, the HQP PlotSafe datasets the user already holds (`external_references/interpine/PlotsSafe_Example_Data`) are probably the best source for estimating feature frequencies by code. Counting Br and F codes there would rank features empirically for HQP stands.
- Butt fluting and ovality are plausibly concentrated in the butt log of fast-grown or wind-exposed PCH and hybrid stems, but I found no Queensland source quantifying this.

### Gaps
- No Queensland data was found on the frequency of forks (tip moth, wind, lightning), fluting in PCH, nodal swelling or spike knots. A HQP or Interpine data request, or analysis of the local PlotSafe files, is needed.
