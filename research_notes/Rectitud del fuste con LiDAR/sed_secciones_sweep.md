# SED-based sweep classification and where a sweep section STARTS and ENDS (field cruising, bucking optimisers, algorithms)

Research notes for PS_LiDAR (HQPlantations, PHYB, Queensland). Compiled 2026-11-19.
Conventions used below: **[sourced]** = statement found in the cited document; **[local]** = derived from the project's own files (read-only); **[inference]** = my reasoning, not stated by any source. Quotes are kept very short; everything else is paraphrased. Older documents are flagged with their year.

Local files read (not modified):
- Quickcard: `/home/jorge/projects/ps_lidar/external_references/interpine/HQP Quickcard LiDAR pt 1.pdf` (sheet 1 "Print Version Date: 3-4-18", sheet 2 "15-09-17").
- Literature review: `/home/jorge/projects/ps_lidar/external_references/interpine/sweep_classification_literature_review.md` (2026-05-28).
- Field data: `/home/jorge/projects/ps_lidar/external_references/interpine/PlotsSafe_Example_Data/stemdescription.csv` + `tree.csv` (population L19RAD05_FY24, client HQP, company Interpine, InvType LIDAR, 2 plots HQP024 [25 trees] and HQP128 [44 trees], 69 PHYB trees, measured Aug–Sep 2024).

---

## Q1. What SED is, which diameter is used for a section on a standing stem, bark, taper, and the circularity problem

### Takeaway
The only primary cruising source found that states which diameter a sweep tolerance refers to is the NZ MARVL bulletin (1987). It uses **the diameter at the top of the section being assessed**, i.e. the small end of that section or candidate log, not a whole-stem or butt diameter. Log specifications and bucking optimisers take SED from the **candidate log** (Swedish rules use top diameter under bark). So the "circularity" is resolved in practice by evaluating each candidate window or log with its own top diameter. None of the cruising sources found says whether a cruiser's SED estimate on a standing tree is over or under bark.

### Cited Findings
- **MARVL (NZ FRI Bulletin 132, Manley, Goulding & Lawrence, 1987 — old but the methodological ancestor of PlotSafe/YTGen cruising).** The sweep-class table footnote defines "D = Diameter at top of section being assessed for sweep", and defines sweep as the maximum deviation from straightness along the length of the log. **[sourced]** — [FRI Bulletin 132 (Scion CONTENTdm scan)](https://scion.contentdm.oclc.org/digital/api/collection/p20044coll6/id/191/download)
- **MARVL tolerance grows with log length** (sweep class 1 / class 2): <3.7 m: D/8 / D; 3.7–4.8 m: D/4 / 2D; 4.9–7.6 m: D/3 / 3D; >7.6 m: D/2 / 4D. Log-grade SED bands in the same appendix are in mm (e.g., P1 400+, S3 200–299, S4 150–199, R 100+). **[sourced]** — [FRI Bulletin 132](https://scion.contentdm.oclc.org/digital/api/collection/p20044coll6/id/191/download)
- **ForestrySA Log Standard Specification, Mt Lofty Ranges (SLSD002, issued 15/01/2016).** Sawlogs (SAW) are cut at 4.85 m and/or 6.05 m with SED ≥150 mm. Sweep is capped at SED/4 for all lengths. Industrial sawlog (ISW) is SED/3 and preservation products are SED/5. The document does not say under or over bark. **[sourced]** — [ForestrySA SLSD002 (2016)](https://www.forestrysa.com.au/?p=2214)
- **Sweden, VMR 1-07 sawlog instruction (Jan 2019 v1).** The straightness test uses a "sågcylinder" (sawing cylinder): a straight cylinder whose diameter is the **top diameter under bark minus 15 mm**, or minus 1 cm when diameter is recorded in whole-cm classes. **[sourced]** — [VMR 1_07 sågtimmer 2019 (Mellanskog copy)](https://www.mellanskog.se/wp-content/uploads/2019/02/matinstruktioner-2019-webb.pdf)
- **Australian market convention (KPMG/APLPI pine log price index, Jun 2017).** Small sawlogs are defined as less than 24.0 cm SED **under bark**, and HQPlantations is listed as a SE Queensland grower. Only a search snippet was seen; the full PDF was not opened. — [APLPI Jun 17 (timberbiz)](https://www.timberbiz.com.au/wp-content/uploads/APLPI-Jun-17-Stumpage-Public.pdf)
- **German and Austrian normalisation (via Gorges et al. 2021).** Germany (RVR/RV WV) divides bow height by **log length**, giving cm/m, and handles diameter through three diameter classes in the grading limits. Austria (ÖNORM L 1021:2015) divides bow height by the **mid-log diameter**. **[sourced]** — [Gorges et al. 2021, Forests 12(7):857 (author PDF)](https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/gorges-2021-forests.pdf)
- **Bucking optimisers compute SED per candidate log.** BuckR (Forests 2025) imputes large- and small-end diameters for every candidate log from a stem profile at 1 cm resolution, then classifies the log by length and SED. **[sourced]** — [Bennemann, Lussier & Labelle 2025, Forests 16(5):780](https://mdpi-res.com/d_attachment/forests/forests-16-00780/article_deploy/forests-16-00780.pdf)
- **NZ MPI grade tolerances** (taken from the local review, which cites the MPI page "last reviewed 27-01-2026"): pruned SED/4, pruned peeler SED/8, structural SED/4, industrial/KIS SED/3. I could not re-verify these this session because the MPI page renders empty to the fetch tool. — [local review](/home/jorge/projects/ps_lidar/external_references/interpine/sweep_classification_literature_review.md); [MPI page](https://www.mpi.govt.nz/forestry/forest-industry-and-workforce/forestry-wood-processing-data/new-zealand-log-grades-for-radiata-pine)
- **HQP Quickcard.** Every sweep code is stated as an SED fraction (8 = max SED/8 over 6 m; L and S = SED/5; 3 = SED/3; 1 = SED/1; X = >SED/1). W is the exception: it is an absolute ">5 cm" back-and-forth in 4 m. The card does not define SED, bark status or the measurement method. **[local]** — [Quickcard](/home/jorge/projects/ps_lidar/external_references/interpine/HQP%20Quickcard%20LiDAR%20pt%201.pdf)
- **Interpine (2013).** The NZ shorts/longs codes were described as "SED 1/4 assessed over 6m". Interpine's framing is that the sweep requirement is essentially the same for shorts and longs, applied over different log lengths. — [Interpine, Herries, 26-Nov-2013](https://interpine.nz/better-predicting-yield-of-long-logs-vs-short-logs-of-the-same-grade-quality/)

### Inferences
- **[inference] Circularity resolution.** SED depends on where the log ends, and the log ends depend on the section boundaries. Standards avoid the loop by never assigning one SED to a "zone": the tolerance is evaluated per **candidate log or window**, using that window's own top diameter (MARVL "D at top of section being assessed"; optimisers impute SED per candidate log). For an automated classifier, the defensible equivalent is:

  `threshold(window [z0, z0+W]) = D(z0+W) / k_code`

  Here D is the fitted stem diameter at the **upper** end of the window. The section's code then follows from which windows pass (see Q6), so boundaries never need an a-priori SED.
- **[inference] Taper matters numerically.** In the example plot, tree 1 has DBH 323 mm and H 26.48 m, a mean taper of about 1.2 cm/m from DBH to tip. Across a 6 m window the top diameter can therefore be about 7 cm smaller than the bottom diameter. With SED/5 that moves the allowed deviation by about 1.4 cm, which is the same order as the SED/8 tolerance itself (about 2.5 cm at SED 20 cm). Using the top diameter is the conservative and MARVL-consistent choice; a whole-zone average or DBH would be too lenient.
- **[inference] Bark.** LiDAR ellipse fits give **over-bark** diameters. The Swedish rule and Australian market definitions use under-bark SED, and the cruisers on a standing tree can only see over bark. Recommendation: compute thresholds from over-bark D but keep a configurable bark-thickness deduction, so the result can be reported both ways until HQP confirms its convention.
- **[inference] Length dependence.** For a constant-curvature (circular) bow, the deviation h grows with L² (h ≈ L²/(8R); the arc geometry is shown in Gorges et al. Eqs. 5–7). MARVL tolerances grow more slowly than L² (D/8 at <3.7 m up to D/2 at >7.6 m), so long logs are judged more strictly in curvature terms. This is why "SED/5 over 6 m" (L) is a stricter stem than "SED/5 over 4 m" (S): (6/4)² = 2.25 times less curvature is allowed.

### Gaps
- No primary HQP/Interpine document states over- vs under-bark SED for cruising sweep codes; the PlotSafe Forest Inventory Procedures Manual is gated. Unverified.
- The NZ "Grading Guidelines for Radiata Pine" PDF (Pine Products NZ) and the full MPI page text were not readable this session.
- The EN 1310 / EN 1311 curvature definitions (paywalled CEN standards) were not verified, so no EN wording is reported here.

---

## Q2. How sweep is measured against SED: reference line, chord, window length, and what "over any 4 m length" means

### Takeaway
Two reference conventions coexist:
- **Centre-line chord:** the maximum distance of the stem axis from a straight line joining the **centres of the two ends**. This is the NZ research practice (6 m pole against end midpoints), the scanner "bow height" (LIMAB, Austria), and the convention Gorges et al. recommend.
- **Surface straight edge:** the deflection of the log surface from a straight edge laid **along the side of the log**. This is the ForestrySA spec and the German contour method.

In every standard the chord spans **the log being graded**, or an explicit fixed window ("over 6 m", "over any 4 m length"). No source uses a whole-stem chord. Read literally, "any" means the test must pass for every placement of the window, which algorithmically is a sliding window.

### Cited Findings
- **NZ research method (Cown et al. 1984, NZJFS 14(1):109–123; old).** Sweep was estimated by holding a **6 m aluminium pole against the mid-points of each end** of the butt log, perpendicular to the plane of sweep, and recording the maximum deviation from the central log axis. Logs were classed straight (0–D/8), D/8–D/4, D/4–D/3, D/3–D/2 and >D/2. I saw this only in search snippets; the full text was blocked (HTTP 403). — [Cown et al. 1984 (Scion PDF)](https://www.scionresearch.com/__data/assets/pdf_file/0004/59701/NZJFS1411984COWN109_123.pdf)
- **ForestrySA (2016).** A non-straight log is accepted only if the sweep is a "uniform bow in one direction". Sweep is measured in mm of deflection from a **straight edge laid alongside the log**. Butt flare is measured separately against a straight edge on the higher side of the butt (max 100 mm). **[sourced]** — [ForestrySA SLSD002](https://www.forestrysa.com.au/?p=2214)
- **MARVL (1987).** Straightness "must be defined for the length of the stem considered". Sweep defined over a long length implies that sweep over any shorter part of that length is no worse. **[sourced]** — [FRI Bulletin 132](https://scion.contentdm.oclc.org/digital/api/collection/p20044coll6/id/191/download)
- **Gorges, Huber, Sauter & Dormann (2021), Forests 12(7):857.** 269 debarked spruce, oak and pine logs were scanned on a Microtec DiShape (360 points per slice, about 5 mm slice spacing). **[sourced]** — [author PDF](https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/gorges-2021-forests.pdf). The paper describes two basic approaches:
  - **Contour method (German RV WV variant 1):** 36 reference lines around the log, 10° apart. Bow height B is the maximum distance between the log contour and a reference line lying outside the trunk; the largest value over all 36 lines is kept. Curvature in cm/m = (B[mm]/10) / (L_N − O), where L_N is nominal length and O is the sum of the end offsets.
  - **Centre-line method:** slice centres (centre of mass, or least-squares ellipse/circle centres) form a centre line, which is smoothed with a moving average of 5–20 cm. Bow height is the maximum distance between this centre line and a straight line through the centre-line points at the (offset) ends.
  - **Segment method (RV WV variant 2, after Wilwerding):** bow heights at 50 cm steps out from the log middle, each reduced by a diameter-dependent acceptable curvature and summed. It has never been implemented in a commercial measuring unit.
  - **Differential method:** mathematical curvature κ of the centre-of-mass curve after heavy 500 mm smoothing.
  - **End offsets:** manual RVR measurement uses a 1.0 m butt offset; RV WV uses 0.5 m butt / 0 m top. The authors recommend 500 mm butt and 200 mm top offsets with 200 mm centre-line smoothing.
  - **Results:** the centre-of-mass and ellipse-fit centre lines were equivalent (ρ = 1). The contour method gave on average 0.21 cm/m higher curvature than the centre-line method because surface hollows and bumps enter it. The differential approach correlated poorly with everything and was judged unusable for practice. Offsets changed spruce values by up to 0.13 cm/m (16 %); smoothing interval changed values by at most 0.06 cm/m (4 %). The authors recommend the centre-line approach.
  - **Limits noted by the authors:** dividing by length makes longer logs of equal radius score higher. Current rules do not distinguish a symmetric "banana" log from a multiply-swept log.
- **Finnish scanner practice (LIMAB 3D log scanner; Silva Fennica article, pdf/388).** "Max bow height" is the largest deviation of the centroid line from a straight line between the mid-points of the bottom and top ends. Bow height was preferred to mm/m sweep because the logs had varying lengths. Seen in a search snippet only. — [Silva Fennica pdf/388](https://silvafennica.fi/pdf/388)
- **US scaling (New Hampshire Best Log Scaling Practices Guide, 1999, built on the USFS Log Scaling Handbook).** Percent deduction = (maximum departure − 2 in) / diameter, with 2 in for a 16 ft log and 1 in for an 8 ft log. Example: a 6 in departure on a 14 in log gives a 29 % deduction. **[sourced]** — [UNH Rep277](https://extension.unh.edu/sites/default/files/migrated_unmanaged_files/Resource000255_Rep277.pdf). Grondin (1998) quotes the same "curve reducing factor" from an earlier bucking model: (maximal deviation − 2) / small-end diameter. — [Grondin 1998, Wood Fiber Sci 30(1)](https://wfs.swst.org/index.php/wfs/article/download/1591/1591)
- **Swedish VMR (2019).** Straightness is expressed as yield loss: the **length of the straight sawing cylinder that does not fit inside the log** because of crook, assessed on one side of the log. Classes: <20 cm is within tolerance; 21–120 cm drops the log to the worst class; >120 cm is a reject. The instruction notes that shortening (bucking off) is not applied, i.e. the loss is not reduced by notionally trimming the log. Sharp crook or top break ("tvärkrök/toppbrott") is not allowed in the better classes. **[sourced]** — [VMR 1_07 2019](https://www.mellanskog.se/wp-content/uploads/2019/02/matinstruktioner-2019-webb.pdf)
- **China (Wang, LeDoux & McNeel 2004, Forest Prod. J. 54(2):45–52).** Sweep is expressed as a percentage of the maximum departure over the distance of a line segment between the top and the butt **of the log**. **[sourced]** — [USFS Treesearch 14464](https://research.fs.usda.gov/download/treesearch/14464.pdf)
- **Field measurement in a 3D bucking study (Wang, Liu & LeDoux, IJFE 20(2), c.2009).** A string was tensioned between sticks nailed at both ends of the stem at equal height. Distances to the stem surface were read every 1.22 m and the measurement repeated after turning the string 90°. **[sourced]** — [IJFE 30318](https://journals.lib.unb.ca/index.php/IJFE/article/download/30318/1882525691/1882526307)
- **HQP Quickcard.**
  - 8 = "max SED/8" drawn over a 6 m arrow.
  - L = SED/5 "over any 6m length", consistent direction, OK for logs >6.1 m; the illustration carries a 4.8 m dimension.
  - S = SED/5 "over any 4m length", back and forth, logs <6.1 m; the illustration carries a 2.4 m dimension.
  - 3 = 4 m of moderate sweep SED/3.
  - W = >5 cm back and forth, drawn over 4 m.
  - K = sharp direction change, max 0.5 m.
  - X = >SED/1 in 0.3–1 m lengths.

  **[local]** — [Quickcard](/home/jorge/projects/ps_lidar/external_references/interpine/HQP%20Quickcard%20LiDAR%20pt%201.pdf)

### Inferences
- **[inference] The reference chord must be local (log or window), never the whole stem.** For a stem with uniform gentle curvature of radius R = 200 m:
  - every 6 m window has h = 6²/(8·200) ≈ 2.3 cm, which is "8" at SED 20 cm;
  - the whole-stem chord over 26 m has a sagitta of about 26²/1600 ≈ 0.42 m, which is "1" or "X".

  A **per-node deviation from a global chord** (the current PS_LiDAR step) therefore has no counterpart in any standard found. It also makes a node's class depend on total tree height. It should be replaced by window or log chords. This is probably also why the pipeline produces 8/3/1 zones that cruisers never call (see Q3).
- **[inference] Centre-line chord vs straight edge.** For a planar bow and roughly linear taper, the concave-side straight-edge deflection and the centre-line chord deviation are approximately equal: the edge touches the ends and the radius at mid-span is about the mean of the end radii. They diverge with butt flare, nodal swellings or out-of-plane (3D) curvature. For LiDAR, the centre line from ellipse centres is the right quantity, as Gorges et al. recommend and as ellipse fitting already provides.
- **[inference] "SED/5 over any 4 m length" as an algorithm.** Slide a window of fixed length W along the centre line at the node step (0.2 m). For each start z0:
  - take the chord from c(z0) to c(z0+W);
  - h(z0) = max perpendicular 3D distance of the nodes inside the window to that chord;
  - pass if h(z0) ≤ D(z0+W)/k.

  "Any" means a section qualifies only if **every** window inside it passes. The "over 6 m" codes (8, L) use W = 6 m. S, 3 and W use W = 4 m. Neither 1 nor X has a stated window; use 4 m for 1 and about 1 m for X.
- **[inference] Equivalent minimum radii at SED 20 cm** (R = W²/(8h)): 8 → h ≤ 2.5 cm over 6 m → R ≥ 180 m; L → h ≤ 4 cm over 6 m → R ≥ 112 m; S → h ≤ 4 cm over 4 m → R ≥ 50 m; 3 → h ≤ 6.7 cm over 4 m → R ≥ 30 m; 1 → h ≤ 20 cm over 4 m → R ≥ 10 m. These are useful sanity checks and could seed an HMM's emission model (Q6).
- **[inference] End offsets.** Following RVR and Gorges et al., exclude roughly the lowest 0.5 m (root run-up / butt flare, which the Quickcard codes separately as B10+) from chord anchoring. Smooth the centre line with about a 0.2 m moving average before computing h.

### Gaps
- No Australian or NZ cruising document was found that says explicitly whether a cruiser sights the stem centre line or the stem edge on a standing tree. The PlotSafe manual is gated.
- EN 1310 wording, the USFS FSH 2409.11 chapter 10 text (HTTP 403) and Microtec/USNR proprietary sweep formulas were not accessed.

---

## Q3. How field cruisers decide where a section starts and ends (MARVL → RAD05/PlotSafe → HQP LiDAR template)

### Takeaway
MARVL cruising (1987) describes each stem **from the butt upward as a sequence of sections of uniform quality**, recording height and length. It deliberately does **not** cut logs in the field, and sections need not match logs. The optimiser later cuts logs only from sections whose codes the grade accepts, so section boundaries are effectively the permissible log-quality boundaries. The HQP/PlotSafe export follows a run-length "top-of-section" encoding: each row's Position is the **upper end** of the section whose code it carries, at 0.1 m resolution, and the last row is the tree height. Cruisers in the example data call few, long sections (median L 9.7 m), never use "8", and do record short butt-sweep "3" sections of 1–3 m.

### Cited Findings
- **MARVL procedure (FRI Bulletin 132, 1987).** **[sourced]** — [FRI Bulletin 132](https://scion.contentdm.oclc.org/digital/api/collection/p20044coll6/id/191/download)
  - The stem is cruised by subdividing it into sections of uniform quality starting at the butt, assessing the height and length of each section and assigning a predefined quality code.
  - Logs are not marked in the field, and sections do not necessarily correspond to logs. The aim is a description of visible features that is independent of the later analysis.
  - The computer then simulates cross-cutting with dynamic programming over log types defined by size and acceptable codes. A log type can only be cut from sections coded with a corresponding quality code.
  - Straightness decision: **straight** sections meet sweep class 1 over their entire length and may go into short logs (≤7.6 m) or long logs (>7.6 m). **Moderately swept** sections cannot be incorporated into long logs and must be cut into short logs. **Swept** sections meeting class 2 are coded industrial; excessive sweep is coded waste.
  - Field lessons: pruned height is measured to the nearest decimetre with a height pole. Many pruned butts showed **"hockey stick" butt sweep** and those portions were coded moderately swept because they could not go into long logs.
  - Internodal quality requires at least 60 % of a section's length to be in internodes of 0.6 m or longer, an example of a section-level rule expressed as a proportion of the section length.
- **MARVL is the origin of L/S.** Interpine traces the shorts/longs code split to late-1990s MARVL dictionaries; RAD05, RAD06 and RAD05A all carry it. NZ shorts are sawlogs up to 6.1 m; longs are export lengths up to 12 m. **Key cruising guidance:** severe wobble that would be unacceptable even in shorts must be noted accurately in the stem description. Milder sinuosity need not be described as wobble or as a short SED/3 section, because over-describing costs field time and can needlessly limit optimisation. Two otherwise similar trees differing only in L vs S showed about a 10 % value difference. **[sourced]** — [Interpine, Herries, 26-Nov-2013](https://interpine.nz/better-predicting-yield-of-long-logs-vs-short-logs-of-the-same-grade-quality/)
- **YTGen cutting strategies (Interpine, Herries, 27-Nov-2013).** Sweep is cruised over the **entire stem**, so a grade should specify sweep as a "must have" list that includes all better codes (e.g., `Sw:8,L,S`). In the example, the long-log grade (7.4, 8.1, 12.1 m) accepts only `Sw:8,L`. Features are recorded only where they occur, so grades exclude them with "must not have" lists. **[sourced]** — [Interpine YTGen article](https://interpine.nz/understanding-a-ytgen-cutting-strategy-from-marvl-to-logmaking/)
- **RAD05 dictionary (Interpine, Lusk, 15-Jun-2005).** A universal NZ PLOTSAFE feature-cruising dictionary. Windblown and resin are tree-level codes; all other codes are stem-description codes. Example grade syntax: `Br=0 Sw:8,L,S F!:B10+,C,D,F10+,N5+,R`. **[sourced]** — [Interpine RAD05](https://interpine.nz/15-rad05-cruising-domain-for-plotsafe-feature-cruising-dictionary/)
- **Cruise quality issues (NZ Tree Grower, May 2005).** MARVL assigns codes and defects to lengths of the bole. One audit found error above acceptable limits, partly from overuse of the default code "N". Changes in branch size and malformation were noted as hard to account for. **[sourced]** — [NZFFA Tree Grower May 2005](https://www.nzffa.org.nz/farm-forestry-model/tree-grower-articles/may-2005/marvl-pre-harvest-inventory-system/)
- **HQP Quickcard rules relevant to boundaries.** **[local]** — [Quickcard](/home/jorge/projects/ps_lidar/external_references/interpine/HQP%20Quickcard%20LiDAR%20pt%201.pdf)
  - **Upgrade rule:** upgrade branch or sweep class only if the better section between zones of lower quality is longer than 3 m.
  - Crutch/reductions have an "assumed length" of 0.5 m; K is max 0.5 m; X is 0.3–1 m.
  - The upgrade-rules illustration shows two example stems. The branch stem has section lengths 6, 2, 4, 2 and 3.5 m (Prune/Medium/Large/…). The sweep stem has 6, 2.5, 3, 2 and 3 m from the butt, annotated SED/8 → SED/4 → SED/1 upward. My reading is that the upper 3 + 2 + 3 m are called SED/1 as one zone because the 2 m straighter piece is ≤3 m; this is a visual interpretation, not text.
- **Example HQP LiDAR field data (Interpine, 2024; my analysis).** **[local]** — [stemdescription.csv](/home/jorge/projects/ps_lidar/external_references/interpine/PlotsSafe_Example_Data/stemdescription.csv)
  - **Position is the top of the section.** In all 69 of 69 trees the last row with a sweep code is at exactly the tree height from `tree.csv`. Trees with a single coded row (e.g., tree 2: Position 12.1 = H, `Br 4, Sw S`) can only mean "whole stem from 0 to 12.1 m is S / branch 4".
  - **Codes run independently.** A row with an empty `Sw` (e.g., a Br-only change or the GCHT marker) does not end the sweep section; Br and Sw sections are encoded on the same rows.
  - **Resolution.** Of 52 internal (non-terminal) Br/Sw boundaries, 45 have 1 decimal, 5 are integers and 2 have 2 decimals, so boundaries are at about 0.1 m resolution. Terminal positions carry the measured height (e.g., 26.48).
  - **Codes used:** S 73, L 40, 3 4 sections; "8", "1", "X", "W" and "K" never occur. Only one feature (S7+) was recorded.
  - **Sequences by tree:** L-S 37; S 27; 3-S-L-S 2; S-3-S 1; S-L-S 1; 3-S 1.
  - **L sections:** min 6.9 m, median 9.7 m, max 24.87 m. Thirty-seven of forty start at 0 m; the other three start above an S zone, at 6.5, 9.5 and 9.9 m. They end at 46–92 % of tree height.
  - **S sections:** non-terminal S (below an L or 3) are 6.5–7.4 m. Terminal S in L-S trees are 2.06–9.43 m. Whole-stem S occurs in 27 mostly small trees.
  - **"3" sections:** 1.0, 2.03, 2.1 and 3.0 m. Three of four start at the butt (0–1.0, 0–2.1, 0–3.0 m); one is at 7.2–9.23 m.

### Inferences
- **[inference] Position semantics for the exporter.** Emit one row per **section top**, carrying only the attribute(s) that change there. The last row is at total height. Positions are rounded to 0.1 m (internal boundaries) and walked from butt to tip. This matches both the MARVL "height and length of each section from the butt" procedure and the observed CSV.
- **[inference] Cruisers segment coarsely and "log-aware".**
  - L zones are at least about 6.9 m long, consistent with "OK for logs >6.1 m".
  - Non-terminal S zones are 6.5–7.4 m, about one short log plus trim.
  - The top S zone covers the small-diameter tip where only short logs or pulp remain.
  - Following Interpine's guidance, cruisers suppress mild sinuosity rather than inserting short SED/3 pieces.

  This independently supports the pipeline's F1.3 symmetric absorption of sub-operative zones.
- **[inference] Butt exception.** Cruisers **do** record short worse zones at the butt: three "3" zones from 0 to 1.0–3.0 m, i.e. hockey-stick butt sweep, as in MARVL 1987. A per-code minimum of 3 m for "3" would absorb two of the four field "3" zones (1.0 m and 2.1 m). Recommendation: allow a butt-anchored "3" or "1" zone down to about 1 m. Equivalently, exempt zones that start at 0 m from absorption into a better neighbour; worst-wins already handles this.
- **[inference] "8" vs "L".** HQP cruisers never used "8" on PHYB, even on stems where L runs from 0 to about 24 m. Either "8" (SED/8 over 6 m) is reserved for exceptional poles or peelers, or cruisers default to L whenever a long sawlog is feasible. A classifier that emits many "8" zones will disagree with field tallies. Options: emit 8 only when the SED/8 test passes over at least 6 m **and** the zone is long (e.g., ≥ 6.1 m), or report 8 ⊂ L as a sub-flag.
- **[inference] Boundary placement rule consistent with MARVL plus the optimiser.** Because a log of grade g can only be cut where every covered section has an acceptable code, place a boundary at the point **where the better class's window test first or last passes**. Do not place it at the deviation peak. Concretely, the top of an S butt zone should be the lowest height from which a 6 m window passes the L test. See the cover formulation in Q6.

### Gaps
- The PlotSafe/Interpine Forest Inventory Procedures Manual (gated) and the HQP LiDAR template documentation (LIR18-HQP) were not accessed. The top-of-section convention is inferred from data and MARVL practice, not read in a manual.
- I found no explicit written rule on rounding to 0.1 m vs 0.5 m, a minimum section length for S/L/3, or how K/X rows are inserted (e.g., as a 0.5 m section ending at Position).

---

## Q4. How bucking optimisers treat sweep inside candidate logs

### Takeaway
Optimisers handle sweep in one of two ways:
- **(a) Code feasibility:** MARVL/YTGen check that every stem section a candidate log overlaps carries an acceptable code. Sweep was already judged by the cruiser over the stem.
- **(b) Geometry-native:** Faaland & Briggs-type models, Grondin, and Chinese and Appalachian 3D buckers compute sweep or crook **for each candidate log** from the stem's centre-line geometry. They use the chord between that log's end centres (or a breakdown simulation of the largest straight cylinder) and compare it with the log's own SED.

In (b), sections are determined by logs, not fixed windows. Open-source BuckR (2025) does not yet include sweep.

### Cited Findings
- **MARVL (1987).** Cross-cutting is a DP over log types defined by diameter, length and acceptable cruise codes, maximising relative value. **[sourced]** — [FRI Bulletin 132](https://scion.contentdm.oclc.org/digital/api/collection/p20044coll6/id/191/download)
- **Grondin (1998), Wood and Fiber Science 30(1).** **[sourced]** — [Grondin 1998](https://wfs.swst.org/index.php/wfs/article/download/1591/1591)
  - DP recursion: stem-value(0, j) = max over available lengths i of [stem-value(0, j−i) + log-value(j−i, j)].
  - log-value(a, b) is the larger of the breakdown (lumber) value and the chip value of the actual segment from a to b. The step p is the greatest common divisor of the bucking lengths, e.g. 1 inch.
  - Test stems had diameters and **log centroid positions every 3 ft**, interpolated every 6 in into elliptical cross-sections "that consider sweep and crook".
  - In Faaland & Briggs (1984), log value comes from a second DP over the **maximum cylinder** that fits inside the log.
- **Wang, LeDoux & McNeel (2004), FPJ 54(2):45–52, NE China.** Sweep is a percentage of maximum departure over the log's butt-to-top line segment, with grade limits. If sweep exceeds the limits, a "must-cut" point is considered. Bucking usually proceeds from the point of maximum departure toward both butt and top. A severely swept portion **≤1.5 m long** that is poorer than grade 3 should be removed. **[sourced]** — [Treesearch 14464](https://research.fs.usda.gov/download/treesearch/14464.pdf)
- **Wang, Liu & LeDoux (IJFE 20(2), Central Appalachian 3D bucking; c.2009).** A network/DP optimiser over 3D stem models with selectable stage intervals, compared from 2.54 cm up to 1 ft (0.30 m). Smaller intervals gave higher value but longer run times. The scale deduction is 2.54 cm of diameter for every 7.62 cm of sweep. Optimal bucking raised stem value 31–38 % over manual bucking. **[sourced]** — [IJFE 30318](https://journals.lib.unb.ca/index.php/IJFE/article/download/30318/1882525691/1882526307)
- **BuckR (Bennemann, Lussier & Labelle 2025, Forests 16(5):780).** **[sourced]** — [Forests 2025 PDF](https://mdpi-res.com/d_attachment/forests/forests-16-00780/article_deploy/forests-16-00780.pdf)
  - R implementation of the Pnevmaticos & Mann (1972) / Dykstra DP. Each stage is a log; candidate logs combine every price-list length with every cumulative length from the previous stage.
  - The stem profile is in 1 cm steps ("the resolution at which the log lengths are set in the industry", following StanForD 2010). Logs are classified by length and SED with an rpart tree.
  - Stem defects and sweep criteria are listed as a **possible future modification**; they are not implemented.
- **optBuck (Noordermeer, R package).** It handles harvester production files and optimal bucking. Whether it implements sweep was not verified. — [optBuck on rdrr](https://rdrr.io/github/lennartnoordermeer/optBuck)

### Inferences
- **[inference] For a LiDAR pipeline both paradigms are available.** The code-based Interpine output (8/L/S/3/1/X) is a summary of the geometry-native feasibility of candidate logs. The most defensible way to derive codes is to **simulate the cruiser's implicit question**: what is the best class of log that can be cut through this point? Concretely, for each node z and each code c with window length W_c and amplitude ratio k_c:

  `code(z) = best c such that ∃ z0 ∈ [z − W_c, z] with h(z0, W_c) ≤ D(z0 + W_c)/k_c (and the direction rule for L)`

  This is a morphological **dilation of the window-pass indicator by W_c**, i.e. an opening of the "good" set. Zone starts and ends fall exactly at the first and last passing windows, which is where a log of that class could begin or end.
- **[inference] Keep a geometry-native product as well.** Store per-node h(z0, W)/D for W ∈ {2.4, 3.7, 4.8, 6.1, …} so that a downstream bucking optimiser (YTGen-like DP or a BuckR extension) can check each candidate log directly against MPI- or HQP-style SED fractions. This also fulfils the "MPI_grade per 6 m window" idea in the local review.
- **[inference] Wang et al.'s "must-cut at the maximum departure; remove ≤1.5 m severe portions"** is the bucking counterpart of the Quickcard X code (0.3–1 m severe sections "excessive to place on trucks"). X zones should be centred on the deviation peak and span the minimal length that removes it.

### Gaps
- AVIS (Geerts & Twaddle 1985) and Marshall & Murphy (2004, NZJFS 34(2), stem scanning for bucking) were inaccessible (Scion HTTP 403), so their exact sweep formulas are unverified.
- Harvester practice (StanForD 2010 hpr stem-grade or quality-change records: are they start positions?) was not verified.

---

## Q5. Single (C) vs compound (S) sweep vs crook/kink, and separating lean

### Takeaway
The terminology distinguishes three shapes:
- **sweep:** gradual curvature;
- **crook:** abrupt bend, often over a short length;
- **bow:** gradual bend in two directions.

Log specifications accept only a **uniform one-direction bow** (ForestrySA). HQP separates L (one consistent direction) from S (back and forth) at the same SED/5 amplitude, and treats W, K and X as bounded-length defects. Lean is removed automatically by any chord-based measure, because a straight leaning stem has zero deviation from its own chord. It only appears in HQP as the "Live Windthrown" tree code (lean > 45°).

### Cited Findings
- **Terminology.** Timell (1986), cited in a Silva Fennica scanner study, defines crook as an abrupt bend, sweep as a gradual bend, and bow as a gradual bend deviating from vertical in two directions. Snippet only. — [Silva Fennica pdf/388](https://silvafennica.fi/pdf/388). US and Canadian scaling glossaries contrast crook (sudden bend) with sweep (gradual curve). — [PNW RN-479](https://www.fs.usda.gov/pnw/pubs/pnw_rn479.pdf); [Yukon scaling glossary (archived)](https://bac-lac.wayback.archive-it.org/web/20061107092051/http://www.emr.gov.yk.ca/pdf/scaling_glossary.pdf)
- **ForestrySA (2016).** Swept logs are acceptable only as a "uniform bow in one direction". **[sourced]** — [ForestrySA SLSD002](https://www.forestrysa.com.au/?p=2214)
- **Interpine (2013).** Longs are a single gentle sweep "with movement in one consistent direction". Shorts may move in more than one direction beyond 6 m. Stems passing SED/4 over 6 m can fail at 8–12 m because of sweep in two or three directions. **[sourced]** — [Interpine 2013](https://interpine.nz/better-predicting-yield-of-long-logs-vs-short-logs-of-the-same-grade-quality/)
- **Gorges et al. (2021).** Current German and Austrian curvature values ignore whether a log is a symmetric "banana" or multiply swept in several planes. A sweep confined to one log end (root run-up) is expected to hurt yield less. Sweep-type classification (after Gjerdrum, Warensjö & Nylinder) is suggested as future work, and the centre-line course is the promising basis. **[sourced]** — [Gorges et al. 2021](https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/gorges-2021-forests.pdf)
- **Swedish VMR.** Sharp crook or top break ("tvärkrök/toppbrott") is a separate binary defect, not allowed in the higher classes, while ordinary crook is expressed as sawing-cylinder yield loss. **[sourced]** — [VMR 1_07 2019](https://www.mellanskog.se/wp-content/uploads/2019/02/matinstruktioner-2019-webb.pdf)
- **US hardwood practice.** Crook deductions use the deviation, the **length of the crook**, the log diameter and total log length, and graders are advised to buck back to remove crook. Snippet only. — [Zena hardwood scaling & grading rules](https://static1.squarespace.com/static/521e264fe4b0e5f492070e32/t/5ecff8e0dc53ee1f1b93d290/1590687969878/Zena+Hardwood+Scaling+%26+Grading+Rules+52820.pdf)
- **HQP Quickcard.** K = kink, sharp change of direction, max 0.5 m. W = wobble back and forth >5 cm in 4 m. X = severe sweep >SED/1 in 0.3–1 m lengths. Live Windthrown = stem on lean >45° and/or root plate lifted. Butt flare (B10+) is a feature, not a sweep code. **[local]** — [Quickcard](/home/jorge/projects/ps_lidar/external_references/interpine/HQP%20Quickcard%20LiDAR%20pt%201.pdf)

### Inferences
- **[inference] C vs S inside a window.**
  1. Project the window's centre-line offsets onto the plane of maximum deviation (PCA of the offsets about the window chord).
  2. Take the signed lateral offset u(s).
  3. A **C (single) bow** has u of one sign over the window and one interior extremum.
  4. An **S (compound) bow** has at least one interior zero crossing of u with prominent extrema on both sides, using a deadband of about 1 cm, as the pipeline already does.

  A useful additional test is that the full-window chord deviation is small while half-window chord deviations are comparable or larger. That signature is unique to S shapes, and it is why a long chord alone under-detects S.
- **[inference] Kink vs sweep.** A kink is a turning-angle event: the angle between two adjacent chords of about 0.25–0.5 m each exceeds a threshold, and the curvature is concentrated in ≤0.5 m. Sweep is distributed curvature over ≥2–4 m. Gorges et al. found that the curvature-based "differential" method correlates poorly with bow-height measures. That supports keeping K detection (local turn angle) separate from sweep classes (window chord / SED).
- **[inference] Lean.** Chord-based measures are invariant to rigid rotation of the stem, so lean does not inflate sweep. Lean only matters as a tree-level windthrow flag (>45°), or if the centre line is referenced to vertical, which no standard does. A butt "hockey stick" (root sweep with an upright upper stem) is real sweep and should appear as a short butt 3/1 zone, as in the field data and MARVL 1987. It should not be removed as lean.
- **[inference] Section boundaries per type.**
  - C bow: zone spans the window(s) that fail the better class.
  - S/W: zone spans the union of windows where reversals occur.
  - K: 0.5 m centred on the turn-angle peak.
  - X: the 0.3–1 m stretch around the deviation peak where h exceeds SED/1 in a ~1 m window.

### Gaps
- No primary source gives quantitative thresholds separating "kink" from "sweep" (degrees, or length of bend) for radiata pine or PHYB. The 0.5 m maximum on the Quickcard is the only number.
- Gjerdrum, Warensjö & Nylinder's sweep-type classes were not accessed directly.

---

## Q6. Algorithms to segment a 3D centre line into straightness sections, and a defensible recipe for PS_LiDAR

### Takeaway
Generic change-point and polyline methods (PELT, binary segmentation, Douglas–Peucker, Bellman DP, BOCPD, HMM/HSMM) can segment a centre line. However, none of them encodes the SED-relative, window-relative definition used by cruisers and log specifications. The defensible design is:
1. **Window-chord deviation fields** with per-window top-diameter thresholds (Q2).
2. **Interval-cover / dilation** to turn passing windows into zones whose starts and ends are where logs of that class could start or end (Q4).
3. A **duration-constrained labelling step** (HSMM-Viterbi or DP with minimum segment lengths and transition penalties) to implement the >3 m upgrade rule and noise-floor absorption in one principled pass.

### Cited Findings
- **Change-point framework.** Offline multiple change-point detection decomposes into a cost function, a search method and a constraint on the number of changes. The Python package **ruptures** implements the main algorithms. — [Truong, Oudre & Vayatis 2020, Signal Processing 167:107299 (arXiv 1801.00718)](https://arxiv.org/abs/1801.00718)
- **PELT.** Exact minimisation of the sum of segment costs plus a penalty λ per change point. Implementations expose a **minimum segment length** parameter (rupturesRcpp `minSize`, default 1). — [rupturesRcpp reference](https://cran.case.edu/web/packages/rupturesRcpp/refman/rupturesRcpp.html); original paper: [Killick, Fearnhead & Eckley 2012, JASA 107(500):1590–1598](https://doi.org/10.1080/01621459.2012.737745) (bibliographic reference, not fetched this session).
- **Gorges et al. (2021) as empirical guidance.**
  - Centre lines from slice centres or ellipse fits are robust and equivalent.
  - About 0.2 m smoothing is enough.
  - Local mathematical curvature computed with forward and central differences after 500 mm smoothing correlated poorly with log-level bow height. Local-curvature-only segmentation is therefore a weak proxy for SED-based sweep.
  - The RV WV "segment" approach (bow heights at 50 cm steps, minus a diameter-dependent allowance, summed) is a published precedent for **sub-log windowed assessment**.

  **[sourced]** — [Gorges et al. 2021](https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/gorges-2021-forests.pdf)
- **Standard references** (bibliographic, from my knowledge; not fetched this session, URLs are DOI or arXiv landing pages):
  - Bellman (1961), optimal approximation of curves by line segments via DP, Commun. ACM 4(6):284 — [doi:10.1145/366573.366611](https://doi.org/10.1145/366573.366611).
  - Douglas & Peucker (1973), polyline simplification, Cartographica 10(2):112–122 — [doi:10.3138/FM57-6770-U75U-7727](https://doi.org/10.3138/FM57-6770-U75U-7727).
  - Adams & MacKay (2007), Bayesian online change-point detection — [arXiv:0710.3742](https://arxiv.org/abs/0710.3742).
  - Yu (2010), hidden semi-Markov models (explicit state durations), Artificial Intelligence 174(2):215–243 — [doi:10.1016/j.artint.2009.11.011](https://doi.org/10.1016/j.artint.2009.11.011).
- **Bucking DP as segmentation.** Grondin's recursion stem-value(0, j) = max_i [stem-value(0, j−i) + log-value(j−i, j)] is structurally the same optimal-partition DP used in Bellman/PELT segmentation, with segment cost replaced by log value. **[sourced for the recursion]** — [Grondin 1998](https://wfs.swst.org/index.php/wfs/article/download/1591/1591)

### Inferences (recommended recipe; all **[inference]**)
1. **Centre line.** Use ellipse centres every 0.2 m (already available). Apply about 0.2 m moving-average smoothing (Gorges). Fit a smooth diameter profile D(z) (monotone taper fit) so that thresholds are not noisy.
2. **Butt handling.** Do not anchor chords inside the lowest about 0.5 m (RVR / Gorges offsets). Treat butt flare via the B10+ feature, not sweep. Allow a butt-anchored 3/1 zone down to about 1 m (field data: 1.0–3.0 m butt "3" zones).
3. **Window deviation fields.** For W ∈ {6 m (8, L), 4 m (S, 3, W, 1), ~1 m (X)} and every start z0 at 0.2 m steps, compute:
   - h(z0, W): the maximum 3D perpendicular distance of nodes to the chord c(z0)→c(z0+W);
   - r(z0, W) = h / D(z0+W);
   - n_rev(z0, W): sign changes of the in-plane signed offset with a 1 cm deadband;
   - h_abs for W.

   Complexity is O(N·W/Δ), trivial for 0.2 m nodes.
4. **Pass masks.**
   - 8: r ≤ 1/8 over 6 m.
   - L: r ≤ 1/5 over 6 m **and** n_rev = 0 (consistent direction).
   - S: r ≤ 1/5 over 4 m.
   - 3: r ≤ 1/3 over 4 m.
   - 1: r ≤ 1 over 4 m.
   - X: r > 1 over ~1 m.
   - W: n_rev ≥ 2 and h > 5 cm over 4 m.
   - K: turn angle over 0.5 m stencil > threshold (existing step).
5. **Cover / dilation (this is what defines START and END).** A node belongs to a class-c candidate zone if some passing window of length W_c covers it. Zone start = first z0 of the first passing window; zone end = z0 + W_c of the last passing window. Assign each node the best class that covers it. This reproduces "OK for logs >6.1 m" semantics directly. L zones are automatically at least 6 m and S zones at least 4 m, with no ad-hoc minimum-length rule. A class that cannot be covered by its own window length simply does not appear.
6. **Duration-constrained clean-up (upgrade rule plus noise floor).** Run a DP or HSMM-Viterbi over node labels:
   - states = codes;
   - emissions = log-likelihood from r and n_rev;
   - explicit **minimum durations**: better zone between worse zones >3 m (Quickcard upgrade rule); K about 0.5 m; X 0.3–1 m; W ≤4 m;
   - transition costs that favour few sections (cruisers record few, long sections; Interpine advises against over-describing).

   This replaces the iterative worst-wins absorption with a single global optimum that has the same intent. Alternatively, PELT on the 2-D signal [r_4m(z), r_6m(z)] with `min_size` ≈ 3 m and a BIC-like penalty gives boundary candidates, which are then labelled by the pass masks.
7. **Export.** One row per section top, 0.1 m rounding, last row at tree height, unchanged attributes left blank. This matches the observed HQP CSV.
8. **Calibration targets from the 69-tree field set:**
   - L min about 6.9 m;
   - non-terminal S about 6.5–7.4 m;
   - terminal S 2–9.4 m;
   - butt "3" 1–3 m;
   - no "8";
   - typical sequences L-S (54 %) and S (39 %).

   Use these to tune penalties. Remaining disagreements should be checked against tree-level LiDAR plots, not forced.

### Gaps
- No forestry paper was found that applies PELT, HSMM or another change-point method specifically to segment a tree or log centre line into sweep classes. The recipe above is a synthesis, not a published method.
- The Python `ruptures` docs (`min_size`, `jump`, `pen`) were not fetched this session; parameter names are from the R port and from general knowledge.
- Microtec/USNR/Inray proprietary scanner algorithms for sweep, crook and "multiple sweep" segmentation are undocumented publicly; only the German RV WV variants (via Gorges et al.) and a LIMAB bow-height definition were found.
