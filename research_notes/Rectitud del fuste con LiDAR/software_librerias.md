# Software and libraries for stem centrelines, stem curves, taper and straightness/sweep from forest LiDAR

Research date: 2026-11-19. Repository metadata (stars, last push, licence, releases) was pulled live from the GitHub API, PyPI JSON and CRAN (crandb) on that date. "Last push" means the last commit to any branch, not necessarily a release.

## Q1. Which tools output a stem curve (centreline + diameters along the stem), and which output straightness/sweep metrics or log grades? With what algorithm?

### Takeaway
Several open-source tools output a stem curve: 3DFin/dendromatics, FSCT, TreeQSM/PyTLidar, TreeLS, RayExtract and SimpleForest. **None of the maintained open-source tools computes sweep, straightness or log grades out of the box.** The only straightness "classifiers" I found are (a) a manual, per-tree 7-point straightness tool in LiDAR360 TLS Forest (Macdonald et al. 2001 scale), (b) in-house or commercial services that are not documented in detail (Interpine's Hovermap workflow, and Treemetrics ForestHQ, which captures sweep implicitly through virtual bucking), and (c) one-off research scripts. The closest public analogue to PS_LiDAR is a Woodknowledge Wales study: 3DFin 0.2 m sections, then R scripts for maximum sagitta (Smax), taper and lean, then a red/green sawlog grade and bucking.

### Cited Findings

**Open-source tools that output a stem curve**
- **3DFin / dendromatics (Python, GPL-3.0).** dendromatics fits circles at user-defined sections, for example one every 0.2 m from 0.3 m up to 25 m. Each circle passes quality tests: inner-circle check, sector occupancy, point count, and "a high value of tilting". The tool also computes a per-tree stem axis. — [dendromatics docs/algorithm.rst](https://github.com/3DFin/dendromatics/blob/main/docs/algorithm.rst)
  - The API exposes `compute_sections()`, which returns X_c, Y_c, R, check_circle, second_time, sector_perct and n_points_in per section. The README example uses `section_len = 0.2`. — [dendromatics README](https://github.com/3DFin/dendromatics)
  - `tilt_detection()` computes the inclination in degrees between every pair of section centres within a tree. It flags outlier sections from a weighted sum of absolute and relative tilt. — [dendromatics sections.py](https://github.com/3DFin/dendromatics/blob/main/src/dendromatics/sections.py)
  - 3DFin exports the axes together with a "matrix that describes the tilt of each axes" (`_export_axes(axes, tilt)` after `dm.generate_axis_cloud`). It writes results as .xlsx or .txt, with as many columns as stem sections. — [3DFin abstract_processing.py](https://github.com/3DFin/3DFin/blob/main/src/three_d_fin/processing/abstract_processing.py)
  - 3DFin does **not** compute a sweep or straightness metric. I found no sweep/straightness function in the code; the "tilt" values are used only for QC and the axis. (Verified by code search; see Gaps.)
- **FSCT (Krisanski; Python, GPL-3.0).** FSCT outputs `taper_data.csv`, described as "simply the largest diameter at a range of given heights above the DTM for each stem". It also outputs `cleaned_cyls.las/csv`, a "point-based cylinder representation with a variety of properties", and a cone-based volume. — [FSCT README](https://github.com/SKrisanski/FSCT)
- **TreeQSM (MATLAB; v2.4.1, 2022).** The `tree_data` step computes `treedata.StemTaper` (a stem taper matrix) as well as vertical profiles and spreads. The stem section is defined "up to the first major branch". — [TreeQSM tree_data.m](https://github.com/InverseTampere/TreeQSM/blob/master/src/main_steps/tree_data.m)
- **PyTLidar (Python, GPL-3.0).** "Version 1 makes the TreeQSM capabilities available through Python, eliminating the need for proprietary tools." It installs with `pip install "PyTLidar[gui]"` and has a CloudCompare plugin. v2.0 (in progress) adds segmentation and DEMs. — [PyPI pytlidar](https://pypi.org/project/pytlidar/), [GitHub Landscape-CV/PyTLidar](https://github.com/Landscape-CV/PyTLidar)
- **TreeLS (R/C++, GPL-3).** Provides `stemPoints()` (Hough or eigen/knn stem point classification), `tlsInventory()` with circle/cylinder shape fitting, and `stemSegmentation()`, which outputs stem segments fitted along the stem (for example `sgt.ransac.circle`). The README's example data is a pine plot. — [TreeLS README](https://github.com/tiagodc/TreeLS)
- **RayExtract / raycloudtools (CSIRO; C++).** `rayextract trees forest.ply forest_mesh.ply` "extracts tree structures to text file, and segments forest". Citation: Devereux, Lowe, …, Calders, …, Levick, Phinn, Woodgate (2026), "RayExtract: A fast, scalable method for tree volume reconstruction from terrestrial laser scanning", *RSE* 334:115162. — [raycloudtools README](https://github.com/csiro-robotics/raycloudtools); [doi:10.1016/j.rse.2025.115162](https://doi.org/10.1016/j.rse.2025.115162)
- **SimpleForest (Computree plugin; C++).** Produces QSM cylinder fits and is used through Computree (for example Computree 5.0.221b with SimpleForest 5.3.2). — [Dryad dataset citing Computree + SimpleForest](https://datadryad.org/dataset/doi:10.5061/dryad.nzs7h44x1)
  - A 2024 preprint says SimpleForest "seems to be similarly capable to TreeQSM, but is only available through Computree, which has been undergoing an extended upgrade". — [EcoEvoRxiv preprint](https://ecoevorxiv.org/repository/object/10250/download/18945)
- **rTwig (R).** Imports SimpleForest and TreeQSM QSMs and corrects overestimated cylinders using measured twig diameters ("Real Twig"). It summarises QSM diameter, height and volume. It is a QSM corrector, not a stem-form tool. — [rTwig GitHub](https://github.com/aidanmorales/rTwig); [rTwig SimpleForest vignette](https://stat.ethz.ch/CRAN/web/packages/rTwig/vignettes/SimpleForest.html)
- **ITSMe (R).** Computes DBH (from the point cloud or QSM), height, crown metrics and DBH fit-quality control. A README grep for stem curve or taper found nothing. — [ITSMe README](https://github.com/lmterryn/ITSMe)
- **pointtree (Python, MIT).** Tree instance segmentation; its outputs include "diameters of the detected trunks at breast height" only. — [pointtree README](https://github.com/ai4trees/pointtree)
- **TreeTool (PyPI 1.0.1, MIT, 2023).** "Python package for tree detection, segmentation and extraction of DBH". — [PyPI treetool](https://pypi.org/project/treetool/)
- **ForestScanner (iOS app).** Estimates stem diameters and positions via real-time instance segmentation and circle fitting. In a 1 ha validation of 672 trees, R² was 0.963 against tape (iPhone). DBH only; no stem curve. — [Tatsumi et al. 2023, MEE, doi:10.1111/2041-210X.13900 (PDF)](https://communityecologist.github.io/pdf/Tatsumi_2023_MethodsEcolEvol.pdf)

**Segmentation / leaf–wood / skeleton tools (no stem curve or straightness output)**
- TLS2trees (Wilkes / UCL) runs an FSCT-derived "lite" semantic segmentation (ground, wood, leaf, cwd) and then graph-based instance segmentation over a tile index. The README targets RIEGL VZ data through `rxp-pipeline`. — [TLS2trees README](https://github.com/tls-tools-ucl/TLS2trees)
- treeseg (Burt) — "Extract individual trees from lidar point clouds". — [GitHub apburt/treeseg](https://github.com/apburt/treeseg)
- SegmentAnyTree, ForestFormer3D (ICCV 2025) and ForAINet: deep-learning instance segmentation. — [SegmentAnyTree](https://github.com/SmartForest-no/SegmentAnyTree), [ForestFormer3D](https://github.com/SmartForest-no/ForestFormer3D), [ForAINet](https://github.com/prs-eth/ForAINet), [ForAINet paper arXiv 2312.15084](https://arxiv.org/pdf/2312.15084)
- TreeLearn: deep-learning segmentation of ground-based LiDAR. — [GitHub](https://github.com/ecker-lab/TreeLearn), [arXiv 2309.08471](https://arxiv.org/pdf/2309.08471)
- Leaf–wood separation tools: PointsToWood (AGPL-3.0) and LeWoS (MATLAB, MIT). — [PointsToWood](https://github.com/harryjfowen/PointsToWood), [LeWoS](https://github.com/dwang520/LeWoS)
- treeiso: TLS tree isolation, also available as a CloudCompare plugin. — [artemis_treeiso](https://github.com/truebelief/artemis_treeiso)
- Skeletonisation and branch modelling: pc-skeletor (Python, MIT) and AdTree (C++, GPL-3.0). — [pc-skeletor](https://github.com/meyerls/pc-skeletor), [AdTree](https://github.com/tudelft3d/AdTree)
- 3D Forest (VUKOZ; C++, GPL-3.0) is a desktop application with descriptor and tree-segmentation tools. — [3D Forest README / manual](https://github.com/VUKOZ-OEL/3d-forest)
- aRchi (R) handles tree architecture from TLS QSMs. — [aRchi](https://github.com/umr-amap/aRchi)
- spanner (R) provides lidar utilities. — [spanner](https://github.com/bi0m3trics/spanner)
- PyForestScan computes forest *structural* metrics from **airborne** point clouds, not stem metrics. — [PyForestScan](https://github.com/iosefa/PyForestScan)
- lidR is focused on "Airborne LiDAR data manipulation and visualisation for forestry". — [lidR](https://github.com/r-lidar/lidR)
- DigiForest DRS (Oxford, `digiforest_analysis`) provides "Python modules to extract tree data". It is minimally documented and has 1 star. — [digiforest_drs](https://github.com/ori-drs/digiforest_drs)

**Tools or studies that do straightness, sweep or log grading**
- **LiDAR360 TLS Forest (GreenValley, commercial).** TLS Forest > TLS Editor > "Measure Individual Tree Attribute" includes a **Straightness tool** that follows MacDonald et al. (2001). It uses a **7-point scale (1 = least straight, 7 = straightest) based on straight log lengths in the first 6 m**. The selection must be taller than 6 m and contain stem points only, because branches and leaves give wrong values. The workflow is **interactive, per tree**; no batch mode is described. **Source is a third-party mirror of the vendor tutorial, not the official manual.** — [Blog mirror of LiDAR360 exercise](https://mendiolatooffer.blogspot.com/2022/10/measuring-and-drawing-trees-stem.html)
  - LiDAR360's TLS Forest toolset also does automatic segmentation and extracts DBH, height and volume. — [Inertial Labs case study](https://inertiallabs.com/efficient-forest-inventory-management-with-resepi-ultra-lite/)
  - It also has a "trunk slicing workflow" for trunk volume. — [ISPRS Archives 2025/2026](https://isprs-archives.copernicus.org/articles/XLVIII-4-W18-2025/295/2026/isprs-archives-XLVIII-4-W18-2025-295-2026.pdf)
- **The 7-point scale behind it** is the UK Forestry Commission scheme, developed for Sitka spruce. Individual trees score 1–7 from straight log lengths in the bottom 6 m. A "straight" log has bow ≤ 1 cm per 1 m, where bow is the maximum deviation between the actual centreline and the straight line joining the end centres. Stand grades A–E are set by the % of trees per score. — [Price 2017, Forest Research Technical Note FCTN021 (PDF)](https://cdn.forestresearch.gov.uk/2017/06/fctn021.pdf); [FR web page](https://www.forestresearch.gov.uk/research/assessing-the-stem-straightness-of-trees)
  - The note says TLS "offers the possibility of acquiring high-resolution data relating to the 3D stem form and straightness", but "most software relating to forestry applications is still in the research and development" stage (2017). — [FCTN021](https://cdn.forestresearch.gov.uk/2017/06/fctn021.pdf)
- **Woodknowledge Wales / Bangor University (Bye; open-source chain).** The workflow is Leica BLK360 multi-scan, manual tree isolation in CloudCompare, then the **3DFin plugin with circles every 0.2 m from 0.3 m**. 3DFin outputs DBH, highest point, and each section's diameter and X, Y, Z. Custom R code then computes **sagitta (sweep), taper and lean**, with straightness defined as **maximum sagitta (Smax)**. A second R script selects sawlogs greedily (4.8 m green, then 4.8 m red, then 3.6 m green … down to 2.4 m) and computes frustum volume and value from 2023 prices.
  - DBH agreed closely with manual measurement; "agreement … was slightly less" for height, and also for taper, sweep and lean. The R code is not published in the report.
  - Source: [Bye, "Estimating Timber Value and Carbon in Complex Woodlands using TLS" (PDF)](https://woodknowledge.wales/wp-content/uploads/Estimating-Timber-Value-and-Carbon-in-Complex-Woodlands-using-Terrestrial-Laser-Scanning-FINAL.pdf)
- **Windrim & Bryson (2020, Univ. Sydney/ACFR; research code).** Deep learning on helicopter ALS over two **radiata pine** forests in **NSW, Australia**. The method isolates trees, labels stem points and "build[s] a segmented model of the main tree stem that encompasses tree height, diameter, taper, and sweep". — [Remote Sensing 12(9):1469, doi:10.3390/rs12091469](https://mdpi.com/2072-4292/12/9/1469); earlier preprint [arXiv 1810.12536](https://arxiv.org/pdf/1810.12536)
- **Pyörälä et al. (2019, Forestry).** Log geometry from TLS was compared against sawmill measurements for Scots pine. Diameter, volume and taper agreed reasonably, but **sweep had a relative mean difference of 78%**; the authors judged that uncertainty in defining sweep limits usefulness. — [UTU research portal record](https://research.utu.fi/converis/portal/detail/Publication/458274175?lang=en_GB)
- **Forests 2022, 13(3):431.** A TLS method for stem straightness and lean that "splits stems into evenly spaced sections and estimates their centres". It states that sweep "is traditionally defined as the maximum sagitta per meter of the stem". — [MDPI PDF](https://mdpi-res.com/d_attachment/forests/forests-13-00431/article_deploy/forests-13-00431.pdf) (full text blocked with 403; details from a search snippet)
- **ForSense (J. Forestry Research 2023).** Measures stem straightness from depth images with computer vision and reports strong correlation with trained forester ratings. Defines straightness as the maximum deviation of the centreline from the chord joining end centres, with sinuosity as an alternative. — [doi:10.1007/s11676-023-01600-x](https://link.springer.com/article/10.1007/s11676-023-01600-x)

### Inferences
- PS_LiDAR's design (ellipse fits in 0.2 m slices, then a centreline, then sweep-zone classification into cruising codes) is functionally **3DFin's `compute_sections` + `tilt_detection` + a straightness/grading layer that no open-source tool provides**. The Wales study confirms that this "3DFin + custom sagitta/bucking script" pattern is the state of practice in open tooling.
- No open-source tool emits Interpine/HQP or MARVL-style sweep codes. PS_LiDAR's classifier appears novel as a published or reusable component, and the published academic methods (Smax, sagitta per metre, Macdonald 7-point) are the natural comparison definitions.
- LiDAR360's straightness tool is a manual scoring aid, not an automated classifier. It is useful only as a semi-independent reference on a handful of trees.

### Gaps
- I could not access GreenValley's official LiDAR360 manual. The straightness-tool description comes from a third-party mirror, and I could not confirm whether LiDAR360 v7/v8 automates straightness in batch or outputs per-section stem curves.
- AdQSM, the Computree ONF plugins (for example stem diameter or curvature steps), 3D Forest's legacy "stem curve" feature (v0.5, Trochta et al. 2017) and TLSeparation (repo `TLSeparation/tlseparation` returned 404) were not verified in this pass.
- No public code for Windrim & Bryson (2020) was found; a GitHub search for "Windrim" returned nothing. The accuracy of their sweep estimate was not retrieved (full text blocked).
- The Forests 2022 13(3):431 paper's authors, species and accuracy were not retrieved (403).

## Q2. Licence, language, last release/commit, maintenance status (as of Nov 2026) and documentation

### Takeaway
The most actively maintained stem-curve tools in late 2026 are:
- **3DFin/dendromatics**: Python, GPL-3.0, PyPI releases in Dec 2025, commits to Jun 2026.
- **raycloudtools/RayExtract**: C++, commits to Sep 2026, RSE paper 2026.
- **PyTLidar**: Python TreeQSM port, PyPI 1.1.2 in Sep 2026.
- **3D Forest**: C++, release Mar 2026.

**FSCT** (last commit Mar 2024), **TreeQSM** (last commit May 2023), **treeseg** (Mar 2023), **SimpleForest's GitHub mirror** (2019) and **TreeLS on CRAN** (2.0.2, Aug 2020) are effectively unmaintained. GPL-3.0 dominates, which matters if PS_LiDAR is ever distributed.

### Cited Findings

Data pulled from the GitHub API, PyPI and crandb on 2026-11-19.

| Tool | Lang | Licence (GitHub SPDX) | Stars | Last push | Latest release / package | Stem curve? | Straightness? |
|---|---|---|---|---|---|---|---|
| [3DFin](https://github.com/3DFin/3DFin) | Python (+ CloudCompare/QGIS plugin) | GPL-3.0 | 105 | 2025-12-05 | v0.6.0 (2025-12-05); [PyPI 3dfin 0.6.0](https://pypi.org/project/3dfin/) | Yes (sections) | No (tilt for QC only) |
| [dendromatics](https://github.com/3DFin/dendromatics) | Python | GPL-3.0 | 33 | 2026-06-25 | [PyPI 0.7.0 (2025-12-05)](https://pypi.org/project/dendromatics/) | Yes | No |
| [dendroptimized](https://github.com/3DFin/dendroptimized) | C++ | GPL-3.0 | 0 | 2025-10-07 | [PyPI 0.3.0 (2025-10-06)](https://pypi.org/project/dendroptimized/) | backend | — |
| [FSCT](https://github.com/SKrisanski/FSCT) | Python | GPL-3.0 | 185 | 2024-03-03 | no releases, not on PyPI | Yes (`taper_data.csv`) | No |
| [TLS2trees](https://github.com/tls-tools-ucl/TLS2trees) | Python | GPL-3.0 | 35 | 2025-01-10 | tag `tls2trees` (2023-10-04) | No (segmentation) | No |
| [TreeQSM](https://github.com/InverseTampere/TreeQSM) | MATLAB | NOASSERTION (custom) | 233 | 2023-05-11 | 2.4.1 (2022-05-11) | Yes (`StemTaper`) | No |
| [PyTLidar](https://github.com/Landscape-CV/PyTLidar) | Python | GPL-3.0 | 32 | 2026-10-04 | [PyPI 1.1.2 (2026-09-03)](https://pypi.org/project/pytlidar/) | Yes (TreeQSM port) | No |
| [SimpleForest](https://github.com/SimpleForest/pluginSimpleForest) | C++ (Computree) | GPL-3.0 | 3 | 2019-07-19 (mirror) | via Computree | QSM | No |
| [TreeLS](https://github.com/tiagodc/TreeLS) | R/C++ | GPL-3.0 | 100 | 2025-06-11 | CRAN 2.0.2 (2020-08-26) | Yes (`stemSegmentation`) | No |
| [lidR](https://github.com/r-lidar/lidR) | R | GPL-3.0 | 716 | 2026-11-18 | CRAN 4.3.3 (2026-09-15) | No (ALS) | No |
| [ITSMe](https://github.com/lmterryn/ITSMe) | R | none detected | 61 | 2026-06-05 | GitHub v2.0.0 (2025-07-17); not on CRAN | DBH/QSM metrics | No |
| [raycloudtools / RayExtract](https://github.com/csiro-robotics/raycloudtools) | C++ (Docker image) | NOASSERTION (custom CSIRO) | 122 | 2026-09-24 | no GitHub releases | Yes (tree structure text) | No |
| [treeseg](https://github.com/apburt/treeseg) | C++ | MIT | 257 | 2023-03-20 | — | No | No |
| [3D Forest](https://github.com/VUKOZ-OEL/3d-forest) | C++ | GPL-3.0 | 98 | 2026-10-31 | release-2026-03-24 | unverified | No |
| [pc-skeletor](https://github.com/meyerls/pc-skeletor) | Python | MIT | 233 | 2025-04-01 | 1.0.1 (2024-04-12); PyPI 1.0.0 | skeleton | No |
| [AdTree](https://github.com/tudelft3d/AdTree) | C++ | GPL-3.0 | 260 | 2026-10-20 | v1.1.2 (2023-02-10) | branch model | No |
| [SegmentAnyTree](https://github.com/SmartForest-no/SegmentAnyTree) | Python | MIT | 173 | 2025-08-22 | — | No | No |
| [ForAINet](https://github.com/prs-eth/ForAINet) | Python | NOASSERTION | 119 | 2026-07-07 | — | No | No |
| [ForestFormer3D](https://github.com/SmartForest-no/ForestFormer3D) | Python | NOASSERTION | 135 | 2026-03-03 | — | No | No |
| [TreeLearn](https://github.com/ecker-lab/TreeLearn) | Python | MIT | 160 | 2026-02-17 | — | No | No |
| [pointtree](https://github.com/ai4trees/pointtree) | Python | MIT | 36 | 2026-10-15 | v1.0.1 (2026-04-19), PyPI | DBH only | No |
| [PointsToWood](https://github.com/harryjfowen/PointsToWood) | Python | AGPL-3.0 | 35 | 2026-05-30 | — | No | No |
| [LeWoS](https://github.com/dwang520/LeWoS) | MATLAB | MIT | 64 | 2026-08-06 | — | No | No |
| [treeiso](https://github.com/truebelief/artemis_treeiso) | Python/C++ | NOASSERTION | 70 | 2025-03-21 | — | No | No |
| [rTwig](https://github.com/aidanmorales/rTwig) | R/C++ | NOASSERTION (CRAN: GPL ≥3) | 23 | 2026-10-16 | CRAN 1.4.0 (2025-03-03) | QSM taper correction | No |
| [aRchi](https://github.com/umr-amap/aRchi) | R | CRAN: CeCILL | 41 | 2025-12-12 | CRAN 2.1.3 (2022-09-03) | QSM | No |
| [spanner](https://github.com/bi0m3trics/spanner) | R/C++ | none detected | 36 | 2026-09-18 | — | ? | No |
| [PyForestScan](https://github.com/iosefa/PyForestScan) | Python | MIT | 89 | 2026-11-15 | [PyPI 0.5.0 (2026-10-15)](https://pypi.org/project/pyforestscan/) | No (ALS structure) | No |
| [optBuck](https://github.com/SmartForest-no/optBuck) | R | NOASSERTION | 5 | 2026-11-16 | GitHub only | input = stem profiles | — |

- Package names checked on PyPI with no package found: `fsct`, `tls2trees`, `treeqsm`, `spatialtree`, `pyforest3d`, `forestscanner`. — [PyPI JSON API](https://pypi.org/)
- TreeLS's README still says "currently on v2.0.2" (the August 2020 major release). — [TreeLS README](https://github.com/tiagodc/TreeLS)
- PyTLidar supports Python 3.8–3.12 and offers a headless install, or a CloudCompare plugin "which uses CloudCompare's own Qt". — [PyPI pytlidar](https://pypi.org/project/pytlidar/)
- RayExtract ships a pre-built Docker container for Linux, macOS and Windows. — [raycloudtools README](https://github.com/csiro-robotics/raycloudtools)

### Inferences
- **Australian relevance.** RayExtract (CSIRO, with Queensland and NSW co-authors Phinn, Levick, Woodgate and others) and Windrim & Bryson (NSW radiata) are the two Australian-origin stem-reconstruction efforts. RayExtract is aimed at tree *volume*, not form or sweep.
- **Licences.** GPL-3.0 tools (3DFin, dendromatics, FSCT, PyTLidar, TreeLS) can be *run* freely as external benchmark baselines. Importing them as libraries inside a distributed PS_LiDAR would make the combined work GPL-bound. MIT tools (SegmentAnyTree, TreeLearn, pointtree, pc-skeletor) carry no such constraint. *(Licence interpretation, not legal advice.)*
- **Effectively unmaintained:** FSCT, TreeQSM (MATLAB), treeseg and the SimpleForest GitHub mirror. FSCT still works, but expect dependency rot (PyTorch versions, for example).

### Gaps
- The exact terms of the "NOASSERTION" licences (TreeQSM, raycloudtools, ForAINet, ForestFormer3D, treeiso) were not read.
- Documentation quality was judged only from README length and content. 3DFin and dendromatics have Read-the-Docs-style docs (the URL `3dfin.readthedocs.io/en/latest/` returned 404 here), and FSCT has a detailed README (377 lines).

## Q3. Commercial and in-house tools: which claim sweep, straightness or log grades, and with what method?

### Takeaway
Vendor claims about sweep are mostly implicit (bucking on measured stem profiles) or undocumented. Only LiDAR360 documents an explicit straightness tool, and it is manual and uses a 7-point Macdonald scale. Interpine (NZ, with Australian operations) has publicly claimed Hovermap-based sweep and diameter along the stem profile since 2020, with no published method. Treemetrics ForestHQ bucks TLS stem profiles "including … the sweep of the stem". Deep Forestry claims diameters along the full stem and lean, but not straightness. I found no evidence that Emesent Aura, FARO Connect or GeoSLAM output sweep.

### Cited Findings
- **Interpine (NZ).** A 22 Sep 2020 post: young (radiata) pine scanned with a backpack-mounted Hovermap, "digitally pruned and then measured", giving "sweep and diameter measurement through the tree stem profile". No method, accuracy or stem-description codes are given. — [Interpine blog](https://interpine.nz/lidar-scanning-forest-at-younger-age-for-selection-trials)
  - Emesent (2020): Interpine used Hovermap with "its own deep-learning algorithms", producing stem diameter, form, quality and height. — [Emesent news 2020-11-18](https://www.emesent.com/news/2020/11/18/interpine-are-speeding-up-forest-imagery-plotting-with-hovermap)
- **PlotSafe is Interpine's field data-capture software,** not Silmetra's. It records overlapping stem-level feature classifications with dictionaries such as RAD05, and exports to YTGen and XML. — [Interpine PlotSafe page](https://interpine.nz/plotsafe); [PlotSafe 1.2 release note](https://interpine.nz/plotsafe-forest-inventory-datacapture-version-1-2-released/)
  - RAD05 is described as a "universal dictionary" for NZ *P. radiata* cruising; most codes are stem description codes. — [RAD05 dictionary post](https://interpine.nz/15-rad05-cruising-domain-for-plotsafe-feature-cruising-dictionary/)
- **Treemetrics ForestHQ (Ireland).** ForestHQ "is designed to work with Terrestrial LiDAR data". Profiles "more precisely represent the real tree's shape, including defects and the sweep of the stem". The virtual harvest simulator "virtually cuts the 3D stems profiles exactly as a harvesting machine would", optimising a weighted product list. There is no explicit sweep metric or quality code. — [ForestHQ docs: crosscut simulations](https://docs.foresthq-en.treemetrics.com/2-measurement-and-valuation/3-6-crosscut-simulations)
  - The predecessor product, AutoStem, was launched in November 2006 under the COFORD TreeScan project. — [COFORD Connects TreeScan](https://www.coford.ie/media/coford/content/publications/projectreports/cofordconnects/ccntreescan.pdf)
- **GreenValley LiDAR360 TLS Forest:** manual 7-point straightness tool; see Q1. — [Blog mirror](https://mendiolatooffer.blogspot.com/2022/10/measuring-and-drawing-trees-stem.html)
  - The TreeLearn authors note that LiDAR360's automatic segmentation "contains numerous errors", such as multiple trees merged. — [TreeLearn arXiv 2309.08471](https://arxiv.org/pdf/2309.08471)
- **Deep Forestry (Uppsala, Sweden).** Products: Neuron Edge (below-canopy drone or backpack capture) and Neuron Mind (detection and measurement). Claims: "diameter along the stem ('measured where seen, fitted above')", stem volume, and lean (for corridors). Its deepest validation is tree-by-tree against **harvester measurements in radiata pine**. The website states no straightness, taper or quality output and gives no numeric accuracy. — [deepforestry.com](https://www.deepforestry.com/)
  - A search snippet of its €3M press release claimed a mean absolute error of 1.6 cm along the full stem, verified by a government authority, but the fetched page did not show this figure. **Unverified.** — [Deep Forestry press release](https://www.deepforestry.com/press-release/deep-forestry-raises-eu3m-to-build-the-forestry-industrys-spatial-intelligence-layer)
- **Emesent Aura, FARO Connect / GeoSLAM:** targeted searches returned no vendor documentation of stem-curve, sweep or straightness outputs. Emesent's own forestry content concerns Hovermap plot capture and "stem extraction with diameter profiles". — [Emesent blog 2022](https://emesent.com/2022/09/15/lidar-data-and-digital-twins-for-better-forest-management/)
- **"Treescope"** turned up only as TreeScope, an academic dataset ("Agricultural Robotics Dataset for LiDAR-Based Mapping of Trees in Forests and Orchards"), not a commercial stem-quality product. — [arXiv 2310.02162](https://arxiv.org/pdf/2310.02162)
- **Scion (NZ)** has tested Hovermap and handheld scanning. A 2017 trade article says pruned height, stem straightness, taper and lean of pruned trees can be measured with a handheld unit (magazine piece, not peer-reviewed). — [NZFFA "Eyes in the sky" (May 2017)](https://www.nzffa.org.nz/farm-forestry-model/tree-grower-articles/may-2017/eyes-in-the-sky/)

**Optimisation and yield tools (do they ingest sweep?)**
- **MARVL / YTGEN (NZ).** MARVL dictionaries split sweep into "Longs" and "Shorts" codes, dating from the late 1990s. For example, "Sw:8,L" lets a stem with the same 6 m sweep class be differentiated for long-log eligibility. YTGEN uses stem features (Br, Sw, F, I) to "buck" or "logmake" stems, and RAD06 "is commonly used for minor sawlog species in NZ and Australia". So these tools **ingest coded sweep, not geometric sweep**. — [Interpine: long vs short logs](https://interpine.nz/better-predicting-yield-of-long-logs-vs-short-logs-of-the-same-grade-quality/); [Interpine: RAD05 / RAD05-A / RAD06](https://interpine.nz/marvl-cruising-dictionaries-what-is-rad05-rad05-a-and-rad06/)
- **optBuck (R).** "Handling production files from forest harvesters and optimal bucking". It reads StanForD 2010 `.hpr` files and optimises bucking with dynamic programming. No sweep input is documented (inputs are diameter profiles and price matrices). The repo has moved to SmartForest-no/optBuck (active, push 2026-11-16). — [rdrr.io optBuck](https://rdrr.io/github/lennartnoordermeer/optBuck/f/README.md); [GitHub SmartForest-no/optBuck](https://github.com/SmartForest-no/optBuck)
- **ForestHQ crosscut simulator** bucks measured 3D profiles, so sweep is implicitly honoured; see above. — [ForestHQ docs](https://docs.foresthq-en.treemetrics.com/2-measurement-and-valuation/3-6-crosscut-simulations)

### Inferences
- For PS_LiDAR's PlotSafe-style export, the realistic downstream consumers are **YTGEN/MARVL-style logmakers that ingest sweep codes** (Sw class plus L/S qualifiers). That justifies converting geometric sweep into codes rather than exporting raw sagitta alone. Exporting both (Smax per log window *and* code) would also enable geometry-aware bucking such as ForestHQ's or optBuck's.
- Interpine appears to be the one commercial actor with an operational LiDAR-to-sweep workflow in the radiata and plantation-pine market of NZ and Australia. Its method is unpublished, so it can serve only as a "commercial comparator" if data-sharing were arranged.

### Gaps
- No primary documentation was found for: Emesent Aura forestry outputs; FARO Connect forestry; Arbonaut; Trimble Forestry; TerraScan/TerraSolid; 4D-Ag; Sensorite; Forest Metrix (only [forestmetrix.com](https://forestmetrix.com/) surfaced, without sweep detail); Remsoft; Forestry Tasmania tools.
- Not covered: Scion AVIS, Atlas Cruiser, BuckR (only a snippet saying it is "published as a script"; possibly [Forests 16:780 (2025)](https://mdpi-res.com/d_attachment/forests/forests-16-00780/article_deploy/forests-16-00780.pdf), unverified) and the FPInnovations tools (Optitek is mentioned in a sweep-measurement paper only via search snippet).
- Silmetra's role could not be confirmed. Every PlotSafe source found is on interpine.nz.

## Q4. Benchmarks comparing these tools, and reported stem-curve accuracy

### Takeaway
The only multi-algorithm benchmark that scores **stem curves** is the EuroSDR/Liang et al. (2018) international TLS benchmark: 18 algorithms, DBH and stem curve to roughly 1–2 cm in easy plots, with detection dropping in dense plots. Sensor comparisons report stem-curve RMSE of 2–15% (Hyyppä et al. 2020). Modern benchmarks (FOR-instance, LAUTx, Wytham Woods, TreeLearn) score **segmentation and DBH, not stem curves or sweep**. No benchmark for straightness or sweep exists. The one validation found (Pyörälä 2019, Scots pine) showed a **78% relative mean difference for sweep** against sawmill scans.

### Cited Findings
- **Liang et al. (2018), "International benchmarking of terrestrial laser scanning approaches for forest inventories"** (ISPRS JPRS 144:137–179; doi:10.1016/j.isprsjprs.2018.06.021). 18 algorithms were compared under EuroSDR, launched in 2014. TLS methods estimated "DBH and stem curve with a 1–2 cm accuracy that are close to what is required in practical applications". Single-scan completeness was about 75% in easy plots versus about 30% in difficult plots (2000 stems/ha). Data are open for non-commercial use. — [TU Delft repository record](https://repository.tudelft.nl/record/uuid:d5fb6947-dc54-424f-a3b0-43af4c75dd9d); [EGU2018 abstract](https://meetingorganizer.copernicus.org/EGU2018/EGU2018-2476.pdf)
- **Hyyppä et al. (2020)** compared backpack, handheld, under-canopy UAV and above-canopy UAV laser scanning: stem curve RMSE 2–15%, DBH 2–8%. — [Aalto ACRIS PDF (ISPRS JPRS 2020)](https://acris.aalto.fi/ws/portalfiles/portal/42219378/1_s2.0_S0924271620300915_main.pdf)
- Mobile mapping systems achieved 4–7 cm stem-curve RMSE (as reported in a Finnish thesis or review). — [core.ac.uk PDF](https://core.ac.uk/download/401692679.pdf)
- **3DFin paper (Laino et al. 2024, *Forestry* 97(4):479):** developers' own TLS evaluation, DBH RMSE 1.75 cm with near-complete detection. — [Oxford Academic](https://academic.oup.com/forestry/article/97/4/479/7680458)
- **2026 low-cost MLS urban comparison.** DBH rRMSE was lowest for ITSMe, then 3DFin (11.27–12.30%), then FSCT (11.75–13.00%). SD of error: ITSMe ±1.19 cm, 3DFin ±1.87 cm, FSCT ±2.09 cm. Height: FSCT RMSE 1.6 m versus 3DFin 2.4 m. — [Frontiers in Remote Sensing 2026, doi:10.3389/frsen.2026.1774149](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2026.1774149/full)
- **Handheld LiDAR, Mediterranean mixed forest (Forests 2026, 17(8):944):** FSCT and 3DFin were comparable for DBH; FSCT "outperformed 3DFin across all species", mainly on height. — [MDPI Forests 17/8/944](https://www.mdpi.com/1999-4907/17/8/944)
- **Older stem-diameter method comparison (~2019):** TLS–QSM showed the lowest bias (−3.7%) across tree sizes. — [Forest Research / Computree three-algorithms comparison (ResearchGate)](https://www.researchgate.net/publication/334562375_Comparison_of_Three_Algorithms_to_Estimate_Tree_Stem_Diameter_from_Terrestrial_Laser_Scanner_Data)
- **FOR-instance (Puliti et al. 2023):** a manually annotated ULS benchmark for semantic and instance segmentation of individual trees, which also includes DBH data. — [arXiv 2309.01279](https://arxiv.org/pdf/2309.01279); [Zenodo 8287792](https://zenodo.org/record/8287792)
  - FOR-instanceV2 adds MLS and tropical data with about 8× the tree instances. ForestFormer3D was evaluated on unseen Wytham Woods and LAUTx data. — [arXiv 2506.16991](https://arxiv.org/abs/2506.16991v2)
- **Sweep validation (Pyörälä et al. 2019, Scots pine, TLS vs sawmill):** diameter, volume and taper agreed reasonably; sweep RMD 78%. — [UTU research portal](https://research.utu.fi/converis/portal/detail/Publication/458274175?lang=en_GB)
- **The Wales 3DFin + R study** reports lower agreement for sweep, taper and lean than for DBH; it gives no numbers in the executive summary. — [Woodknowledge Wales PDF](https://woodknowledge.wales/wp-content/uploads/Estimating-Timber-Value-and-Carbon-in-Complex-Woodlands-using-Terrestrial-Laser-Scanning-FINAL.pdf)
- "Derivation of Tree Stem Curve and Volume Using Point Clouds" (ISPRS Archives 2024) is a further stem-curve method paper; its details were not read. — [ISPRS Archives XLVIII-4-W11-2024](https://isprs-archives.copernicus.org/articles/XLVIII-4-W11-2024/81/2024/isprs-archives-XLVIII-4-W11-2024-81-2024.pdf)

### Inferences
- There is **no public benchmark against which PS_LiDAR's sweep codes could be scored**. Validation will need in-house reference data, such as field sweep assessments by trained cruisers, harvester or sawmill scanner profiles, or felled-stem measurements.
- The Liang 2018 benchmark data (non-commercial licence) is the only open dataset with reference stem curves. It is boreal Scots pine, spruce and birch, so it suits validating PS_LiDAR's centreline and diameter-profile stage, not the plantation sweep classifier.
- The Pyörälä result suggests that sweep from TLS or MLS is noise-sensitive. PS_LiDAR's sweep-zone thresholds should be tested for robustness against centreline noise, for example by bootstrapping slice fits.

### Gaps
- Per-algorithm stem-curve RMSE values from Liang et al. 2018 were not extracted (full paper not fetched).
- I found no benchmark of RayExtract, PyTLidar or TreeLS stem curves against each other, and none on Australian pine plantations.
- Wytham Woods (Calders et al.) and LAUTx (Tockner et al. 2022) primary sources were not fetched. They are referenced only through ForestFormer3D and FOR-instance.

## Q5. Practical reuse: what could feed PS_LiDAR (segmentation front-end) or serve as a validation baseline, and at what effort?

### Takeaway
The cheapest high-value reuse is to run **dendromatics/3DFin (pip-installable, the same 0.2 m section logic) as a stem-curve baseline** on the same plots. PS_LiDAR's ellipse centres and diameters would be compared section by section against 3DFin's circle centres and radii, and 3DFin's axis and tilt output against PS_LiDAR's centreline. The Wales Smax and log-selection logic can then serve as a published straightness definition to cross-check the sweep-zone codes. For the segmentation front-end, FSCT and TLS2trees (Python, GPL-3.0, CPU or GPU) or MIT-licensed SegmentAnyTree, TreeLearn and pointtree are drop-in candidates. PyTLidar (Python TreeQSM) or RayExtract (Docker CLI) provide an independent QSM-based stem-taper baseline.

### Cited Findings
- dendromatics exposes `generate_dtm`, `clean_ground`, individualisation, `compute_sections(stems, sections)` with `section_len = 0.2`, `tilt_detection` and `tree_locator`. These are numpy-based functions callable from Python. — [dendromatics README](https://github.com/3DFin/dendromatics); [sections.py](https://github.com/3DFin/dendromatics/blob/main/src/dendromatics/sections.py)
- 3DFin also runs as a CloudCompare plugin. The Wales study used it there, after manual tree isolation in CloudCompare. — [Woodknowledge Wales PDF](https://woodknowledge.wales/wp-content/uploads/Estimating-Timber-Value-and-Carbon-in-Complex-Woodlands-using-Terrestrial-Laser-Scanning-FINAL.pdf)
- FSCT produces semantic segmentation (terrain, vegetation, CWD, stems) and per-stem `taper_data.csv` and `cleaned_cyls`. TLS2trees reuses FSCT's semantic step with `--batch_size` and `--num_procs` controls and tiled processing, and adds graph-based instance segmentation. — [FSCT README](https://github.com/SKrisanski/FSCT); [TLS2trees README](https://github.com/tls-tools-ucl/TLS2trees)
- PyTLidar provides TreeQSM through Python with a headless install. TreeQSM's `treedata.StemTaper` provides a QSM-based taper curve. — [PyPI pytlidar](https://pypi.org/project/pytlidar/); [TreeQSM tree_data.m](https://github.com/InverseTampere/TreeQSM/blob/master/src/main_steps/tree_data.m)
- RayExtract runs from the command line (`rayextract terrain`, `rayextract trees`) and ships as a Docker image; it is designed as "fast, scalable" for tree volume. — [raycloudtools README](https://github.com/csiro-robotics/raycloudtools)
- pointtree is a pip-installable (MIT) instance-segmentation package. — [pointtree](https://github.com/ai4trees/pointtree)
- TreeLearn and SegmentAnyTree are MIT-licensed deep-learning segmenters for ground-based and sensor-agnostic LiDAR. — [TreeLearn](https://github.com/ecker-lab/TreeLearn); [SegmentAnyTree](https://github.com/SmartForest-no/SegmentAnyTree)
- TreeLS's `stemSegmentation()` (RANSAC circle or cylinder per stem segment) is an alternative R baseline. — [TreeLS README](https://github.com/tiagodc/TreeLS)

### Inferences

Effort estimates are rough, based on README and API inspection, not on runs. Ranked by value for effort:

1. **dendromatics/3DFin as the stem-curve baseline** (low effort, about 1–2 days).
   - Both use numpy and LAS, and both slice at 0.2 m, so a per-section comparison is direct.
   - Feed PS_LiDAR-segmented stems in through the "assigned_cloud" path, or let 3DFin individualise.
   - Run it as a separate process to avoid GPL linkage.
2. **Implement the Wales Smax definition and Macdonald/FC 7-point score as alternative straightness outputs** (low effort).
   - This makes PS_LiDAR's cruising codes comparable to the published definitions and to LiDAR360's manual tool.
3. **Segmentation front-end** (medium effort).
   - FSCT or TLS2trees (GPL-3.0, PyTorch; GPU recommended) or TreeLearn, SegmentAnyTree or ForestFormer3D (MIT/other; CUDA stacks, heavier installs).
   - Benefit: better stem separation in dense young plantations.
   - Cost: dependency weight, and models trained mostly on European, boreal and temperate data, not Queensland hybrid pine.
4. **PyTLidar or RayExtract QSM taper as an independent baseline** (medium effort).
   - Useful to check whether centreline and sweep are stable under a different geometric representation (cylinders versus slice ellipses).
5. **LiDAR360 TLS Forest** (commercial licence).
   - Useful only for manual spot checks of straightness scores on a few trees, not as a batch baseline.

Novelty: given that no open tool emits sweep codes, PS_LiDAR's classifier plus a documented algorithm and validation could be a publishable contribution, especially on Australian hybrid pine.

### Gaps
- I did not test the tools on PS_LiDAR data, so actual runtime, memory and scalability on Queensland plots are unknown.
- The sensors PS_LiDAR uses (TLS, MLS/Hovermap or ULS) were not specified in the brief. Tool suitability differs: FSCT and 3DFin target TLS, MLS and dense ULS; Windrim & Bryson used helicopter ALS.
- Whether the 3DFin and dendromatics hosted docs are current: the `latest` Read-the-Docs URL returned 404 on 2026-11-19.
