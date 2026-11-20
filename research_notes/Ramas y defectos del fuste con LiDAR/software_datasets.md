# Software, libraries and datasets for branch, wood/leaf, fork and stem-defect extraction from forest LiDAR

Research date: 2026-11-20. Repository metadata (stars, last push, SPDX licence, language) was pulled live from the GitHub API on that date. This note covers **branch and defect capabilities only**. Stem-curve and sweep content is in `../Rectitud del fuste con LiDAR/software_librerias.md` and is not repeated here. Tool-call budget was limited, so several items are flagged as **unverified**.

## Q1. Which tools output per-branch diameters and positions along the stem? What branch-level accuracy is reported, and what is the minimum branch size reliably reconstructed?

### Takeaway
Only QSM-type tools output an explicit **per-branch record**: diameter, order, insertion height, angle and azimuth. The main ones are TreeQSM, its Python port PyTLidar, SimpleForest and AdTree. RayExtract's output fields could not be verified.

Destructive validations show TreeQSM is reliable for **branches above about 10 cm** (95% of branches over 30 cm found; diameters of 20–60 cm branches within about 8%). Below about 5 cm, branch diameters are **strongly overestimated** (+73% for branches under 2.5 cm).

Branch-specific TLS methods for conifers (Pyörälä et al., Scots pine; Hollaus et al. 2021, Norway spruce) detect about 70% of whorls. They **underestimate** maximum branch diameter, with RMSE about 1.4 cm (46%).

For PlotSafe branch codes (4/7/10/15/>15 cm), this means:
- the 4 cm class boundary is below what has been reliably validated;
- 7–15 cm sits in the zone where QSMs have about 4–8% diameter bias but unknown detection rates in dense conifer crowns.

### Cited Findings

**QSM tools with per-branch output**
- **TreeQSM (MATLAB, v2.4.1).** `branches.m` "Determines the branching structure and computes branch attributes". The output `branch` structure contains "Branch order, parent, volume, length, angle, height, azimuth and diameter". The trunk is branch 1 with order 0. The source header says "free software … GNU General Public License … version 3", so GitHub's "NOASSERTION" SPDX tag hides a GPL-3 licence. — [TreeQSM branches.m](https://github.com/InverseTampere/TreeQSM/blob/master/src/main_steps/branches.m)
  - The repo's last push was 2023-05-11 (233 stars), so it is unmaintained. — [GitHub API, InverseTampere/TreeQSM](https://github.com/InverseTampere/TreeQSM)
- **PyTLidar (Python, GPL-3.0).** A Python port of TreeQSM, last push 2026-10-04. Its README CLI options concern patch diameters (`--ipd`, `--minpd`, `--maxpd`) and choosing the best model by "lowest mean distance to trunk". The README does not document branch-table export explicitly; because it ports TreeQSM, branch outputs are *inferred*, not verified. — [PyTLidar README](https://github.com/Landscape-CV/PyTLidar)

**Destructive branch-level validations of TreeQSM**
- **Lau et al. 2018** (*Trees*): nine tropical trees in Guyana, TLS, TreeQSM versus destructive harvest. The reference set was 279 branches over 10 cm.
  - TreeQSM reconstructed 95% of branches over 30 cm.
  - It underestimated diameters of 20–60 cm branches by 8%.
  - Branching order was correct in 99% of cases.
  - The supplementary table shows a 10–20 cm class with modelled 19.25 cm versus measured 14.10 cm, i.e. small branches are overestimated.
  - Sources: [Trepo record](https://trepo.tuni.fi/handle/10024/126544); [CIFOR PDF](https://cifor-icraf.org/publications/pdf_files/articles/ALau1802.pdf); [ESM table](https://media.springernature.com/original/springer-static/esm/art%3A10.1007%2Fs00468-018-1704-1/MediaObjects/468_2018_1704_MOESM3_ESM.pdf)
- **Demol et al. 2022** (*Silva Fennica* 56(1):10550): two *Fraxinus excelsior*, RIEGL VZ-400i with 21 scans leaf-off, TreeQSM v2.4.0. Reference diameters were measured by tape (>2.5 cm) or caliper.
  - Branches over 10 cm were overestimated by about +4%; branches under 2.5 cm by about +73%.
  - "The accuracy of QSM branch diameter estimates decreased with smaller branch diameters."
  - Branches under 5 cm caused 80–83% of the 38–52% total-volume overestimation.
  - All 265 measured branch locations matched a cylinder within 5 cm. This is a location match, not a detection rate.
  - Source: [Silva Fennica 10550](https://www.silvafennica.fi/article/10550)
- **Gap in the literature:** a review statement says no experiment has destructively validated individual branch-diameter reconstruction for branches under 10 cm in field conditions (paraphrased from a search snippet of the Silva Fennica paper). — [Silva Fennica PDF](https://silvafennica.fi/pdf/10550)
- **rTwig validation vignette** (TreeQSM v2.4.1 against destructive branch biomass): mean relative error of branch biomass 949% versus 4.6% for the main stem (*mass*, not diameter). rTwig corrects small-branch radii using measured twig diameters. — [rTwig Validation vignette](https://packages.oit.ncsu.edu/cran/web/packages/rTwig/vignettes/Validation.html)

**Conifer-specific branch/whorl methods**
- **Pyörälä et al. 2018** (IEEE J-STARS): 158 Scots pines in six mature stands, TLS.
  - Whorl detection accuracy was 69.9%, with 1.9% false positives.
  - Maximum branch diameter per whorl was underestimated by 0.34 cm (11.1%), RMSE 1.42 cm (46.0%).
  - Distance to the scanner, occlusion and wind were the main error factors.
  - Source: [UTU portal](https://research.utu.fi/converis/portal/detail/Publication/458277313?lang=en_US); [core.ac.uk PDF](https://core.ac.uk/download/286389376.pdf)
- **Pyörälä et al. 2018** (*Canadian J. Remote Sensing*): branch diameter (bd) and insertion angle (bα) in crown sections, compared with manual point-cloud references.
  - bd was underestimated (tree-specific bias 0.89 cm); angle bias was 1.98°.
  - Accuracy decreased above the live-crown base, mainly because of scanner distance.
  - Source: [DOAJ](https://doaj.org/article/4d584b7c2f1e491fa804d49ffc152ab0)
- **Pyörälä et al. 2018** (*Scand. J. For. Res.*): TLS versus sawmill X-ray on 162 Scots pine log sections.
  - X-ray found 37.25 whorls per tree versus 22.93 from TLS.
  - Tree-level maximum branch diameter from TLS was not significantly different from X-ray maximum knot diameter.
  - Source: [UEF eRepo record](https://erepo.uef.fi/handle/123456789/7163?show=full) (summary via search snippet)
- **Hollaus, Otepka, Galle & Bronner 2021** (SilviLaser 2021, doi:10.34726/wim.2002): Norway spruce, about 120 years old, unpruned, RIEGL VZ-400i with 18 scans.
  - Stems are modelled with robust cylinder fits.
  - Branch points are taken from a **cylindrical shell between stem radius +5 cm and +10 cm**, projected **equirectangularly (cylinder angle × height)**. Branch positions come from point density and distance distributions.
  - **Branch diameter is estimated from quantiles of the horizontal extent only**, because the upper side of branches is not seen.
  - The method outputs **branch diameter classes**. It is "fully automatic" with "high degree of completeness", but no numbers are given.
  - Implementation is OPALS (TU Wien, commercial/academic) plus Python. No code was released.
  - Source: [TU Wien repositum PDF](https://repositum.tuwien.at/bitstream/20.500.12708/19141/3/Hollaus-2021-Branch%20detection%20based%20on%20TLS%20data-vor.pdf)
- **BranchPoseNet** (Puliti, Fischer, Astrup, arXiv 2409.14755, 2024; NIBIO): pose-estimation deep learning on "sectional images" of the point cloud to find whorl and branch keypoints along the stem. It was tested on destructively sampled trees with whorls located on felled stems. The abstract gives no accuracy figures, and code and data availability were not confirmed. — [arXiv 2409.14755](https://arxiv.org/abs/2409.14755)
- **Olofsson & Holmgren 2023** (ISPRS Open J. Photogramm. RS): density-based branch extraction from harvester-mounted scanner point clouds compared with X-ray. They cite about 65–70% detection accuracy for voting-based branch detection. Full text was blocked (403), so only a search snippet was available. — [DOAJ record](https://doaj.org/article/2c3abf4218c24307b7a3180e25970d01)
- **Forest Ecosystems (Jan 2024)**: drone laser scanning slice metrics used to predict the "diameter of the largest branch per log (DLB)" with "high accuracy". Species, authors and numbers were not retrieved (403); only a search snippet was available. — [DOAJ record](https://doaj.org/article/2a945a67449d46c9a0cd9916ce39abed)

**Other skeleton/branch modellers**
- **AdTree** (TU Delft; C++, GPL-3.0; last push 2026-10-20; 260 stars): branch modelling from point clouds. Branch-level accuracy was not retrieved in this pass. — [GitHub AdTree](https://github.com/tudelft3d/AdTree)
- **smart-tree** (Univ. Canterbury NZ, uc-vision; Python, MIT; last push 2026-01-13): "Neural Medial Axis Approximation of Point Clouds for 3D Tree Skeletonization". It outputs a mesh and skeleton and is linked to a "synthetic-trees" dataset.
  - The README does not state whether radii are exported, the GPU needs, or the species tested.
  - Source: [GitHub uc-vision/smart-tree](https://github.com/uc-vision/smart-tree)
- **pc-skeletor** (Python, MIT; last push 2025-04-01): generic Laplacian/semantic skeletonisation. It gives a skeleton graph, not radii. — [GitHub pc-skeletor](https://github.com/meyerls/pc-skeletor)
- **RayExtract** (CSIRO raycloudtools; C++; last push 2026-09-24): `rayextract trees` "extracts tree structures to text file". The README does not document the fields; per-segment radii and parents are *likely* but **unverified**. The licence is CSIRO custom (`licence.txt`, not read). — [raycloudtools README](https://github.com/csiro-robotics/raycloudtools)
- **AdQSM**: a 2020 *Remote Sensing* PDF hosted at Tsinghua surfaced, which confirms that it exists as a QSM method. Code, licence and branch outputs were not verified. — [AdQSM RS 2020 PDF](https://archive.ymsc.tsinghua.edu.cn/pacm_download/61/11848-adqsm-RS-2020.pdf)

### Inferences
- **The Hollaus 2021 shell-and-unwrap approach is the closest published analogue to what PS_LiDAR needs.** It works per stem section, uses an existing cylinder/circle stem model, and outputs branch diameter *classes*. PS_LiDAR already fits slice circles, so the approach could be reimplemented in numpy/scipy in a few days without any GPL dependency: take a radial shell outside the fitted radius, unwrap it to an (angle, height) image, cluster it, and estimate the horizontal extent.
- Because every TLS study shows branch diameter **underestimated** from point extent (occluded upper side) while QSMs **overestimate** small branches, a PS_LiDAR classifier should be calibrated per class boundary against field or destructive data, not used raw.
- In unpruned hybrid pine, the 4 cm and 7 cm boundaries are the risky ones. The >10 cm and >15 cm classes are within the validated envelope of QSMs and of the Pyörälä methods.
- Forks (a codominant leader) are essentially "branch order 1 with diameter close to the stem diameter" in a QSM. TreeQSM/PyTLidar's branch table (order, diameter, height, angle) is therefore a ready fork-detector baseline. This is inferred; no fork-specific validation was found.

### Gaps
- No tool found outputs **spike knots, nodal swelling, fluting, ovality or damage** as named defects. No open "stem-surface defect map" tool was found. Open3D/PCL surface-deviation methods (residuals from a fitted cylinder, unwrapped to a raster) were not researched as packaged tools; this would be custom code.
- RayExtract's tree-file fields, AdTree's branch-diameter accuracy, SimpleForest's branch exports and 3D Forest's branch features were not verified.
- No published branch-detection accuracy for *Pinus radiata*, *P. elliottii × caribaea* or other plantation pines was found. A targeted search returned only stem-DBH studies of radiata. — [Remote Sensing 17(8):1456](https://mdpi.com/2072-4292/17/8/1456)
- LiDAR360: no documentation of a branch-extraction tool was found (only TLS-Forest segmentation/DBH). — [arXiv 2108.01552](https://arxiv.org/pdf/2108.01552)
- Commercial (Treemetrics, Interpine, Deep Forestry): no branch-size or knot output claims were found in this pass. See the stem note for their stem claims.

## Q2. Which tools are maintained, licence-compatible, CPU-feasible and Python-callable (including wood/leaf and semantic branch segmentation)?

### Takeaway
For **wood/leaf or branch-class semantic labels**, the maintained options are:
- PointsToWood (AGPL-3.0, deep learning);
- LeWoS (MIT, MATLAB, unsupervised, CPU);
- ForAINet / ForestFormer3D (stem / live branch / woody branch classes from FOR-instance; GPU; licence NOASSERTION);
- FSCT / TLS2trees (GPL-3.0; stem and vegetation, but no branch class).

For **per-branch geometry**, PyTLidar is the only maintained, pip-installable Python option, and it is GPL-3.0. TreeQSM itself is also GPL-3 (per its source headers) and unmaintained.

MIT-licensed geometry helpers (pc-skeletor, smart-tree) give skeletons, not validated branch diameters. TLSeparation could not be located on GitHub under its known names.

### Cited Findings

GitHub API, 2026-11-20:

| Tool | Branch/defect-relevant output | Lang | Licence | Last push | Stars | GPU? | Python-callable |
|---|---|---|---|---|---|---|---|
| [TreeQSM](https://github.com/InverseTampere/TreeQSM) | per-branch order, parent, diameter, length, angle, height, azimuth | MATLAB | GPL-3 (source header; SPDX NOASSERTION) | 2023-05-11 | 233 | No | No (MATLAB) |
| [PyTLidar](https://github.com/Landscape-CV/PyTLidar) | TreeQSM port (branch table inferred) | Python | GPL-3.0 | 2026-10-04 | 32 | No | Yes (pip) |
| [AdTree](https://github.com/tudelft3d/AdTree) | branch skeleton + cylinder model | C++ | GPL-3.0 | 2026-10-20 | 260 | No | CLI only |
| [raycloudtools / RayExtract](https://github.com/csiro-robotics/raycloudtools) | tree-structure text file (fields unverified) | C++ | CSIRO custom | 2026-09-24 | 122 | No | CLI / Docker |
| [smart-tree](https://github.com/uc-vision/smart-tree) | neural medial-axis skeleton + mesh | Python | MIT | 2026-01-13 | 56 | Likely (unverified) | Yes |
| [pc-skeletor](https://github.com/meyerls/pc-skeletor) | skeleton graph | Python | MIT | 2025-04-01 | 233 | No | Yes |
| [SimpleForest](https://github.com/SimpleForest/pluginSimpleForest) | QSM (via Computree) | C++ | GPL-3.0 | 2019-07-19 (mirror) | 3 | No | No |
| [rTwig](https://github.com/aidanmorales/rTwig) | corrects QSM branch radii | R/C++ | CRAN GPL≥3 | 2026-10-16 | 23 | No | via rpy2 |
| [aRchi](https://github.com/umr-amap/aRchi) | tree architecture from QSM | R | CRAN CeCILL | 2025-12-12 | 41 | No | via rpy2 |
| [ITSMe](https://github.com/lmterryn/ITSMe) | branch angle/radius/length metrics from QSMs | R | none detected | 2026-06-05 | 61 | No | via rpy2 |
| [LeWoS](https://github.com/dwang520/LeWoS) | wood/leaf labels (unsupervised) | MATLAB | MIT | 2026-08-06 | 64 | No | No |
| [PointsToWood](https://github.com/harryjfowen/PointsToWood) | wood/leaf labels (deep learning, TLS) | Python | **AGPL-3.0** | 2026-05-30 | 35 | Yes (DL) | Yes |
| [ForAINet](https://github.com/prs-eth/ForAINet) | ground / low veg / stem / live branch / dead branch + instances | Python | NOASSERTION | 2026-07-07 | 119 | Yes | Yes |
| [ForestFormer3D](https://github.com/SmartForest-no/ForestFormer3D) | semantic + instance (FOR-instanceV2) | Python | NOASSERTION | 2026-03-03 | 135 | Yes | Yes |
| [SegmentAnyTree](https://github.com/SmartForest-no/SegmentAnyTree) | instances (no branch class) | Python | MIT | 2025-08-22 | 173 | Yes | Yes |
| [TreeLearn](https://github.com/ecker-lab/TreeLearn) | instances (no branch class) | Python | MIT | 2026-02-17 | 160 | Yes | Yes |
| [FSCT](https://github.com/SKrisanski/FSCT) | terrain / vegetation / CWD / stem (no branch class) | Python | GPL-3.0 | 2024-03-03 | 185 | Recommended | Yes |
| [TLS2trees](https://github.com/tls-tools-ucl/TLS2trees) | ground / wood / leaf / cwd + instances | Python | GPL-3.0 | 2025-01-10 | 35 | Optional | Yes |
| [3D Forest](https://github.com/VUKOZ-OEL/3d-forest) | desktop app (branch features unverified) | C++ | GPL-3.0 | 2026-10-31 | 98 | No | No |
| [treeiso](https://github.com/truebelief/artemis_treeiso) | tree isolation (CloudCompare plugin) | Python/C++ | NOASSERTION | 2025-03-21 | 70 | No | Yes |
| [pointtree](https://github.com/ai4trees/pointtree) | instances | Python | MIT | 2026-10-15 | 36 | Optional | Yes (pip) |

Further findings:
- **TLSeparation:** the GitHub API returned "Not Found" for both `TLSeparation/tlseparation` and `mattbv/tlseparation`. The repo may have moved or been removed; this is **unverified**. — [GitHub API](https://api.github.com/repos/TLSeparation/tlseparation)
- **FOR-instance classes** (used to train ForAINet/ForestFormer3D):
  - Classes are stem (4), live branches (5) and woody branches (6), plus unclassified, low vegetation, terrain and out-points.
  - Live branches are 63.8% of annotated points; woody (needle-less) branches 4.66%.
  - Separating woody from live branches "was hardest for coniferous trees".
  - Some collections (e.g. NIBIO) annotated parts of live branches as stem near tree tops.
  - Source: [Puliti et al. 2023, arXiv 2309.01279](https://arxiv.org/pdf/2309.01279)
- **ForAINet:** semantic classes are ground, low vegetation, stems, live branches and dead branches. It reports >85% F-score for individual trees and >73% mean IoU across the five classes on FOR-instance (ULS). — [Xiang et al., arXiv 2312.15084](https://arxiv.org/html/2312.15084v2)
- **ITSMe:** computes "branch angle-, radius- and length-related metrics" from QSMs (per a search summary of the ITSMe docs). — [ITSMe GitHub](https://github.com/lmterryn/ITSMe)

### Inferences
- **Licence:** the GPL-3 tools (TreeQSM, PyTLidar, AdTree, FSCT, TLS2trees, 3D Forest, SimpleForest) should be run as **external processes** (subprocess or Docker) to produce benchmark outputs, not imported into PS_LiDAR if PS_LiDAR will be distributed under a non-GPL licence. PointsToWood is **AGPL-3.0**, which is stricter (it also covers network use). MIT tools (smart-tree, pc-skeletor, SegmentAnyTree, TreeLearn, pointtree, LeWoS) can be vendored. ForAINet, ForestFormer3D, treeiso and RayExtract have custom or unclear licences that need reading before reuse. *(Licence interpretation, not legal advice.)*
- **CPU feasibility:** QSM tools (PyTLidar, TreeQSM, AdTree, RayExtract) and LeWoS run on CPU. ForAINet, ForestFormer3D, PointsToWood and smart-tree are GPU deep-learning stacks.
- **Branch-class semantic segmentation is a poor fit for PlotSafe branch codes.** FOR-instance's "live branch" class covers foliage plus branch wood and gives no diameters. It helps only as a mask to remove crown points before stem/branch geometry.
- **Recommended integration order:**
  1. Native Hollaus-style shell/unwrap branch detector (no dependencies).
  2. PyTLidar QSM as a GPL-isolated baseline for per-branch diameter and fork candidates.
  3. LeWoS or PointsToWood wood/leaf masks if foliage biases branch extents.

### Gaps
- PyTLidar's actual export of branch tables and CPU runtime on plantation plots are untested.
- The NOASSERTION licence texts (ForAINet, ForestFormer3D, treeiso, raycloudtools) were not read.
- The sensors on which each branch tool was validated are mostly TLS (RIEGL VZ-400/400i in the destructive studies above). MLS/Hovermap branch validation was not found.
- FSCT's branch handling was taken from the prior note (README classes); no FSCT "branch" class exists in its outputs per that README. — [FSCT README](https://github.com/SKrisanski/FSCT)

## Q3. Which datasets could validate a branch-size classifier for conifers?

### Takeaway
**No open dataset pairs conifer LiDAR with branch-diameter or knot labels at the per-branch level.**
- **Closest by sensor and labels:** FOR-instance / FOR-instanceV2. These have conifer ULS/MLS with stem / live-branch / woody-branch point labels but **no branch diameters**.
- **Closest by reference quality:** the destructive QSM studies (Lau 2018 tropical; Demol 2022 ash) and the Scots pine TLS-versus-X-ray studies (Pyörälä 2018). Their data are not confirmed open, and none is a plantation pine.
- **BranchPoseNet (NIBIO):** has destructively sampled whorl positions, but release is unconfirmed.
- **BioDiv-3DTrees (Sci. Data 2025):** offers 3,386 QSMs, but only for broadleaves.

Practically, PS_LiDAR will need its own reference data: field calipering of the largest branch per section on felled or climbed trees, or harvester/sawmill knot data.

### Cited Findings
- **FOR-instance:** a manually annotated UAV laser scanning benchmark for semantic and instance segmentation, including conifer plots, with stem / live branch / woody branch classes. — [arXiv 2309.01279](https://arxiv.org/pdf/2309.01279); [Zenodo 8287792](https://zenodo.org/record/8287792)
  - FOR-instanceV2 adds MLS and tropical data, with about 8× the instances. — [ForestFormer3D, arXiv 2506.16991](https://arxiv.org/abs/2506.16991v2)
- **Point2Tree (arXiv 2305.02651):** a coniferous MLS dataset with manual instance labels and semantic labels for ground, vegetation (branches, leaves, low vegetation), CWD and stems. It has no branch diameters or QSMs. — [arXiv 2305.02651](https://arxiv.org/pdf/2305.02651)
- **BioDiv-3DTrees (Scientific Data, 2025):** 4,952 single-tree TLS/ULS point clouds of 19 species, plus 3,386 QSMs and tree graphs for broadleaves of 14 species. — [PMC12712040](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12712040/)
- **WSL "Individual tree TLS point clouds for tree volume estimation" (EnviDat):** individual-tree TLS clouds for volume estimation. Species composition and branch references were not checked. — [EnviDat](https://envidat.ch/dataset/c82b9485-0b40-41f4-a798-4e4e725cafdc)
- **Lau et al. 2018** (destructive, 279 branches >10 cm, 9 tropical trees): per-class errors are in the electronic supplementary material. Data availability was not confirmed. — [ESM](https://media.springernature.com/original/springer-static/esm/art%3A10.1007%2Fs00468-018-1704-1/MediaObjects/468_2018_1704_MOESM3_ESM.pdf)
- **Demol et al. 2022** (destructive, 2 ash trees, branches down to 3.3 mm calipered): no data repository is stated in the text. — [Silva Fennica 10550](https://www.silvafennica.fi/article/10550)
- **Pyörälä et al. 2018** (Scots pine, TLS versus X-ray knot data, 162 log sections; 158 trees for whorl detection): data availability was not confirmed. — [UEF eRepo](https://erepo.uef.fi/handle/123456789/7163?show=full); [UTU portal](https://research.utu.fi/converis/portal/detail/Publication/458277313?lang=en_US)
- **BranchPoseNet** (NIBIO): destructively sampled trees with whorl locations along felled stems. Release of the dataset is unconfirmed. — [arXiv 2409.14755](https://arxiv.org/abs/2409.14755)
- **Finnish boreal open TLS dataset** (Silva Fennica 2024): 55 plots of 32 × 32 m, not described as having knot or CT reference data. — [Silva Fennica 24066](https://silvafennica.fi/pdf/24066)
- **Synthetic data:** HELIOS++ (Heidelberg) simulations are used for leaf/wood training (Esmorís, Weiser et al. 2024; Albert, Weiser et al. 2025). The "Scaling Up Forest Vision with Synthetic Data" paper (2025) notes that branch-level labels "require customised tree assets". No public synthetic dataset with per-branch diameters was confirmed. — [arXiv 2509.11201](https://arxiv.org/pdf/2509.11201); [ISPRS Annals X-G-2025-25](https://isprs-annals.copernicus.org/articles/X-G-2025/25/2025/)
  - smart-tree links a "synthetic-trees" dataset, whose content was not verified. — [smart-tree](https://github.com/uc-vision/smart-tree)

### Inferences
- **Most useful public validation path for conifer branch size:**
  - (a) Use FOR-instance conifer plots to test the *crown-masking* step (stem versus branch points).
  - (b) Use published Scots pine and spruce numbers (whorl detection about 70%, max branch diameter RMSE about 1.4 cm) as **external accuracy targets** to report against.
  - (c) Generate synthetic pine stems with known whorl and branch diameters (L-system or HELIOS++ with custom assets) to test the 0/4/7/10/15 cm boundaries under occlusion and noise. This is inferred; no ready synthetic pine-branch set exists.
- A small in-house destructive or calipered set (for example 10–20 PHYB trees, largest branch per 1 m section plus fork positions) is unavoidable for a defensible Queensland classifier. HQP harvester/sawmill knot data, if available, would be the scalable alternative.

### Gaps
- **Not verified in this pass (tool budget):**
  - LAUTx (Tockner et al. 2022): labels believed to be instances only.
  - Wytham Woods (Calders et al.): believed to include QSMs for broadleaves.
  - TreeLearn data: instances.
  - FOR-species20K: species labels only.
  - Weiser et al. synthetic tree clouds.
  - None of these is believed to carry branch-diameter labels, but this was not confirmed.
- No open CT/X-ray knot dataset matched to TLS scans was found, for any species.
- The Liang et al. 2018 EuroSDR benchmark (in the stem note) has stem curves but no branch references, as far as known (unverified).
