# Detección de defectos de forma del fuste y rasgos estructurales con LiDAR terrestre/móvil (TLS, MLS, SLAM, UAV bajo dosel)

> Notas de investigación (20-nov-2026). Fuentes web verificadas en esta sesión salvo donde se marca **[NO VERIFICADO]**. Definiciones de códigos tomadas de los PDF locales de Interpine/HQP (solo lectura).
> Referencias locales: `external_references/interpine/PLOTSAFE Overlapping Feature Cruising Forest Inventory Procedures - Feb 2007.pdf` (§4.1.4.3, 4.1.5–4.1.8, pp. 25–32) y `external_references/interpine/HQP Quickcard LiDAR pt 1.pdf`.

## Pregunta 1 — ¿Qué trabajos detectan cada rasgo con LiDAR, con qué método y qué exactitud?

### Takeaway
No existe un trabajo que emita códigos tipo PlotSafe/HQP desde nube de puntos; la literatura útil está fragmentada: (a) mapas de residuos radiales "desenrollados" respecto a un círculo/centro robusto (escáneres de trozas USFS; Kretschmer 2013; Nguyen/Kerautret/Delconte 2016–2022) para protuberancias, cicatrices y nudos; (b) ajuste de sección (círculo/elipse/spline/Fourier) para ovalidad y fluting; (c) QSM/esqueletos para bifurcaciones; (d) clasificadores sobre árbol completo (mayoritariamente ALS/ULS) para árboles muertos y copas rotas. Exactitudes típicas: F1 ≈ 0.74 para defectos superficiales en TLS de alta densidad (frondosas), ~83–93 % de defectos severos detectados en trozas escaneadas, ~70 % de verticilos en pino silvestre con TLS.

### Cited Findings

**Definiciones operativas de los códigos (fuente local, manual PlotSafe Feb 2007 / Quickcard HQP)**
- Fluting: profundidad máxima (fuera de corteza, cm) medida desde una línea recta que puentea la depresión; F5+ (>5 cm) no permitido en grados peeler; F10+ (>10 cm) no permitido en grados sawlog; se debe "bajar" la clase a lo largo de su longitud (F10+ → F5+) — PlotSafe §4.1.4.3, p. 25–27 (archivo local).
- Ovalidad: relación entre el diámetro más corto y el más largo a través del centro; los ejes no son necesariamente perpendiculares; O1.2+ = >1.2× (ej. 500 mm menor y >601 mm mayor) — PlotSafe §4.1.4.3 y Quickcard HQP (archivos locales).
- Butt flare: aumento súbito del diámetro en la base, medido desde el borde del abultamiento hasta la superficie del fuste sobre la conicidad proyectada; B10+ = >10 cm — PlotSafe p. 26; Quickcard HQP.
- Nodal swelling: hinchamiento alrededor de un nudo/verticilo, distancia máxima entre el tope del hinchamiento y la línea paralela a la conicidad general; N5+ (manual) / N10+ (Quickcard HQP) — archivos locales.
- Spike knots: rama con ángulo <30° respecto al fuste; clase por longitud del nudo duro a lo largo del fuste (S7+ 7–10 cm, S10+ 10–16, S16+ 16–25, S25+ ≥25 cm) — PlotSafe p. 26–27.
- Damage (D): daño medido en profundidad, >10 % del diámetro (agujeros, ramas arrancadas, cicatrices de raleo); si hay madera podrida o insectos → R — PlotSafe p. 26–27.
- Rot (R): evidenciada por infestación de insectos y/o hongos; árboles muertos/podridos se codifican R — PlotSafe p. 25; Quickcard HQP.
- Forks: se insertan cuando ≥2 fustes resultantes tienen >100 mm; la altura de bifurcación es la base de la "zona de horqueta" (A), no el punto de separación (B); el software asume una zona de horqueta C de 0.5 m sobre la bifurcación; si es más larga se codifica C explícitamente. Fork en la cima del árbol = basket top. Broken top se usa también si todos los fustes de la bifurcación son <100 mm. Reducciones de diámetro asumen zona ">" de 0.5 m — PlotSafe §4.1.5–4.1.8, pp. 29–32.
- Live windthrown: fuste con inclinación >45° y/o plato radical levantado; árboles muertos y windthrown vivos no requieren descripción de features — Quickcard HQP; PlotSafe p. 28.

**Desenrollado del fuste / mapas de residuos radiales (protuberancias, cicatrices, nudos, daño)**
- Thomas, Shaffer, Mili & Thomas (2007), *Forest Products Journal* 57(4): ajuste robusto de círculos 2-D (Generalized-M estimator) por sección transversal; residuos radiales punto–círculo formando una matriz ángulo × altura ("imagen de distancias radiales"); curvas de nivel en 6 niveles, se retiene el nivel más alto y se filtran regiones por área, relación ancho/largo y anidamiento. Datos: escáner industrial USNR, secciones cada 0.8 pulg (~2 cm), ~1 000 puntos por sección (~0.05 pulg ≈ 1.3 mm entre puntos); frondosas (red oak, yellow-poplar), >160 trozas. Resultado: 63 de 68 defectos severos correctos (92.6 %), 10 falsos positivos. La resolución longitudinal de 0.8 pulg limitó la detección fiable a defectos ≥5 pulg (~12.7 cm) de diámetro y ≥1 pulg (~2.5 cm) de elevación; los defectos sin cambio de altura (p. ej. cicatrices planas) no se detectan — [Thomas et al. 2007 PDF](https://people.cs.vt.edu/~shaffer/Papers/Thomas07.pdf)
- Thomas & Thomas (2011, USFS): algoritmo para defectos ≥0.5 pulg (~1.3 cm) de altura y ≥3 pulg (~7.6 cm) de diámetro en trozas/fustes de frondosas con corteza; en 14 trozas detectó 53/64 defectos severos (~83 %), 9 falsos positivos; 90.2 % del área de defecto localizada — [USFS Treesearch 38044](https://research.fs.usda.gov/treesearch/38044) (cifras vía snippet de búsqueda; **[parcialmente verificado]**).
- Kretschmer, Kirchner, Morhart & Spiecker (2013), *Silva Fennica* 47(5), doi:10.14214/sf.1071: nube TLS convertida a coordenadas cilíndricas {L, z, d} sobre una cadena de cilindros ajustados (d = distancia sobre la superficie del cilindro), interpolación Delaunay → modelo de corteza desenrollado; método por intensidad para cicatrices sin relieve (siguiendo Schütt et al. 2004). Escáner Z+F Imager 5006, "super high" (20 000 px/360°), espaciado 2.4 mm a 7.8 m. Haya (*Fagus sylvatica*). Altura mínima de rasgo detectable ≈1.2 cm; 18 cicatrices en 3 árboles: 58 % de alturas de cicatriz dentro de ±1 cm del manual; diferencias restantes 0–4.85 cm (media 1.1 cm) tras excluir 12 valores >5 cm como falsas identificaciones; el ancho de nudos ramificados tuvo el mayor error. No desarrolla ovalidad ni curvatura — [Silva Fennica 1071](https://silvafennica.fi/article/1071)
- Nguyen, Kerautret, Debled-Rennesson, Colin, Piboule et al. (ICPR 2016) "Segmentation of defects on log surface from terrestrial lidar data": malla reconstruida desde TLS → línea central del tronco (Kerautret et al. 2016, acumulación de votos de normales) → representación local del relieve en cada punto ("delta distance", distancia relativa a la vecindad en lugar de a un cilindro global, porque el tronco no es un cilindro perfecto) → umbral — [HAL hal-02740896](https://hal.inrae.fr/hal-02740896); detalles de umbral y exactitud no verificados.
- Nguyen, Constant & Colin (2021), *Annals of Forest Science*, doi:10.1007/s13595-020-01022-3: segmentación de singularidades + Random Forest para tipo de defecto (ramas, cicatrices de rama, epicórmicos) y eliminación de falsos positivos; estima posición y dimensiones horizontal/vertical. Roble (*Quercus petraea*, corteza rugosa) y haya (corteza lisa), TLS terrestre. F1 medio 0.74; buena concordancia en número de defectos, peor en área; defectos detectados desde ~0.5 cm de diámetro. Datos: doi:10.15454/EOBUM0 — [INRAE/AFS blog](https://ist.blogs.inrae.fr/afs/2021/04/08/an-innovative-and-automated-method-for-characterizing-wood-defects-on-trunk-surfaces-using-high-density-3d-terrestrial-lidar-data/); [artículo Springer](https://link.springer.com/article/10.1007/s13595-020-01022-3)
- Delconte, Ngo, Kerautret, Debled-Rennesson, Nguyen & Constant (2022), IPOL 12:1–26, doi:10.5201/ipol.2022.369: construye un *relief map* (imagen 2-D del relieve del tronco) y segmenta singularidades con CNN (U-Net según la presentación RRPR 2020); código abierto en IPOL — [IPOL 2022/369](https://www.ipol.im/pub/art/2022/369/); [RRPR slides](https://rrpr2020.sciencesconf.org/data/presRRPR2021_Knot.pdf). Métricas no extraídas **[gap]**.

**Sección transversal: ovalidad, fluting, butt flare**
- Witzmann et al. (2022), *Remote Sensing* 14(8):1923: compara ajuste de círculo, elipse y spline a secciones TLS vs discos de referencia; spline fue el más preciso para área de sección (RMSD 3.66 %, sesgo 0.17 %) — [MDPI rs14081923](https://mdpi.com/2072-4292/14/8/1923/html)
- Pfeifer & Winterhalder (ISPRS 2004): B-splines cerrados con ~±1 cm de exactitud de ajuste; derivan ovalidad desde los splines — [ISPRS PFEIFER.pdf](https://www.isprs.org/proceedings/XXXVI/8-W2/PFEIFER.pdf)
- Wang et al. (2017): secciones nunca exactamente circulares; reconstrucción por series de Fourier — citado en [Reconstructing Stem Cross Section Shapes From TLS (UTU)](https://research.utu.fi/converis/portal/detail/Publication/458340302?lang=en_US) **[detalles no verificados]**.
- Bauwens et al. (2016), *Methods in Ecology and Evolution*: perímetro por envolvente convexa (convex hull) a 1–3 m como medida estándar para fustes con contrafuertes; menciona métodos "convex–concave" que modelan el tronco como cilindro + contrafuertes/flautas — [ORBi PDF](https://orbi.uliege.be/bitstream/2268/207146/1/Bauwens_et_al-2016-Methods_in_Ecology_and_Evolution.pdf)
- Con nubes ruidosas o incompletas, la envolvente convexa sobre/subestima el perímetro; propuesta: sectores angulares + GMM para puntos de borde antes del hull — [MDPI rs17162880](https://www.mdpi.com/2072-4292/17/16/2880) (según snippet; año reportado distinto en el resumen de búsqueda) **[parcialmente verificado]**.
- Plataformas móviles suelen ver solo parte de la circunferencia, lo que sesga radio/perímetro de ajustes circulares — [IJAT, Fujipress](https://www.fujipress.jp/ijat/au/ijate002000040331/?full=1)
- No se encontró ningún trabajo que mida "fluting depth" ni "butt flare" con LiDAR en *Pinus radiata* o PHYB.

**Verticilos / nudos / hinchamiento nodal / spike knots**
- Pyörälä et al. (2018), IEEE J-STARS: TLS en 158 pinos silvestres (*Pinus sylvestris*), 6 rodales maduros; detección de verticilos con 69.9 % de exactitud y 1.9 % de falsos positivos; diámetro e ángulo de inserción de ramas evaluados; distancia al escáner, oclusión y viento como factores principales — [UEF erepo](https://erepo.uef.fi/handle/123456789/7163?show=full) (cifras vía snippet).
- arXiv 2409.14755 (2024): modelo de *pose estimation* sobre cortes de la nube para detectar verticilos y ramas a lo largo del fuste; datos TLS, MLS y ULS; especies incluyen *P. sylvestris* y *Pseudotsuga menziesii* — [arXiv 2409.14755](https://arxiv.org/abs/2409.14755) (autores y métricas no verificados **[NO VERIFICADO]**).
- Ángulo de inserción de rama está disponible en QSM (TreeQSM, etc.), lo que permitiría la regla <30° de spike knots — ver hallazgos QSM abajo; no se encontró validación de ángulo de rama en radiata.

**Bifurcaciones (forks), horquetas (crutch), basket tops**
- Los QSM automáticos (p. ej. Raumonen et al. 2013) recuperan la estructura de ramificación y permiten extraer frecuencia de bifurcación y ángulos de ramificación — [MDPI rs5020491](https://www.mdpi.com/2072-4292/5/2/491)
- Comparación de cuatro QSM (TreeQSM, AdQSM, AdTree, SimpleForest) sobre TLS de 191 árboles con propuesta híbrida; no reporta métricas de detección de horquetas en el snippet — vía [búsqueda MDPI](https://www.mdpi.com/1999-4907/16/6/1023) **[no verificado que sea ese artículo]**.
- En segmentación individual, troncos bifurcados cerca de la base se cuentan como dos árboles (error típico) — [Hierarchical min-cut, MDPI rs8050372](https://www.mdpi.com/2072-4292/8/5/372)
- En inventario de plantaciones con MLS, los árboles bifurcados se señalan como problema para la segmentación individual — [ResearchGate 387907719](https://www.researchgate.net/publication/387907719_Single_tree_detection_algorithm_oriented_to_plantation_forest_inventories_using_mobile_laser_scanning_data)
- En *Pinus pinaster* la "forma de fuste" se usa genéricamente incluyendo horquetas — [MDPI Forests 13(3):431](https://mdpi.com/1999-4907/13/3/431)
- No se encontró un trabajo publicado que valide, contra campo, la detección de forks con umbral de diámetro (≥100 mm) desde TLS/MLS en pino de plantación.

**Copas rotas, árboles muertos**
- Yao, Krzystek & Heurich (2012): detección de *snags* (árboles muertos, a menudo sin cima) con ALS full-waveform + SVM; OA 73 % hojas presentes / 71 % sin hojas — [ISPRS Annals I-7/359](https://isprs-annals.copernicus.org/articles/I-7/359/2012/)
- Räty et al. (2024, *Forestry*): árboles rotos por nieve con ALS + RF; F1 0.66 nivel árbol, 0.85 nivel parcela; el CHM superó ligeramente al método basado en nube — vía [jukuri.luke.fi](https://jukuri.luke.fi/server/api/core/bitstreams/9eadba5d-0727-4664-a40c-710641e07e8c/content) (cifras vía snippet).
- Briechle, Krzystek & Vosselman: PointNet++ sobre ULS + multiespectral para pino, abedul, aliso y árboles muertos en pie, OA 90.2 % vs 85.3 % RF con atributos manuales — [ISPRS Annals V-2-2020/203](https://isprs-annals.copernicus.org/articles/V-2-2020/203/2020/)
- Heinaro (2023, tesis): con ULS de alta densidad, los atributos geométricos de retornos discretos no bastaron para identificar árboles muertos en pie de forma fiable — [Dissertationes Forestales 23009](https://dissertationesforestales.fi/pdf/23009)
- No se encontraron estudios TLS/MLS específicos de árboles muertos en pie ni de detección de "broken top" con métricas.

**Inclinación (lean) y windthrow**
- Douglas-fir con MLS: inclinación estimada (i) como desplazamiento horizontal entre centros del fuste a 1.3 m y 18 m y (ii) media de siete ángulos sucesivos; incluir la inclinación mejora el volumen cuando >2°; MLS superior a ALS/UAS en detección de fustes, inferior en ápices — [CJFR 10.1139/cjfr-2020-0484](https://www.sciencegate.app/document/10.1139/cjfr-2020-0484)
- TLS para deslizamientos superficiales: inclinación del fuste como indicador; fustes curvados por movimiento de suelo — [Remote Sensing 12:3744 PDF](https://ubt.opus.hbz-nrw.de/opus45-ubtr/files/1661/remotesensing-12-03744-v2.pdf)
- No se encontró validación de detección de "plato radical levantado" desde LiDAR terrestre.

**Torceduras (crook/kink, relacionado con sweep)**
- Detección de *crook* con CNN entrenada con torceduras sintéticas sobre TLS real; desviación mínima 5 cm según práctica de clasificación de trozas; más comisiones en la parte alta (nube rala) y más omisiones en trozas basales — [Forestry 99(3) cpag026](https://academic.oup.com/forestry/article/99/3/cpag026/8607024)

### Inferences
- El patrón algorítmico común y transferible a PS_LiDAR es: (1) línea central robusta, (2) referencia local (círculo/elipse robusto por rebanada, o superficie suavizada a lo largo del eje), (3) imagen desenrollada r(θ, z) − r_ref, (4) segmentación por umbral/contornos/CNN, (5) clasificación de la región por forma (área, aspecto, altura, posición respecto a verticilos). Thomas 2007 y Kretschmer 2013 son las recetas más explícitas; Nguyen/Delconte la versión moderna con referencia local + RF/CNN.
- Para los códigos HQP, las métricas geométricas se derivan casi directamente:
  - O1.2+: d_max/d_min por rebanada desde elipse o spline/Fourier (no círculo); los ejes del código no exigen perpendicularidad, así que conviene calcular diámetros por pares de puntos opuestos (calipers rotatorios sobre el contorno), no solo los semiejes de la elipse.
  - F5+/F10+: profundidad = distancia máxima entre envolvente convexa del contorno de la rebanada y el contorno real (equivale a la "línea recta que puentea la flauta" del manual).
  - B10+: residuo del perfil de radio en la base respecto a la conicidad proyectada desde más arriba (p. ej. modelo de ahusamiento ajustado entre 1.3 y ~4 m extrapolado hacia abajo).
  - N5+/N10+: residuo positivo localizado del perfil de radio a la altura de verticilos respecto a la conicidad suavizada; requiere detrending del perfil y un detector de picos con anchura acotada.
  - C: zona de horqueta 0.5 m sobre la altura de bifurcación (convención del software, no medición); bifurcación = nodo del esqueleto/QSM con ≥2 hijos de diámetro >100 mm.
  - Lean >45°: ángulo de la línea central respecto a la vertical (trivial con buena centerline).
- Árboles muertos y copas rotas en plantación probablemente se resuelvan mejor con atributos de copa (densidad de puntos/intensidad cerca del ápice, altura vs vecinos, ausencia de follaje) que con el fuste; la evidencia publicada es sobre todo ALS/ULS.

### Gaps
- No obtuve el texto completo de Nguyen et al. 2016/2021 ni Delconte 2022: resolución exacta del relief map (mm/píxel), umbrales y métricas por clase sin verificar.
- El informe FWPA PNC377-1516 (LiDAR denso para evaluación de fustes en *P. radiata*) excedió el límite de descarga; podría contener evaluación de forks/defectos en radiata — [FWPA PNC377](https://fwpa.com.au/wp-content/uploads/2018/10/PNC377-1516_Final_Report_27_Sept18-.pdf) **[pendiente de lectura]**.
- Sin literatura encontrada sobre fluting o butt flare medidos con LiDAR en coníferas de plantación, ni sobre spike knots (<30°) desde nube.
- Sin estudio TLS/MLS que valide detección de broken/basket tops o dead trees con métricas.

## Pregunta 2 — Tamaño mínimo de defecto alcanzable dado el ruido MLS (≈5–20 mm) y la densidad; qué es realista detectar

### Takeaway
Con TLS estático de alta resolución (espaciado ~2–3 mm, ruido ~1 mm) se han detectado relieves de ~0.5–1.2 cm; con escáneres de trozas industriales (~2 cm entre secciones) el mínimo fiable fue ~2.5 cm de altura y ~13 cm de diámetro. El MLS/SLAM tiene ruido radial mayor (sesgo de DBH ≈1 cm reportado para ZEB Horizon) y cobertura angular incompleta, de modo que solo rasgos de escala centimétrica alta (≥3–5 cm de relieve, ≥10 cm de extensión) son realistas: B10+, F10+, O1.2+, forks, lean y probablemente N10+; F5+, N5+, D y spike knots son marginales; R (pudrición interna) no es detectable geométricamente.

### Cited Findings
- Kretschmer et al. 2013 (TLS Z+F 5006, ~2.4 mm de espaciado): altura mínima de rasgo ≈1.2 cm; orientación del escaneo ≈1 mm — [Silva Fennica 1071](https://silvafennica.fi/article/1071)
- Nguyen et al. 2021 (TLS denso, roble/haya): defectos desde ~0.5 cm de diámetro — [INRAE/AFS](https://ist.blogs.inrae.fr/afs/2021/04/08/an-innovative-and-automated-method-for-characterizing-wood-defects-on-trunk-surfaces-using-high-density-3d-terrestrial-lidar-data/)
- Thomas et al. 2007 (escáner de trozas USNR, 0.8 pulg ≈2 cm entre secciones): detección fiable limitada a defectos ≥5 pulg de diámetro y ≥1 pulg de elevación por la resolución longitudinal; defectos sin cambio de altura no detectables — [Thomas07 PDF](https://people.cs.vt.edu/~shaffer/Papers/Thomas07.pdf)
- Thomas & Thomas 2011: umbral de diseño ≥0.5 pulg (1.3 cm) de altura y ≥3 pulg (7.6 cm) de diámetro — [USFS 38044](https://research.fs.usda.gov/treesearch/38044)
- SLAM (GeoSLAM ZEB Horizon, picea/haya): más ruido estocástico que TLS; sesgo negativo consistente de ~1 cm en diámetro por la distribución del ruido sobre la superficie; el ruido son puntos *inliers* dispersos a lo largo de la superficie real, no outliers — [MDPI rs16071261](https://www.mdpi.com/2072-4292/16/7/1261/xml) (vía snippet).
- MLS en plantaciones maduras (Scion): RMSE de diámetro 1.5 cm (3.9 %) con sotobosque denso (dispositivo no confirmado) — [Scion 2020slam](https://scionresearch.com/redirects/2020slam)
- Hovermap ST-X en túnel minero: ~10 mm de exactitud en la mejor configuración (no forestal) — [Acta Montanistica Slovaca 2024](https://actamont.fberg.tuke.sk/pdf/2024/n3/10kremen.pdf)
- Plataformas móviles ven parte de la circunferencia → sesgo sistemático en radio/perímetro — [IJAT](https://www.fujipress.jp/ijat/au/ijate002000040331/?full=1)
- Detección de crook en TLS: más errores de comisión en la parte alta por nube rala; mínimo operativo 5 cm de desviación — [Forestry cpag026](https://academic.oup.com/forestry/article/99/3/cpag026/8607024)
- Pyörälä 2018: distancia al escáner, oclusión y viento degradan la detección de verticilos — [UEF erepo](https://erepo.uef.fi/handle/123456789/7163?show=full)
- La pudrición se diagnostica en campo por insectos/hongos visibles (síntomas externos), no por geometría — PlotSafe §4.1.4.3 (archivo local). Los métodos geométricos de log scanners no detectan defectos sin cambio de altura — [Thomas07](https://people.cs.vt.edu/~shaffer/Papers/Thomas07.pdf); Kretschmer usó intensidad para cicatrices sin relieve — [Silva Fennica 1071](https://silvafennica.fi/article/1071).

### Inferences
- Regla práctica (inferencia, no publicada): un rasgo es detectable si su amplitud radial es ≳3σ del ruido radial efectivo tras promediar dentro de la ventana del rasgo. Con σ_MLS ≈ 5–20 mm por punto, promediar N puntos por celda reduce σ en √N, pero el sesgo de cobertura parcial no se promedia. Para celdas de ~5 cm × 5° con decenas de puntos, rasgos de ≥2–3 cm de relieve son plausibles en la parte baja del fuste; por encima de ~8–10 m (nube rala/ocluida en MLS) el umbral sube.
- Viabilidad por código con MLS (Hovermap/GeoSLAM/mochila), en orden: lean/windthrow (alta), forks ≥100 mm por debajo de la copa viva (alta-media), O1.2+ (media-alta en 0.3–6 m; exige cobertura angular ≥~270° por rebanada), B10+ (alta: es una anomalía de 10 cm de radio), F10+ (media), N10+ (media), F5+/N5+ (baja-media), D >10 % del diámetro (media para cicatrices grandes con relieve; baja si son planas, considerar intensidad), spike knots S7+–S25+ (baja: requiere ramas individuales y ángulo <30°, y los nudos de 7–16 cm están en el límite del ruido), R (no detectable geométricamente; a lo sumo como proxy de árbol muerto/agujeros), broken/basket top y dead (dependen de la copa, mejor con ULS/ALS o atributos de follaje).
- PS_LiDAR usa rebanadas de 0.2 m: esto ya fija la resolución vertical; rasgos de longitud <0.4 m (spike knots, nódulos estrechos) quedan sub-muestreados. Para un mapa desenrollado conviene trabajar con los puntos crudos binned (p. ej. Δz 2–5 cm, Δθ 2–5°) en lugar de los círculos de 0.2 m.

### Gaps
- No encontré un valor publicado de σ radial (mm) sobre corteza para Hovermap o GeoSLAM en pino de plantación; los 5–20 mm del encargo no están respaldados por una fuente específica encontrada aquí.
- Ningún estudio cuantifica la detectabilidad de fluting, nodal swelling o butt flare en función del ruido MLS.

## Pregunta 3 — ¿Hay trabajos que clasifiquen automáticamente rasgos de calidad en códigos de crucero o grados de trozas?

### Takeaway
Hay clasificación automática de defectos superficiales en tipos (RF/CNN) y simulaciones de trozado/grado en aserradero, pero no encontré ningún trabajo que emita códigos de crucero tipo PlotSafe/Interpine/HQP desde nubes de puntos; la integración con grados de troza es el hueco a llenar por PS_LiDAR.

### Cited Findings
- Nguyen et al. 2021: Random Forest asigna tipo de defecto (rama, cicatriz de rama, epicórmico) y elimina falsos positivos, F1 0.74 — [INRAE/AFS](https://ist.blogs.inrae.fr/afs/2021/04/08/an-innovative-and-automated-method-for-characterizing-wood-defects-on-trunk-surfaces-using-high-density-3d-terrestrial-lidar-data/)
- Revisión (MDPI Forests 15(6):943, 2024): el láser es tecnología extendida para características del fuste; ML se ha aplicado a clasificar defectos superficiales (cicatrices de rama, brotes epicórmicos, burls); la sinuosidad (sweep) del árbol en pie no correlacionó bien con la medida en aserradero, aunque dimensiones sí — [MDPI Forests 15/6/943](https://www.mdpi.com/1999-4907/15/6/943)
- Wang, Thomas et al. (2018): combinar ensayo acústico + escaneo láser en 21 trozas de yellow-poplar mejora la predicción de calidad y rendimiento de grado de tabla frente a cada método solo — [USFS 56761 PDF](https://research.fs.usda.gov/download/treesearch/56761.pdf)
- Picea abies (Annals of Forest Science 2026): en 479 árboles los atributos láser a nivel de troza explicaron ~25 % de la variación de calidad; 73 % de la variabilidad era intra-árbol; los atributos externos no representan de forma robusta la mayoría de factores de calidad — [Springer s13595-026-01326-w](https://link.springer.com/article/10.1186/s13595-026-01326-w)
- PlotSafe indica la consecuencia de cada código en grados de troza (p. ej. D → solo pulp/waste; F10+ no permitido en sawlog; O1.2+ y F5+ no permitidos en peeler; C → pulp/waste; R → waste) — PlotSafe p. 27 (archivo local), lo cual da la tabla de mapeo código→restricción.
- Detección automática de crook con CNN sobre TLS usando umbral de desviación de práctica de grading (5 cm) — [Forestry cpag026](https://academic.oup.com/forestry/article/99/3/cpag026/8607024)

### Inferences
- Para PS_LiDAR, la vía más defendible es basada en reglas geométricas con los umbrales del manual (B10+, F5+/F10+, O1.2+, N5+/N10+, C=0.5 m sobre fork, lean>45°), aplicadas sobre ventanas de altura y con la regla de "upgrade" (>3 m entre zonas) del Quickcard, y dejar ML (RF sobre atributos del parche) para separar tipos de anomalías (nudo/verticilo vs cicatriz vs ruido), al estilo Nguyen 2021.
- La validación deberá ser contra los datos de crucero PlotSafe del cliente (`PlotsSafe_Example_Data`), sabiendo que la literatura sugiere concordancia modesta para atributos de calidad (≈25 % de varianza explicada en picea; sweep mal predicho).

### Gaps
- No encontré trabajos de Scion/Interpine/FWPA que conviertan nubes TLS/MLS en códigos de crucero; tampoco detalles públicos de algoritmos comerciales (Microtec, USNR) más allá de Thomas et al.
- No verifiqué si el informe FWPA PNC377 o trabajos de Interpine con LiDAR (p. ej. "remotesensing-13-01413.pdf" en la carpeta local) incluyen clasificación de features; revisar esos documentos localmente es el siguiente paso.
