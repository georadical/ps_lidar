# Detección de ramas, verticilos y tamaño de nudo con LiDAR terrestre/móvil (TLS, MLS, HHLS, ULS bajo dosel)

> Notas de investigación (nov-2026) para estimar el código de ramas (Br) por sección de fuste en PS_LiDAR (HQP/PHYB, Queensland).
> Convención: cada dato lleva su fuente. Lo que no pude verificar con texto primario va en "Gaps". Varias cifras provienen solo de resúmenes (abstracts) porque los textos completos devolvieron 403; se indica cuando es así.

## 1. ¿Qué trabajos detectan ramas y miden diámetro, altura de inserción, ángulo y verticilos en coníferas con TLS/MLS?

### Takeaway
El núcleo de evidencia en coníferas viene de Finlandia (Pyörälä, Yrttimaa, Kankare, Holopainen, Vastaranta: pino silvestre con TLS; Winberg/Pyörälä: abeto rojo con HHLS ZEB Horizon) y de NIBIO (Puliti: detección de verticilos con deep learning). No encontré ningún trabajo revisado por pares sobre *P. elliottii*, *P. caribaea* ni PHYB; para *P. radiata* solo hay una tesis de máster (Canterbury/Scion) y una presentación de Scion, ambas con resultados pobres en árboles jóvenes.

### Cited Findings
**Pino silvestre (*Pinus sylvestris*), TLS — Pyörälä et al. 2018, IEEE JSTARS 11(10):3598–3607 (DOI 10.1109/JSTARS.2018.2819598)**
- 158 pinos en seis rodales maduros; se analizaron secciones de troza con diámetro de fuste >15 cm. Detección cuantitativa de verticilos: exactitud 69,9 % con 1,9 % de falsos positivos — [research.utu.fi](https://research.utu.fi/converis/portal/detail/Publication/458277313?lang=en_US)
- El diámetro máximo de rama por verticilo y su ángulo de inserción se subestimaron en 0,34 cm (11,1 %) y 0,67° (1,0 %) respectivamente (RMSE mayores, no recuperados) — [research.utu.fi](https://research.utu.fi/converis/portal/detail/Publication/458277313?lang=en_US) (vía resumen)
- Factores externos principales: distancia al escáner, oclusión y viento; los autores advierten que la completitud y densidad de puntos deben atenderse antes de usar modelos TLS para parámetros de rama — [research.utu.fi](https://research.utu.fi/converis/portal/detail/Publication/458277313?lang=en_US)

**Pino silvestre, TLS — Pyörälä et al. 2018, Canadian Journal of Remote Sensing 44(5):462–475**
- La exactitud de detección de ramas cae significativamente por encima de la base de copa viva, principalmente por la mayor distancia al escáner y no tanto por oclusión del follaje; el diámetro de rama se subestimó en general — [DOAJ](https://doaj.org/article/4d584b7c2f1e491fa804d49ffc152ab0)

**Pino silvestre, TLS vs rayos X de aserradero — Pyörälä et al. 2018, Scandinavian Journal of Forest Research**
- 162 secciones de troza: TLS detectó en promedio ~23 verticilos por árbol frente a ~37 en rayos X; el diámetro máximo de nudo por rayos X no difirió significativamente del diámetro máximo de rama medido en la nube TLS — [research.utu.fi](https://research.utu.fi/converis/portal/detail/Publication/458276467?lang=en_US) (vía resumen)

**Pino silvestre, TLS — Yrttimaa et al. 2024, Forestry 97(4):531 (online dic-2023, DOI 10.1093/forestry/cpad062)**
- Escáner Riegl VZ-400i (1550 nm), 23 posiciones multi-scan, rodal boreal sur gestionado (Evo, Finlandia), 0,4 ha. Método: transformación a coordenadas cilíndricas, filtro morfológico (SMRF) para separar superficie del fuste de ramas, origen de rama = intersección rama-superficie del fuste, segmentación por componentes conexas; código MATLAB abierto en GitHub — [Forestry/OUP](https://academic.oup.com/forestry/article/97/4/531/7458501)
- Debajo del umbral de reconstrucción del fuste: recall 75 %, precisión 92 %, F1 0,82; F1 por sección de fuste (de abajo hacia arriba) 0,95 / 0,86 / 0,80 / 0,63; recall 0,50 en la sección más alta — [Forestry/OUP](https://academic.oup.com/forestry/article/97/4/531/7458501)
- Solo identifica ramas hasta donde se reconstruye el fuste: 59,3–81,2 % de la altura relativa en el 90 % de los árboles. Validación por interpretación visual de 364 secciones de 100 árboles. Los autores declaran explícitamente que "no branch diameters were measured in this study" — [Forestry/OUP](https://academic.oup.com/forestry/article/97/4/531/7458501)

**Abeto rojo (*Picea abies*), HHLS GeoSLAM ZEB Horizon — Winberg et al. 2023, ISPRS Open J. Photogramm. Remote Sens. 9:100040**
- 457 árboles de 13 rodales (SE de Finlandia), rastreados hasta aserradero con rayos X. Dos métodos: DBSCAN propio y QSM basado en segmentación — [DOAJ](https://doaj.org/article/2c3abf4218c24307b7a3180e25970d01)
- Un método sobreestimó el número de verticilos en 19 % frente a rayos X (probablemente por nudos pequeños que el rayo X no registra); DBSCAN solo detectó ramas grandes, con sesgo −11 % en número de verticilos — [búsqueda, resumen de SLU](https://pub.epsilon.slu.se/id/document/20429569)
- Variables de nudosidad medidas en aserradero predichas con R² 0,47–0,64; comparable a resultados previos con TLS estático — [DOAJ](https://doaj.org/article/2c3abf4218c24307b7a3180e25970d01)
- Pyörälä et al. 2026 (Annals of Forest Science 83:9) afirman que treeQSM y el método DBSCAN no capturaron completamente la estructura de ramificación del abeto en datos MLS frente a referencias internas de aserradero — [SLU](https://pub.epsilon.slu.se/id/document/20461965)

**Coníferas, CNN sobre imágenes de TLS — grupo Holopainen (Finlandia)**
- Detectores de objetos CNN para segmentar verticilos y secciones de troza a partir de imágenes derivadas de TLS: RMSE del conteo de verticilos 7,73 (64,41 %); los autores señalan que inferir la estructura interna de nudos a partir de verticilos externos es un reto — [búsqueda; arXiv 2312.15084](https://arxiv.org/pdf/2312.15084v2) (cifra vista en snippet; no verifiqué el texto completo)

**Pino radiata (*P. radiata*), TLS + escáner personal (PLS) — tesis de máster, Univ. of Canterbury / Scion**
- Ensayo de mejoramiento de 9 años en Kaingaroa (3 ha), 5 árboles con mapeo de copa como verdad de campo — [UC Research Repository](https://ir.canterbury.ac.nz/items/5013fbd0-488b-4fca-91b0-e31e52e5fe2d)
- Enfoque QSM: diámetro de rama R² = 0,82, RMSE = 20,2 %, pero detección de ramas muy baja (exactitud 0,07; F1 0,13); prácticamente sin correlación entre ángulos de rama medidos en campo y por láser; conclusión del autor: ningún escáner evaluado fue adecuado para caracterizar la estructura de ramas de radiata joven con QSM — [UC Research Repository](https://ir.canterbury.ac.nz/items/5013fbd0-488b-4fca-91b0-e31e52e5fe2d) (cifras vía snippet; la página dio 403)
- Presentación de Scion: PLS para diámetro, ángulo y patrón de ramas contra mediciones manuales en dos ensayos de radiata (5 y 8 años); el resumen no da cifras — [Scion/eventsair](https://airdrive.eventsair.com/eventsairseasiaprod/production-tcc-public/7db5f800cce24735918283a8c99433d9)
- Hartley et al. 2022 (Remote Sensing 14:3344): MLS de mochila en radiata madura, orientado a métricas de fuste/forma, no de rama — [MDPI](https://www.mdpi.com/2072-4292/14/14/3344)

### Inferences
- La familia de métodos con mejor respaldo para coníferas es "geométrica sobre el fuste": modelar la superficie del fuste (cilindro/círculo por rebanada) y tratar los puntos fuera de ella como ramas, localizando el origen de cada rama en la intersección con el fuste. PS_LiDAR ya tiene rebanadas de 0,2 m con círculos/elipses y línea central, que es justamente el insumo de Yrttimaa et al. 2024 y Pyörälä et al. 2018.
- Para el código Br solo hace falta el diámetro máximo por sección, no la topología completa, lo que favorece métodos de "residuo radial alrededor del fuste" por sobre QSM completos.

### Gaps
- No encontré estudios de detección de ramas con LiDAR en *P. elliottii*, *P. caribaea* ni sus híbridos, ni trabajo australiano (CSIRO RayCloudTools `rayextract`, Univ. of the Sunshine Coast) con validación de diámetro de rama; `rayextract` no apareció en las búsquedas.
- No obtuve los RMSE completos de Pyörälä 2018 (JSTARS, CJRS) ni la densidad de puntos usada; los textos completos dieron 403.
- No verifiqué la tesis de Canterbury en texto completo (autor/año exactos sin confirmar).
- No busqué específicamente Kankare (QSM pino), Hackenberg/SimpleForest en coníferas ni AdQSM con validación de rama en coníferas; no hallé ningún estudio de validación destructiva de diámetro de rama en coníferas con QSM.

## 2. Exactitud reportada del diámetro de rama, tasa de detección por tamaño, diámetro mínimo detectable y degradación con altura y oclusión

### Takeaway
El diámetro máximo de rama por verticilo en pino silvestre con TLS se subestima ~11 % (0,34 cm); en ramas pequeñas los QSM tienden a sobreestimar el diámetro (hasta +40 %) y a perder ramas. La detección se degrada con la altura: F1 de 0,95 en la sección basal a 0,63 en la más alta, y por encima de la base de copa viva la distancia al escáner es el factor dominante. No hay una cifra publicada robusta de "diámetro mínimo detectable" para MLS en coníferas.

### Cited Findings
- Pino silvestre TLS: diámetro máximo de rama subestimado 0,34 cm (11,1 %); ángulo de inserción subestimado 0,67° (1,0 %) — [research.utu.fi](https://research.utu.fi/converis/portal/detail/Publication/458277313?lang=en_US)
- Pino silvestre TLS: detección peor sobre la base de copa viva, sobre todo por distancia al escáner; diámetro generalmente subestimado — [DOAJ/CJRS](https://doaj.org/article/4d584b7c2f1e491fa804d49ffc152ab0)
- Pino silvestre TLS (Riegl VZ-400i): F1 por sección 0,95 → 0,86 → 0,80 → 0,63 de abajo hacia arriba; techo de detección en 59–81 % de la altura relativa — [Forestry/OUP](https://academic.oup.com/forestry/article/97/4/531/7458501)
- Radiata joven, TLS/PLS + QSM: diámetro R² 0,82 / RMSE 20,2 %, pero F1 de detección 0,13 — [UC Research Repository](https://ir.canterbury.ac.nz/items/5013fbd0-488b-4fca-91b0-e31e52e5fe2d)
- Fresno (*Fraxinus excelsior*, latifoliada, no conífera), TLS + QSM vs 265 mediciones manuales en 2 árboles: la exactitud del diámetro empeora con ramas más pequeñas; las ramas <5 cm aportan 80 % y 83 % de la sobreestimación de volumen leñoso; los autores la atribuyen al escáner (huella del haz) y al co-registro más que a la parametrización del QSM; señalan que no hay validación destructiva de diámetro de ramas <10 cm en campo — [Silva Fennica 10550](https://www.silvafennica.fi/article/10550)
- Árboles tropicales (Lau et al. 2018, 9 árboles cosechados): TreeQSM asignó correctamente el orden de ramificación en 99 % de casos, reconstruyó 87 % de la longitud de ramas y 97 % del volumen; validación de diámetro solo para ramas >10 cm — [CIFOR-ICRAF PDF](https://cifor-icraf.org/publications/pdf_files/articles/ALau1802.pdf)
- En un estudio tropical citado por Silva Fennica, en árboles de 10–20 cm de diámetro de fuste se reconstruyó menos de la mitad de las ramas, y en las reconstruidas el diámetro se sobreestimó 40 % en promedio; otro estudio halló volumen de ramas <7 cm "severamente sobreestimado", posiblemente por ruido asociado a la distancia de escaneo — [Silva Fennica PDF](https://silvafennica.fi/pdf/10550)
- HHLS ZEB Horizon en abeto: DBSCAN solo detecta ramas grandes (−11 % en verticilos); el otro método sobrecuenta +19 % — [SLU](https://pub.epsilon.slu.se/id/document/20429569)
- ULS en radiata: sigue sufriendo oclusión de dosel pese a mayor densidad; TLS y MLS son mejores para forma detallada del fuste — [MDPI Remote Sensing 14:3344](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-03344/article_deploy/remotesensing-14-03344.pdf)
- Pyörälä et al. 2026: en 479 abetos (HHLS + ALS), >70 % de la variabilidad de calidad de madera ocurre dentro del árbol; Random Forest con métricas láser capturó 25 % de esa variabilidad con RMSE medio 39,9 % — [SLU](https://pub.epsilon.slu.se/id/document/20461965)

### Inferences
- Hay dos sesgos de signo opuesto: el máximo por verticilo tiende a subestimarse (ramas incompletas, oclusión), mientras que el diámetro de ramas pequeñas por ajuste de cilindros tiende a sobreestimarse (ruido y huella del haz del orden de cm). Para la frontera de códigos HQP 4/7 (≤4 vs ≤7 cm) el ruido de MLS es del mismo orden que el intervalo de clase, así que conviene calibrar con datos propios.
- Con MLS SLAM (Hovermap, ZEB) el ruido de rango típico (~1–3 cm, no verificado aquí) hace poco confiable ajustar cilindros a ramas <3–4 cm; la tarea realista es distinguir ≤4 / 4–7 / 7–10 / >10 cm en el tercio inferior y medio del fuste, y marcar "sin dato" más arriba.
- Como la mayoría de las secciones HQP son código 4, un clasificador conservador con prior fuerte hacia 4 y detección de ramas gruesas (>7 cm) como anomalía probablemente rinde más que una estimación de diámetro continua.

### Gaps
- No encontré una cifra publicada de diámetro mínimo detectable por densidad de puntos para MLS/HHLS en coníferas.
- Faltan los RMSE absolutos del diámetro de rama en pino silvestre (solo tengo sesgo).
- No hay validación destructiva conífera de diámetro de rama con QSM en las fuentes encontradas.
- No encontré resultados de Hovermap ni de UAV bajo dosel para ramas.

## 3. Predicción del tamaño de nudo: externo (o métricas LiDAR) → nudo interno / calidad de madera aserrada

### Takeaway
En pino silvestre, el diámetro máximo de rama medido con TLS no difiere significativamente del diámetro máximo de nudo por rayos X, y existe un modelo de conicidad de nudo calibrado con atributos TLS. En abeto, métricas HHLS predicen nudosidad de aserradero con R² 0,47–0,64. Para radiata existen modelos de crecimiento de ramas (TreeBLOSSIM, Grace et al.) pero no encontré un vínculo CT↔LiDAR.

### Cited Findings
- Pino silvestre: diámetro máximo de nudo (rayos X) vs diámetro máximo de rama (TLS) sin diferencia estadísticamente significativa; TLS detecta menos verticilos (~23 vs ~37) — [research.utu.fi](https://research.utu.fi/converis/portal/detail/Publication/458276467?lang=en_US)
- Pino silvestre: ecuación de conicidad de nudo (knot taper) calibrada con atributos de rama derivados de TLS, para simulación de aserrado; motivación: la falta de estructura interna de nudos introduce error en simulaciones previas a la cosecha — [WSE Network](https://wsenetwork.org/modelling-the-knot-taper-in-scots-pine-pinus-sylvestris-l-from-terrestrial-laser-scanning-derived-branch-attributes/)
- Abeto, HHLS: propiedades de nudosidad de aserradero (rayos X) predichas con R² 0,47–0,64 — [DOAJ](https://doaj.org/article/2c3abf4218c24307b7a3180e25970d01)
- Abeto, HHLS + ALS (Pyörälä et al. 2026): RF capturó 25 % de la variabilidad industrial de calidad (RMSE 39,9 %) — [SLU](https://pub.epsilon.slu.se/id/document/20461965)
- Pino Douglas (CT): relación entre tamaño de nudo interno y diámetro de la rama más grande del verticilo; diferencia media externa vs interna ~13,8 mm (el diámetro externo es mayor) — [Springer, Wood Sci. Technol.](https://link.springer.com/article/10.1007/s00226-021-01352-z) (cifra vía snippet)
- Radiata, NZ: modelo de ramas que predice tamaños hasta la copa verde a los 20 años a partir de altura, DAP y área basal, con "branch factor" opcional (rama mayor del primer verticilo sobre 6 m); longitud de entrenudo y número de ramas por verticilo independientes del tamaño del árbol, índice de sitio y densidad — [registro bibliográfico](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_proquest_miscellaneous_18301625)
- Radiata: Grace et al. — modelo de desarrollo de ramas que predice crecimiento del diámetro en el punto de inserción; Branch Index predicho con función empírica de variables de rodal — [Scion NZJFS PDF](https://www.scionresearch.com/__data/assets/pdf_file/0007/59443/NZJFS293_1999_391_408GRACE.pdf)
- Radiata: TreeBLOSSIM predice ubicación y diámetro de ramas junto al fuste anualmente — [Scion PDF](https://www.scionresearch.com/__data/assets/pdf_file/0009/59490/NZJFS2821998-GRACE182-194.pdf)
- Radiata (tesis, red neuronal sobre datos sintéticos del Branch Growth Model): predicción de la rama mayor de la mitad inferior del verticilo a partir de la superior, R² 0,72–0,74 — [Lincoln Research Archive](https://researcharchive.lincoln.ac.nz/entities/publication/87079587-a005-45a8-a0e4-7b3aa11f2d3e)

### Inferences
- El código Br de HQP se mide externamente (diámetro de rama sobre corteza), igual que lo que mide el LiDAR; no hace falta pasar por el nudo interno para replicar el código de campo. El vínculo con nudo interno solo importa si se quiere ir a grado de troza.
- Los modelos tipo "Branch Index" de NZ sugieren usar variables de árbol (DAP, altura, espaciamiento) como prior del diámetro máximo cuando el LiDAR no ve la sección (copa alta).

### Gaps
- No encontré estudios CT o de aserradero que vinculen LiDAR con nudos en radiata, slash, caribaea o PHYB, ni estudios australianos (QDAF/HQP) de este tipo.
- No verifiqué el texto del estudio Douglas-fir ni la cifra de 13,8 mm.

## 4. Métodos que mapean a "diámetro máximo de rama por sección" y predicción de clases categóricas / grados de troza

### Takeaway
No encontré ningún trabajo que prediga directamente códigos categóricos de rama por sección (tipo HQP Br o PlotSafe RAD05) desde LiDAR. Lo más cercano: diámetro máximo por verticilo (Pyörälä 2018), segmentación de ramas por sección de fuste (Yrttimaa 2024), clasificación de secciones de troza con CNN (Holopainen) y regresión de calidad de aserradero (Winberg 2023, Pyörälä 2026).

### Cited Findings
- Diámetro máximo de rama por verticilo y su ángulo, restringido a secciones de fuste >15 cm de diámetro — [research.utu.fi](https://research.utu.fi/converis/portal/detail/Publication/458277313?lang=en_US)
- Segmentación de ramas por sección de fuste con F1 por sección (0,95 a 0,63), método abierto en MATLAB — [Forestry/OUP](https://academic.oup.com/forestry/article/97/4/531/7458501)
- CNN que detecta verticilos y "sawlog sections" en imágenes de TLS — [arXiv 2312.15084](https://arxiv.org/pdf/2312.15084v2)
- BranchPoseNet deriva ángulo de inserción (ángulo de apertura) y longitud máxima de rama por verticilo; según sus autores, el ángulo de inserción informa sobre cómo aparecerá el nudo en los productos — [arXiv 2409.14755](https://arxiv.org/pdf/2409.14755)

### Inferences
- Propuesta de mapeo para PS_LiDAR (inferencia, no validada en la literatura): por cada sección de fuste, (a) tomar los puntos fuera del cilindro/elipse ajustado en la rebanada de 0,2 m (residuo radial > r + margen); (b) agrupar por ángulo azimutal y altura para obtener candidatos de rama; (c) estimar diámetro de cada candidato con un círculo ajustado en un plano perpendicular al eje de la rama a 5–15 cm del fuste (donde se mide en campo, "a ángulo recto del fuste, sobre corteza"); (d) diámetro máximo por sección → umbrales 4/7/10/15 cm → código; (e) si la sección está por encima de la altura de reconstrucción confiable, emitir "sin dato" o usar un prior (moda = 4).
- Para spike knots (<30°), el ángulo entre el eje del candidato y el eje del fuste es directamente computable; Pyörälä reporta sesgo de ángulo de solo ~1 % en pino silvestre TLS, pero la tesis de radiata reporta casi nula correlación de ángulos con PLS en árboles jóvenes: hay que validar.
- Dado el desbalance (casi todo código 4), conviene evaluar con matriz de confusión y F1 por clase, no exactitud global.

### Gaps
- Ningún estudio encontrado predice grados de troza o códigos de rama categóricos desde MLS en pinos subtropicales.
- No encontré trabajos que validen la detección de spike knots con LiDAR.

## 5. Deep learning para segmentación rama/madera (FSCT, ForAINet, ForestFormer3D, SegmentAnyTree, PointsToWood, TreeLearn) — ¿entregan diámetros de rama?

### Takeaway
Los modelos de segmentación semántica/instancias actuales separan madera de follaje e incluso "ramas vivas/muertas", pero ninguno de los revisados entrega diámetros por rama; esa medición siempre requiere un paso geométrico posterior (QSM, cilindros, círculos). BranchPoseNet es el único que entrega propiedades por verticilo (posición, ángulo, longitud), no diámetro.

### Cited Findings
- BranchPoseNet (Puliti, Fischer, Astrup 2024, NIBIO): YOLOv8-pose sobre imágenes de perfiles seccionales de árboles individuales; entrenado con 1.071 imágenes de TLS, MLS y ULS (FOR-instance, FOR-species20K), especies *P. abies* y *P. sylvestris*; precisión 0,69, recall 0,68, F1 0,68 en detección de verticilos (vs 0,70/0,63/0,66 de un modelo local previo de Puliti et al. 2023); ~5,7 s por árbol (16 núcleos + GPU Quadro P6000); entrega ángulo de apertura de rama y longitud máxima de rama; código en https://github.com/stefp/whorl_pose_detector — [arXiv 2409.14755 PDF](https://arxiv.org/pdf/2409.14755)
- BranchPoseNet no reporta diámetro de rama; las métricas de evaluación son F1 de verticilos y RMSE de distancia entre verticilos — [arXiv 2409.14755](https://arxiv.org/abs/2409.14755)
- ForAINet (Xiang et al.): >85 % F-score en árboles individuales y >73 % mIoU en cinco clases (suelo, vegetación baja, fustes, ramas vivas, ramas muertas); sin diámetro de rama — [arXiv 2312.15084](https://arxiv.org/html/2312.15084v2)
- SegmentAnyTreeV2 (Wielgosz, Puliti, Astrup 2026): segmentación semántica + instancias de árbol multi-sensor basada en Transformer; sin diámetros de rama en lo encontrado — [arXiv 2606.08206](https://arxiv.org/html/2606.08206v1)
- PointsToWood: segmentación hoja-madera de TLS en bosques europeos (boreal, templado, mediterráneo, tropical); supera al enfoque PointNet más usado en ramitas y ramas de la copa externa según una métrica ponderada por longitud de camino; no reporta diámetros — [arXiv 2503.04420](https://arxiv.org/pdf/2503.04420)
- TLS2trees (Wilkes et al. 2023, Methods Ecol. Evol.): segmentación de instancias de árbol (10.557 árboles vs 1.281 segmentados a mano), peor en árboles pequeños (DAP ≤10 cm), en parte por la segmentación semántica; no es una herramienta de rama — [DOAJ](https://doaj.org/article/bd799e8ffc524618a468fb72926e3251)

### Inferences
- Un pipeline práctico combinaría un segmentador madera/follaje (p. ej. clase "branch" de ForAINet o PointsToWood) para limpiar acículas con la geometría propia de PS_LiDAR para medir diámetros. Las acículas de pino (fascículos largos de PHYB) generan puntos lineales que pueden confundirse con ramas finas, lo cual es relevante para filtros basados en linealidad.
- BranchPoseNet podría servir para ubicar verticilos y ángulos (spike knots) a bajo costo, y luego medir diámetro solo cerca del fuste en cada verticilo detectado.

### Gaps
- No revisé FSCT, ForestFormer3D ni TreeLearn en esta ronda (presupuesto de búsquedas); por lo encontrado, ninguno está diseñado para diámetros por rama, pero no lo verifiqué.
- No hay modelos entrenados en pinos subtropicales ni con datos Hovermap en lo encontrado.

## Anexo: el módulo existente `src/core/branch_extraction.py` (lectura, sin modificar)

### Takeaway
Es un extractor de máscara de puntos de rama, no un medidor de ramas: sirve como paso de limpieza/candidatos, pero no produce nada que se pueda convertir en código Br sin trabajo adicional.

### Cited Findings
- Qué hace (lectura directa de `/home/jorge/projects/ps_lidar/src/core/branch_extraction.py`): (1) toma puntos no-fuste con z > 0,3 m; (2) calcula linealidad con pgeof (escala 0,1 m) y conserva los de linealidad ≥ 0,5; (3) voxeliza fuste + candidatos a 5 cm y arma un grafo de 26 vecinos; (4) guarda solo componentes conexas que tocan vóxeles de fuste; (5) descarta puntos a más de `max_branch_length` (5 m) del punto de fuste más cercano; (6) devuelve `wood_mask`, `branch_only_mask` y conteos.
- Limitaciones observadas en el código: no hay instancias de rama individuales, ni punto de inserción, ni altura, ni ángulo, ni diámetro; `n_branch_clusters = min(n_vóxeles_únicos, n_componentes)` no es un conteo real de ramas; el bucle Python sobre vóxeles para la adyacencia será lento en nubes grandes; todo lo lineal conectado al fuste (incluidas acículas en fascículos o ramas de árboles vecinos que tocan el fuste) se acepta.

### Inferences
- Punto de partida útil solo para el filtrado de candidatos (linealidad + conectividad al fuste). Para el código Br habría que añadir: instanciación de ramas desde la superficie del fuste (estilo Yrttimaa 2024, usando los círculos de 0,2 m ya existentes), ajuste de diámetro cerca de la inserción, ángulo rama-fuste y agregación del máximo por sección. El filtro de linealidad puede incluso eliminar los tramos basales gruesos de rama (que localmente son más cilíndricos/planos que lineales a 0,1 m de escala), justamente los que definen el código.

### Gaps
- No ejecuté el módulo con datos reales; las observaciones de desempeño son por lectura de código.
