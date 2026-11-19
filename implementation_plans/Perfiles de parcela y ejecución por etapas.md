# Perfiles de parcela y ejecución por etapas

Estado: diseño, sin implementar. Fecha: 2026-11-19.

## 1. Problema

- **Los parámetros viven en la notebook.** Están repartidos en 6 celdas, con unas 47 constantes y unos 160 argumentos. Al procesar otra parcela se pisan, y la configuración con que se procesó una parcela anterior se pierde.
- **Las rutas `J:/` están escritas a mano en cada celda y no coinciden.** La celda 16 escribe en `J:/OUTPUTS/T460298B` y las celdas 19 y 21 en `J:/OUTPUTS/T460298B_diag`. Las salidas se versionan a mano: `tree_inventory06.xlsx`, `centerlines_cilindro_04.laz`.
- **El checkpoint actual (pickle) guarda resultados sin los parámetros que los produjeron.** Incluye `trunk_result`, `section_result`, las líneas centrales, el centro y el radio, pero no la configuración del CSF, del DTM ni del tracking. Además:
  - el resultado de troncos registra la configuración **por defecto**, no la usada;
  - un pickle se rompe cuando cambian las clases;
  - cada corrida pisa la anterior, y la celda 21 carga lo que encuentre sin comprobar que sigue siendo válido.

## 2. Principios

1. **La configuración sale del código.** Cada parcela tiene un archivo de perfil legible.
2. **Tres niveles con prioridad clara:** valores por defecto del método (en el código) < base de rodal o sensor (opcional) < perfil de parcela < ajustes de línea de comando para experimentar.
3. **Cada salida se puede rastrear** hasta su configuración completa, el commit del código y la huella del archivo de entrada.
4. **Solo se recalcula lo que cambió.** Cada etapa guarda su resultado con una clave que depende de sus parámetros, de los de las etapas anteriores y de su código.
5. **Formatos abiertos, sin pickle:** LAZ, `npz` (sin pickle) y CSV/XLSX.
6. **Sin dependencias nuevas.** `tomllib` viene con Python 3.11 o superior. `openpyxl` ya hace falta hoy para el XLSX y **falta en requirements**.
7. **La primera versión reproduce exactamente** los resultados de la notebook (estado aceptado de la Fase 1B). Mejorar resultados viene después.

## 3. Niveles de configuración

| Nivel | Dónde | Qué contiene | Cuándo cambia |
|---|---|---|---|
| Valores por defecto del método | `src/pipeline/defaults.py` | Las ~100 constantes internas, con los valores actuales de la notebook | Solo con un cambio de código, con tests |
| Base de rodal o sensor (opcional) | `bases/<nombre>.toml` | Lo que comparten varias parcelas: rango de DAP, tela del CSF, altura de inicio de la detección | Rara vez |
| Perfil de parcela | `perfiles/<id>.toml` | Entrada, centro, radio, salida y lo que difiera de la base | Una vez por parcela |
| Ajustes de línea de comando | `--set ground.class_threshold=0.3` | Pruebas puntuales | Por corrida; quedan registrados en el manifiesto |

Una base sola se hereda en un único nivel (`base = "..."`), sin cadenas de herencia. Su propósito es que dos parcelas del mismo rodal no diverjan en 60 perillas.

## 4. Esquema del perfil

Ejemplo de perfil para T460298B, con los valores actuales de la notebook:

```toml
# perfiles/T460298B.toml
schema = 1
base = "../bases/hqp_phyb.toml"        # opcional

[plot]
id = "T460298B"
input = "${PS_LIDAR_DATA}/LiDAR QAs/T460298_subsampled_laz1_4.laz"
output_dir = "${PS_LIDAR_OUT}/T460298B"  # opcional; por defecto <workspace>/<id>
center = [-0.311372, -1.461118]          # punto elegido en CloudCompare
center_source = "manual"                 # "manual" | "mat_auto" (sección 14)
radius = 20.0

[stand]
dbh_min = 0.15         # m → r_min de secciones = dbh_min / 2
dbh_max = 0.80         # m → r_max de secciones = dbh_max / 2
height_max = 36.0      # m → altura máxima de secciones

[ground]
cloth_resolution = 1.0
class_threshold = 0.5
rigidness = 1

[detection]
start_height = 3.0     # m → tracking z_min y basal_stripe_z_low
seed = 2026

[methods]
sections = "circle"          # "circle" | "ellipse"
sweep = "global_chord"       # "global_chord" | "windowed"
sweep_dictionary = "HQP"     # "HQP" | "RAD05"

# Opcional: reemplaza cualquier campo de las clases de configuración.
# Las claves desconocidas son un error, no se ignoran.
[advanced.tracking]          # TrackingAssignmentConfig
# basal_stripe_z_high = 15.0
[advanced.ellipse]           # EllipseFitConfig
# ransac_tau_sampson = 0.02
[advanced.stem_cleaning]     # StemCleaningConfig
# min_points_section = 80
```

**Los ~12 parámetros esenciales y su equivalente actual:**

| Clave | Hoy en la notebook | Destino |
|---|---|---|
| `plot.id` | Se deduce del nombre del LAZ en la celda 21 | Filas de `Stem_Description`, nombres de salida |
| `plot.input` | `FILE_PATH` (celda 3) | Etapa de recorte |
| `plot.center`, `plot.radius` | `CENTER_X/Y`, `PLOT_RADIUS` (celda 6) | Recorte; filtro de árboles a `radius − 0.5` |
| `plot.output_dir` | `OUTPUT_DIR` (celdas 16, 19 y 21) | Todas las salidas |
| `stand.dbh_min/max` | `DBH_MIN/MAX` (celda 19) | `StemCleaningConfig.r_min/r_max` |
| `stand.height_max` | `HEIGHT_MAX` | `StemCleaningConfig.maximum_height` |
| `ground.*` | Argumentos de `classify_ground` (celda 12) | Etapa de suelo |
| `detection.start_height` | `z_min = 3.0`, `basal_stripe_z_low = 3.0` | `TrackingAssignmentConfig` |
| `detection.seed` | `TRACKING_RANDOM_SEED`, `ELLIPSE_RANSAC_SEED` | Generadores RANSAC |
| `methods.*` | `USE_ELLIPSE_FIT` y el método de sweep | Elección de función |

**Lo que deja de existir como parámetro:**
- `USE_TRACKING_ASSIGNMENT`: queda solo tracking.
- `STEM_CLEANING_MODE`: se fuerza a `global` con tracking.
- `EXPORT_RICH_FEATURES` y `SAVE_PIPELINE_CHECKPOINT`.
- Los 22 `BASAL_ANCHOR_*` y el resto de la configuración de la ruta vieja.
- Los diagnósticos pasan a ser una opción de la línea de comandos (`--diagnostics`).

**Validación al cargar:**
- tipos y rangos: `radius > 0`, `dbh_min < dbh_max`, `0 < class_threshold`;
- claves desconocidas;
- que el archivo de entrada exista;
- rutas relativas al propio perfil;
- expansión de `${VAR}`, para que el mismo perfil funcione en Windows y Linux.

## 5. Etapas

| # | Etapa | Funciones actuales | Parámetros que entran en la clave | Salida | Costo estimado |
|---|---|---|---|---|---|
| 1 | `clip` | `PointCloudLoader`, `clip_circular_plot` | Huella de la entrada, `center`, `radius` | `01_clip/points.laz` (con intensidad y retornos) | Lectura |
| 2 | `ground` | `classify_ground` | `ground.*` | `02_ground/ground_mask.npz` (máscara sobre `points.laz`) | ~15 s |
| 3 | `normalize` | `normalize_heights` | Resolución del DTM | `03_normalize/veg_norm.laz`, `ground_norm.laz` | ~5 s |
| 4 | `trunks` | `assign_trees_by_tracking` y su adaptador | `tracking`, `ellipse`, `start_height`, `seed` | `04_trunks/trunks.npz` (`tree_ids` por punto, `trunk_mask`), `tree_axes.csv`, `tracking_audit.csv` | Minutos (GS.4) |
| 5 | `stems` | `clean_stems`, secciones, `filter_trees`, `build_centerline_from_sections` | `stem_cleaning`, `stand.*`, `methods.sections` | `05_stems/sections.npz`, `stem_mask.npz`, `centerlines.csv`, `filter.csv` | Segundos a minutos |
| 6 | `inventory` | dendrometría, cobertura, descripción del fuste, taper | `methods.sweep*`, `z_step`, `crown_buffer` | `06_inventory/tree_inventory.xlsx` y una copia CSV de cada hoja; `centerlines.laz` | Segundos |
| 7 | `viewer` (opcional) | Export del visor (celda 22) | — | `07_viewer/plot_<id>.json` | Segundos |
| 8 | `potree` (opcional) | PotreeConverter sobre capas coloreadas (sección 13) | Capas elegidas, paleta | `08_potree/<capa>/` (octree de Potree) y `index.html` | Segundos a minutos por capa |

El análisis de normalización (M3) no entra al pipeline: su resultado no se usaba. Si se quiere control de calidad, va como métricas en el `meta.json` de la etapa 3, por ejemplo los residuos del suelo.

## 6. Caché por etapa

- **Clave de una etapa:** `sha256` de cuatro cosas:
  1. el nombre de la etapa;
  2. sus parámetros ya resueltos, como JSON canónico;
  3. la clave de la etapa anterior;
  4. la huella del código, es decir el contenido de los archivos fuente que usa esa etapa.

  Cambiar el umbral del CSF invalida desde `ground` hacia abajo. Cambiar solo el sweep invalida solo `inventory`. Editar `tracking_assignment.py` invalida desde `trunks`.
- **`meta.json` por etapa:** la clave, los parámetros, la clave de la etapa anterior, la huella del código, los tiempos, los conteos (puntos, árboles, secciones válidas) y las advertencias.
- **Al ejecutar:** si la clave guardada coincide, la etapa se reusa; si no, se recalcula y se sobrescribe. `--force <etapa>` fuerza el recálculo y `--from <etapa>` / `--to <etapa>` acotan la corrida.
- **Huella de la entrada:** ruta + tamaño + fecha de modificación + `sha256`. El hash se guarda para no recalcularlo si el archivo no cambió.
- **Coherencia:** las salidas de una etapa se escriben en una carpeta temporal y se renombran al terminar. Una corrida cortada a la mitad no deja un caché corrupto.
- **Disco:** se guarda solo la última salida de cada etapa. El historial guarda manifiestos, no nubes. Para conservar una corrida completa existe `--keep-copy <nombre>`.

## 7. Manifiesto de corrida

Cada corrida escribe `run.json` y una copia en `runs/<fecha>.json` con:

- el perfil completo ya resuelto (valores por defecto + base + parcela + ajustes de línea de comando), y una copia de `profile.toml` tal como estaba;
- el commit de git y si había cambios sin commitear;
- las versiones de numpy, laspy, CSF y pgeof;
- la huella de la entrada;
- por etapa: si se reusó o se calculó, la clave, el tiempo, los conteos y las advertencias.

Con esto `tree_inventory06.xlsx` deja de hacer falta: el nombre de salida es fijo y la historia está en `runs/`.

## 8. Cómo se usa

```bash
# crear el perfil de una parcela nueva (copia la plantilla y valida);
# sin --center detecta el mat central y propone candidatos (sección 14)
python -m src.pipeline init --id T460298B --input "J:/LiDAR QAs/T460298_subsampled_laz1_4.laz" --radius 20 --base bases/hqp_phyb.toml

# elegir otro candidato del mat, o fijar el centro a mano
python -m src.pipeline recenter perfiles/T460298B.toml --candidate 2
python -m src.pipeline recenter perfiles/T460298B.toml --center -0.311372 -1.461118

# correr todo, reusando lo que siga válido
python -m src.pipeline run perfiles/T460298B.toml

# probar un cambio sin editar el perfil (queda en el manifiesto)
python -m src.pipeline run perfiles/T460298B.toml --set ground.class_threshold=0.3

# ver qué está vigente, qué quedó desactualizado y por qué
python -m src.pipeline status perfiles/T460298B.toml

# comparar la configuración de dos parcelas o de dos corridas
python -m src.pipeline diff perfiles/T460298B.toml perfiles/T460299A.toml

# lote
python -m src.pipeline run perfiles/*.toml
```

La notebook queda para inspeccionar:

```python
from src.pipeline import Project
p = Project("perfiles/T460298B.toml")
p.run(to="stems")              # usa el caché
sections = p.load("stems").section_result
```

## 9. Dónde viven los perfiles

`georadical/ps_lidar` es un **repositorio público**, y los perfiles contienen rutas y coordenadas de parcelas de un cliente. Por eso:

- `perfiles/` y `bases/` viven **fuera del repo**, por ejemplo en `J:/PS_LiDAR/perfiles`, o en un repositorio privado aparte;
- el repo trae solo `perfiles/ejemplo.toml`, `bases/ejemplo.toml` y entradas en `.gitignore`;
- `PS_LIDAR_DATA` y `PS_LIDAR_OUT` apuntan a las carpetas de datos en cada máquina.

## 10. Código nuevo

```
src/pipeline/
  __init__.py      Project
  config.py        esquema, carga, herencia, validación, resolución a dataclasses
  defaults.py      valores actuales de la notebook (estado aceptado de la Fase 1B)
  stages.py        una función por etapa, que envuelve src/core sin cambiarlo
  cache.py         claves, huellas, meta.json, escritura atómica
  serialize.py     guardar y cargar TrunkExtractionResult, SectionResult y líneas centrales (npz/CSV)
  cli.py           init, run, status, diff
  __main__.py
```

`src/core` no se toca en esta fase. Estimación: ~900–1.200 líneas más los tests. Dependencia nueva en `requirements/core.txt`: `openpyxl`, que ya hace falta hoy.

## 11. Plan de migración y criterios de aceptación

1. **Configuración.** `config.py` y `defaults.py`, más un `perfiles/T460298B.toml` generado con los valores actuales.
   - **Test:** el perfil resuelto produce dataclasses idénticas campo por campo a las que arma la notebook.
2. **Etapas, caché y serialización.**
   - **Tests de ida y vuelta:** guardar y cargar da objetos idénticos.
   - **Tests de invalidación:** cambiar `ground.*` recalcula de `ground` en adelante; cambiar `methods.sweep` recalcula solo `inventory`.
   - **Test de punta a punta** sobre una nube sintética chica.
3. **Línea de comandos y manifiesto.** Corres `run` sobre T460298B en tu máquina.
   - **Aceptación:** mismo número de árboles, mismas secciones válidas y el XLSX idéntico al de la notebook, salvo redondeo. Se compara con un script `compare_runs`.
4. **Notebook como visor.** Se reemplazan las celdas de parámetros por `Project(...)` y se elimina el checkpoint pickle.
5. Recién ahí, la **fase 0 de simplificación** (borrar la ruta vieja y los módulos muertos). Pasa a ser segura porque el paso 3 da una referencia contra la cual comparar.

## 12. Riesgos y decisiones abiertas

- **Reproducibilidad exacta.** Depende de que el RANSAC consuma el generador en el mismo orden. Con la misma semilla y la misma entrada debería coincidir (verificar en el paso 3).
- **Recálculos de más.** La huella del código recalcula etapas caras aun ante cambios inocuos, como un comentario. Se acepta por seguridad; la alternativa, versionar etapas a mano, se olvida.
- **Disco.** `points.laz` más `veg_norm.laz` por parcela; con 27 M de puntos son cientos de MB. Medir.
- **Decisiones del usuario:**
  1. Dónde guardar los perfiles: carpeta en `J:` o repositorio privado.
  2. Si usar bases de rodal o solo perfiles de parcela.
  3. Si los ~12 parámetros esenciales son los correctos, en particular si `detection.start_height` y `stand.*` cambian por parcela o por rodal.
  4. Si conservar solo la última salida por etapa o todas.

## 13. Opcional: visor Potree

**Para qué.** Revisar los resultados con una calidad visual comparable a la de CloudCompare sin salir de la herramienta. No reemplaza a CloudCompare para editar ni medir; solo cumple la función de visor.

**Por qué Potree.** Es un visor web abierto (WebGL) que tiene lo que da la calidad visual de CloudCompare:
- EDL (*eye-dome lighting*, el sombreado que marca bordes y profundidad);
- niveles de detalle por octree, para navegar millones de puntos con fluidez;
- color por campo y tamaño de punto ajustable;
- navegación orbital y en primera persona.

Ejemplo: https://potree.org/potree/examples/viewer.html

**Cómo se integra:**

1. **Capas.** La etapa `potree` escribe LAZ temporales con el color ya calculado en RGB, por ser el atributo que Potree lee siempre (verificar el soporte de atributos extra en PotreeConverter 2.x si se quiere colorear en el visor):
   - `arboles`: la nube normalizada coloreada por id de árbol, en gris lo que no pertenece a ninguno;
   - `fustes`: los puntos de fuste coloreados por el código de sweep de su zona;
   - `lineas_centrales` y `suelo`.
2. **Conversión.** Cada capa pasa por PotreeConverter, un ejecutable externo cuya ruta se configura una vez por máquina, por ejemplo en `PS_LIDAR_POTREE_CONVERTER`. La caché de la sección 6 evita reconvertir si la entrada no cambió.
3. **Página.** Se genera un `index.html` a partir de la plantilla de ejemplo de Potree, con las capas cargadas y el EDL activado por defecto.
4. **Cómo se ve.** Potree no abre con `file://`; necesita servirse por HTTP. Hay dos formas:
   - `python -m src.pipeline view perfiles/T460298B.toml`, que levanta un servidor local en la carpeta `08_potree/` y abre el navegador;
   - embebido en la interfaz opcional de Streamlit, mediante un `iframe` a ese mismo servidor.

   PotreeDesktop permite además arrastrar las carpetas convertidas, sin servidor.

**Activación en el perfil:**

```toml
[viewer.potree]
enabled = false                       # opcional; apagado por defecto
layers = ["arboles", "fustes", "lineas_centrales"]
```

**Costos y riesgos:**
- Es una dependencia externa: un binario por sistema operativo (Windows y Linux), fuera de pip. Sin él, la etapa avisa y se saltea; el resto del pipeline no depende de ella.
- Disco: un octree por capa, del mismo orden que el LAZ de entrada.
- Mantenimiento de Potree: la versión 1.8 usa un three.js antiguo y hay una nueva generación en desarrollo (verificar el estado actual antes de implementar).
- Licencias de Potree y PotreeConverter: BSD de dos cláusulas, según recuerdo (verificar).

## 14. Captura del centro de la parcela

**Hoy.** El centro se busca a mano en CloudCompare: se ubica el mat reflectivo central, se toma su coordenada con *Point Picking* y se pega en la celda 6. Es el único paso del pipeline que obliga a salir de la herramienta en cada parcela nueva.

**Lo que ya existe.** `src/core/sampling.py` tiene `detect_reflective_mats` y `find_plot_center` (commit `da185f2`, 2026-01-14). La notebook pasó a coordenadas manuales dos días después (`c89efd3`), sin un motivo registrado. Debilidades visibles en el código:
- el umbral de intensidad es un percentil 99,5 sobre **toda** la nube, sin restringirse a la franja del suelo, así que compiten cintas en los árboles, equipos y bordes brillantes;
- el agrupamiento es voraz, con un radio fijo;
- elige el mat más cercano al baricentro de **todos** los mats, lo que supone que hay varios dispuestos simétricamente;
- si no detecta nada devuelve el centroide de la nube con solo un *warning*, es decir, falla en silencio.

**Propuesta: detección semiautomática con confirmación, guardada en el perfil.** Se ejecuta una sola vez, en `init`, y su resultado queda escrito en `plot.center`. Así cada corrida posterior es determinista y no repite la detección.

1. **Franja del suelo sin CSF.** Se toma el piso aproximado como el Z más bajo de cada celda de 1–2 m (percentil 2 para resistir ruido), y la franja de 0 a 0,3 m sobre ese piso. Solo esos puntos compiten.
2. **Candidatos.** Dentro de la franja, alta intensidad, con un umbral relativo al fondo de la propia franja y no a toda la nube. Se agrupan con DBSCAN y se filtran por forma:
   - extensión compatible con el tamaño del mat (parámetro `mat_size`);
   - plano y horizontal: PCA, con la normal cerca de la vertical;
   - número mínimo de puntos.
3. **Puntaje y orden.** Combina el ajuste a la forma, el contraste de intensidad y, si se confirma la hipótesis, la cercanía al origen del escaneo. En T460298B el centro está a ~1,5 m del origen; si el SLAM siempre arranca junto al mat, es un criterio muy fuerte (**verificar** con varias parcelas).
4. **Salida para revisar.**
   - `center_candidates.csv`: posición, puntaje y medidas de cada candidato.
   - `center_check.png`: vista en planta de la franja del suelo coloreada por intensidad, con los candidatos numerados y el círculo de la parcela dibujado sobre el elegido.
   - En el perfil: `center`, `center_source = "mat_auto"` y `center_candidate = 1`.
5. **Corrección.** `recenter --candidate N` elige otro candidato y `recenter --center X Y` fija el centro a mano (`center_source = "manual"`). La coordenada de CloudCompare sigue siendo siempre una opción válida.
6. **Sin candidatos.** El pipeline **se detiene** y pide el centro a mano. Nunca usa el centroide en silencio.

**Imagen de revisión con contexto (`center_check.png`).** Ver solo los puntos de los mats no basta para saber cuál es el central. La imagen muestra:
- **Fondo:** solo la franja del suelo en planta, submuestreada y en gris. Su contorno muestra la extensión de la nube. Troncos, árboles y sotobosque no se dibujan.
- **Encima:** los candidatos en color y numerados.
- **El círculo de la parcela** (radio del perfil) dibujado alrededor del candidato elegido, y otro en línea punteada alrededor de cada uno de los demás. Así se ve cuál deja el círculo centrado en la nube y cuál lo saca del borde.
- **Indicadores por candidato**, en la imagen y en `center_candidates.csv`:
  - distancia al centro de la nube;
  - porcentaje del círculo que cae dentro de la nube;
  - distancia al origen del escaneo.
- **Recuadro ampliado** del mat elegido, con el umbral de intensidad aplicado, para ver su forma y su centro.

**Aprobación manual hasta demostrar el método:**
- `init` deja `center_approved = false` en el perfil, y `run` se niega a correr mientras sea falso.
- `approve perfiles/<id>.toml` acepta el candidato propuesto. `recenter --candidate N` o `recenter --center X Y` lo corrigen y también cuentan como aprobación.
- Cada decisión queda en un registro común a todas las parcelas (`center_reviews.csv`) con: parcela, candidato propuesto, candidato o coordenada elegidos, distancia entre propuesto y elegido, fecha y versión del detector.
- **Paso a automático.** Cuando el registro muestre que el candidato propuesto fue el elegido en todas las parcelas revisadas, sobre una cantidad mínima a definir, por ejemplo 20, se activa `center_auto_approve = true` en la configuración del espacio de trabajo. Desde ahí `init` aprueba solo y la imagen sigue generándose como control.
- **Vuelta atrás.** Si una corrección manual posterior contradice al detector, o cambia la versión del detector, `center_auto_approve` vuelve a falso y se exige aprobación otra vez.

**Interfaz (opcional).** En la app de Streamlit, la revisión y la aprobación pasan a ser la misma vista: un botón para aprobar, y un clic sobre un candidato o sobre cualquier punto para corregir.

**Validación antes de confiar en el modo automático.** Correr la detección sobre las parcelas que ya tienen centro manual y medir la distancia entre ambos. Criterio propuesto: dentro del tamaño del mat (~0,1 m) en todas, o al menos el centro correcto entre los tres primeros candidatos. Si no se cumple, el modo automático queda solo como sugerencia.

**Datos que hacen falta.**
- Medidas, forma y diseño del mat. ¿Es el damero de 0,6 m que supone el código actual?
- Si hay un solo mat por parcela o también marcas en el borde.
- El sensor y su escala de intensidad.
- Si el escaneo arranca siempre junto al mat.
- Por qué se abandonó el detector automático en enero.
