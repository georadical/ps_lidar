# Auditoría de los módulos M3 a M7B

Alcance: Normalisation Analysis (M3), Ground Filtering (M4), Height Normalisation (M5), Trunk Extraction (M7) y Stem Cleaning + Sectioning (M7B), tal como los configura `notebooks/01_playground.ipynb`. Fecha: 2026-11-19.

**Cómo leer las cifras.** No hay nubes reales en este equipo (están en `J:/`). Salvo que diga *(real)* o *(código)*, cada número viene de simulaciones sintéticas con la configuración de la notebook; los guiones están en el scratchpad de la sesión. *(Real)* son salidas guardadas en la notebook de la corrida sobre T460298B. *(Código)* está comprobado leyendo el código. Lo marcado **verificar** no se comprobó.

---

## Conclusión: los cambios de mayor retorno

### Arreglos inmediatos (minutos, sin riesgo)

1. **Módulo 7 falla con numpy 2.** `diagnose_slab_clusters` usa `ndarray.ptp()` (`tracking_assignment.py:1254-1255`), que no existe en numpy 2.5, la versión de este entorno. Con `diagnostic_basal_slabs=True`, como está en la celda, se cae después de GS.6 *(código)*. Cambiar por `np.ptp(...)`.
2. **Secciones rechazadas que se aceptan como válidas.** En `_fit_circle_check` (`trunk_validation.py:826-834`), si el reintento también falla, la función devuelve el ajuste rechazado con R > 0 y estado 1 *(código)*. Todo lo que viene después trata R > 0 como válido: DAP, SED, taper y línea central. En simulación se aceptaron:
   - arcos de 120° (99–100 %);
   - anillos más chicos que `r_min`;
   - una mancha de 3 cm como sección de 6,8 cm;
   - radios de metros.

   Corregirlo es devolver R = 0, como ya hace la ruta de elipse (`ellipse_fitting.py:948-951`), y usar un único criterio de validez en todo el código.

### Los cinco cambios de fondo, por retorno

| # | Cambio | Módulo | Qué mejora (medido en sintético) |
|---|---|---|---|
| 1 | Filtro de continuidad en las secciones y SED desde un taper monótono, junto con el arreglo inmediato 2 | M7B | Error de SED: de +16,2 cm a +0,5 cm. Error p95 de la línea central: de 49,7 mm a 1,0 mm. Amplitud de sweep: de +127 mm a −0,3 mm de error |
| 2 | Refinar el suelo en dos etapas y DTM por TIN en vez de promedio por celda | M4 + M5 | Sesgo de altura en la base del tronco: mediana de −15,7 cm a −0,6 cm; p95 de 21–25 cm a 1,4–1,9 cm |
| 3 | Ajuste de círculo robusto (RANSAC más recorte de atípicos) | M7B | Con muñones de rama o follaje pegado: sesgo de DAP de +12–41 mm a ≤ 0,2 mm; error del centro de 12–65 mm a ~1 mm |
| 4 | Detectar árboles chicos e inclinados: `r_min` 0,03–0,035, sub-rodajas finas y huecos internos como fracción | M7 | Hoy no se detecta ningún árbol con diámetro < 10 cm a 3 m (8 de 69 árboles de campo, ~12 %). Inclinaciones de 4° o más hacen perder árboles de DAP 0,20 m |
| 5 | Unir tramos partidos (`max_gap_slabs` 2–3, fusión colineal) y evaluar con F1 en vez de con un conteo | M7 | En una parcela sintética tipo HQP128 el conteo dio exacto, 44 de 44, pero solo 40 eran correctos: 4 perdidos, 3 duplicados y 1 falso |

El cambio 2 es el único con evidencia real directa. En la corrida sobre T460298B, el suelo normalizado contra sí mismo va de **−0,81 a +0,79 m** y la vegetación baja hasta **−1,21 m** *(real)*. Es la firma esperada del umbral de 0,5 m más el promedio por celda.

---

## Sustento por módulo

### M3 · Normalisation Analysis (`normalization.py`)

Hoy no afecta ninguna métrica: ninguna celda usa su resultado *(código)*.

- **Veredictos falsos.** Se basa en estadísticas globales de Z con tolerancias amplias. Declara "NORMALIZED" una nube con origen local cerca del suelo, una en coordenadas absolutas a 1 m AHD y una normalizada con +0,4 m de sesgo.
- **Error menor.** El patrón `x or default` ignora un umbral explícito de 0,0 (`:127-129`).
- **Propuesta.** Quitarlo, o reemplazarlo por un control de calidad después de normalizar: residuos del suelo y residuo en un anillo alrededor de cada tronco. Un prototipo acertó en 13 de 14 casos sintéticos.

### M4 · Ground Filtering (`ground.py`)

1. **`class_threshold=0.5` mete el pie del tronco en el suelo** *(alta)*.
   - El CSF etiqueta como suelo el 98–100 % de la corteza entre 0 y 0,3 m, el 85–99 % entre 0,3 y 0,5 m, y el 90–97 % de los arbustos de 0,3–1,0 m.
   - Consecuencias: el pie desaparece de la nube de vegetación, lo que deja sin puntos a la sección de 0,3 m y al anclaje basal (0,15–0,80 m), y esos puntos contaminan el DTM de M5.
   - Bajar el umbral solo no alcanza: con tela de 1 m en un 30 % de pendiente se pierde el 7,8 % del suelo verdadero.
2. **La tela de 1,0 m se sube a los arbustos en pendiente** *(media)*.

   | Pendiente | Arbustos > 0,5 m etiquetados como suelo | Suelo perdido |
   |---|---|---|
   | 20 % | 62 % | 0,87 % |
   | 30 % | 76 % | 1,99 % |

   Con tela de 0,3 m se pierde ≤ 0,03 % del suelo.
3. **El docstring de `rigidness` parece invertido** (`:73-76`). En CSF, 1 corresponde a pendiente fuerte y 3 a terreno plano (**verificar** contra el README de CSF). El valor de la notebook, 1, es el correcto; el riesgo es que alguien siga el docstring.

### M5 · Height Normalisation (`height.py`)

1. **El promedio por celda no resiste un suelo contaminado** (`:77-83`).
   - Un punto a 1,3 m reales se reporta con error mediano de −15,7 cm y máximo de 24–41 cm según la pendiente.
   - Por eso la sección "de 1,3 m" cae en realidad a 1,46–1,55 m, y toda la escala de alturas se corre.
   - Usar TIN sobre esos mismos promedios no lo arregla (−13 cm). El problema es promediar puntos contaminados.
2. **Consulta en escalón sin interpolar, con relleno por la celda más cercana** (`:85-91`, `:138-147`). El docstring dice "interpolation", pero el código no interpola *(código)*.

   | Pendiente | Error p95 en suelo limpio con huecos de 0,5–1 m | Con TIN |
   |---|---|---|
   | 20 % | 16–27 cm | ≤ 1,3 cm |
   | 30 % | 24–41 cm | ≤ 1,3 cm |

   Además, el desempate de `distance_transform_edt` hace que el signo del error dependa de la orientación de la pendiente: ±12 cm según hacia dónde mire.
3. **No hay una referencia de altura por árbol** *(media)*.
   - Cada punto se normaliza contra su propia celda, así que las rodajas siguen el terreno en lugar de ser horizontales.
   - El DAP queda referido al centro del tronco y no al lado alto que pide HQP: −6,9 cm de media con 30 % de pendiente y R = 0,225 m.

### M7 · Trunk Extraction (`tracking_assignment.py`, `trunk_extraction.py`)

**Lo que corre de verdad.** Con `USE_TRACKING_ASSIGNMENT = True`, `extract_trunks` y su `trunk_config` no se ejecutan *(código)*.
- Unas 40 constantes de la celda son inertes: `DBH_MIN`, `DBH_MAX`, la franja 8–11 m, `VERTICALITY_THRESH 0.7`, `CROWN_DISTANCE_*` y los 22 `BASAL_ANCHOR_*`.
- Las ~12 perillas que de verdad deciden son valores por defecto del módulo que la celda no muestra: `r_min`/`r_max`, `slab_half_thickness`, `max_inner_points`, `min_sectors`, `max_gap_slabs`, `min_track_length`, entre otras.
- El resultado informa `config=TrunkExtractionConfig()` por defecto, no lo que se usó (`:1742`) *(código)*.

**Defectos, del más grave al menos grave:**

1. **Los árboles chicos no se pueden detectar** *(alta, recall)*.
   - El piso de radio de la elipse, `r_min = 0.05` (`ellipse_fitting.py:706`, `:921`), se aplica a partir de 3 m de altura (`z_min = 3.0`).
   - Regla resultante: un árbol se detecta solo si su diámetro a ~3 m es ≥ 10 cm.
   - En los 69 árboles de campo de HQP quedan fuera 8: cuatro de regeneración en HQP024 (DAP 80–100 mm) y cuatro en HQP128 (DAP 108–120 mm, altura 6,8–9,9 m).
   - La regeneración que HQP pide al menos contar no se puede contar.
   - El mismo piso corta la parte alta de cada tramo, lo que probablemente explica parte de la cobertura de fuste de ~0,61 (**verificar**).
2. **Los troncos inclinados se rechazan** *(alta)*.
   - La rodaja de 1 m (`slab_half_thickness = 0.5`) estira el corte de un tronco inclinado, y el umbral absoluto de huecos internos (`max_inner_points = 5`) lo descarta.
   - Un DAP de 0,20 m se pierde con 4° o más.
   - Con 2°, la captura a la altura del pecho ya baja a 0,62.
   - La extensión "fantasma" vertical de 3 m hacia abajo no alcanza la sección de 1,3 m de un tronco inclinado.
3. **Los tramos se parten y no hay paso de fusión** *(alta, precisión)*.
   - `max_gap_slabs = 1` y `min_track_length = 1`.
   - La ventana basal de 3 a 15 m acepta tramos que empiezan en plena copa.
   - Cada rodaja que falta (verticilo, oclusión, ruido) crea un árbol nuevo.
   - Un objetivo por conteo lo oculta.
4. **Tramos de un solo nodo sin sección de 1,3 m** *(alta para el DAP)*. En la parcela sintética realista, 28 de 61 árboles detectados capturan menos del 90 % de los puntos a la altura del pecho.
5. **Sobreparametrización** *(media hoy, alta para usar otra parcela)*.
   - Todo se ajustó a mano en una sola parcela, T460298B, sobre una submuestra del 10 % y contra un conteo objetivo, sin mapa de fustes ni parcela de control.
   - Los 69 árboles de campo son de otras parcelas (HQP024 y HQP128).
6. **Fallas silenciosas y reproducibilidad** *(media)*.
   - Los rechazos no guardan su motivo.
   - Un árbol sin puntos sigue contando en `n_trees`.
   - Un fallo de LM cae en silencio al ajuste algebraico.
   - Hay un solo generador RANSAC para todo, así que un cambio aguas arriba baraja todos los ajustes siguientes y mete ruido al comparar palancas.
   - Rendimiento: GS.4 ocupa ~95 % del tiempo (~0,4 s por clúster).

### M7B · Stem Cleaning + Sectioning (`trunk_validation.py`)

El ajuste en sí es bueno. En secciones limpias y completas, el DAP tiene sesgo ≤ 0,3 mm, RMSE ≤ 0,8 mm y error de centro < 1 mm; no es un ajuste Kåsa sesgado. Los problemas están en los filtros de validez y en la robustez.

1. **Reintento fallido aceptado como válido** *(crítica)*. Ver el arreglo inmediato 2. En sintético, corregirlo:
   - lleva el error de SED de +4,4 a +1,4 cm;
   - lleva el error de amplitud de sweep de +16,1 a +1,0 mm;
   - pero baja el alcance de 9,7 a 8,3 m, así que conviene entregarlo junto con el punto 5.
2. **El círculo no tiene manejo de atípicos** *(alta)*. Un muñón de rama del 10–20 % agrega +13–23 mm al DAP y desplaza el centro 14–24 mm, y aun así la sección pasa todos los filtros. El retorno está en el cambio 3 de la tabla.
3. **SED, taper y línea central no tratan secciones aisladas** *(alta)*.
   - El SED es la sección válida más alta, sin control de continuidad.
   - El DAP sale de una sola rodaja, la de 1,3 m, sin alternativa en 1,1 o 1,5 m.
   - La mediana de 3 de la línea central ignora los huecos en z.
   - En un árbol sintético con ramas, el SED salió de 20,6 cm donde el diámetro real era 4,4 cm.
4. **Círculo sobre arcos parciales de troncos ovalados** *(alta para centros y sweep)*.
   - El centro se desplaza 15–40 mm con ovalidad 1,2–1,3 y 180–240° de arco visible. Eso rompe el ~1 cm que necesita el clasificador de sweep.
   - Con `USE_ELLIPSE_FIT = False` la ovalidad vale siempre 1 y el código `O1.2+` nunca se emite *(código)*.
   - La elipse recupera la ovalidad (1,2 ± 0,04) solo con cobertura casi completa. Con 180° da 12 % de falsos `O1.2+`.
5. **Los mismos umbrales a toda altura** *(media hoy, alta tras el arreglo 2)*.
   - `r_min = DBH_MIN/2 = 0.075` prohíbe secciones de menos de 15 cm, así que nunca se llega a SED10.
   - La regla de 9 de 16 sectores rechaza todo arco menor de ~180°, aunque arcos de 120° en troncos redondos dan centros con 1,7–2,9 mm de error.
   - La cobertura de arco pesa más que el número de puntos.

---

## Barrido de sensibilidad

Una variable a la vez alrededor del valor actual, más un hipercubo latino chico. Cada parcela se reporta por separado.

| Módulo | Parámetro | Rango | Métrica de salida |
|---|---|---|---|
| M4 | `class_threshold` × banda de refinamiento | 0,05–0,5 × 0,05–0,20 m | Sesgo del DTM en el anillo de cada tronco; puntos en la sección de 0,3 m |
| M4 | `cloth_resolution` × `rigidness` | 0,25–1,0 × 1–3 | Suelo perdido; arbustos marcados como suelo |
| M5 | Interpolación × estadístico por celda | escalón / bilineal / TIN × media / mediana / P10 | Altura real de la sección de 1,3 m |
| M7 | `ellipse.r_min` × `z_min` | 0,03–0,05 × 1–3 m | Recall por clase de DAP (< 150, 150–250, ≥ 250 mm) |
| M7 | `slab_half_thickness` × huecos internos como fracción | 0,10–0,5 × 1–5 % | Recall por clase de inclinación |
| M7 | `max_gap_slabs` × `min_track_length` | {1, 2, 3} × {1, 2} | Duplicados por acierto; F1 |
| M7B | `min_points_section` × `min_sectors` | 20–80 × 6–13 | Altura de la última sección válida; ruido de la línea central |
| M7B | Tolerancia robusta | 8–30 mm | DAP contra cinta; tasa de aceptación |
| M7B | Filtro de continuidad | tolerancia 10–25 % × hueco 0,6–1,5 m | Error del SED; cortes falsos |
| M7B | Cobertura mínima para usar elipse | 10–14 sectores | Falsos `O1.2+` en troncos redondos |

---

## Protocolo de validación externa

Los umbrales se fijan **antes** de medir. Los valores son propuestas, no normas.

| Qué | Referencia | Métrica | Confirma | Descarta el cambio |
|---|---|---|---|---|
| DTM (M4–M5) | Puntos de estación total bajo dosel, incluido el pie de los troncos | RMSE y sesgo según la pendiente | RMSE ≤ 5 cm, sin tendencia con la pendiente | Sesgo sistemático en la base de los troncos |
| Detección (M7) | Mapa de fustes de campo con posición y DAP | F1 emparejando dentro de 0,5 m; recall por clase de DAP e inclinación | F1 ≥ 0,9 y regeneración contable | Recall < 0,8 en DAP < 150 mm, o más duplicados |
| DAP (M7B) | Cinta a 1,3 m en el lado alto | Sesgo y RMSE | Sesgo \|b\| ≤ 1 cm | Sesgo > 2 cm |
| Transferencia | Una parcela de control, con los parámetros congelados | Las mismas métricas | No empeoran | Hay que volver a calibrar |
| Ojo humano | Un cruiser de HQP o Interpine, a ciegas, sobre 10 árboles en CloudCompare | Discrepancias por árbol | Sin discrepancias sistemáticas | Patrón repetido de error |

---

## Cierre

**Supuestos críticos.**
- Casi todo lo medido es sintético. La densidad real, la inclinación, el arco visible por altura, el ruido y la rugosidad de la corteza en T460298B están sin verificar.
- El checkpoint guarda `check` y `sector_pct`. Contar las secciones con `R > 0 & check == 1` por altura mostraría cuánto depende hoy la salida del defecto del reintento.
- Los 69 árboles de campo no pertenecen a la parcela usada para calibrar.

**Cuándo descartar cada cambio:**
- **Refinamiento del suelo:** si en la nube real reduce el suelo detectado bajo los arbustos sin bajar el residuo en el anillo de los troncos.
- **Arreglo del reintento:** si el alcance cae y el filtro de incertidumbre no lo recupera.
- **Ajuste robusto:** si sube el DAP en troncos limpios, por ejemplo con una tolerancia demasiado ajustada sobre ovalidad 1,15, que en sintético dio +22 mm.
- **Cambios de M7:** si el F1 contra el mapa de fustes no mejora aunque el conteo sí.

**Los tres próximos pasos de menor costo:**
1. Aplicar los dos arreglos inmediatos (`np.ptp` y reintento) más el filtro de continuidad, correr T460298B desde el checkpoint y comparar alcance, SED y códigos de sweep. No requiere datos nuevos.
2. Medir el DTM real con lo que ya existe: el residuo en el anillo de cada tronco de T460298B. Luego probar refinamiento + TIN y ver cuánto se mueve la sección de 1,3 m.
3. Conseguir referencia: un mapa de fustes con DAP de cinta de T460298B, o pedir a HQP las nubes de HQP024 y HQP128. Sin eso, ningún cambio de M7 se puede validar.
