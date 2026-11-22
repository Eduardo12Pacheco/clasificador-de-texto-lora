# Protocolo de evaluación y guía de reportes

Este documento describe cómo mide el proyecto los dos clasificadores y cómo leer cada reporte
versionado. Las decisiones que lo sustentan son vinculantes: D2 (métricas), D3 (política de fugas), D4
(roles de las particiones y bloqueo de test), D10 (calibración y abstención), D11 (perturbaciones), D19
(veredicto de comparación) y la ADR-0004.

## Roles de las particiones y bloqueo de test

Las particiones oficiales de MASSIVE se conservan. Solo se limpia train; dev y test se reportan tal como
se publicaron.

| Partición | Filas | Rol |
|---|---:|---|
| train | 10,898 tras la limpieza (11,514 originales) | Ajuste de la línea base y del transformer. |
| dev | 2,033 | Búsqueda en grid, screening de backbones, selección con Optuna, selección de semilla, temperatura `T`, umbral de revisión `τ` y puerta de cuantización. |
| test | 2,974 | Una comparación final auditada por release candidate. |

`DevOnlyRepository` envuelve el repositorio de datos para todo caso de uso no final, y
`require_test_grant` lanza `TestSplitLocked` (código de salida 6) cuando se pide la partición de test sin
un `TestAccessGrant`. Hay exactamente dos lectores auditados: el caso de uso `FinalReport`
(`stc report final`) y `stc evaluate --split test --final`. Cada uno añade una línea JSON a
`reports/test_access_log.jsonl` con la marca de tiempo UTC, el comando, el git sha, el sha256 del modelo
y el número de ejemplos. Las líneas del registro nunca se borran.

La limpieza sigue D3 (`normalize_for_dedup` y después tres etapas solo sobre train), que elimina 616
filas: 356 por solapamiento con la evaluación, 43 duplicados con conflicto de etiquetas y 217 duplicados
exactos. `reports/data_audit.json` registra cada conteo y los ids eliminados por motivo;
`docs/DATA.md` explica la política.

## Definiciones de métricas

Todas las métricas se calculan con el espacio de 60 intenciones en orden léxico. El conjunto de clases
evaluadas son las etiquetas con al menos un ejemplo real en la partición (`present_labels`): 59
intenciones en test (`cooking_query` no aparece) y 59 en dev (`audio_volume_other` no aparece). Las
divisiones indefinidas valen 0.

- **Exactitud** — `(1/N) · Σ_i 1[y_i = ŷ_i]`.
- **Precisión / recall / F1 por clase** — para la clase `c`, con `TP_c = CM[c, c]`,
  `FP_c = Σ_r CM[r, c] - TP_c` y `FN_c = Σ_p CM[c, p] - TP_c`:
  `P_c = TP_c / (TP_c + FP_c)`, `R_c = TP_c / (TP_c + FN_c)`, `F1_c = 2·P_c·R_c / (P_c + R_c)`.
- **Macro-F1** — media no ponderada de `F1_c` solo sobre las etiquetas presentes. Es la métrica principal
  de selección (D2).
- **F1 ponderado** — `Σ_c support_c · F1_c / Σ_c support_c`.
- **Exactitud top-3** — fracción de filas cuya intención real está entre las tres puntuaciones más altas
  (`argsort(..., kind="stable")`).
- **Exactitud por escenario** — se asigna a la intención real y a la predicha su escenario (el prefijo de la
  intención antes del primer `_`) y se comparan los escenarios.

### Intervalos de confianza y significancia

- **Bootstrap percentil**, `B = 2000`, `numpy.random.default_rng(13)`, intervalo del 95 % (percentiles `2.5`
  y `97.5`). El intervalo reportado se ajusta para incluir su propia estimación puntual, porque una métrica
  discreta sobre un remuestreo pequeño podría producir un intervalo que contradiga la estimación.
- **Bootstrap pareado del Δ** — los modelos A y B se evalúan sobre los *mismos* índices de filas
  remuestreados, de modo que el intervalo describe la diferencia pareada `metric(A) - metric(B)` y no dos
  intervalos independientes.
- **McNemar exacto** — con `b` filas correctas solo para el modelo A y `c` correctas solo para el modelo B,
  `n = b + c` y `p = min(1, 2 · Σ_{k=0}^{min(b,c)} C(n, k) / 2^n)`. La implementación usa `math.comb` con
  enteros, no una aproximación normal.
- **Veredicto (D19)** — `transformer_better` solo cuando el intervalo pareado del Δmacro-F1 no incluye cero y
  McNemar `p < 0.05` favorece al transformer; la regla simétrica define `baseline_better`; en otro caso,
  `tie`. La frase del README se genera a partir de `reports/summary.json`, así que no puede exagerar.

## Calibración y abstención

Para logits `z` y una única temperatura `T`, la distribución calibrada es `softmax(z / T)`. `T` se ajusta
sobre los logits de dev minimizando NLL con una búsqueda golden-section de 80 iteraciones sobre
`log T ∈ [log 0.05, log 20]`; la línea base obtiene su propio `T`, ajustado sobre sus puntuaciones de
`decision_function`.

- **NLL** — `-mean(log p_y)`.
- **Brier** — `mean(Σ_c (p_c - 1[c = y])²)`.
- **ECE** — 15 bins de ancho igual sobre la probabilidad máxima calibrada:
  `ECE = Σ_m (n_m / N) · |accuracy_m - confidence_m|`, donde `n_m` es el número de filas del bin `m`,
  `confidence_m` es su probabilidad máxima media y `accuracy_m` es su exactitud empírica.
- **Umbral de revisión `τ`** — el punto de corte más pequeño de la probabilidad máxima calibrada en el que la
  exactitud selectiva en dev es de al menos 0.95 y el punto de corte cubre al menos el 20 % de dev. Si ningún
  punto de corte cumple, `τ = 1.0` y el reporte indica que la abstención no tiene un punto de operación útil.
- **Cobertura / exactitud selectiva** — en `τ`, la cobertura es la fracción de filas con `confidence >= τ`; la
  exactitud selectiva es la exactitud sobre ese subconjunto cubierto.
- **`needs_review`** — `confidence < τ`, calculado una vez por `CalibratedClassifier` y expuesto por la CLI, la
  API y la UI. El escalado de temperatura nunca cambia el argmax; solo se mueven la confianza y la decisión de
  revisión.

## Perturbaciones de robustez

`src/stc/domain/perturbations.py` implementa siete perturbaciones puras. El subconjunto evaluado es el conjunto
de enunciados que la función cambió realmente.

| Nombre | Definición |
|---|---|
| `no_accents` | NFD, quitar las marcas combinantes salvo la tilde de `ñ`/`Ñ`, y NFC. |
| `uppercase` | Pasar todo el enunciado a mayúsculas. |
| `typos` | Para palabras alfabéticas de longitud ≥ 4, con probabilidad 0.06 se aplica una edición determinista (sustitución, borrado o duplicación de un carácter) usando `random.Random(f"{seed}:{text}")` con semilla 13. La idea anterior de intercambiar adyacentes se reemplazó por sustitución para que cada edición tenga distancia de Levenshtein 1, lo que fijan los tests de propiedades. |
| `chat_abbrev` | Abreviaturas de chat de palabra completa de `src/stc/domain/lexicon.py`. |
| `regionalize:mx` | Sustituciones de léxico del español mexicano **sintético**. |
| `regionalize:ec` | Sustituciones de léxico del español ecuatoriano **sintético**. |
| `regionalize:ar` | Sustituciones de léxico rioplatense y de voseo **sintéticas**. |

Para cada perturbación y cada modelo, el reporte indica el número de enunciados afectados, la exactitud antes
y después en ese subconjunto, el Δ de exactitud pareado por bootstrap con su IC del 95 % y la confianza
calibrada media antes y después. El macro-F1 **no** se reporta a propósito en los subconjuntos de perturbación:
son demasiado pequeños para métricas por clase estables, y el reporte lo indica en su campo `note`.

Las tablas de regionalización son **sintéticas**. MASSIVE no tiene un locale de español latinoamericano. Estas
sondas miden la fragilidad léxica; no son evidencia de precisión real en LatAm. El aviso se repite en la
ADR-0009, la model card y el README.

## Cortes naturales

`src/stc/domain/slices.py` produce las siguientes máscaras sobre test:

- escenario (18 escenarios de MASSIVE);
- bucket de longitud `le4`, `5_8`, `9_12` y `ge13`, según el número de palabras separadas por espacios;
- `digit:yes` / `digit:no` (al menos un dígito decimal);
- `accent:yes` / `accent:no` (un diacrítico distinto de la tilde de `ñ`);
- traductor `worker:<id>` para los workers con al menos 30 filas de test.

Cada corte reporta `n`, la exactitud con IC del 95 % y, solo cuando `n >= 200`, el macro-F1 con IC del 95 %
(`min_macro_f1_support` en el reporte). El corte por worker también expone el mínimo, el máximo y la
dispersión, para que el lector pueda juzgar si una puntuación depende de un solo anotador.

## Archivos de reporte

Todos los reportes son deterministas: claves ordenadas, flotantes de cuatro decimales, UTF-8 y `\n`.

| Archivo | Contenido | Generado por |
|---|---|---|
| `reports/data_audit.json` | Conteos originales y procesados, cada conteo de solapamiento y duplicado, conteos de eliminación e ids eliminados por motivo. | `stc data audit` |
| `reports/data_audit.md` | Resumen de auditoría legible para personas. | `stc data audit` |
| `reports/baseline/search.json` | Cada configuración del grid D5 con exactitud en dev, macro-F1, segundos de ajuste y número de features, más el ganador. | `stc train baseline` |
| `reports/baseline/dev_metrics.json` | Métricas en dev del ganador, filas por clase, matriz de confusión y errores. | `stc train baseline` |
| `reports/experiments/environment.json` | GPU, driver, CPU, RAM, versiones de bibliotecas, sha256 de `uv.lock`, git sha y segundos de reloj por etapa. | `stc doctor --gpu --json` + recolector |
| `reports/experiments/screening.json` | Seis ejecuciones backbone × semilla con métricas en dev, parámetros entrenables, segundos por época, VRAM pico y latencia de CPU. | `stc train screen` + recolector |
| `reports/experiments/selection.json` | Medias por backbone, la regla D6 y la rama y backbone seleccionados. | `stc train screen` + recolector |
| `reports/experiments/full_ft_reference.json` | Grid de fine-tuning completo por tasa de aprendizaje. | `stc train full-ft` + recolector |
| `reports/experiments/search_trials.json` | Cada trial de Optuna, su estado y objetivo, además del mejor trial y configuración. | `stc train search` + recolector |
| `reports/experiments/final_seeds.json` | Tres semillas finales, semilla y ejecución seleccionadas, media y desviación en dev. | `stc train lora` + recolector |
| `reports/experiments/search_summary.md` | Narrativa de la búsqueda y mejores trials. | recolector |
| `reports/final/metrics.json` | Exactitud, macro-F1 y F1 ponderado, top-3, exactitud por escenario, IC, calibración y matriz de confusión en test para ambos modelos. | `stc report final` |
| `reports/final/comparison.json` | Δ pareados de exactitud y macro-F1 con IC, McNemar `b`/`c`/`p`, veredicto y frases EN/ES. | `stc report final` |
| `reports/final/per_class.csv` | Una fila por modelo × intención: precisión, recall, F1 y soporte (un soporte 0 se marca en la propia fila). | `stc report final` |
| `reports/final/confusion_matrix.csv` | Conteos completos de 60 × 60 por modelo, en formato largo. | `stc report final` |
| `reports/final/calibration_dev.json` | `T`, `τ`, ECE y NLL antes y después en dev, y cobertura y exactitud selectiva en dev. | `stc calibrate` |
| `reports/final/calibration_test.json` | La misma calibración aplicada a test, incluidos Brier y cobertura. | `stc report final` |
| `reports/final/robustness.json` | Tabla de perturbaciones por modelo (número afectado, antes y después, IC del Δ, confianza). | `stc report final` |
| `reports/final/slices.json` | Tabla de cortes naturales por modelo. | `stc report final` |
| `reports/final/errors.json` | Una entrada por error de test, con texto, intención real y predicha, confianza calibrada y las predicciones de ambos modelos. | `stc report final` |
| `reports/final/export_parity.json` | ONNX fp32 frente a torch: diferencia absoluta máxima de logits, acuerdo y resultado frente a la puerta de 5e-3. | `stc export` |
| `reports/final/quantization_dev.json` | Cuatro variantes int8, exactitud y macro-F1 en dev, acuerdo de etiquetas, tamaño, resultado de la puerta y `shipped_format`. | `stc export --quantize all` |
| `reports/final/benchmark.json` | Modelo de CPU, versión de ORT, latencias p50, p95 y media por número de hilos, y rendimiento batch-32. | `stc benchmark` |
| `reports/figures` | Figuras de fiabilidad, confusión, F1 por clase, robustez y búsqueda (PNG). | reporte final, recolector, exportación |
| `reports/summary.json` | Cifras compactas que consumen los renderizadores del README y de la model card. | `scripts/render_results.py` |
| `reports/results_table.md` | Tabla de resultados en Markdown generada a partir del resumen. | `scripts/render_results.py` |
| `reports/test_access_log.jsonl` | Registro de auditoría de solo anexar: una línea por cada lectura autorizada de test. | lectores auditados |
| `docs/ERROR_ANALYSIS.md` | Categorías cualitativas de error con ids reales verificados por `scripts/check_error_analysis.py`. | manual, a partir de `errors.json` |

## Volver a ejecutar los reportes

Los comandos de datos y de modelos suponen que `STC_DATA_DIR` apunta a los datos locales preparados y que
`artifacts/` contiene la línea base ajustada y la semilla final. Las etapas pesadas de GPU están documentadas
con duraciones medidas en `docs/TRAINING.md`.

```bash
uv run stc data fetch
uv run stc data verify
uv run stc data prepare
uv run stc data audit --out reports
uv run stc train baseline --out artifacts/baseline --reports reports/baseline
uv run stc export --run artifacts/final/seed-42 --out artifacts/bundle --quantize all --reports reports/final
uv run stc calibrate --bundle artifacts/bundle --baseline artifacts/baseline --reports reports/final
uv run stc benchmark --bundle artifacts/bundle --threads 1,4 --runs 200 --out reports/final/benchmark.json
uv run stc report final --bundle artifacts/bundle --baseline artifacts/baseline --out reports/final --figures reports/figures
uv run python scripts/collect_experiment_reports.py --artifacts artifacts --out reports/experiments --figures reports/figures
uv run python scripts/render_results.py --summary reports/summary.json --table reports/results_table.md --registry src/stc/model_registry.json --readme README.en.md --readme-es README.md
uv run python scripts/render_model_card.py --template docs/model_card.template.md --summary reports/summary.json --out MODEL_CARD.md
uv run python scripts/validate_reports.py all
```

El renderizador de resultados reescribe el bloque de resultados de `README.en.md` (inglés) y de `README.md`
(español) a partir de `reports/summary.json`; sin `--registry` el bloque no incluye el tamaño del bundle.

Los comandos pesados de entrenamiento (screening, fine-tuning completo, búsqueda con Optuna y las tres
semillas finales) están en `docs/TRAINING.md`; sus resultados compactos los produce
`scripts/collect_experiment_reports.py`.

## Reejecuciones de la evaluación final

El registro de auditoría contiene actualmente **una** línea, así que hubo una sola lectura auditada de test para
v0.1.0 y no hay reejecuciones que explicar:

| Marca de tiempo (UTC) | Comando | git sha | sha256 del modelo | Ejemplos |
|---|---|---|---|---:|
| 2026-11-20T17:02:19 | `stc report final --bundle artifacts/bundle --baseline artifacts/baseline --dataset massive` | `c6523f4` | `8a438c1c…75fbd5` | 2,974 |

Si una futura corrección de un error obliga a una segunda lectura, añade aquí una fila nueva con el motivo;
nunca borres ni reescribas una línea del registro.

## Notas de reproducibilidad

- La semilla de Python, NumPy y torch se fija mediante `set_all_seeds(13)`; torch usa
  `use_deterministic_algorithms(True, warn_only=True)` y `CUBLAS_WORKSPACE_CONFIG=:4096:8`. No se afirma
  reproducción bit a bit entre GPUs y drivers.
- Los sorteos del bootstrap y del bootstrap pareado usan `default_rng(13)` con `B = 2000` fijo; McNemar es exacto.
- La línea base usa un grid fijo (`configs/baseline.toml`) y un desempate fijo: macro-F1, luego exactitud y
  después menos features.
- La selección usa solo dev. Los números de test de `reports/final` se produjeron una sola vez con el comando
  registrado en `reports/test_access_log.jsonl`.
- Los tests nunca usan la red. Las muestras bajo `src/stc/_fixture` y `tests/fixtures` son deterministas y están
  versionadas.
- Los reportes registran el git sha y el sha256 de `uv.lock` de la ejecución en
  `reports/experiments/environment.json`; el manifiesto del modelo publicado registra la ejecución de
  entrenamiento, la longitud del tokenizer y el modo de cuantización.

## Limitaciones conocidas

- Solo se evalúa `es-ES`; las sondas de LatAm son sintéticas (ver arriba).
- El conjunto de test no tiene ningún ejemplo de `cooking_query`, así que el macro-F1 se calcula sobre 59 de las 60 intenciones.
- Las perturbaciones reportan exactitud, no macro-F1, por diseño.
- El escalado de temperatura corrige la confianza, no el ranking; los errores de alta confianza por ruido en las
  etiquetas siguen visibles en `docs/ERROR_ANALYSIS.md`.
- La latencia de CPU depende del host; `reports/final/benchmark.json` registra el modelo de CPU y la versión de
  ONNX Runtime junto a las cifras.
