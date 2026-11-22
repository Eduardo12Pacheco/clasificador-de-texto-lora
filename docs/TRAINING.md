# Guía de entrenamiento

Esta guía reproduce los reportes de entrenamiento versionados en una máquina con una GPU NVIDIA de 8 GB.
La ruta de ejecución no necesita nada de esto: la inferencia es ONNX Runtime solo para CPU. El
entrenamiento vive en el extra `train` y en `src/stc/adapters/training`, el único paquete que importa
torch, transformers, PEFT y Optuna.

## Requisitos de hardware y software

- GPU NVIDIA con al menos 8 GB de VRAM (la máquina de referencia es una RTX 4060 con 7,848.6 MiB
  utilizables), un driver CUDA actual y un wheel de torch con soporte CUDA.
- Python 3.12 gestionado con `uv`, además de git.
- RAM de CPU: 16 GiB bastan para las ejecuciones LoRA; el host medido tenía 31.08 GiB.
- El archivo fijado de MASSIVE (39,500,415 bytes, sha256 `7df623fd…153f8`) se descarga una vez con
  `stc data fetch`.

El entorno de referencia registrado en `reports/experiments/environment.json`:

| Elemento | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, driver 620.70.06, CUDA 13.2, bf16 disponible |
| CPU | 12th Gen Intel Core i5-12400T, 12 hilos |
| RAM | 31.08 GiB |
| Python | 3.12.13 (uv) |
| torch | 2.15.0+cu132 |
| transformers | 4.57.6 |
| peft | 0.22.0 |
| optuna | 5.1.0 |
| mlflow | 3.18.0 |
| ONNX Runtime | 1.32.0 |

## Configuración

```bash
uv sync --locked --extra train --group dev
uv run stc doctor --gpu --json
```

`stc doctor --gpu` debe informar `"cuda_available": true` y un nombre de dispositivo. Si no lo hace, falta
el wheel de CUDA o el driver no es visible; no entrenes en CPU esperando los tiempos versionados.

El directorio de datos está excluido de git a propósito. Apunta `STC_DATA_DIR` a un directorio local antes
del primer `prepare`:

```bash
export STC_DATA_DIR="$PWD/data"
uv run stc data fetch
uv run stc data prepare
```

`stc data prepare` debe imprimir `train=10898 dev=2033 test=2974 dropped=616`. `artifacts/baseline` debe
existir ya desde la tarea de la línea base (o reconstruirse con
`uv run stc train baseline --out artifacts/baseline --reports reports/baseline`).

En una máquina compartida con otros trabajos de GPU, envuelve cada comando largo con el lock compartido
(los comandos siguientes lo omiten para facilitar la lectura):

```bash
flock /tmp/portfolio-gpu.lock uv run stc train screen --config configs/screening.toml
```

## Registrar el entorno

```bash
mkdir -p artifacts reports/experiments reports/figures logs
uv run stc doctor --gpu --json > artifacts/environment.json
```

El recolector añade después `stages_seconds` sumando el `run.json` de cada ejecución.

## Paso 1 — screening de backbones

La receta de screening D7 entrena tres backbones × dos semillas con r=16, alpha=32, dropout 0.05, weight
decay 0.01, warm-up 0.06, batch 32 y cuatro épocas. La tasa de aprendizaje es 3e-4 para los encoders y 1e-4
para la referencia decoder.

```bash
uv run stc train screen --config configs/screening.toml --out artifacts/screening --tracker jsonl,mlflow
```

Duración medida: **1972.199 s** (unos 32 min 52 s). Genera `artifacts/screening/screening.json` y
`artifacts/screening/selection.json`. La regla D6 preregistrada se ejecuta en
`src/stc/application/services/backbone_selection.py`. La ejecución versionada seleccionó
`BSC-LT/MrBERT-es` con la rama `mrbert_ahead`.

Si ocurre un error de memoria de CUDA (OOM), el trainer reintenta la ejecución una vez con
`batch_size=16` y acumulación de gradientes 2, y luego registra `oom_retry: true`. Qwen3-0.6B necesitó ese
reintento; sigue siendo una referencia no exportable.

## Paso 2 — referencia de fine-tuning completo

Es el punto de comparación, no el modelo publicado. Actualiza todos los pesos y usa el mismo trainer con el
adaptador desactivado.

```bash
uv run stc train full-ft --backbone BSC-LT/MrBERT-es --lrs 2e-5,3e-5,5e-5 --epochs 4 --out artifacts/full_ft --tracker jsonl,mlflow
```

Duración medida: **506.509 s** (unos 8 min 27 s). VRAM pico: 2,964 MiB. La mejor ejecución de referencia
(lr 5e-5) alcanzó un macro-F1 de dev de 0.8645, por lo que se prefiere la receta LoRA: entrena ~4.4 % de los
parámetros con 1,261 MiB en las ejecuciones finales.

## Paso 3 — búsqueda con Optuna de hiperparámetros

El espacio D7 está preregistrado en `configs/lora_search.toml`: 24 trials TPE con `seed=13`,
`multivariate=True` y `MedianPruner(n_startup_trials=6, n_warmup_steps=1)`. Los trials se podan según el
macro-F1 de dev por época; el objetivo es el macro-F1 de dev tras la última época.

```bash
uv run stc train search --backbone BSC-LT/MrBERT-es --trials 24 --config configs/lora_search.toml --out artifacts/search --tracker jsonl,mlflow
```

Duración medida: **1315.813 s** (unos 21 min 56 s), 24/24 trials (8 COMPLETE, 16 PRUNED). Genera, entre
otros, `artifacts/search/search_trials.json`, `artifacts/search/best_config.toml` y las figuras de historial
e importancia. El mejor trial versionado (trial 5) es r=32, alpha=32, dropout 0.0, lr≈3.63e-4, weight decay
0.01, 6 épocas, batch 16, warm-up 0.06 y sin ponderación de clases.

Copia la configuración generada al repositorio cuando quieras que las ejecuciones finales la usen:

```bash
cp artifacts/search/best_config.toml configs/best_lora.toml
```

## Paso 4 — semillas finales

Entrena la configuración ganadora con las tres semillas preregistradas y selecciona la semilla publicada en
dev (mayor macro-F1 de dev; en empate, menor NLL de dev).

```bash
uv run stc train lora --config configs/best_lora.toml --backbone BSC-LT/MrBERT-es --seeds 13,42,2026 --out artifacts/final --tracker jsonl,mlflow
```

Duración medida: **754.476 s** (unos 12 min 34 s). VRAM pico: 1,261 MiB por semilla. La ejecución versionada
seleccionó la semilla 42 (macro-F1 de dev 0.8699); la media de tres semillas es 0.8636 ± 0.0059. Genera
`artifacts/final/seed-13`, `seed-42`, `seed-2026` y `artifacts/final/selected.json`.

## Paso 5 — recolectar y validar los reportes compactos

Los directorios grandes de ejecución permanecen en `artifacts/` (excluido de git). Solo se versionan los
resúmenes.

```bash
uv run python scripts/collect_experiment_reports.py --artifacts artifacts --out reports/experiments --figures reports/figures
uv run python scripts/validate_reports.py experiments
```

El recolector escribe `reports/experiments/environment.json`, `screening.json`, `selection.json`,
`full_ft_reference.json`, `search_trials.json`, `final_seeds.json` y `search_summary.md`, y copia las figuras
de búsqueda en `reports/figures`.

## Monitorización, seguimiento y reanudación

- **Tracker JSONL** escribe un evento por línea en `<run>/tracker.jsonl` y guarda `run.json` y
  `metrics.json` junto al checkpoint. Es el valor por defecto.
- **MLflow** registra los mismos parámetros, métricas y artefactos en `sqlite:///artifacts/mlflow.db`.
  Inspecciónalo en local en un puerto dentro del rango 19000–19999 del proyecto:

  ```bash
  uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 19081
  ```

- **Optuna** guarda su estudio en `artifacts/search/optuna.db`. Reanuda una búsqueda interrumpida con
  `--resume`; los trials completados no se repiten.
- **Screening y semillas finales** omiten una ejecución cuyo `run.json` ya existe. Vuelve a ejecutar el mismo
  comando para continuar tras un fallo o el fin de una sesión.
- `stc doctor --gpu --json` es el primer diagnóstico cuando una ejecución no arranca.

## Solución de problemas

| Síntoma | Qué revisar | Solución |
|---|---|---|
| `cuda_available: false` | `nvidia-smi`, `stc doctor --gpu --json` | Instala un wheel de torch con CUDA; no caigas en entrenamiento por CPU. |
| CUDA out of memory | Log de la ejecución, `peak_vram_mib` | El trainer reintenta una vez con batch 16 y acumulación 2; si no, reduce `batch_size` o `max_length` en el TOML. |
| Pérdida NaN o macro-F1 de dev < 0.78 en screening | Orden de `labels.json`, id de pad, `max_length`, tasa de aprendizaje | El umbral mínimo de D7 es 0.78; corrige el pipeline de entrada antes de continuar. |
| Un trial de búsqueda falla por un motivo distinto de OOM | Estado en `search_trials.json` | La búsqueda registra el trial y continúa; más de seis trials fallidos significan parar e investigar. |
| El puerto de la UI de MLflow está ocupado | Nota sobre el rango de `STC_API_PORT` | Usa otro puerto dentro de 19000–19999. |
| El tokenizer no carga | `tokenizer.json` en la caché del backbone | D8 usa el archivo `tokenizers` de Rust, no `AutoTokenizer`; el adaptador recurre al token de fin de texto del decoder cuando falta `<pad>`. |

## Cambiar el backbone

Pasa `--backbone <hf-id>` a `stc train full-ft`, `stc train search` y `stc train lora`, o edita la lista de
candidatos en `configs/screening.toml` y vuelve a ejecutar el screening. El backbone debe ser un encoder que
incluya `tokenizer.json` y que pueda fusionarse y exportarse a ONNX; los modelos solo decoder son referencias
de screening y nunca se publican. Un cambio en el conjunto de candidatos o en la regla de selección es un
cambio de decisión (D6/ADR-0003), no un experimento local: regístralo en la ADR y vuelve a ejecutar la
selección antes de tocar test.

Para cambiar la receta del adaptador, edita `configs/best_lora.toml` (el archivo generado contiene los valores
del trial) y vuelve a ejecutar las semillas finales. Mantén las piezas fijas de D7 (longitud máxima 64, atención
eager y el bucle de entrenamiento personalizado) salvo que una decisión las cambie.

## Reentrenar con otro conjunto de datos

1. Implementa `RawDatasetSource` y `DatasetParser` para el nuevo formato y conéctalos en `build_data_services`
   de `src/stc/bootstrap.py`.
2. Mantén los roles de train, dev y test y ejecuta `stc data prepare`; la auditoría de fugas y la limpieza D3
   deben pasar antes de entrenar.
3. Reconstruye la línea base (`stc train baseline`) y vuelve a ejecutar el pipeline D7 completo. Las etiquetas
   se derivan de los datos procesados en orden léxico, así que un espacio de etiquetas nuevo cambia
   `labels.json` e invalida los bundles anteriores.
4. Actualiza `docs/DATA.md` con la fuente, la licencia, el checksum y el linaje, y la model card con la nueva
   declaración de datos. Cambiar el conjunto de datos afecta a D1, así que necesita una actualización de la
   decisión antes de publicar los resultados.

El conjunto de test sigue bloqueado: un conjunto de datos nuevo implica una comparación auditada nueva y de una
sola vez, y cada lectura sigue añadiendo una línea a `reports/test_access_log.jsonl`.

## Presupuesto de VRAM y de tiempo (medido)

| Etapa | Tiempo de pared | VRAM pico | Notas |
|---|---:|---:|---|
| Screening (6 ejecuciones) | 1972.199 s | 1,666 MiB (LoRA), 6,707 MiB (reintento de Qwen3) | Incluye la referencia decoder. |
| Fine-tuning completo (3 lr) | 506.509 s | 2,964 MiB | Actualiza los 150 M de parámetros. |
| Búsqueda con Optuna (24 trials) | 1315.813 s | 1,666 MiB | El podado reduce la media. |
| Semillas finales (3) | 754.476 s | 1,261 MiB | Configuración ganadora r=32. |

El pipeline medido completo dura unos 76 minutos en la máquina de referencia. Son valores medidos, no
garantías; la fuente de verdad son los reportes versionados.

## Advertencia sobre la inferencia solo en CPU

El entrenamiento requiere el extra `train` y una GPU. La inferencia en CPU es la ruta de producto soportada:
instala el wheel sin el extra, descarga el bundle ONNX y ejecuta `stc predict` o `stc serve`. El job
`smoke-install` de CI verifica que `import torch` falla dentro del entorno de la herramienta instalada, mientras
la inferencia sigue funcionando.
