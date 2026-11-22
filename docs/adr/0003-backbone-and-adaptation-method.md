# ADR-0003 — Backbone y método de adaptación

- **Estado:** Aceptado
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto

## Contexto

El clasificador necesita un encoder fuerte en español, lo bastante pequeño para ejecutarse en
CPU tras la exportación a ONNX y con una licencia que permita redistribuir los pesos ajustados.
También necesita un método de adaptación que quepa en una RTX 4060 de 8 GB, que pueda
reproducirse con un bucle explícito y que después pueda fusionarse en un único grafo ONNX. D6
pre-registra los candidatos y la regla de selección; D7 fija los hiperparámetros de LoRA y el
bucle de entrenamiento personalizado.

## Tabla de candidatos

| Backbone | Arquitectura | Parámetros | Licencia | Rol |
|---|---|---|---|---|
| `BSC-LT/MrBERT-es` | encoder ModernBERT | 150,295,040 | Apache-2.0 | Principal |
| `bertin-project/bertin-roberta-base-spanish` | encoder RoBERTa | 124,695,640 | CC BY 4.0 | Respaldo para el screening |
| `Qwen/Qwen3-0.6B` | decoder | 751,632,384 | Apache-2.0 | Solo referencia; nunca se publica |

BETO fue descartado porque sus metadatos de licencia faltan y su README advierte que las
licencias de los corpus de entrenamiento podrían ser incompatibles. `roberta-base-bne` fue
descartado porque sus pesos no están disponibles (repositorio vacío, `config.json` con 404 el
2026-11-19). `xlm-roberta-base` fue descartado porque duplica el tamaño del artefacto sin
evidencia de calidad. `docs/DATA.md` registra la lista completa.

## Decisión

1. **Regla mecánica.** Entre los candidatos exportables (MrBERT-es y bertin), se elige la mayor
   macro-F1 media en dev. MrBERT-es gana si queda dentro de 1.0 punto porcentual de bertin;
   bertin gana solo si queda más de 1.0 punto por encima de MrBERT-es. Qwen3-0.6B nunca es
   seleccionable (`exportable = false`) porque es una referencia de decoder, no un encoder
   desplegable. La regla vive en `application/services/backbone_selection.py`; los tests fijan
   las tres ramas.
2. **LoRA** (`peft.LoraConfig(task_type=SEQ_CLS, target_modules="all-linear")`) es el método de
   adaptación. Mantiene los parámetros entrenables por debajo del 25 % (medido: 2.24 % en
   MrBERT-es con r=16), cabe holgadamente en 8 GB con batch 32 y se fusiona con los pesos base
   antes de la exportación a ONNX.
3. **Bucle personalizado en PyTorch, no el `Trainer` de HF.** La tokenización, el batching, la
   semilla, el recorte de gradientes, el registro y la evaluación en dev son explícitos y se
   comparten con la inferencia. Así, las entradas de calibración y análisis son idénticas al
   artefacto entrenado y se evitan los formatos de registro propios de `Trainer`.

## Alternativas

- Fine-tuning completo para el modelo publicado: pico medido de 2.90 GiB frente a 1.52 GiB de
  LoRA, y sin evidencia de mejor macro-F1 en dev en la sonda.
- Encoder congelado más una cabeza lineal: barato, pero la ejecución de LoRA de 4 épocas de la
  sonda ya igualaba una línea base TF-IDF ajustada, así que congelar el encoder renunciaría a la
  comparación principal.
- Destilación o un modelo multilingüe más pequeño: fuera de alcance para v0.1.0.

## Consecuencias

- El checkpoint seleccionado debe poder exportarse a ONNX con atención eager; los decoders quedan
  excluidos por construcción.
- La regla de selección se basa en datos pero está pre-registrada: nadie puede cambiar de
  backbone después de ver los números de test.
- Los adaptadores LoRA se publican como un asset de release pequeño y separado, mientras que el
  grafo ONNX fp32 fusionado es el artefacto de runtime.

## Evidencia

- `uv run pytest tests/unit/application/test_backbone_selection.py -q`
- `uv run pytest -q -m torch` (trainer, búsqueda y screening con un backbone diminuto)
- Mediciones de la sonda en la tabla "Local measurements" del plan (2026-11-19).

## Resultados

Medidos el 2026-11-20 en la RTX 4060 de 8 GB del responsable (`reports/experiments/`, comandos
en `docs/TRAINING.md`; cada número siguiente proviene de esos reportes versionados o de los
metadatos de ejecución que citan). El entrenamiento usó el conjunto de entrenamiento de MASSIVE
es-ES ya limpio (10,898 filas) y seleccionó solo sobre dev (2,033 filas).

### Screening de backbones (receta D7, dos semillas cada uno)

| Backbone | Exportable | Macro-F1 dev (media ± desv.) | Exactitud dev (media) | Parámetros entrenables | Época media (s) | VRAM pico (MiB) | p50 CPU (ms)* |
|---|---|---|---|---|---|---|---|
| `BSC-LT/MrBERT-es` | sí | 0.8313 ± 0.0006 (0.8309, 0.8317) | 0.8667 | 3,496,056 | 31.9 | 1,666 | 116.4 |
| `bertin-project/bertin-roberta-base-spanish` | sí | 0.7997 ± 0.0031 (0.8019, 0.7975) | 0.8463 | 3,927,672 | 21.5 | 1,666 | 82.9 |
| `Qwen/Qwen3-0.6B` | no (decoder) | 0.8418 ± 0.0049 (0.8453, 0.8383) | 0.8579 | 10,153,984 | 193.2 | 6,707 | 224.9 |

\* Latencia de CPU de una sola frase medida *mientras el job de GPU estaba en marcha*, con la
CPU ocupada en otras tareas; solo sirve para el orden relativo en esta tabla. T12 la reemplaza
por los números versionados de `stc benchmark`. Qwen3-0.6B tuvo un OOM de CUDA con batch 32 y,
como permite D7, reintentó con batch 16 y acumulación de gradientes 2 (`oom_retry: true`); sigue
siendo una referencia de screening y nunca es exportable.

**Selección:** `BSC-LT/MrBERT-es`, rama `mrbert_ahead`. MrBERT-es supera en 3.16 puntos de
macro-F1 en dev a bertin (0.8313 frente a 0.7997), muy por encima del margen de 1.0 punto, así
que no se activó la regla de respaldo de D6. Qwen3-0.6B tuvo la media de screening más alta
(0.8418), pero queda excluido por construcción por ser un decoder; además, su coste por época fue
unas 6 veces el de MrBERT-es.

### Referencia de fine-tuning completo (MrBERT-es, semilla 13, 4 épocas, batch 32)

| Tasa de aprendizaje | Exactitud dev | Macro-F1 dev | VRAM pico (MiB) | Tiempo total (s) |
|---|---|---|---|---|
| 2e-5 | 0.8716 | 0.8428 | 2,961 | 176.9 |
| 3e-5 | 0.8682 | 0.8371 | 2,964 | 166.3 |
| 5e-5 | 0.8795 | **0.8645** | 2,964 | 163.3 |

El fine-tuning completo entrena 150,289,980 parámetros y alcanza un pico de ~2.96 GiB. Su mejor
ejecución individual (0.8645) está 0.09 puntos por encima de la media de LoRA con tres semillas
(0.8636) y 0.54 puntos por debajo de la mejor semilla de LoRA (0.8699): no hay evidencia de que
actualizar todos los pesos valga ~43 veces los parámetros entrenables ni ~2.3 veces la VRAM.

### Búsqueda con Optuna (24 trials, TPE, semilla 13)

La búsqueda completó 24/24 trials (8 COMPLETE, 16 PRUNED) con la mejor macro-F1 en dev de
**0.8627** (trial 5). Mejor configuración:

`r=32`, `alpha=32`, `lora_dropout=0.0`, `lr=3.632e-4`, `weight_decay=0.01`, `epochs=6`,
`batch_size=16`, `warmup_ratio=0.06`, `class_weighting=none`.

Los 16 trials podados fueron las configuraciones más lentas o con más épocas que quedaron por
debajo de la mediana en ejecución tras la primera época; el sampler es el TPE pre-registrado en
D7, y el mejor valor debe leerse junto con el ruido de semilla de ~±0.005 que se ve más abajo. El
objetivo de la búsqueda ordena configuraciones, no modelos finales: la reejecución con tres
semillas de la configuración ganadora es el número usado para la selección.

### Semillas finales de LoRA (mejor configuración)

| Semilla | Exactitud dev | Macro-F1 dev | NLL dev | Tiempo total (s) | VRAM pico (MiB) |
|---|---|---|---|---|---|
| 13 | 0.8741 | 0.8627 | 0.7857 | 239.4 | 1,261 |
| 42 | 0.8795 | **0.8699** | 0.7695 | 249.0 | 1,261 |
| 2026 | 0.8746 | 0.8583 | 0.7673 | 266.1 | 1,261 |

Semilla enviada = **42** (mayor macro-F1 en dev; no hizo falta desempate). Media dev de tres
semillas = **0.8636**, desviación muestral = **0.0059**. LoRA r=32 entrena 6,899,832 de
157,143,672 parámetros (4.39 %) con un pico de 1.26 GiB.

### Interpretación honesta

En dev, el modelo LoRA ajustado supera en 2.07 puntos de macro-F1 a la línea base TF-IDF más
lineal (0.8636 frente a 0.8429) y en 2.8 puntos de exactitud (0.8795 frente a 0.8495). La brecha,
mucho menor en la sonda, no se reprodujo con la configuración buscada; pero dev se usa para la
selección y no puede zanjar la cuestión: T12 reporta la comparación única auditada en test, los
intervalos bootstrap y el veredicto de McNemar. La referencia de fine-tuning completo y la
búsqueda muestran la variabilidad habitual de un problema de 60 clases con pocos ejemplos en la
cola; cualquier diferencia en test menor a ~1 punto debe tratarse como no resuelta, salvo que los
intervalos pareados y McNemar coincidan. Aquí no se afirma ningún número de test.

Los tiempos de ejecución y de pared por etapa están versionados en
`reports/experiments/environment.json` (screening 1,972 s, fine-tuning completo 507 s, búsqueda
1,316 s, semillas finales 754 s; git sha `d5cb1f3`, sha256 de `uv.lock` registrado allí).
