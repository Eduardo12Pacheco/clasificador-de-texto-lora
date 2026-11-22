# ADR-0005 — Inferencia con ONNX Runtime y la puerta de cuantización

- **Estado:** Aceptado
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto

## Contexto

S11/S12 exigen una ruta de inferencia solo para CPU que nunca importe `torch`, una exportación a
ONNX del modelo LoRA fusionado con paridad fp32 medida y una decisión honesta sobre la
cuantización int8. La sonda de planificación en la máquina del responsable mostró que el ONNX
fp32 es exacto y ya es rápido, mientras que la int8 dinámica por defecto pierde varios puntos de
macro-F1 en una arquitectura ModernBERT. Publicar int8 a ciegas degradaría el producto en
silencio; publicar fp32 a ciegas descartaría una ganancia limpia si alguna variante la supera.

## Decisión

1. **ONNX Runtime es el único evaluador en runtime.** `OnnxClassifier`
   (`adapters/inference/onnx_classifier.py`) carga `model.onnx`, el archivo `tokenizers` de Rust y
   `labels.json`, usa `CPUExecutionProvider`, respeta un `intra_op_num_threads` explícito y
   devuelve logits float32 en el orden de entrada. El contrato de paquetes de runtime prohíbe
   `torch`, `transformers`, `peft`, `onnx` y `optuna` en todo adaptador de runtime; solo se
   importan `onnxruntime` y `tokenizers`.
2. **Pipeline de exportación.** `OnnxExporter` reconstruye el backbone base con atención eager,
   añade el adaptador PEFT, llama a `merge_and_unload()`, envuelve el modelo para que el grafo
   emita una única salida `logits` y exporta con `opset_version=17`, `dynamo=False` y ejes
   dinámicos de batch y secuencia. Copia el tokenizer, escribe `labels.json` y registra un
   manifiesto.
3. **La paridad fp32 es una puerta obligatoria.** La comparación entre el modelo fusionado y ONNX
   en dev solo pasa si `max |Δlogit| <= 5e-3` y el acuerdo de argmax es exactamente `1.0`.
   `stc export` escribe `export_parity.json` y falla la ejecución si la verificación de paridad
   no se cumple.
4. **Matriz de candidatos de cuantización (D12).** Se construyen y evalúan en dev cuatro variantes:
   `int8_dynamic`, `int8_dynamic_per_channel`, `int8_dynamic_mlp_only` (excluye por nombre los nodos
   de atención, cabeza y clasificador) e `int8_static_qdq` (QDQ, por canal, 512 enunciados de
   calibración de ancho fijo, percentil). Una variante se publica solo si su caída de macro-F1
   frente a fp32 es `<= 0.5` puntos **y** el acuerdo de argmax es `>= 99 %`. Se publica la variante
   aprobada más pequeña; si ninguna pasa, se publica fp32. `quantization_dev.json` registra cada
   variante y el formato publicado.
5. **Respaldo de calibración estática.** El calibrador por percentiles de ONNX Runtime no puede
   construir un histograma de 2048 bins cuando un tensor tiene un único valor (la máscara de
   atención constante del grafo de prueba diminuto es uno de esos casos). El exportador recurre a
   la calibración MinMax solo para esa exportación; la exportación real de MrBERT-es usa percentil,
   como se decidió.

## Alternativas

- Inferencia con TorchScript: mantiene `torch` en el runtime, lo que viola S12 y la historia de
  instalación solo para CPU.
- Exportación con `optimum`: dependencia extra para una ruta que ya controlamos de forma explícita.
- Publicar int8 incondicionalmente: la sonda midió una pérdida de 3–4.5 pt de macro-F1.
- Publicar fp32 incondicionalmente: el resultado negativo debe medirse, no suponerse; un futuro
  backbone puede pasar la puerta.

## Consecuencias

- La herramienta instalada funciona en una CPU sin el extra de entrenamiento; un smoke test de CI
  verifica que `import torch` falla dentro del entorno de la herramienta.
- La decisión sobre int8 es reproducible a partir de los reportes de dev versionados; la puerta
  puede cambiar sin tocar código cuando una ejecución la supere.
- `scripts/list_onnx_nodes.py` existe para que la primera exportación real confirme los patrones de
  nombres de nodo de `mlp_only` antes de confiar en ellos.
- La primera exportación real (T12) añade resultados medidos a esta ADR.

## Evidencia

- Sonda de planificación (no versionada): paridad fp32 con 100 % de acuerdo de argmax y
  max |Δlogit| 1.39e-3; int8 por defecto 0.8234 de exactitud / 0.7827 de macro-F1 / 88.5 % de
  acuerdo; por canal 0.8328 / 0.7983 / 90.1 %; fp32 0.8613 / 0.8282.
- `uv run pytest tests/unit/adapters/test_onnx_classifier.py -q`
- `uv run pytest tests/unit/adapters/test_parity.py -q`
- `uv run python -m pytest tests/unit/adapters/test_export_torch.py -q -m torch`
- `uv run python -c "import sys, stc.adapters.inference.onnx_classifier; assert 'torch' not in sys.modules"`

## Resultados (T12, exportación real)

Comando: `uv run stc export --run artifacts/final/seed-42 --out artifacts/bundle --quantize all --reports reports/final`
en el host RTX 4060 del responsable. El grafo fp32 ocupa 601,760,862 bytes
(574 MiB), opset 17, con ejes dinámicos de batch y secuencia. `onnx.checker` y `stc bundle verify`
pasan.

**Paridad fp32 (dev, 2,033 enunciados).** `max |Δlogit| = 3.92e-3` (límite 5e-3), acuerdo de argmax
`1.0`, `passed: true`. La ejecución de sanidad en dev reprodujo exactamente las métricas de torch de
la semilla seleccionada: exactitud 0.8795, macro-F1 0.8699.

**Patrones de `mlp_only`.** La primera exportación real confirma los nombres de nodo que supone
`quantization.py`: los MatMul de atención viven bajo `/inner/model/layers.N/attn/...` (por ejemplo
`/inner/model/layers.0/attn/Wqkv/MatMul`) y la cabeza del clasificador bajo
`/inner/head/dense/MatMul`; por tanto, `/attn/`, `/head/` y `/classifier/` coinciden con nodos
reales y excluyen los MatMul previstos. No hizo falta corregir ningún patrón.

**Matriz de cuantización (dev).** Puerta: caída de macro-F1 ≤ 0.5 pt y acuerdo de etiquetas ≥ 99 %.
Las cuatro variantes no pasan; se publica fp32.

| Variante | Exactitud | Macro-F1 | Acuerdo | Tamaño (bytes) | Puerta |
|---|---|---|---|---|---|
| fp32 | 0.8795 | 0.8699 | 1.0000 | 601,760,862 | referencia |
| `int8_dynamic` | 0.8490 | 0.8223 | 0.9139 | 151,317,198 | no pasa |
| `int8_dynamic_per_channel` | 0.8593 | 0.8318 | 0.9179 | 151,997,624 | no pasa |
| `int8_dynamic_mlp_only` | 0.8603 | 0.8430 | 0.9228 | 309,082,247 | no pasa |
| `int8_static_qdq` | 0.0246 | 0.0083 | 0.0251 | 152,669,567 | no pasa |

Los resultados dinámicos van en la misma dirección que la sonda de planificación, pero son algo
peores: la atención de ModernBERT es sensible al redondeo de pesos int8, y la variante mlp-only paga
con un archivo más grande y aun así pierde 2.7 pt de macro-F1. La exportación QDQ estática falló en
esta ejecución (2.5 % de acuerdo de etiquetas con fp32, macro-F1 de dev 0.0083), así que no es
candidata incluso antes de la puerta de calidad. La causa no fue investigada. La exportación usó la
calibración MinMax descrita abajo, así que esta medición no demuestra que la arquitectura no pueda
cuantizarse de forma estática.

**Desviación en la calibración estática.** D12 eligió calibración por percentiles para
`int8_static_qdq`. Los histogramas por percentiles por canal sobre este encoder de 150 M de
parámetros alcanzan un pico de 19.1 GiB de RSS (medido con ONNX Runtime 1.32.0); el host compartido
de 31 GiB terminó repetidamente el proceso por OOM con ~6.0 GiB de RSS anónimo bajo trabajos
concurrentes del portafolio. Por ello, la variante usa `CalibrationMethod.MinMax` con el mismo grafo
QDQ, pesos por canal y 512 enunciados de calibración de ancho fijo. Es la única desviación de D12 y
afecta a un candidato que falla la puerta por un margen amplio; no cambia el formato publicado.
`tests/unit/adapters/test_export_torch.py::test_static_variant_uses_minmax_calibration` fija esta
elección.

**Benchmark de CPU (S12, `reports/final/benchmark.json`).** Intel Core i5-12400T, ONNX Runtime 1.32.0,
fp32, 200 ejecuciones de un solo enunciado: 1 hilo p50 30.32 ms / p95 43.93 ms; 4 hilos p50 15.80 ms /
p95 19.90 ms; rendimiento batch-32 de 28.9 / 93.2 enunciados/s. El manifiesto publicado registra
`quantization.mode = "fp32"`.
