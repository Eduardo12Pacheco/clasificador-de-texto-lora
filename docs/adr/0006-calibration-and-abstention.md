# ADR-0006 — Calibración y abstención

- **Estado:** Aceptado
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto

## Contexto

El clasificador debe exponer una confianza en la que el usuario pueda basarse. El máximo de un
softmax no es una probabilidad calibrada: un transformer recién ajustado suele ser demasiado
confiado, y los márgenes de `decision_function` de una línea base lineal no son probabilidades en
absoluto. El producto también necesita una política de abstención: cuando el modelo no está seguro,
la CLI, la API y la UI deben decirlo en lugar de adivinar.

S8 fija los entregables: ECE, NLL y Brier antes y después de la calibración en test; un umbral de
revisión elegido en dev para una exactitud selectiva de 0.95; la cobertura y la exactitud selectiva
en test con ese umbral; y `needs_review` expuesto por cada interfaz. D4 prohíbe usar el conjunto de
test para cualquiera de estas decisiones.

## Decisión

1. **Temperature scaling** (`p = softmax(z / T)`) es el método de calibración. `T` se ajusta solo
   con logits de dev minimizando NLL, mediante una búsqueda golden-section en numpy puro sobre
   `log T ∈ [log 0.05, log 20]`, 80 iteraciones (D10). La línea base recibe su propio `T` ajustado
   sobre sus puntuaciones de `decision_function`.
2. **ECE usa 15 bins de ancho igual sobre la probabilidad máxima** (D10). NLL y Brier siempre se
   reportan junto con ECE, para que un solo artefacto del binning no oculte una distribución de
   probabilidades peor.
3. **El umbral de revisión `τ`** es el punto de corte más pequeño de la probabilidad máxima
   calibrada en el que la exactitud selectiva en dev es de al menos 0.95 y cubre al menos el 20 %
   de dev. Si ningún punto de corte cumple, `τ = 1.0` y el reporte indica que la abstención no tiene
   un punto de operación útil (`Calibration` con umbral 1.0 significa que todas las predicciones se
   revisan).
4. **`needs_review` es `confidence < τ`**, calculado por el decorador `CalibratedClassifier`, para
   que la CLI, la API y la UI compartan una única regla.
5. **El argmax nunca cambia con temperature scaling**; solo cambian la confianza y la decisión de
   abstención. Un test verifica esta invariante.

## Alternativas

- **Platt scaling** (un ajuste logístico por clase o sobre la puntuación máxima): más parámetros y un
  segundo modelo que ajustar, sin una ganancia esperada para una distorsión monótona única de un
  vector de logits ya entrenado.
- **Regresión isotónica**: flexible pero constante por tramos, propensa al sobreajuste con el tamaño
  de dev (2,033 filas) e incapaz de extrapolar más allá del rango de confianza observado.
- **Ensambles de calibración o ensambles profundos**: requieren reentrenar el backbone varias veces,
  lo que choca con el presupuesto medido y añade horas de GPU sin mejorar la historia de abstención.
- **Umbral fijo (p. ej. 0.9)**: no es reproducible entre modelos y conjuntos de datos; la regla de D10
  se deriva de la evidencia de dev y se reporta con su cobertura.
- **Ajustar `T` sobre test**: prohibido por D4 y haría optimistas todas las métricas de test.

## Consecuencias

- Cada interfaz puede mostrar una confianza calibrada y una marca de revisión; la model card y el
  README pueden citar el ECE antes y después y la cobertura en test del umbral. Esta es la evidencia
  de que el número de "confianza" es significativo.
- Temperature scaling es un parámetro monótono único: no puede corregir un modelo cuyo ranking es
  erróneo y no cambia la exactitud. Por eso los reportes mantienen la exactitud, el macro-F1, NLL y
  Brier junto al ECE.
- El ECE depende de la regla de binning. La elección de 15 bins está fijada en el código y
  documentada, y el ECE nunca es la única métrica de calibración.
- El `decision_function` de la línea base se trata como un logit, de modo que la calibración de la
  línea base es comparable con la del transformer bajo la misma regla.
- `stc calibrate` (T8) escribirá `calibration.json` con `temperature` y `review_threshold` junto al
  bundle; T12 completa el addendum de resultados.

## Evidencia

- `uv run pytest tests/unit/domain/test_calibration.py -q` — recuperación de la temperatura dentro
  del 5 % en logits sintéticos, límites del ECE, regla del umbral y métricas selectivas.
- `uv run pytest tests/unit/application/test_evaluate_model.py -q` — el caso de uso de evaluación
  reporta ECE, NLL y Brier sin calibrar y calibrados, y la cobertura en `τ`.
- `src/stc/domain/calibration.py` — `fit_temperature`, `choose_review_threshold`,
  `selective_metrics`, `Calibration`.
- `src/stc/application/services/calibrated_classifier.py` — la implementación compartida de
  `predict`, `predict_proba` y `needs_review`.
- `docs/adr/0004-evaluation-protocol-and-leakage-policy.md` — selección de modelo solo con dev, que
  esta ADR sigue.

## Resultados (T12, modelos reales)

Ajustado en dev (2,033 enunciados) con la regla D10 y luego aplicado sin cambios al conjunto de test
auditado. `reports/final/calibration_dev.json` y `reports/final/calibration_test.json` contienen las
cifras.

| Modelo | T | τ | ECE dev antes → después | NLL dev antes → después | ECE test antes → después | Cobertura en τ (test) | Exactitud selectiva (test) |
|---|---|---|---|---|---|---|---|
| Línea base TF-IDF | 0.7454 | 0.7195 | 0.0832 → 0.0265 | 0.6366 → 0.5842 | 0.0849 → 0.0253 | 0.7690 | 0.9353 |
| Transformer LoRA | 2.2872 | 0.7655 | 0.0826 → 0.0169 | 0.7695 → 0.4814 | 0.0938 → 0.0181 | 0.8083 | 0.9509 |

El transformer está infraconfiado tras el fine-tuning (T > 1); la línea base lineal está
sobreconfiada (T < 1). Temperature scaling reduce el ECE en test del transformer un 81 % y su NLL en
test un 38 %, sin cambiar ningún argmax. Con el τ elegido en dev, el transformer cubre el 80.8 % de
los enunciados de test con una exactitud selectiva de 95.1 %; la línea base cubre el 76.9 % con
93.5 %. El flag `needs_review` usa exactamente estos umbrales mediante `CalibratedClassifier` en la
CLI, la API y la UI.

Hay una salvedad visible en el análisis de errores: algunos errores de alta confianza (por ejemplo
`ex-612` y `ex-1639`) parecen ruido en las etiquetas de referencia y no un fallo de calibración.
Temperature scaling no puede reparar un ranking erróneo; los umbrales solo deciden cuándo abstenerse.
