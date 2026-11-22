# ADR-0009 — Sonda sintética de robustez dialectal

- **Estado:** Aceptado
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto

## Contexto

El modelo se entrena con Amazon MASSIVE `es-ES`, pero los usuarios del responsable están en
Latinoamérica. La investigación del 2026-11-19 no encontró ningún corpus de clasificación en español
con licencia abierta, redistribuible y con etiquetas naturales de dialectos latinoamericanos: el
candidato más grande (MeLiSA) tiene una licencia de solo demostración y no redistribuible, y
TASS/InterTASS no publica una licencia clara. Reportar la exactitud sobre un corpus que no podemos
distribuir volvería irreproducible la métrica.

S9 aún exige evidencia de robustez. Las perturbaciones deben ser puras, deterministas y lo bastante
baratas para ejecutarse en cada evaluación final, y cada reporte, README y mención en la model card
debe decir con claridad que la sonda dialectal es sintética.

## Decisión

1. La sonda dialectal aplica **sustituciones de léxico** a enunciados de test reales, con tres perfiles:
   `mx` (México), `ec` (Ecuador) y `ar` (rioplatense: Argentina y Uruguay). Las tablas viven en
   `src/stc/domain/lexicon.py` y son entradas compartidas para toda la región (`COMMON_LATAM`) más
   superposiciones específicas de cada perfil.
2. El emparejamiento es **por palabra completa, sin distinguir mayúsculas y con la clave más larga
   primero**, de modo que `ahora mismo` se reemplaza como una sola unidad y `que` nunca coincide dentro
   de `quesito`. Una sustitución conserva la mayúscula inicial.
3. Cada sustitución **preserva el género y el número**. Un test estructural verifica que ningún valor de
   reemplazo sea también una clave, lo que hace `regionalize` idempotente y evita que una segunda pasada
   encadene sustituciones.
4. Las demás perturbaciones de D11 (`no_accents`, `uppercase`, `typos`, `chat_abbrev`) también son
   funciones puras del dominio. `typos` edita cada palabra seleccionada en un carácter; una sustitución
   reemplaza la transposición adyacente de D11, porque una transposición tiene distancia de Levenshtein 2
   mientras que S9 fija cada edición a una distancia <= 1.
5. El subconjunto de evaluación es el conjunto de enunciados que la perturbación cambió realmente. Sobre
   ese subconjunto, el reporte da la exactitud antes y después, el delta con un IC 95 % bootstrap pareado
   (B = 2000, semilla 13) y la confianza media calibrada. No se reporta el macro-F1 allí: los subconjuntos
   son demasiado pequeños por clase, y el reporte lo dice.
6. `scripts/lexicon_coverage.py` imprime cuántos enunciados de dev cambia cada perfil. Rechaza
   `--split test` con código de salida 6; el conjunto de test permanece tras la ruta de evaluación final
   auditada.

## Alternativas

- **Entrenar o evaluar con un corpus real de LatAm.** Descartado: no existe un corpus adecuado y
  redistribuible (la evidencia de licencia está en `docs/DATA.md`).
- **Parafrasear o traducir el conjunto de test con un LLM.** Descartado: no es determinista, necesita un
  LLM local, cambia el significado más allá del dialecto y vuelve irreproducible la métrica.
- **Solo ruido de caracteres.** Descartado: no puede mostrar la deriva léxica del dialecto, que es el
  fallo que los usuarios realmente notan.
- **Un conjunto de test de LatAm etiquetado a mano.** Fuera de alcance para v0.1.0, y no sería una sonda de
  sensibilidad sobre los mismos enunciados.

## Consecuencias

La sonda puede mostrar que el modelo es sensible al léxico dialectal, pero no puede estimar la precisión
real en LatAm y nunca debe reportarse como tal. El léxico son datos, así que ampliarlo es barato y no toca
el evaluador. Los homógrafos (por ejemplo, un sustantivo que también es una forma verbal) pueden producir
sustituciones agramaticales; las tablas conservadoras y el test de "ningún valor es una clave" limitan, pero
no eliminan, ese riesgo. La cobertura se mide, no se supone.

## Evidencia

- `uv run pytest tests/unit/domain/test_lexicon.py tests/unit/domain/test_perturbations.py -q` cubre el
  emparejamiento por palabra completa, la preservación de mayúsculas, la idempotencia, la distancia de los
  typos, el determinismo y el registro.
- `uv run python scripts/lexicon_coverage.py --split dev` sobre la partición dev de MASSIVE fijada imprimió
  `mx 99`, `ec 105`, `ar 387` (2,033 enunciados).
- `uv run python scripts/lexicon_coverage.py --split test` sale con código 6 sin leer la partición.
- El reporte final escribe la sonda en `reports/final/robustness.json`, con la nota de que el macro-F1 no se
  reporta en los subconjuntos de perturbación.
