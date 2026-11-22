# ADR-0004 — Protocolo de evaluación y política de fuga de datos

- **Estado:** Aceptado
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto

## Contexto

MASSIVE se ensambla a partir de traducciones de SLURP y contiene duplicados genuinos de
enunciados entre particiones. Leer el conjunto de test antes de tiempo, o entrenar con filas
cuyo texto también aparece en dev o test, infla silenciosamente la comparación final y vuelve
no fiable el veredicto entre la línea base y el transformer. El protocolo debe estar
pre-registrado, ser reproducible y verse en artefactos versionados.

## Decisión

### Roles de las particiones (D4)

- **train** — solo para ajustar el modelo.
- **dev** — búsqueda de hiperparámetros, screening de backbones, selección de modelo,
  temperatura, elección del umbral de revisión y la puerta de cuantización.
- **test** — exactamente una evaluación final auditada por release candidate. Solo se lee
  mediante un `TestAccessGrant` emitido a `stc evaluate --split test --final` y a
  `stc report final`; cada lectura añade una línea a `reports/test_access_log.jsonl`. Cualquier
  otro caso de uso recibe un `DevOnlyRepository` que lanza `TestSplitLocked` (código de salida 6)
  para test.

La varianza entre semillas se mide en **dev** (tres semillas), nunca en test.

### Política de limpieza (D3)

`normalize_for_dedup` aplica, en este orden: `casefold`, `NFKD`, quitar las marcas combinantes,
quitar `[^\w\s]`, colapsar los espacios y `strip`. Las filas de train se ordenan por id entero y
tres etapas se ejecutan solo sobre train:

1. eliminar las filas cuyo texto normalizado aparece en dev o test (`OVERLAP_WITH_EVAL`);
2. eliminar todas las filas de un grupo cuyo texto normalizado tiene más de una intención
   (`LABEL_CONFLICT_DUPLICATE`);
3. conservar la fila de menor id de cada grupo de duplicados restante (`EXACT_DUPLICATE` para
   las demás).

Dev y test nunca se modifican. El `worker_id` del traductor no puede usarse para reagrupar
particiones: 19 de los 20 workers aparecen en más de una partición.

### Métricas y estadística (D2)

Métrica principal: macro-F1 sobre las clases con al menos un ejemplo real en la partición
evaluada, `zero_division=0`. Secundarias: exactitud, F1 ponderado, exactitud top-3, exactitud
por escenario y ECE/NLL/Brier. Intervalos de confianza: bootstrap percentil, B = 2000,
`numpy.random.default_rng(13)`, remuestreo pareado para el Δ entre línea base y modelo.
Significancia: McNemar bilateral exacto sobre la corrección por ejemplo.

## Alternativas

- **Volver a particionar el corpus desde cero**: descarta las particiones oficiales diseñadas
  por personas y hace que los resultados no sean comparables con las cifras publicadas de
  MASSIVE.
- **Deduplicar todas las particiones**: ocultaría la contaminación real en las cifras; el
  protocolo reporta la duplicación entre dev y test en lugar de borrarla.
- **Entrenar con todas las filas y reportar test después de ajustar**: filtra el conjunto de test
  a la selección del modelo y no puede auditarse.
- **Agrupar particiones por `worker_id`**: imposible sin perder la mayoría de los datos, porque
  los workers abarcan varias particiones.

## Consecuencias

- Train se reduce de 11,514 a 10,898 filas (616 eliminadas: 356 por solapamiento, 43 por conflicto
  de etiquetas y 217 duplicados exactos).
- Dev y test conservan 45 textos normalizados solapados y sus propios grupos de duplicados; esos
  números quedan versionados en `reports/data_audit.json` en lugar de ocultarse.
- La evaluación de test es costosa de repetir por diseño; las reejecuciones deben registrarse en el
  registro de evaluación (`docs/EVALUATION.md`).
- La comparación final es estadística: una pequeña ventaja del transformer puede ser un empate.

## Evidencia

- `uv run stc data prepare` → `train=10898 dev=2033 test=2974 dropped=616`.
- `uv run stc data audit --out reports` → conteos exactos de la lista de aceptación (dev∩test 45,
  dev en train 123 con 7 conflictos, test en train 152 con 6, 264 grupos de duplicados en train,
  19 grupos con conflicto de etiquetas, 18 grupos en dev y 31 en test).
- `tests/unit/domain/test_leakage.py` y `tests/unit/application/test_dev_only_repository.py`
  hacen cumplir la política y el bloqueo en la suite de tests.
