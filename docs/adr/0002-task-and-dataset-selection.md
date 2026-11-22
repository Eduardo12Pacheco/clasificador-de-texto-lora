# ADR-0002 — Selección de tarea y conjunto de datos

- **Estado:** Aceptado
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto

## Contexto

El proyecto necesita una tarea de clasificación de enunciados en español que sea lo bastante
pequeña para entrenar en una sola GPU de 8 GB, lo bastante grande para un análisis
significativo por clase y de calibración, y con una licencia que permita publicar código,
métricas y una muestra redistribuible. La mayoría de los conjuntos de datos en español
encontrados durante la investigación eran de uso no comercial, tenían una licencia ambigua,
usaban etiquetas derivadas automáticamente o eran demasiado pequeños para un estudio de 60
clases.

## Decisión

Usar **Amazon MASSIVE 1.0, locale `es-ES`** (CC BY 4.0), una versión en español localizada
por personas de SLURP:

- etiqueta principal: `intent` (60 clases, formato `scenario_intent`);
- etiqueta secundaria: `scenario` (18 clases, el prefijo de la intención antes del primer
  guion bajo, verificado igual al campo `scenario` de MASSIVE en las 16,521 filas);
- orden de etiquetas: intenciones ordenadas lexicográficamente y guardadas en `labels.json`;
- particiones: las particiones oficiales de train, dev y test (11,514 / 2,033 / 2,974 filas),
  con la política de limpieza solo sobre train de la ADR-0004.

El archivo fijado, su tamaño y su sha256 están en `src/stc/adapters/data/massive_archive.py`
y se documentan en `docs/DATA.md`. Solo se extraen tres miembros del archivo (`es-ES.jsonl`,
`LICENSE`, `NOTICE.md`); cualquier otro miembro se ignora y cualquier miembro inseguro
rechaza el archivo completo.

## Alternativas

| Candidato | Motivo de descarte |
|---|---|
| MeLiSA (reseñas de Mercado Libre, 8 países de LatAm) | Licencia "other" ligada a un acuerdo con Amazon, publicación solo para demostración; no redistribuible. |
| TASS / InterTASS | No se encontró declaración de licencia; distribución de tuits solo para investigación. |
| MLSUM (es) | Términos no comerciales; solo El País (España). |
| XL-Sum | CC BY-NC-SA 4.0; no comercial. |
| SIB-200 | Siete clases temáticas; demasiado pequeño para un estudio por clase. |
| WikiCAT_es v2 | Etiquetas derivadas automáticamente de las categorías de Wikipedia; metadatos de licencia inconsistentes. |
| MTEB SpanishNewsClassification | La procedencia original no puede verificarse. |
| cardiffnlp tweet sentiment | Sin metadatos de licencia; tuits. |

No existe un corpus en español con licencia abierta, redistribuible y con etiquetas naturales
de dialectos latinoamericanos, así que el proyecto no afirma precisión para LatAm; sondea la
sensibilidad dialectal con una suite de perturbaciones sintéticas (ADR-0009).

## Consecuencias

- La tarea es un problema realista de clasificación de 60 clases desbalanceado: entre 4 y 810
  filas de entrenamiento por clase (`cooking_query` tiene cuatro filas de train y ninguna en
  dev ni test).
- El conjunto de datos contiene duplicados reales entre particiones; la auditoría y la política
  de limpieza son obligatorias y quedan registradas en `reports/data_audit.json`.
- El español es solo el locale `es-ES`. Los resultados no deben presentarse como evidencia para
  el español latinoamericano, y la sonda dialectal sintética se etiqueta así en el README, la
  model card y los reportes.
- Redistribuir una muestra pequeña está permitido con atribución; la muestra incluida (fixture)
  trae `LICENSE-CC-BY-4.0.txt` y `NOTICE.md`.

## Evidencia

- `uv run stc data fetch` → `sha256 OK 7df623fd…153f8`, 16,521 filas, 60 intenciones, 18
  escenarios.
- `uv run stc data audit --out reports` → `reports/data_audit.json` reproduce dev∩test 45,
  dev en train 123, test en train 152.
- `uv run stc data fixture info` → 358 filas de train / 118 de dev / 117 de test.
