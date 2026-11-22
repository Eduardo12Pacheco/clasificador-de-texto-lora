# Registros de decisiones de arquitectura

Este directorio registra las decisiones que dan forma al proyecto. Cada ADR es inmutable
una vez aceptada; los cambios posteriores se añaden como addendum o como una ADR nueva.

## Índice

| ADR | Título | Estado |
|---|---|---|
| [0001](0001-hexagonal-architecture.md) | Arquitectura hexagonal con dirección de dependencias forzada | Aceptado |
| [0002](0002-task-and-dataset-selection.md) | Selección de tarea y conjunto de datos | Aceptado |
| [0003](0003-backbone-and-adaptation-method.md) | Backbone y método de adaptación | Aceptado |
| [0004](0004-evaluation-protocol-and-leakage-policy.md) | Protocolo de evaluación y política de fuga de datos | Aceptado |
| [0005](0005-onnx-runtime-inference-and-quantization.md) | Inferencia con ONNX Runtime y la puerta de cuantización | Aceptado |
| [0006](0006-calibration-and-abstention.md) | Calibración y abstención | Aceptado |
| [0007](0007-artifact-distribution-and-integrity.md) | Distribución de artefactos, bundles e integridad | Aceptado |
| [0008](0008-experiment-tracking.md) | Seguimiento de experimentos | Aceptado |
| [0009](0009-synthetic-dialect-robustness.md) | Sonda sintética de robustez dialectal | Aceptado |

Las ADR-0003, ADR-0005 y ADR-0006 incluyen addenda con resultados medidos de la ejecución
en GPU y de la evaluación final auditada. Los números de esos addenda provienen de los
archivos versionados en `reports/experiments` y `reports/final`; los comandos que los
regeneran están en `docs/TRAINING.md` y `docs/EVALUATION.md`.

Cada ADR usa la plantilla siguiente.

## Plantilla

```markdown
# ADR-NNNN — Título

- **Estado:** Propuesto | Aceptado | Reemplazado por ADR-NNNN
- **Fecha:** YYYY-MM-DD
- **Decisores:** responsable del proyecto

## Contexto

¿Qué fuerzas están en juego? ¿Cuál es el problema?

## Decisión

Qué elegimos, expresado de forma que pueda verificarse.

## Alternativas

Qué más se consideró y por qué se descartó.

## Consecuencias

Qué se vuelve más fácil, qué se vuelve más difícil y qué debe revisarse.

## Evidencia

Comandos, tests, mediciones o documentos que demuestran que la decisión sigue vigente.
```
