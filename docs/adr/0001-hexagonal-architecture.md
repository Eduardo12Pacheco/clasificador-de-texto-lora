# ADR-0001 — Arquitectura hexagonal con dirección de dependencias forzada

- **Estado:** Aceptado
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto

## Contexto

`spanish-text-classifier-lora` crece desde un modelo de clasificación puro hasta incluir un
pipeline de datos, una pila de entrenamiento, una suite de reportes, una CLI, una API HTTP
y una interfaz web local. Estas preocupaciones cambian a ritmos muy distintos: las reglas
de etiquetas y la política de fuga de datos deben permanecer estables y ser triviales de
probar, mientras que las elecciones de frameworks y bibliotecas (Typer, FastAPI, torch,
ONNX Runtime) son detalles reemplazables. Mezclarlas dificulta auditar el bloqueo del
conjunto de test, la política de fuga de datos y las métricas.

## Decisión

Usar una arquitectura hexagonal (puertos y adaptadores) con tres capas:

- `stc.domain` — objetos de valor y reglas puras (etiquetas, normalización de texto, fuga
  de datos, métricas, calibración). Solo importa la biblioteca estándar, `numpy` y otros
  módulos de `stc.domain`. Sin sistema de archivos, red, frameworks ni reloj.
- `stc.application` — casos de uso, puertos (`typing.Protocol`) y DTOs expresados como
  dataclasses. Orquesta el dominio y depende solo de abstracciones.
- `stc.adapters` — adaptadores concretos, conductores y conducidos (datos, modelos,
  inferencia, entrenamiento, exportación, seguimiento, reportes, hub, almacenamiento, CLI,
  API, web).

`stc.config`, `stc.logging` y `stc.bootstrap` quedan fuera de las capas. `bootstrap` es el
único composition root: los adaptadores nunca se importan entre sí y todo el cableado ocurre
allí, de modo que la dirección de dependencias es acíclica.

El conjunto de test queda protegido por un token de capacidad (`TestAccessGrant`) que solo
se emite a los casos de uso de evaluación final auditada. Así, la regla de "leer el conjunto
de test una sola vez" es una propiedad del dominio y no una convención en los scripts.

## Alternativas

- **MVC clásico de tres capas** — mapear controladores, modelos y servicios invita a importar
  frameworks en el núcleo y no ofrece un límite exigible; las reglas de fuga de datos y de
  bloqueo del test serían imposibles de probar de forma aislada.
- **Estructura única tipo script** — es lo más rápido para empezar, pero impide cambiar los
  adaptadores de entrenamiento, exportación e inferencia, y deja el dominio sin poder probarse
  sin dependencias pesadas instaladas.
- **Sin estructura forzada, solo convenciones documentadas** — las convenciones se degradan;
  el repositorio ya necesita tests del dominio que usen solo `numpy` y una ruta de runtime sin
  torch, y ambas cosas son fáciles de romper sin querer.

## Consecuencias

- Los tests del dominio se ejecutan en milisegundos sin extras opcionales instalados.
- La ruta de inferencia en CPU no puede importar `torch` por accidente: un contrato falla en
  lugar de que el usuario lo descubra en ejecución.
- Cierta indirección es inevitable: los casos de uso hablan con protocolos y DTOs, y los
  adaptadores se conectan en `bootstrap`. La indirección se limita a las costuras reales
  (fuente de datos, clasificador, exportador, tracker, almacén de modelos, reloj).
- Un nuevo adaptador debe registrarse en `bootstrap`, nunca ser importado por otro adaptador.

## Evidencia

Se hace cumplir de forma mecánica, no mediante revisión:

- `[tool.importlinter]` en `pyproject.toml` declara seis contratos: dirección de capas,
  aislamiento del dominio (sin capas externas ni bibliotecas de terceros), aislamiento de la
  aplicación, un conjunto de paquetes de runtime sin torch e independencia entre adaptadores.
  `uv run lint-imports` debe imprimir `0 broken`.
- `tests/architecture/test_layers.py` ejecuta `lint-imports` y, además, recorre el AST de cada
  módulo `src/stc/domain/*.py`, verificando que solo importe la biblioteca estándar, `numpy`
  y `stc.domain`.
- `tests/unit/domain/` ejercita las reglas puras sin ningún adaptador, base de datos, red ni
  dependencia opcional.
