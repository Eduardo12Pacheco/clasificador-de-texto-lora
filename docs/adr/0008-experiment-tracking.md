# ADR-0008 — Seguimiento de experimentos

- **Estado:** Aceptado
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto

## Contexto

T7 ejecuta el screening, una referencia de fine-tuning completo, una búsqueda con Optuna y las
ejecuciones de semillas finales. La evidencia debe ser reproducible y revisable sin versionar
checkpoints grandes ni bases de datos. El proyecto es local-first y no debe requerir una cuenta ni un
servidor de seguimiento alojado. D9 fija un puerto y sus adaptadores; D5 exige que la búsqueda de
Optuna pueda reanudarse.

## Decisión

1. **Puerto `ExperimentTracker`** (`application/ports.py`) con cuatro adaptadores:
   - `JsonlTracker` (por defecto): un flujo de eventos JSONL por directorio de ejecución, sin dependencia
     extra y trivialmente comparable tras la recolección.
   - `MlflowTracker`: SQLite local (`sqlite:///artifacts/mlflow.db`), que almacena parámetros, métricas
     por época, etiquetas (git sha, backbone, semilla) y `run.json`.
   - `NullTracker`: Null Object para los comandos que se ejecutan sin seguimiento.
   - `CompositeTracker`: distribución a varios hijos con aislamiento de errores por hijo, para que un
     almacén de MLflow roto no pierda una ejecución de entrenamiento.
2. **`TrackerFactory` puro.** Interpreta `STC_TRACKER` (`jsonl`, `jsonl,mlflow`, `null`), rechaza los
   nombres desconocidos con `InvalidInputError` y recibe funciones constructoras desde el composition
   root. La capa de aplicación nunca importa adaptadores, lo que mantiene intactos los contratos
   hexagonales.
3. **El almacenamiento de Optuna es un archivo SQLite local** (`<out>/optuna.db`) con
   `load_if_exists=True`. `--resume` ejecuta solo los trials que faltan hasta el presupuesto solicitado;
   una segunda invocación imprime `0 new trials`.
4. **Solo se versionan resúmenes compactos.** `scripts/collect_experiment_reports.py` elimina las rutas
   absolutas y las cargas útiles grandes de los trials, escribe JSON determinista en
   `reports/experiments/` y copia las figuras de búsqueda. Las bases de datos, los checkpoints y los
   directorios de trials permanecen en `artifacts/`, que está ignorado.

## Alternativas

- Weights & Biases o un MLflow alojado: requieren cuenta o red y violan la restricción de cero cuentas.
- Logs simples en stdout: no se pueden consultar y son fáciles de perder.
- Versionar las bases de datos SQLite: son binarias, grandes y no revisables.

## Consecuencias

- Cada ejecución de entrenamiento tiene un flujo de eventos portable junto a sus artefactos, incluso
  cuando MLflow no está instalado.
- La recolección de reportes es un paso separado y determinista; T11 la valida con
  `scripts/validate_reports.py experiments`.
- Los fallos del tracker son visibles mediante `CompositeTracker.errors` en lugar de tumbar una
  ejecución de entrenamiento de varias horas.

## Evidencia

- `uv run pytest tests/unit/adapters/test_trackers.py -q`
- `uv run pytest tests/unit/application/test_tracker_factory.py -q`
- `uv run pytest tests/unit/scripts/test_collect_experiment_reports.py -q`
