# ADR-0007 — Distribución de artefactos, bundles e integridad

- **Estado:** Aceptado
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto

## Contexto

El modelo ocupa ~574 MiB en fp32, muy por encima de lo que pertenece a git, y el proyecto se
distribuye como una herramienta de Python sin servicio alojado. Los usuarios deben poder obtener
exactamente los bytes que describen los reportes, detectar una descarga corrupta o manipulada, y
hacerlo sin conexión. El artefacto base es un objeto de sklearn serializado con pickle, lo que supone
un riesgo de ejecución de código si se carga desde una fuente no confiable.

## Decisión

1. **Layout del bundle (`bundle_version: 1`, D13).** Un directorio con `manifest.json`,
   `model.onnx` (o la variante publicada), `tokenizer.json`, `labels.json`, `calibration.json`,
   `baseline/` opcional, `MODEL_CARD.md`, `LICENSE` y `NOTICE`. El manifiesto registra las versiones
   del bundle y de la herramienta, los parámetros del modelo, el id, la revisión y la licencia del
   modelo base, la receta del adaptador, el id y los conteos del conjunto de datos, el op-set de ONNX
   y los nombres de entrada y salida, la configuración del tokenizer y una lista `files` con
   `{path, sha256, size}` para cada archivo de contenido.
2. **El manifiesto como ancla de confianza.** `verify_bundle` calcula el hash de cada archivo listado,
   rechaza discrepancias de tamaño o de checksum, rechaza rutas relativas inseguras e informa de los
   archivos inesperados en lugar de ignorarlos en silencio. Cargar un bundle lo verifica por defecto;
   la ruta explícita de variante de la CLI omite el escaneo del bundle porque las variantes de
   cuantización viven fuera del bundle.
3. **Releases deterministas.** `stc bundle pack` vuelve a calcular el hash del layout ensamblado,
   escribe entradas zip ordenadas con una marca de tiempo fija (1980-01-01) y deflate nivel 9, e
   imprime el sha256 y el tamaño. Dos empaquetados de las mismas entradas son idénticos byte a byte.
   El adaptador LoRA se publica como un segundo archivo.
4. **Registro versionado, pesos externos.** `src/stc/model_registry.json` almacena solo
   `{name, version, url, sha256, size_bytes, license}`. `stc model pull` descarga con `urllib`,
   transmite a un archivo temporal, verifica el sha256 y extrae con una protección contra zip-slip
   (rechaza rutas absolutas, `..`, barras invertidas y enlaces simbólicos). `STC_OFFLINE=1` rechaza
   las descargas de red con código de salida 5; `--from-file` sigue funcionando sin conexión. Los
   bundles instalados viven en `${STC_HOME}/models/<name>/<version>/`.
5. **Lista de tipos permitidos de skops.** La línea base se carga mediante `skops.io` con una lista de
   tipos restringida a sklearn, numpy, scipy.sparse y builtins; cualquier otro tipo lanza
   `UntrustedArtifactError` (código de salida 3). Su propio manifiesto calcula el hash de
   `baseline.skops` y de `labels.json`.
6. **Transporte de distribución.** El bundle es un asset de GitHub Release que adjunta
   `scripts/publish_model.sh` (T13); nunca lo hace CI ni va a git. GitHub permite assets de menos de
   2 GiB. `scripts/check_no_weights.py` falla si los pesos o los archivos comprimidos quedan
   versionados.

## Alternativas

- Git LFS: aun así hace pesados los clones y acopla los pesos al historial.
- Hugging Face Hub en v0.1.0: opcional más adelante; no es necesario para una herramienta local-first.
- Un tarball sin hashes por archivo: no hay forma de detectar un bundle reemplazado parcialmente.
- `pickle` para la línea base: permite ejecutar código arbitrario; descartado a favor de skops con
  lista de tipos permitidos.

## Consecuencias

- Cada modelo instalado es reproducible a partir de la entrada del registro y verificable sin conexión.
- El empaquetado es determinista, así que el sha256 del registro puede comparar una persona o un paso
  de CI.
- Cambiar el layout del bundle requiere subir `bundle_version` y añadir un nuevo addendum a esta ADR.
- Una descarga corrupta falla antes de instalar nada; un estado parcial no puede ocultar un bundle
  bueno porque la instalación copia en un árbol nuevo.

## Evidencia

- `uv run pytest tests/unit/adapters/test_bundle_io.py -q`
- `uv run pytest tests/unit/adapters/test_model_hub.py -q`
- `uv run pytest tests/unit/adapters/test_registry_and_store.py -q`
- `uv run stc model verify tests/fixtures/tiny_bundle`
- `uv run stc bundle pack --bundle tests/fixtures/tiny_bundle --out "$(mktemp -d)/a.zip"`
