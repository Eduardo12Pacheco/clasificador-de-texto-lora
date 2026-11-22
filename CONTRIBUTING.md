# Contribuir

Gracias por echarle un vistazo. Este repositorio es un proyecto de portafolio local-first, así que el flujo de
contribución es deliberadamente pequeño y reproducible.

## Configuración del entorno de desarrollo

Requisitos: Python 3.12 (gestionado con `uv`), git y, opcionalmente, una GPU NVIDIA para los tests de
entrenamiento.

```bash
git clone https://github.com/Eduardo12Pacheco/clasificador-de-texto-lora.git
cd clasificador-de-texto-lora
uv sync --locked --group dev
uv run stc --version
```

`uv.lock` está versionado. Actualízalo de forma deliberada con `uv lock` y explica los cambios de dependencias
en el pull request.

## La puerta de calidad

Todo cambio debe pasar la misma puerta que ejecuta CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run lint-imports
uv run pytest -q -m "not e2e and not torch"
```

Añade cobertura cuando el cambio toque `src/stc`:

```bash
uv run pytest -q --cov=src/stc --cov-fail-under=85
```

Los tests de entrenamiento y de navegador son opcionales porque necesitan paquetes adicionales:

```bash
uv sync --locked --extra train --group dev
uv run pytest -q -m torch

uv run playwright install chromium
uv run pytest -q -m e2e
```

## Test-first y límites

Lee `docs/ARCHITECTURE.md` antes de cambiar una capa y `docs/EVALUATION.md` antes de cambiar una métrica o una
regla de particiones. El pipeline de GPU está documentado en `docs/TRAINING.md`.

- Escribe primero el test que falla y después la implementación más pequeña que lo haga pasar.
- Mantén el hexágono intacto: `domain` es Python puro más NumPy, `application` es responsable de los puertos y
  los casos de uso, y los adaptadores contienen la E/S. `lint-imports` y `tests/architecture/test_layers.py`
  hacen cumplir la dirección.
- No debilites, no omitas y no borres un test para lograr una ejecución en verde.
- Ejecuta `uv run python scripts/check_no_weights.py` antes de hacer commit: los pesos, los conjuntos de datos y
  los archivos grandes nunca entran en git.

## Commits

Conventional Commits en inglés, una unidad coherente por commit (unas 400 líneas cambiadas o menos). Incluye los
tests y la documentación junto con el comportamiento que describen. No añadas líneas de atribución a la IA. Parte
de `main`; no hagas force-push.

## Releases

Las releases se disparan con tags y los assets del modelo se adjuntan de forma manual. Nunca subas pesos desde CI.

1. Confirma que la versión en `pyproject.toml` y el bundle del modelo en `src/stc/model_registry.json` coinciden
   con la release que vas a publicar.
2. Sube `main` y, después, crea y sube el tag anotado:

   ```bash
   git tag -a v0.1.0 -m "v0.1.0"
   git push origin main
   git push origin v0.1.0
   ```

3. El workflow `Release` comprueba que el tag coincide con la versión del paquete, construye el wheel y el sdist,
   genera las notas de la release a partir de los Conventional Commits con git-cliff, escribe `SHA256SUMS` y crea
   la GitHub Release. Rechaza cualquier asset de más de 100 MB.
4. Adjunta a esa release el bundle del modelo verificado:

   ```bash
   scripts/publish_model.sh v0.1.0
   ```

   El helper requiere un `gh` autenticado, una release existente y un bundle cuyo sha256 sea igual al del
   registro. Sube `dist-model/stc-es-intent-*.zip` y `dist-model/SHA256SUMS-model.txt`; nunca toca git.
5. El workflow `Verify release` se ejecuta sobre la release publicada (y puede lanzarse con un tag): instala el
   wheel publicado en Linux, Windows y macOS, espera el asset del modelo, descarga y verifica el bundle,
   comprueba `tests/fixtures/smoke_phrases.json` y ejecuta `stc doctor`.

La instalación local se comporta como el paquete publicado:

```bash
bash scripts/smoke_install.sh --check   # prerequisites only
bash scripts/smoke_install.sh           # build, install, exercise, verify torch is absent
bash scripts/smoke_install.sh --uninstall
pwsh ./scripts/smoke_install.ps1 -Check # Windows equivalent
```

La instalación de humo usa un `UV_TOOL_DIR` y un `STC_HOME` temporales, así que nunca reemplaza tu `stc` normal.
CI ejecuta los mismos scripts en el job de matriz `smoke-install`.

Las comprobaciones de seguridad previas a una release son `gitleaks git --no-banner`, la auditoría de dependencias
descrita en `SECURITY.md` y `uv run python scripts/validate_workflows.py`.
