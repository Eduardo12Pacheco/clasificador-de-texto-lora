# Política de seguridad

## Versiones soportadas

| Versión | Soportada |
|---------|-----------|
| 0.1.x   | Sí        |

## Reportar una vulnerabilidad

No abras una incidencia pública por una posible vulnerabilidad. Usa el reporte privado de vulnerabilidades de
GitHub ("Report a vulnerability" en la pestaña Security) en `Eduardo12Pacheco/clasificador-de-texto-lora`, o el
formulario privado de Security Advisory del repositorio. Incluye la versión afectada, los pasos para reproducirla
y el impacto. Nuestro objetivo es acusar recibo de los reportes en un plazo de siete días.

## Modelo de amenazas

El proyecto es una herramienta local-first. Descarga un conjunto de datos público y un bundle de modelo público
solo con comandos explícitos, sirve una UI en loopback y nunca envía datos a terceros. Los siguientes activos y
límites están dentro del alcance.

### Bundles de modelo no confiables

- `src/stc/model_registry.json` fija la URL de la release, el tamaño y el sha256 del bundle publicado.
  `stc model pull` transmite la descarga a un archivo temporal y se niega a instalarla si el checksum o el tamaño no
  coinciden (`IntegrityError`, código de salida 3).
- Cada bundle descomprimido se verifica contra su `manifest.json`: cada archivo listado debe coincidir con su
  sha256 y tamaño registrados, los archivos inesperados se rechazan y las rutas no seguras (rutas absolutas, `..`,
  barras invertidas, enlaces simbólicos) se rechazan antes de extraer.
- `stc model verify` vuelve a comprobar un bundle instalado en cualquier momento.
- Los artefactos `skops` de la línea base se cargan mediante una lista de tipos permitidos restringida a
  scikit-learn, NumPy, SciPy y contenedores integrados; cualquier otro tipo lanza `UntrustedArtifactError`
  (código de salida 3) en lugar de deserializarse.
- La sesión de ONNX Runtime nunca habilita operadores personalizados ni descargas remotas de modelos.

### Servidor local

- `stc serve` enlaza `127.0.0.1:19080` por defecto (configurable con `STC_API_HOST` y `STC_API_PORT`; el rango de
  puertos 19000-19999 evita colisiones con los demás proyectos locales de esta máquina). Imprime una advertencia
  cuando se le pide enlazar una interfaz que no es loopback, y no es un servicio público endurecido.
- Las respuestas incluyen `Content-Security-Policy: default-src 'self'`, `X-Content-Type-Options: nosniff` y
  `Referrer-Policy: no-referrer`. La UI no hace peticiones a terceros y no carga fuentes ni scripts remotos.
- El texto de entrada nunca se registra; los logs de peticiones contienen solo longitudes, identificadores y
  tiempos. Los límites de validación (64 textos, 512 caracteres cada uno y `top_k` de 1 a 10) limitan el
  agotamiento accidental de recursos.

### Cadena de suministro e higiene del repositorio

- Las dependencias están fijadas en `uv.lock` y se auditan en CI con
  `uv export --locked --no-dev ... | uvx pip-audit --no-deps`.
- El historial completo de git se analiza con gitleaks. Las referencias de GitHub Actions las comprueban
  `scripts/validate_workflows.py` y actionlint, y los workflows piden los `permissions` mínimos.
- Los pesos del modelo, los conjuntos de datos y los archivos comprimidos nunca entran en git:
  `scripts/check_no_weights.py` falla si hay archivos versionados de más de 1 MB o con una extensión de peso o de
  bundle fuera de las pequeñas muestras de test.
- Las releases se disparan con tags. CI construye solo el wheel y el sdist y rechaza assets de más de 100 MB; el
  bundle del modelo verificado lo adjunta después el responsable con `scripts/publish_model.sh`, que primero
  comprueba el checksum del registro.

## Excepciones de la auditoría de dependencias

Actualmente no hay advisories ignorados. Si un advisory no puede corregirse de inmediato, el paquete exacto, el
identificador del advisory, la justificación y la fecha de revisión se registran aquí y en el commit que añade la
excepción, en lugar de desactivar la auditoría en silencio.

## Ejecutar las comprobaciones en local

```bash
uv sync --locked --group dev
uv run python scripts/check_no_weights.py
uv run python scripts/validate_workflows.py
uv export --locked --no-dev --no-hashes --format requirements-txt -o /tmp/req.txt
uvx pip-audit -r /tmp/req.txt --no-deps --disable-pip
gitleaks git --no-banner
```

Los pasos de release e instalación están documentados en `CONTRIBUTING.md` y los ejercita el job `smoke-install`
de CI.
