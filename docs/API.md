# Guía de la API

`stc serve` inicia la aplicación FastAPI en la interfaz de loopback. El servicio carga el bundle
ONNX publicado una sola vez al arrancar y expone el mismo clasificador calibrado que `stc predict`.
Nunca descarga modelos ni datos.

## Iniciar el servicio

```bash
stc serve                       # 127.0.0.1 and STC_API_PORT (19080 by default)
stc serve --host 127.0.0.1 --port 19080 --model stc-es-intent
```

Si se enlaza una dirección que no es loopback, se imprime una advertencia: el servicio no tiene
autenticación y está pensado para uso local.

## Endpoints

| Método | Ruta | Propósito |
|---|---|---|
| `GET` | `/healthz` | Comprobación de vida más `model_loaded` |
| `GET` | `/v1/model` | Identidad del manifiesto, calibración y disponibilidad de la línea base |
| `GET` | `/v1/labels` | Las 60 intenciones y los 18 escenarios |
| `POST` | `/v1/predict` | Predicciones top-k calibradas para entre 1 y 64 textos |
| `GET` | `/` y `/static/*` | Interfaz web incluida (sin CDN y sin peticiones a terceros) |

### Predicción

```bash
curl -s -X POST http://127.0.0.1:19080/v1/predict \
  -H 'content-type: application/json' \
  -d '{"texts": ["apaga la luz", "despiértame a las cinco"], "top_k": 2, "model": "default"}'
```

```json
{
  "schema_version": 1,
  "model": {
    "id": "stc-es-intent",
    "kind": "onnx-seqcls",
    "version": "0.1.0",
    "quantization": "fp32",
    "temperature": 1.0,
    "review_threshold": 0.95
  },
  "predictions": [
    {
      "index": 0,
      "intent": "iot_hue_lightoff",
      "scenario": "iot",
      "confidence": 0.93,
      "needs_review": false,
      "top_k": [
        {"intent": "iot_hue_lightoff", "scenario": "iot", "probability": 0.93},
        {"intent": "iot_hue_lightdim", "scenario": "iot", "probability": 0.04}
      ]
    }
  ],
  "latency_ms": 18.4
}
```

El orden de entrada se conserva: `predictions[i].index == i`. `model` puede ser `"baseline"` cuando el
bundle instalado incluye una línea base; si no, la API responde `404`.

## Límites

- `texts`: de 1 a 64 elementos, cada uno de 1 a 512 caracteres tras quitar los espacios; los textos
  vacíos se rechazan.
- `top_k`: de 1 a 10, nunca mayor que el espacio de etiquetas.
- `model`: `"default"` o `"baseline"`.
- El cuerpo de la petición no puede contener campos desconocidos (`extra="forbid"`).

## Errores

Toda respuesta de error es RFC 7807 `application/problem+json`:

```json
{
  "type": "about:blank",
  "title": "invalid request",
  "status": 422,
  "detail": "request validation failed: body.texts"
}
```

| Estado | Causa |
|---|---|
| `404` | Se pidió `model="baseline"` pero no hay línea base instalada |
| `422` | Incumplimiento de límites o de esquema |
| `500` | Fallo inesperado (detalle genérico; sin texto de entrada) |
| `503` | Modelo no disponible, artefacto no seguro o rechazo por modo sin conexión |

## Seguridad y privacidad

- El servidor enlaza `127.0.0.1` por defecto; `STC_API_HOST` o `--host` lo cambian, y un valor que no
  es loopback registra una advertencia.
- Las respuestas incluyen `Content-Security-Policy: default-src 'self'`,
  `X-Content-Type-Options: nosniff` y `Referrer-Policy: no-referrer`.
- Cada respuesta repite o genera `X-Request-ID`.
- Los logs de acceso contienen método, ruta, estado, duración y `n_texts`; el texto de entrada nunca
  se registra.
- Los bundles de modelo se verifican contra el sha256 de su manifiesto antes de usarse.

## Snapshot de OpenAPI

`docs/openapi.json` lo genera `scripts/export_openapi.py`; el test de contrato
`tests/contract/test_openapi_snapshot.py` falla cuando el archivo se desvía.
