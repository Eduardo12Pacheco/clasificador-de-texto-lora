export const STRINGS = {
  "es": {
    "app": {
      "title": "Clasificador de intenciones",
      "subtitle": "Enrutamiento de peticiones para asistentes virtuales en español",
      "skip": "Saltar al contenido"
    },
    "header": {
      "theme": "Cambiar tema",
      "language": "Cambiar idioma",
      "modelLoading": "Cargando modelo…"
    },
    "input": {
      "label": "¿Qué quieres hacer?",
      "placeholder": "Escribe una petición, por ejemplo: ponme música de Juanes",
      "hint": "Ctrl/Cmd + Enter para clasificar",
      "counter": "{count}/{max}"
    },
    "actions": {
      "classify": "Clasificar",
      "clear": "Limpiar",
      "retry": "Reintentar"
    },
    "examples": {
      "label": "Ejemplos"
    },
    "empty": {
      "title": "Sin resultados todavía",
      "body": "Escribe una petición o elige un ejemplo para ver la intención, el escenario y la confianza calibrada."
    },
    "loading": {
      "label": "Clasificando…"
    },
    "error": {
      "title": "No se pudo clasificar",
      "unreachable": "No se puede contactar con el servidor local. Comprueba que «stc serve» sigue en marcha.",
      "model": "El modelo no está instalado. Ejecuta «stc model pull» y vuelve a intentarlo.",
      "generic": "El servidor devolvió un error. Puedes reintentar la petición.",
      "service": "El servicio no está disponible en este momento."
    },
    "result": {
      "label": "Resultado",
      "topIntent": "Intención principal",
      "confidence": "Confianza calibrada",
      "top3": "Tres mejores opciones",
      "review": "Confianza baja: conviene revisar manualmente",
      "latency": "{ms} ms en CPU",
      "scenario": "Escenario"
    },
    "compare": {
      "toggle": "Comparar con el baseline",
      "title": "Comparación de modelos",
      "default": "Transformer LoRA",
      "baseline": "TF-IDF + lineal",
      "agree": "Ambos modelos coinciden",
      "disagree": "Los modelos no coinciden",
      "confidence": "Confianza: {value}"
    },
    "modelPanel": {
      "summary": "Información del modelo",
      "id": "Identificador",
      "version": "Versión",
      "quantization": "Cuantización",
      "parameters": "Parámetros",
      "license": "Licencia",
      "temperature": "Temperatura (T)",
      "threshold": "Umbral de revisión (τ)",
      "ece": "ECE en dev (antes → después)",
      "baseline": "Baseline disponible",
      "yes": "Sí",
      "no": "No",
      "unavailable": "No disponible"
    },
    "footer": {
      "license": "Código MIT · Pesos Apache-2.0",
      "data": "Datos: Amazon MASSIVE es-ES (CC BY 4.0)",
      "repo": "Repositorio"
    },
    "offline": {
      "banner": "Sin conexión a la red. La aplicación local sigue funcionando si el servidor está activo."
    },
    "scenarios": {
      "alarm": "Alarmas y temporizadores",
      "audio": "Audio y volumen",
      "calendar": "Calendario",
      "cooking": "Cocina",
      "datetime": "Fecha y hora",
      "email": "Correo electrónico",
      "general": "General",
      "home": "Hogar",
      "iot": "Domótica",
      "lists": "Listas",
      "music": "Música",
      "news": "Noticias",
      "play": "Reproducción",
      "qa": "Preguntas y respuestas",
      "recommendation": "Recomendaciones",
      "social": "Redes sociales",
      "takeaway": "Comida a domicilio",
      "transport": "Transporte"
    }
  },
  "en": {
    "app": {
      "title": "Intent classifier",
      "subtitle": "Request routing for Spanish virtual assistants",
      "skip": "Skip to content"
    },
    "header": {
      "theme": "Toggle theme",
      "language": "Switch language",
      "modelLoading": "Loading model…"
    },
    "input": {
      "label": "What do you want to do?",
      "placeholder": "Type a request, for example: play some Juanes",
      "hint": "Ctrl/Cmd + Enter to classify",
      "counter": "{count}/{max}"
    },
    "actions": {
      "classify": "Classify",
      "clear": "Clear",
      "retry": "Retry"
    },
    "examples": {
      "label": "Examples"
    },
    "empty": {
      "title": "No results yet",
      "body": "Type a request or pick an example to see the intent, the scenario and the calibrated confidence."
    },
    "loading": {
      "label": "Classifying…"
    },
    "error": {
      "title": "Classification failed",
      "unreachable": "The local server cannot be reached. Check that «stc serve» is still running.",
      "model": "The model is not installed. Run «stc model pull» and try again.",
      "generic": "The server returned an error. You can retry the request.",
      "service": "The service is unavailable right now."
    },
    "result": {
      "label": "Result",
      "topIntent": "Top intent",
      "confidence": "Calibrated confidence",
      "top3": "Top 3 candidates",
      "review": "Low confidence: review this prediction manually",
      "latency": "{ms} ms on CPU",
      "scenario": "Scenario"
    },
    "compare": {
      "toggle": "Compare with baseline",
      "title": "Model comparison",
      "default": "LoRA transformer",
      "baseline": "TF-IDF + linear",
      "agree": "Both models agree",
      "disagree": "Models disagree",
      "confidence": "Confidence: {value}"
    },
    "modelPanel": {
      "summary": "Model information",
      "id": "Identifier",
      "version": "Version",
      "quantization": "Quantization",
      "parameters": "Parameters",
      "license": "License",
      "temperature": "Temperature (T)",
      "threshold": "Review threshold (τ)",
      "ece": "Dev ECE (before → after)",
      "baseline": "Baseline available",
      "yes": "Yes",
      "no": "No",
      "unavailable": "Unavailable"
    },
    "footer": {
      "license": "Code MIT · Weights Apache-2.0",
      "data": "Data: Amazon MASSIVE es-ES (CC BY 4.0)",
      "repo": "Repository"
    },
    "offline": {
      "banner": "No network connection. The local application keeps working while the server is running."
    },
    "scenarios": {
      "alarm": "Alarms and timers",
      "audio": "Audio and volume",
      "calendar": "Calendar",
      "cooking": "Cooking",
      "datetime": "Date and time",
      "email": "Email",
      "general": "General",
      "home": "Home",
      "iot": "Smart home",
      "lists": "Lists",
      "music": "Music",
      "news": "News",
      "play": "Playback",
      "qa": "Questions and answers",
      "recommendation": "Recommendations",
      "social": "Social media",
      "takeaway": "Takeaway food",
      "transport": "Transport"
    }
  }
};
