# Análisis de errores — conjunto de test auditado

Este documento analiza los 603 enunciados de `reports/final/errors.json` en los que al menos uno de los dos
clasificadores publicados se equivoca: la línea base TF-IDF y el transformer LoRA. El modelo LoRA comete 397
errores (13.3 % de los 2,974 enunciados de test); la línea base comete 473 (15.9 %). Ambos modelos fallan en
267 enunciados, solo la línea base falla en 206 y solo el modelo LoRA falla en 130. Las confianzas citadas son
calibradas.

Las categorías son diagnósticos cualitativos, no particiones mutuamente excluyentes: un mismo enunciado puede
ser ambiguo y tener además una etiqueta errónea. Cada `ex-<id>` citado existe en
`reports/final/errors.json` y lo comprueba `scripts/check_error_analysis.py`.

## Las 15 confusiones principales (modelo LoRA; conteo de la línea base para el mismo par)

| # | Real → predicha | LoRA | Línea base |
|---|---|---|---|
| 1 | `general_quirky` → `qa_factoid` | 14 | 25 |
| 2 | `general_quirky` → `calendar_query` | 11 | 10 |
| 3 | `calendar_query` → `calendar_set` | 10 | 14 |
| 4 | `play_radio` → `play_music` | 10 | 6 |
| 5 | `calendar_set` → `calendar_query` | 8 | 15 |
| 6 | `general_quirky` → `news_query` | 8 | 9 |
| 7 | `qa_factoid` → `general_quirky` | 6 | 8 |
| 8 | `play_game` → `play_music` | 5 | 0 |
| 9 | `play_music` → `music_query` | 5 | 0 |
| 10 | `weather_query` → `general_quirky` | 5 | 0 |
| 11 | `calendar_query` → `general_quirky` | 4 | 4 |
| 12 | `email_sendemail` → `email_addcontact` | 4 | 0 |
| 13 | `general_quirky` → `calendar_set` | 4 | 0 |
| 14 | `iot_hue_lightup` → `iot_hue_lighton` | 4 | 0 |
| 15 | `news_query` → `qa_factoid` | 4 | 0 |

### Intenciones de dominio abierto frente a noticias y QA factual

`general_quirky` es el cajón de sastre de MASSIVE para las peticiones que no encajan en una habilidad concreta,
y el modelo dirige esos enunciados, de forma repetida, a la habilidad concreta más cercana. `ex-6301` ("qué está
pasando") pasa a `news_query` con 0.96 de confianza; `ex-6382` ("oye esta pasando algo interesante en los
deportes ultimamente") también pasa a `news_query`; `ex-6448` ("trump") pasa a `news_query`, y `ex-6405` ("que
sonido hace un perro") pasa a `qa_factoid`. La frontera es realmente difusa sin contexto de conversación, pero la
confianza calibrada es alta, lo que hace peligrosa esta clase de error para el enrutamiento automático.

**Mitigación.** Añadir una característica de decisión `general_quirky` frente a habilidad concreta (por ejemplo,
un pequeño conjunto de ejemplos negativos escritos a mano por habilidad en los datos de entrenamiento), o bajar la
confianza de las predicciones de `general_quirky` para que la política de abstención las envíe a revisión. Ambos
cambios deben validarse en dev y darían lugar a un nuevo release candidate.

### Acción de calendario frente a consulta de calendario

El cúmulo temático más grande es la familia de calendario. `ex-6962` ("puedo saber lo que tengo yo que hacer el
próximo sabado a las seis de la tarde") es una consulta, pero se clasifica como `calendar_set`; `ex-6968` pide
confirmar reuniones y enviar invitaciones, es decir, una acción, pero se clasifica como `calendar_query`; `ex-7219`
("pusiste el recordatorio sobre la reunión de mañana") es una consulta sobre una acción pasada y se clasifica como
`calendar_set`; `ex-7569` ("planificar para mañana") es una acción pero pasa a `calendar_query`. El tiempo verbal y
la presencia de una pregunta de confirmación son las pistas discriminativas que el encoder no capta.

**Mitigación.** Añadir características basadas en reglas para marcadores interrogativos ("puedo", "qué tengo",
"cómo será") y marcadores imperativos ("pon", "planifica", "confirma") en una pequeña capa de postprocesado, o
sobremuestrear las formulaciones minoritarias de acción durante el fine-tuning.

### Ontología de música y reproducción

Varias intenciones comparten el mismo vocabulario superficial. `ex-1710` ("cambiar el modo de música a rock") es
`music_settings`, pero pasa a `play_music`; `ex-4268` ("cambia a la siguiente canción de la lista") también es
`music_settings` y pasa a `play_music`; `ex-5911` ("reproducir la música") es un control genérico de reproducción y
se clasifica como `play_music`; `ex-2365` ("que canción hay en la radio") es una consulta, pero pasa a
`play_radio`, y `ex-2647` (crear un canal de Pandora) pasa a `play_radio` en lugar de `music_settings`. Las
entidades de radio y de streaming dominan la señal léxica.

**Mitigación.** Añadir el nombre del servicio (Pandora, Spotify, radio) como característica explícita, o entrenar
una cabeza jerárquica que prediga primero el escenario (`music`, `play`) y después la intención; la cabeza de
escenario ya es casi perfecta y restringiría la intención.

### Alcance de las acciones de luces IoT de Hue

Las intenciones de luz solo se diferencian por el atributo objetivo (encender o apagar, subir o bajar, cambiar o
atenuar). `ex-1071` ("encender las luces del pasillo") es `iot_hue_lightup`, pero pasa a `iot_hue_lighton`;
`ex-1756` ("enciende las luces") muestra el mismo colapso; `ex-2128` ("el color de la luz brillo bajo") es
`iot_hue_lightdim`, pero pasa a `iot_hue_lightchange`; `ex-2263` ("enseñame un color suave") y `ex-2264` ("luz de
color para la habitación de estudio") son peticiones de cambio de color clasificadas como atenuar o subir. Los
datos de entrenamiento usan "encender" tanto para `lighton` como para `lightup`, así que la distinción depende en
parte de quien anota.

**Mitigación.** Tratar la familia de luces como una decisión en dos etapas (encendido frente a atributo), o fusionar
`lighton` y `lightup` si el producto acepta esa pérdida semántica; documentar la elección en una ADR nueva en lugar
de aceptar la confusión en silencio.

### Noticias frente a respuestas estructuradas a preguntas

`news_query` y las habilidades `qa_*` se solapan siempre que una petición de noticias contiene un número. `ex-281`
("cuál es el tipo de cambio del dólar estadounidense a la libra esterlina") es `news_query`, pero pasa a
`qa_currency` con 0.996 de confianza; `ex-451` ("ponme lo último sobre el precio de las acciones de mercadona")
pasa a `qa_stock`; `ex-680` ("mantenme informado sobre el precio del bitcoin") pasa a `qa_stock`; `ex-1423`
("dime sobre pedro sánchez") pasa a `qa_factoid`. La etiqueta depende de si el usuario quiere una actualización
actual o un hecho puntual, algo que el enunciado rara vez indica.

**Mitigación.** Añadir una característica de señal temporal ("lo último", "actual", "actualización") y una compuerta
binaria `news_query` frente a `qa_*` antes de la cabeza de 60 clases. Además, bajar el umbral de confianza para
estos pares en la política de revisión.

### Envío de correo frente a gestión de contactos

`ex-15828` ("aquí tienes una nueva dirección de correo electrónico para añadir a los contactos") es
`email_sendemail`, pero pasa a `email_addcontact`; `ex-16131` ("este nuevo correo electrónico debería agregarse a
mi contacto") sigue el mismo patrón; `ex-15966` ("envía una respuesta si es muy importante") es `email_query`, pero
pasa a `email_sendemail`; `ex-16225` ("abre contactos") es `email_addcontact`, pero pasa a `email_querycontact`. El
vocabulario de la agenda de contactos es la señal confusora.

**Mitigación.** Añadir una característica explícita de patrón "añadir o guardar contacto" frente a "enviar o
responder", e incluir más ejemplos de gestión de contactos en la muestra que se usa para las comprobaciones de
regresión.

### Errores de alta confianza como candidatos a ruido en las etiquetas

Estos errores tienen una confianza calibrada ≥ 0.94, así que o el modelo está sobreconfiado o la etiqueta de
referencia es cuestionable. Son los primeros candidatos para una auditoría manual de etiquetas; la última columna
indica la explicación más probable.

| Ejemplo | Texto | Referencia → predicha | Confianza | Lectura |
|---|---|---|---|---|
| `ex-612` | por favor borra la alarma despertador a las ocho de la mañana el próximo sábado | `alarm_set` → `alarm_remove` | 0.985 | el texto pide borrar; la referencia probablemente es errónea |
| `ex-1639` | por favor sube el volumen | `audio_volume_down` → `audio_volume_up` | 0.980 | el texto pide subir; la referencia probablemente es errónea |
| `ex-5911` | reproducir la música | `music_settings` → `play_music` | 0.987 | petición genérica de reproducción; la etiqueta es discutible |
| `ex-555` | jugar mf doom cualquier cosa | `play_music` → `play_game` | 0.935 | "jugar" y DOOM son pistas de juego; la referencia probablemente es errónea |
| `ex-3205` | jugar poker face de lady gaga | `play_music` → `play_game` | 0.938 | "jugar" es un distractor; la referencia puede ser correcta |
| `ex-2780` | hay algún concierto próximo de | `music_likeness` → `recommendation_events` | 0.974 | consulta de evento truncada; la referencia probablemente es errónea |
| `ex-4166` | cambia el nivel del volumen al maximo | `audio_volume_other` → `audio_volume_up` | 0.944 | "al maximo" se acerca más a subir; la etiqueta es ambigua |
| `ex-50` | siri dime la hora en g. m. t. más cinco | `datetime_convert` → `datetime_query` | 0.912 | conversión aritmética de husos horarios; la referencia es discutible |
| `ex-1744` | qué piensas | `music_likeness` → `general_quirky` | 0.961 | opinión de forma libre; `general_quirky` es defendible |
| `ex-6301` | qué está pasando | `general_quirky` → `news_query` | 0.964 | sin contexto; ambas etiquetas son plausibles |
| `ex-451` | ponme lo último sobre el precio de las acciones de mercadona | `news_query` → `qa_stock` | 0.991 | "lo último" es una pista de noticias; la referencia puede ser correcta |
| `ex-281` | cuál es el tipo de cambio del dólar estadounidense a la libra esterlina | `news_query` → `qa_currency` | 0.996 | pregunta de divisas pura; la referencia probablemente es errónea |

**Mitigación.** Auditar estos 12 enunciados frente a la anotación original de MASSIVE; si la referencia es errónea,
mantenerlos en el conjunto de test (sin editar los datos de test), pero documentar el suelo de ruido esperado. En
los casos ambiguos, el umbral de revisión ya envía a revisión manual las variantes de menor confianza.

### Enunciados cortos y sin contexto

Los enunciados de una a tres palabras aportan poca evidencia: `ex-46` ("que pasa") es `general_greet`, pero pasa a
`calendar_query`; `ex-9291` ("pon") es `play_radio`, pero pasa a `play_music`; `ex-10333` ("abre saga") es
`play_game`, pero pasa a `play_audiobook`; `ex-10828` ("borrar todo") es `lists_remove`, pero pasa a
`calendar_remove`. La línea base también falla en la mayoría de estos casos, así que la brecha no es específica del
transformer.

**Mitigación.** Para entradas muy cortas, preferir la ruta de abstención y pedir al usuario una aclaración en lugar
de enrutar en silencio; la API y la UI ya exponen `needs_review` para ese fin.

## Lo que el análisis no muestra

Estas categorías provienen de un único conjunto de test de un corpus sintético localizado por personas. No demuestran
que el modelo falle con el español latinoamericano real; la sonda dialectal de `reports/final/robustness.json` es
sintética y así se etiqueta en la ADR-0009. Ninguna de las mitigaciones anteriores está implementada en la v0.1.0:
son candidatas para una versión futura.
