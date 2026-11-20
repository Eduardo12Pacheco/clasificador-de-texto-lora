# Error analysis — audited test split

This document analyses the 603 utterances in `reports/final/errors.json` where at
least one of the two shipped classifiers is wrong: the TF-IDF baseline and the
LoRA transformer. The LoRA model makes 397 errors (13.3 % of the 2,974 test
utterances); the baseline makes 473 (15.9 %). Both models are wrong on 267
utterances, only the baseline is wrong on 206 and only the LoRA model is wrong
on 130. Confidences below are calibrated.

The categories are qualitative diagnoses, not mutually exclusive partitions:
one utterance can be ambiguous and mislabelled at the same time. Every cited
`ex-<id>` exists in `reports/final/errors.json` and is checked by
`scripts/check_error_analysis.py`.

## Top-15 confusions (LoRA model; baseline count for the same pair)

| # | True → predicted | LoRA | Baseline |
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

### Open-domain intents versus news and factoid QA

`general_quirky` is MASSIVE's catch-all for requests that do not fit a concrete
skill, and the model repeatedly routes those utterances to the nearest concrete
skill. `ex-6301` ("qué está pasando") becomes `news_query` with 0.96 confidence;
`ex-6382` ("oye esta pasando algo interesante en los deportes ultimamente")
also becomes `news_query`; `ex-6448` ("trump") becomes `news_query`; and
`ex-6405` ("que sonido hace un perro") becomes `qa_factoid`. The boundary is
genuinely fuzzy without a conversation context, but the calibrated confidence is
high, which makes this class of error dangerous for automatic routing.

**Mitigation.** Add a `general_quirky`-vs-concrete-skill decision feature (for
example a small set of hand-written negative examples per skill in the training
data), or lower the confidence of `general_quirky` predictions so the abstention
policy sends them to review. Both changes must be validated on dev and would be
a new release candidate.

### Calendar action versus calendar query

The largest topical cluster is the calendar family. `ex-6962` ("puedo saber lo
que tengo yo que hacer el próximo sabado a las seis de la tarde") is a query but
is classified as `calendar_set`; `ex-6968` asks to confirm meetings and send
invitations, an action, but is classified as `calendar_query`; `ex-7219`
("pusiste el recordatorio sobre la reunión de mañana") is a query about a past
action and is classified as `calendar_set`; `ex-7569` ("planificar para mañana")
is an action but becomes `calendar_query`. Verb tense and the presence of a
confirmation question are the discriminative cues the encoder misses.

**Mitigation.** Add rule-based features for interrogative markers ("puedo",
"qué tengo", "cómo será") and imperative markers ("pon", "planifica",
"confirma") in a small post-processing layer, or oversample the minority action
formulations during fine-tuning.

### Music and playback ontology

Several intents share the same surface vocabulary. `ex-1710` ("cambiar el modo
de música a rock") is `music_settings` but becomes `play_music`; `ex-4268`
("cambia a la siguiente canción de la lista") is also `music_settings` and
becomes `play_music`; `ex-5911` ("reproducir la música") is a generic playback
control and is classified as `play_music`; `ex-2365` ("que canción hay en la
radio") is a query but becomes `play_radio`; and `ex-2647` (creating a Pandora
channel) becomes `play_radio` instead of `music_settings`. The radio/streaming
entities dominate the lexical signal.

**Mitigation.** Add the service name (Pandora, Spotify, radio) as an explicit
feature, or train a hierarchical head that first predicts the scenario
(`music`, `play`) and then the intent; the scenario head is already almost
perfect and would constrain the intent.

### IoT Hue light action scope

The light intents differ only by the target attribute (on/off, up/down,
change/dim). `ex-1071` ("encender las luces del pasillo") is `iot_hue_lightup`
but becomes `iot_hue_lighton`; `ex-1756` ("enciende las luces") shows the same
collapse; `ex-2128` ("el color de la luz brillo bajo") is `iot_hue_lightdim` but
becomes `iot_hue_lightchange`; `ex-2263` ("enseñame un color suave") and
`ex-2264` ("luz de color para la habitación de estudio") are colour-change
requests classified as dim/up. The training data uses "encender" for both
`lighton` and `lightup`, so the distinction is partly annotator-dependent.

**Mitigation.** Treat the light family as a two-stage decision (power vs
attribute) or merge `lighton`/`lightup` if the product accepts that semantic
loss; document the choice in a new ADR instead of silently accepting the
confusion.

### News versus structured question answering

`news_query` and the `qa_*` skills overlap whenever a news request contains a
number. `ex-281` ("cuál es el tipo de cambio del dólar estadounidense a la libra
esterlina") is `news_query` but becomes `qa_currency` with 0.996 confidence;
`ex-451` ("ponme lo último sobre el precio de las acciones de mercadona")
becomes `qa_stock`; `ex-680` ("mantenme informado sobre el precio del bitcoin")
becomes `qa_stock`; `ex-1423` ("dime sobre pedro sánchez") becomes
`qa_factoid`. The label depends on whether the user wants a current update or a
single fact, which the utterance rarely states.

**Mitigation.** Add a "latest/current/update" temporal cue feature and a
`news_query` vs `qa_*` binary gate before the 60-way head. Also lower the
confidence threshold for these pairs in the review policy.

### Email send versus contact management

`ex-15828` ("aquí tienes una nueva dirección de correo electrónico para añadir a
los contactos") is `email_sendemail` but becomes `email_addcontact`;
`ex-16131` ("este nuevo correo electrónico debería agregarse a mi contacto") is
the same pattern; `ex-15966` ("envía una respuesta si es muy importante") is
`email_query` but becomes `email_sendemail`; `ex-16225` ("abre contactos") is
`email_addcontact` but becomes `email_querycontact`. The contact-book vocabulary
is the confounding signal.

**Mitigation.** Add an explicit "add/guardar contacto" vs "enviar/responder"
pattern feature, and include more contact-management examples in the fixture
used for regression checks.

### High-confidence errors as label-noise candidates

These errors have calibrated confidence ≥ 0.94, so either the model is
overconfident or the gold label is questionable. They are the first candidates
for a manual label audit; the last column states the most likely explanation.

| Example | Text | Gold → predicted | Confidence | Reading |
|---|---|---|---|---|
| `ex-612` | por favor borra la alarma despertador a las ocho de la mañana el próximo sábado | `alarm_set` → `alarm_remove` | 0.985 | text asks to delete; gold is likely wrong |
| `ex-1639` | por favor sube el volumen | `audio_volume_down` → `audio_volume_up` | 0.980 | text asks to raise; gold is likely wrong |
| `ex-5911` | reproducir la música | `music_settings` → `play_music` | 0.987 | generic play request; label is arguable |
| `ex-555` | jugar mf doom cualquier cosa | `play_music` → `play_game` | 0.935 | "jugar" and DOOM are game cues; gold is likely wrong |
| `ex-3205` | jugar poker face de lady gaga | `play_music` → `play_game` | 0.938 | "jugar" is a distractor; gold may be right |
| `ex-2780` | hay algún concierto próximo de | `music_likeness` → `recommendation_events` | 0.974 | truncated event query; gold is likely wrong |
| `ex-4166` | cambia el nivel del volumen al maximo | `audio_volume_other` → `audio_volume_up` | 0.944 | "al maximo" is closer to up; label is ambiguous |
| `ex-50` | siri dime la hora en g. m. t. más cinco | `datetime_convert` → `datetime_query` | 0.912 | arithmetic/time-zone conversion; gold is arguable |
| `ex-1744` | qué piensas | `music_likeness` → `general_quirky` | 0.961 | free-form opinion; `general_quirky` is defensible |
| `ex-6301` | qué está pasando | `general_quirky` → `news_query` | 0.964 | context-free; both labels are plausible |
| `ex-451` | ponme lo último sobre el precio de las acciones de mercadona | `news_query` → `qa_stock` | 0.991 | "lo último" is a news cue; gold may be right |
| `ex-281` | cuál es el tipo de cambio del dólar estadounidense a la libra esterlina | `news_query` → `qa_currency` | 0.996 | pure currency question; gold is likely wrong |

**Mitigation.** Audit these 12 utterances against the upstream MASSIVE
annotation; if the gold label is wrong, keep them in the test split (do not
edit test data) but document the expected noise floor. For the ambiguous cases,
the review threshold already sends lower-confidence variants to manual review.

### Short and context-free utterances

Utterances of one to three tokens carry little evidence: `ex-46` ("que pasa")
is `general_greet` but becomes `calendar_query`; `ex-9291` ("pon") is
`play_radio` but becomes `play_music`; `ex-10333` ("abre saga") is `play_game`
but becomes `play_audiobook`; `ex-10828` ("borrar todo") is `lists_remove` but
becomes `calendar_remove`. The baseline is also wrong on most of these, so the
gap is not transformer-specific.

**Mitigation.** For very short inputs, prefer the abstention path and ask the
user for a clarifying follow-up instead of routing silently; the API and UI
already expose `needs_review` for this purpose.

## What the analysis does not show

These categories are drawn from a single test split of a synthetic,
human-localised corpus. They do not prove that the model fails on real
Latin-American Spanish; the dialect probe in `reports/final/robustness.json` is
synthetic and is labelled as such in ADR-0009. None of the mitigations above is
implemented in v0.1.0: they are candidates for a future release.
