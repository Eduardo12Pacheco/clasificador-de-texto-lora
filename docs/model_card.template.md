# Model card — Spanish intent classifier (`stc-es-intent`)

<!-- Every number in this card is injected from reports/summary.json by
     scripts/render_model_card.py. Prose is completed in T12; do not edit
     generated numbers by hand. -->

## Model details

- **Model:** `stc-es-intent` 0.1.0, ONNX sequence classifier served on CPU.
- **Base model:** `BSC-LT/MrBERT-es` (ModernBERT, Apache-2.0), adapted with LoRA.
- **Code licence:** MIT. **Weights licence:** Apache-2.0.
- **Data:** Amazon MASSIVE 1.0 `es-ES` (CC BY 4.0), a human-localised version
  of SLURP (CC BY 4.0).

## Intended use

Route short Spanish virtual-assistant utterances into one of 60 intents and
18 scenarios, locally and offline. The calibrated confidence and the
`needs_review` flag are part of the interface.

## Out-of-scope use

No natural Latin-American accuracy claim: the dialect probe is synthetic. Do
not use the model for safety-critical, legal, medical or financial decisions,
for long documents, or for languages other than Spanish.

## Training data

Amazon MASSIVE 1.0 `es-ES`, official train/dev/test partitions, cleaned with
the leakage policy of ADR-0004 (10,898 train rows after removing cross-split
overlaps, label-conflicting duplicates and exact duplicates). Dev was used for
all selection and calibration; test was read once and audited.

## Evaluation (audited test split)

| Model | Test accuracy (95 % CI) | Test macro-F1 (95 % CI) |
|---|---|---|
| TF-IDF baseline | {{models.baseline.test.accuracy}} [{{models.baseline.test.accuracy_lower}}, {{models.baseline.test.accuracy_upper}}] | {{models.baseline.test.macro_f1}} [{{models.baseline.test.macro_f1_lower}}, {{models.baseline.test.macro_f1_upper}}] |
| LoRA transformer | {{models.lora.test.accuracy}} [{{models.lora.test.accuracy_lower}}, {{models.lora.test.accuracy_upper}}] | {{models.lora.test.macro_f1}} [{{models.lora.test.macro_f1_lower}}, {{models.lora.test.macro_f1_upper}}] |

**Verdict:** {{comparison.verdict}}. {{comparison.verdict_sentence_en}}

Top-3 accuracy is {{models.lora.test.top3_accuracy}}; scenario accuracy is
{{models.lora.test.scenario_accuracy}}.

## Calibration and abstention

Temperature scaling uses T = {{models.lora.calibration.temperature}} and the
review threshold is τ = {{models.lora.calibration.review_threshold}}. Test ECE
moves from {{models.lora.calibration.ece_before}} to
{{models.lora.calibration.ece_after}}; at τ the model covers
{{models.lora.calibration.coverage_at_threshold}} of test utterances with
selective accuracy {{models.lora.calibration.selective_accuracy}}.

## Robustness and biases

Synthetic perturbations (accents, case, typos, chat abbreviations and LatAm
lexicon substitutions) measure sensitivity, not real dialect accuracy. On the
synthetic Rioplatense profile the accuracy changes from
{{models.lora.robustness.regionalize:ar.accuracy_before}} to
{{models.lora.robustness.regionalize:ar.accuracy_after}} on the affected
subset ({{models.lora.robustness.regionalize:ar.n_affected}} utterances).
The model inherits the es-ES bias of MASSIVE and was not trained on natural
Latin-American Spanish.

## Ethical and legal notes

MASSIVE contains assistant-style utterances collected through SLURP; see the
dataset NOTICE for attribution. Weights are a derivative of MrBERT-es and the
modification is stated here. No personal data is collected at inference time
and input texts are never logged.

## Citation

Cite MrBERT (Tamayo et al.), Amazon MASSIVE and SLURP when using this model.
