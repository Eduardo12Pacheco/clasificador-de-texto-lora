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

## Training procedure

The shipped adapter is a LoRA fine-tune of `BSC-LT/MrBERT-es` selected by the
pre-registered Optuna search of ADR-0003 (24 trials, dev macro-F1 objective).
The shipped seed is 42, chosen on dev only. The adapter uses rank 32, alpha 32,
dropout 0.0, learning rate 3.63e-4, weight decay 0.01, 6 epochs, batch size 16
and 6 % linear warm-up; class weighting was disabled. Training used bf16
autocast with fp32 master weights on an NVIDIA RTX 4060 (8 GB), then the adapter
was merged and exported to ONNX Runtime fp32 (opset 17). The TF-IDF baseline is
word 1–2 grams plus `char_wb` 2–5 grams with a LinearSVC, selected on dev from
the grid in D5. The test split was read once by the audited `stc report final`
command; its access log is committed.

## Evaluation (audited test split)

| Model | Test accuracy (95 % CI) | Test macro-F1 (95 % CI) |
|---|---|---|
| TF-IDF baseline | 0.8410 [0.8268, 0.8534] | 0.8175 [0.7919, 0.8330] |
| LoRA transformer | 0.8665 [0.8544, 0.8786] | 0.8305 [0.8064, 0.8458] |

**Verdict:** tie. The difference in accuracy is statistically significant: the LoRA transformer is more accurate than the TF-IDF baseline on the audited test split (McNemar p = 0.000); the macro-F1 difference is not significant (Δmacro-F1 +0.0131 (95 % CI [-0.0084, +0.0344])).

Top-3 accuracy is 0.9597; scenario accuracy is
0.9072.

## Calibration and abstention

Temperature scaling uses T = 2.2872 and the
review threshold is τ = 0.7655. Test ECE
moves from 0.0938 to
0.0181; at τ the model covers
0.8083 of test utterances with
selective accuracy 0.9509.

## Robustness and biases

Synthetic perturbations (accents, case, typos, chat abbreviations and LatAm
lexicon substitutions) measure sensitivity, not real dialect accuracy. On the
synthetic Rioplatense profile the accuracy changes from
0.8795 to
0.8664 on the affected
subset (614 utterances).
The model inherits the es-ES bias of MASSIVE and was not trained on natural
Latin-American Spanish.

## Limitations

- The training and evaluation data are European Spanish (`es-ES`). The LatAm
  regionalisation suite is synthetic and cannot establish real dialect accuracy.
- The corpus is assistant-style and human-localised; it is not a sample of
  spontaneous customer-support language.
- The model is a 60-way closed-set classifier. It cannot say "none of these" and
  has no generative fallback; use `needs_review` for uncertain cases.
- Rare intents have very few training rows (the smallest class has four), so
  per-class precision and recall are uneven. See `reports/final/per_class.csv`.
- The audited comparison with the baseline is a tie under the D19 rule despite a
  small positive macro-F1 estimate, because the paired confidence interval
  includes zero. Do not read the point estimate as a proven win.
- Dynamic and static int8 quantisation variants failed the pre-registered gate;
  only the fp32 ONNX graph ships. See ADR-0005.

## Licence and attribution

Code is MIT (`Copyright (c) 2026 Eduardo Pacheco`). The released weights are
Apache-2.0, a derivative of `BSC-LT/MrBERT-es` (Apache-2.0); the LoRA
modification is stated here. Amazon MASSIVE 1.0 is CC BY 4.0, Copyright
Amazon.com Inc., human-localised from SLURP (CC BY 4.0). See `NOTICE` and
`docs/DATA.md` for the full attribution and the pinned dataset checksum. No
personal data is collected at inference time and input texts are never logged.

## Citation

If you use this model, cite MrBERT (Tamayo et al., "MrBERT: Modern Multilingual
Encoders via Vocabulary, Domain, and Dimensional Adaptation"), Amazon MASSIVE
(FitzGerald et al., "MASSIVE: A 1M-Example Multilingual Natural Language
Understanding Dataset with 51 Typologically-Diverse Languages") and SLURP
(Bastianelli et al., "SLURP: A Spoken Language Understanding Resource Package").
