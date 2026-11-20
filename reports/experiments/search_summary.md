# LoRA hyper-parameter search

- Study: `stc-lora-search` (TPE, seed 13, multivariate)
- Trials: 24 (requested 24)
- Best dev macro-F1: 0.8627

## Best parameters

- `alpha`: `32`
- `batch_size`: `16`
- `class_weighting`: `none`
- `epochs`: `6`
- `lora_dropout`: `0.0`
- `lr`: `0.0003632297554518523`
- `r`: `32`
- `warmup_ratio`: `0.06`
- `weight_decay`: `0.01`

## Trials

| trial | state | dev macro-F1 |
|---|---|---|
| 0 | COMPLETE | 0.8352 |
| 1 | COMPLETE | 0.8384 |
| 2 | COMPLETE | 0.8505 |
| 3 | COMPLETE | 0.8265 |
| 4 | COMPLETE | 0.8265 |
| 5 | COMPLETE | 0.8627 |
| 6 | PRUNED | 0.6972 |
| 7 | PRUNED | 0.7241 |
| 8 | COMPLETE | 0.8501 |
| 9 | PRUNED | 0.7238 |
| 10 | PRUNED | 0.7490 |
| 11 | PRUNED | 0.6903 |
| 12 | PRUNED | 0.8142 |
| 13 | PRUNED | 0.8446 |
| 14 | PRUNED | 0.6802 |
| 15 | PRUNED | 0.7183 |
| 16 | PRUNED | 0.7555 |
| 17 | PRUNED | 0.7493 |
| 18 | PRUNED | 0.7316 |
| 19 | PRUNED | 0.7329 |
| 20 | PRUNED | 0.8265 |
| 21 | PRUNED | 0.7245 |
| 22 | COMPLETE | 0.8430 |
| 23 | PRUNED | 0.7509 |
