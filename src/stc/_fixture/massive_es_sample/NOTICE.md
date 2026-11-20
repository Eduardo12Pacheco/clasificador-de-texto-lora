Amazon MASSIVE
==============

# NOTICE

MASSIVE was created by translating and localizing the English text data from SLURP into 50 languages. Additionally, some of the text data from SLURP is included as part of MASSIVE. SLURP is licensed under CC BY 4.0. The license can be found here:

> [https://github.com/pswietojanski/slurp/blob/master/LICENSE.txt](https://github.com/pswietojanski/slurp/blob/master/LICENSE.txt)

For additional information on SLURP, please see [their paper](https://aclanthology.org/2020.emnlp-main.588.pdf).

Sample selection and reformatting
---------------------------------
The JSONL files in this directory are a small sample selected and reformatted
by spanish-text-classifier-lora, 2026. Rows are drawn from the cleaned training
split and the official development/test splits of Amazon MASSIVE 1.0 es-ES,
licensed CC BY 4.0. The sample keeps up to 6 train, 2 dev and 2
test rows per intent, chosen deterministically by ascending sha256(id).
