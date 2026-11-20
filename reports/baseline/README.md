# Classical baseline (dev)

TF-IDF word (1,2) + `char_wb` (2,5) with `LogisticRegression(C=10, class_weight="balanced")`
wins the 42-configuration D5 grid on dev: **accuracy 0.8495, macro-F1 0.8429**
(weighted-F1 0.8509, top-3 0.9424, scenario accuracy 0.8952).

Reproduce with `STC_DATA_DIR=$PWD/data uv run stc train baseline --out artifacts/baseline --reports reports/baseline`
(42 configs, 40 LinearSVC + 2 logistic, selected on dev macro-F1 only). The fitted
`baseline.skops` is 56.4 MB, so the D5 float32 coefficient rule was applied and
recorded as `float32_cast: true` in the artifact manifest; the artifact stays ignored.
