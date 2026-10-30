# Jackfruit vs. original notebook results

## Run protocol

Ran the packaged Jackfruit detector on the seed 0 clean scenario for each dataset. I used the original notebook code for data loading, cleanup, train/test splits, feature processing, and the Random Forest model fit. Compared each result with the seed 0 clean Random Forest record in the matching notebook results archive. This is a single-run clean-baseline comparison; it does not repeat all leakage-injection scenarios or evaluation seeds.

## Results

| Dataset | Train / test rows | Features | Jackfruit AUC | Notebook AUC | Δ AUC | Jackfruit scores T / C / Temp / P | Notebook scores T / C / Temp / P | Jackfruit verdict |
|---|---:|---:|---:|---:|---:|---|---|---|
| Adult Income | 29,274 / 9,758 | 13 | 0.9216 | 0.9160 | +0.0056 | 0.1094 / 0.1053 / N/A / 0.0000 | 0.1157 / 0.0000 / N/A / 0.0000 | clean |
| California Housing | 12,384 / 4,128 | 8 | 0.9502 | 0.9550 | -0.0049 | 0.3312 / 0.0000 / N/A / 0.0000 | 0.1907 / 0.0000 / N/A / 0.0000 | clean |
| Credit Card Fraud | 24,284 / 6,071 | 29 | 0.9791 | 0.9791 | -0.0000 | 0.2474 / 0.0000 / 0.0000 / 0.0000 | 0.2474 / 0.0000 / 0.0000 / 0.0000 | clean |
| Rossmann | 30,000 / 10,000 | 17 | 0.8897 | 0.8902 | -0.0005 | 0.5019 / 0.0000 / 0.0000 / 0.0000 | 0.5112 / 0.0000 / 0.0000 / 0.0000 | clean |

T = target, C = contamination, Temp = temporal, P = preprocessing. Jackfruit returned `False` for all flags on the four clean splits. Temporal detection returned N/A when the dataset had no timestamp. All four overall verdicts agree with the original notebook records: `clean`.

## Differences to review

- Adult Income: the contamination score is 0.1053 vs. 0.0000 in the saved notebook row. Both remain below threshold and unflagged. Target score is 0.1094 vs. 0.1157.
- California Housing: the target score is 0.3312 vs. 0.1907. Both remain below threshold and unflagged. This is the largest target-score gap.
- Credit Card Fraud: scores match to floating-point precision; test AUC differs by less than 0.0001.
- Rossmann: target score is 0.5019 vs. 0.5112; all other scores match. Both verdicts remain clean.

The notebook reference values come from archived results CSVs in the project ZIPs, not a fresh rerun of the notebooks’ complete multi-seed, multi-scenario benchmarks. The target-score differences above are real differences between the packaged detector output and those saved reference rows; they should not be interpreted as equivalent scores merely because flags agree.

## Data and source notes

- Adult Income: UCI Adult records combined from the UCI `adult.data` and `adult.test` files.
- California Housing: scikit-learn California Housing archive, exported temporarily in the schema expected by the notebook.
- Credit Card Fraud: public `creditcard.csv` mirror used by the notebook.
- Rossmann: public mirror of the competition training data and store metadata; prepared using the notebook’s chronological blocks.

Downloaded raw data and generated intermediate files were kept under `/private/tmp/jackfruit_data`; the project notebooks were not edited. The JSON companion contains per-detector flags and evidence for every run.
