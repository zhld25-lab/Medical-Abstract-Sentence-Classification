# Dataset Notes

## Dataset

This project uses the PubMed RCT sentence classification dataset, commonly known as **PubMed 20k RCT / PubMed 200k RCT**.

## Task

The task is sequential sentence classification for medical abstracts. Each sentence in an abstract is assigned one role label:

- `BACKGROUND`
- `OBJECTIVE`
- `METHODS`
- `RESULTS`
- `CONCLUSIONS`

## Data Used in This Project

This repository is built around the smaller `20k_abstracts` folder first. The PubMed 200k RCT split is useful for future work, but it is intentionally not used in the initial experiments because it is much larger.

Expected local files:

```text
data/raw/20k_abstracts/train.txt
data/raw/20k_abstracts/dev.txt
data/raw/20k_abstracts/test.txt
```

Processed CSV files are also provided:

```text
data/processed/train_processed.csv
data/processed/val_processed.csv
data/processed/test_processed.csv
data/processed/processed_data_summary.csv
```

The processed files contain these columns:

- `target`
- `text`
- `line_number`
- `total_lines`
- `relative_position`

## Repository Data Policy

The PubMed 20k RCT raw files used by this project are included in this repository so the notebook, scripts, and Streamlit app can be reproduced without a separate dataset download. Larger future datasets, such as PubMed 200k RCT, should still be managed carefully because they can make the repository much heavier.

## How to Add the Data Locally

The included 20k split is expected here:

```text
data/raw/20k_abstracts/train.txt
data/raw/20k_abstracts/dev.txt
data/raw/20k_abstracts/test.txt
```

If the processed files need to be regenerated, run:

```bash
python src/train_baseline_models.py
```

or load and save the parsed splits manually with `src/data_preprocessing.py`.

Then run the training script from the project root:

```bash
python src/train_baseline_models.py
```
