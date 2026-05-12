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

## Why Raw Files Are Not Committed

The raw dataset text files are not committed to GitHub because dataset files can be large and should be managed separately from source code. The `.gitignore` file excludes `data/raw/`, so users can place the data locally without accidentally committing it.

## How to Add the Data Locally

After downloading or extracting the dataset archive, place the 20k split here:

```text
data/raw/20k_abstracts/train.txt
data/raw/20k_abstracts/dev.txt
data/raw/20k_abstracts/test.txt
```

Then run the training script from the project root:

```bash
python src/train_baseline_models.py
```

