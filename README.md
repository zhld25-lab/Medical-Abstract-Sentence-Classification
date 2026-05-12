# Medical Abstract Sentence Classification

## 1. Project Overview

This project builds a machine learning NLP system that predicts the role of each sentence in a medical research abstract. The target labels are `BACKGROUND`, `OBJECTIVE`, `METHODS`, `RESULTS`, and `CONCLUSIONS`.

The main idea is not only to train a text classifier, but to test whether abstract structure improves classification. Medical abstracts usually follow a logical order: background, objective, methods, results, and conclusions. I compare pure TF-IDF text baselines with a model that combines TF-IDF features with `line_number`, `total_lines`, and `relative_position`.

## 2. Why This Project Matters

Medical papers are difficult to read quickly. This project helps automatically identify whether a sentence is describing background, objective, methods, results, or conclusions. This can support medical literature search, evidence extraction, and healthcare information systems.

## 3. Dataset

Dataset: PubMed 20k RCT / PubMed 200k RCT

This project uses the `20k_abstracts` folder first:

```text
data/raw/20k_abstracts/train.txt
data/raw/20k_abstracts/dev.txt
data/raw/20k_abstracts/test.txt
```

The PubMed 20k RCT raw split used in this project is included in this repository under `data/raw/20k_abstracts/`. See [data/README_data.md](data/README_data.md) for setup notes.

## 4. Problem Definition

Given one sentence from a structured medical abstract, predict its sentence role:

| Label | Meaning |
|---|---|
| BACKGROUND | Context or motivation for the research |
| OBJECTIVE | Study aim, question, or hypothesis |
| METHODS | Study design, participants, interventions, or measurements |
| RESULTS | Findings, measurements, and statistical outcomes |
| CONCLUSIONS | Interpretation, implication, or final summary |

## 5. Project Structure

```text
Medical-Abstract-Sentence-Classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README_data.md
│   └── sample/
│       └── sample_abstract_sentences.csv
├── notebooks/
│   └── 01_medical_abstract_sentence_classification.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_baseline_models.py
│   ├── evaluate_models.py
│   └── predict.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── predictions/
└── app/
    └── streamlit_app.py
```

## 6. Methodology

The workflow includes:

- Data parsing from structured abstract text files.
- Label extraction from tab-separated sentence rows.
- Sentence position feature engineering with `line_number`, `total_lines`, and `relative_position`.
- TF-IDF text representation.
- Logistic Regression baseline.
- LinearSVC baseline.
- TF-IDF plus structural feature model.
- Model comparison using validation accuracy, macro F1, and weighted F1.
- Error analysis on misclassified test examples.

## 7. Models

The project trains three models:

| Model | Features |
|---|---|
| TF-IDF + Logistic Regression | Sentence text only |
| TF-IDF + LinearSVC | Sentence text only |
| TF-IDF + Position Logistic Regression | Sentence text plus abstract structure features |

## 8. Evaluation Metrics

The main metrics are:

- Accuracy
- Macro F1
- Weighted F1
- Per-class precision, recall, and F1
- Confusion matrix

Macro F1 is important because it treats each label equally, which helps reveal whether a model performs poorly on less frequent sentence roles.

## 9. Results

After running `python src/train_baseline_models.py`, the generated model comparison table is saved to:

```text
outputs/tables/model_comparison.csv
```

Current validation comparison:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF + Position Logistic Regression | 0.8748 | 0.8256 | 0.8750 |
| TF-IDF + Logistic Regression | 0.8272 | 0.7681 | 0.8278 |
| TF-IDF + LinearSVC | 0.8215 | 0.7576 | 0.8219 |

The best model is selected by validation macro F1 and evaluated on the test set. Test metrics are saved to:

```text
outputs/tables/best_model_test_metrics.csv
outputs/tables/best_model_test_classification_report.csv
```

Current best test result:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF + Position Logistic Regression | 0.8687 | 0.8194 | 0.8689 |

## 10. Error Analysis

Misclassified test examples are saved to:

```text
outputs/predictions/misclassified_examples.csv
```

Typical error patterns include:

- `BACKGROUND` vs `OBJECTIVE`: both often appear near the beginning of abstracts and may share motivation language.
- `METHODS` vs `RESULTS`: both can contain experimental terminology, measurements, and clinical details.
- Position features help, but they cannot fully solve semantic overlap between sentence roles.

## 11. How to Run

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Place the 20k PubMed RCT files under:

```text
data/raw/20k_abstracts/
```

Run the full training and evaluation workflow:

```bash
python src/train_baseline_models.py
```

Predict one sentence from the command line:

```bash
python src/predict.py --text "The aim of this study was to evaluate the effect of treatment."
```

For a position-aware prediction:

```bash
python src/predict.py --text "Participants were randomly assigned to placebo or treatment." --line-number 4 --total-lines 10
```

## 12. Streamlit Demo

Run the web app from the project root:

```bash
streamlit run app/streamlit_app.py
```

The app lets users enter a medical sentence, optionally provide its line number and total abstract length, and view the predicted sentence role with a short label explanation.

## 13. Limitations

- This model classifies sentences independently.
- It does not fully model abstract-level sequence dependency.
- Traditional TF-IDF models cannot deeply understand medical semantics.
- Results on PubMed 20k may not fully generalize to other clinical text.
- Future work should include SciBERT, BioBERT, or sequence models.

## 14. Future Work

- Fine-tune SciBERT or BioBERT.
- Use the full PubMed 200k RCT dataset.
- Build an abstract-level sequence model.
- Add Streamlit web app deployment.
- Add external medical abstract testing.
