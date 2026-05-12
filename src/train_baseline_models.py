"""Train baseline models for PubMed RCT sentence classification.

This script creates the saved artifacts used throughout the project:
tables, figures, model files, predictions, and a best-model pointer for the
Streamlit app and command-line predictor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_preprocessing import LABEL_ORDER, get_lines, load_pubmed_dataset
from evaluate_models import (
    calculate_metrics,
    plot_confusion_matrix,
    save_classification_report,
    save_metrics,
    save_model_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_FEATURE = "text"
STRUCTURAL_FEATURES = ["line_number", "total_lines", "relative_position"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train PubMed RCT baseline models.")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "20k_abstracts",
        help="Folder containing train.txt, dev.txt, and test.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Folder where tables, figures, models, and predictions are saved.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
        help="Maximum number of TF-IDF features.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create output folders and return their paths."""

    paths = {
        "figures": output_dir / "figures",
        "tables": output_dir / "tables",
        "models": output_dir / "models",
        "predictions": output_dir / "predictions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_raw_line_counts(raw_data_dir: Path, table_dir: Path) -> pd.DataFrame:
    """Save the number of raw lines in each split."""

    records = []
    split_files = {
        "train": raw_data_dir / "train.txt",
        "validation": raw_data_dir / "dev.txt",
        "test": raw_data_dir / "test.txt",
    }

    for split, filepath in split_files.items():
        records.append({"split": split, "raw_line_count": len(get_lines(filepath))})

    line_count_df = pd.DataFrame(records)
    line_count_df.to_csv(table_dir / "raw_line_counts.csv", index=False)
    return line_count_df


def save_processed_previews(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    table_dir: Path,
) -> None:
    """Save small processed previews for notebook display."""

    train_df.head(20).to_csv(table_dir / "train_processed_preview.csv", index=False)
    val_df.head(20).to_csv(table_dir / "val_processed_preview.csv", index=False)
    test_df.head(20).to_csv(table_dir / "test_processed_preview.csv", index=False)


def save_label_distribution(splits: dict[str, pd.DataFrame], table_dir: Path, figure_dir: Path) -> pd.DataFrame:
    """Save label counts and a grouped bar chart."""

    records = []
    for split, df in splits.items():
        counts = df["target"].value_counts().reindex(LABEL_ORDER, fill_value=0)
        for label, count in counts.items():
            records.append(
                {
                    "split": split,
                    "target": label,
                    "count": int(count),
                    "percentage": count / len(df),
                }
            )

    distribution_df = pd.DataFrame(records)
    distribution_df.to_csv(table_dir / "label_distribution.csv", index=False)

    plot_df = distribution_df.pivot(index="target", columns="split", values="count").loc[LABEL_ORDER]
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df.plot(kind="bar", ax=ax, color=["#3366CC", "#DC3912", "#109618"])
    ax.set_title("Label Distribution by Split")
    ax.set_xlabel("Sentence role")
    ax.set_ylabel("Number of sentences")
    ax.legend(title="Split")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(figure_dir / "label_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return distribution_df


def save_text_length_outputs(splits: dict[str, pd.DataFrame], table_dir: Path, figure_dir: Path) -> pd.DataFrame:
    """Save sentence length summary statistics and a histogram."""

    records = []
    fig, ax = plt.subplots(figsize=(9, 5))

    for split, df in splits.items():
        lengths = df["text"].str.split().map(len)
        df["sentence_length"] = lengths
        records.append(
            {
                "split": split,
                "count": int(lengths.count()),
                "mean_words": lengths.mean(),
                "median_words": lengths.median(),
                "min_words": lengths.min(),
                "max_words": lengths.max(),
                "p90_words": lengths.quantile(0.90),
            }
        )
        ax.hist(lengths, bins=50, alpha=0.45, label=split)

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(table_dir / "text_length_summary.csv", index=False)

    ax.set_title("Sentence Length Distribution")
    ax.set_xlabel("Words per sentence")
    ax.set_ylabel("Number of sentences")
    ax.legend(title="Split")
    fig.tight_layout()
    fig.savefig(figure_dir / "text_length_histogram.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return summary_df


def save_label_mapping(y_train: pd.Series, table_dir: Path) -> LabelEncoder:
    """Fit a LabelEncoder and save the integer-to-label mapping."""

    encoder = LabelEncoder()
    encoder.fit(y_train)
    mapping_df = pd.DataFrame(
        {
            "encoded_label": range(len(encoder.classes_)),
            "target": encoder.classes_,
        }
    )
    mapping_df.to_csv(table_dir / "label_mapping.csv", index=False)
    return encoder


def make_text_logreg_pipeline(max_features: int) -> Pipeline:
    """Create the TF-IDF + Logistic Regression baseline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1_000,
                        class_weight="balanced",
                        random_state=42,
                    )
                ),
            ),
        ]
    )


def make_text_linearsvc_pipeline(max_features: int) -> Pipeline:
    """Create the TF-IDF + LinearSVC baseline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(class_weight="balanced", dual="auto", random_state=42, max_iter=5_000),
            ),
        ]
    )


def make_position_logreg_pipeline(max_features: int) -> Pipeline:
    """Create a TF-IDF + structural features Logistic Regression model."""

    features = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
                TEXT_FEATURE,
            ),
            ("position", StandardScaler(), STRUCTURAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("features", features),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1_000,
                        class_weight="balanced",
                        random_state=42,
                    )
                ),
            ),
        ]
    )


def evaluate_and_save(
    model: Pipeline,
    model_name: str,
    X_val,
    y_val: pd.Series,
    metric_path: Path,
    report_path: Path,
    table_dir: Path,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate a model on validation data and save metrics/report."""

    predictions = model.predict(X_val)
    metrics = calculate_metrics(y_val, predictions)
    save_metrics(metrics, table_dir / metric_path, model_name=model_name)
    save_classification_report(y_val, predictions, LABEL_ORDER, table_dir / report_path)
    return metrics, predictions


def plot_model_comparison(comparison_df: pd.DataFrame, figure_dir: Path) -> None:
    """Save a bar chart comparing validation performance."""

    plot_df = comparison_df.set_index("model")[["accuracy", "macro_f1", "weighted_f1"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax, color=["#3366CC", "#DC3912", "#109618"])
    ax.set_title("Validation Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(figure_dir / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_top_words_by_class(model: Pipeline, output_path: Path, top_n: int = 15) -> pd.DataFrame:
    """Save the highest positive TF-IDF features for each Logistic Regression class."""

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    records = []
    for class_index, class_label in enumerate(classifier.classes_):
        if hasattr(classifier, "coef_"):
            coefficients = classifier.coef_[class_index]
        else:
            coefficients = classifier.estimators_[class_index].coef_.ravel()
        top_indices = np.argsort(coefficients)[-top_n:][::-1]
        for rank, feature_index in enumerate(top_indices, start=1):
            records.append(
                {
                    "target": class_label,
                    "rank": rank,
                    "word_or_phrase": feature_names[feature_index],
                    "coefficient": coefficients[feature_index],
                }
            )

    top_words_df = pd.DataFrame(records)
    top_words_df.to_csv(output_path, index=False)
    return top_words_df


def predict_with_model(model: Pipeline, input_mode: str, examples_df: pd.DataFrame) -> np.ndarray:
    """Predict with either a text-only or position-aware model."""

    if input_mode == "position":
        return model.predict(examples_df[[TEXT_FEATURE, *STRUCTURAL_FEATURES]])
    return model.predict(examples_df[TEXT_FEATURE])


def main() -> None:
    """Run the full baseline training and evaluation workflow."""

    args = parse_args()
    output_paths = ensure_output_dirs(args.output_dir)
    figure_dir = output_paths["figures"]
    table_dir = output_paths["tables"]
    model_dir = output_paths["models"]
    prediction_dir = output_paths["predictions"]

    print("Loading and preprocessing PubMed 20k RCT data...")
    train_df, val_df, test_df = load_pubmed_dataset(args.raw_data_dir)
    splits = {"train": train_df, "validation": val_df, "test": test_df}

    save_raw_line_counts(args.raw_data_dir, table_dir)
    save_processed_previews(train_df, val_df, test_df, table_dir)
    save_label_distribution(splits, table_dir, figure_dir)
    save_text_length_outputs(splits, table_dir, figure_dir)
    save_label_mapping(train_df["target"], table_dir)

    X_train_text = train_df["text"]
    X_val_text = val_df["text"]
    y_train = train_df["target"]
    y_val = val_df["target"]

    X_train_position = train_df[[TEXT_FEATURE, *STRUCTURAL_FEATURES]]
    X_val_position = val_df[[TEXT_FEATURE, *STRUCTURAL_FEATURES]]

    models = {
        "TF-IDF + Logistic Regression": {
            "pipeline": make_text_logreg_pipeline(args.max_features),
            "input_mode": "text",
            "model_path": model_dir / "tfidf_logreg.joblib",
            "metric_file": "logreg_validation_metrics.csv",
            "report_file": "logreg_classification_report.csv",
            "X_train": X_train_text,
            "X_val": X_val_text,
        },
        "TF-IDF + LinearSVC": {
            "pipeline": make_text_linearsvc_pipeline(args.max_features),
            "input_mode": "text",
            "model_path": model_dir / "tfidf_linearsvc.joblib",
            "metric_file": "linearsvc_validation_metrics.csv",
            "report_file": "linearsvc_classification_report.csv",
            "X_train": X_train_text,
            "X_val": X_val_text,
        },
        "TF-IDF + Position Logistic Regression": {
            "pipeline": make_position_logreg_pipeline(args.max_features),
            "input_mode": "position",
            "model_path": model_dir / "tfidf_position_logreg.joblib",
            "metric_file": "position_logreg_validation_metrics.csv",
            "report_file": "position_logreg_classification_report.csv",
            "X_train": X_train_position,
            "X_val": X_val_position,
        },
    }

    comparison_records = []
    for model_name, config in models.items():
        print(f"Training {model_name}...")
        model = config["pipeline"]
        model.fit(config["X_train"], y_train)
        joblib.dump(model, config["model_path"])

        metrics, _ = evaluate_and_save(
            model=model,
            model_name=model_name,
            X_val=config["X_val"],
            y_val=y_val,
            metric_path=Path(config["metric_file"]),
            report_path=Path(config["report_file"]),
            table_dir=table_dir,
        )
        comparison_records.append({"model": model_name, **metrics})

    comparison_df = save_model_comparison(comparison_records, table_dir / "model_comparison.csv")
    plot_model_comparison(comparison_df, figure_dir)

    best_model_name = comparison_df.iloc[0]["model"]
    best_config = models[best_model_name]
    best_model = best_config["pipeline"]
    best_input_mode = best_config["input_mode"]

    print(f"Best validation model: {best_model_name}")
    joblib.dump(best_model, model_dir / "best_model.joblib")
    metadata = {
        "best_model_name": best_model_name,
        "model_filename": "best_model.joblib",
        "source_model_filename": best_config["model_path"].name,
        "input_mode": best_input_mode,
        "labels": LABEL_ORDER,
        "structural_features": STRUCTURAL_FEATURES,
    }
    (model_dir / "best_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    X_test = test_df[[TEXT_FEATURE, *STRUCTURAL_FEATURES]] if best_input_mode == "position" else test_df["text"]
    y_test = test_df["target"]
    test_predictions = best_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_predictions)
    save_metrics(test_metrics, table_dir / "best_model_test_metrics.csv", model_name=best_model_name)
    save_classification_report(
        y_test,
        test_predictions,
        LABEL_ORDER,
        table_dir / "best_model_test_classification_report.csv",
    )
    plot_confusion_matrix(
        y_test,
        test_predictions,
        LABEL_ORDER,
        figure_dir / "confusion_matrix.png",
        table_dir / "confusion_matrix.csv",
    )

    misclassified_df = test_df.loc[y_test != test_predictions, [TEXT_FEATURE, *STRUCTURAL_FEATURES]].copy()
    misclassified_df.insert(1, "true_label", y_test[y_test != test_predictions].values)
    misclassified_df.insert(2, "predicted_label", test_predictions[y_test != test_predictions])
    misclassified_df.head(30).to_csv(prediction_dir / "misclassified_examples.csv", index=False)

    save_top_words_by_class(models["TF-IDF + Logistic Regression"]["pipeline"], table_dir / "top_words_by_class.csv")

    demo_df = pd.DataFrame(
        [
            {
                "text": "The aim of this study was to evaluate the effect of treatment.",
                "line_number": 1,
                "total_lines": 10,
            },
            {
                "text": "Participants were randomly assigned to receive placebo or active medication.",
                "line_number": 4,
                "total_lines": 10,
            },
            {
                "text": "The intervention group showed a statistically significant reduction in symptoms.",
                "line_number": 7,
                "total_lines": 10,
            },
            {
                "text": "These findings suggest that the treatment may improve patient outcomes.",
                "line_number": 9,
                "total_lines": 10,
            },
        ]
    )
    demo_df["relative_position"] = demo_df["line_number"] / demo_df["total_lines"]
    demo_df["predicted_label"] = predict_with_model(best_model, best_input_mode, demo_df)
    demo_df.to_csv(prediction_dir / "demo_predictions.csv", index=False)

    print("Training complete. Outputs saved under:", args.output_dir)


if __name__ == "__main__":
    main()
