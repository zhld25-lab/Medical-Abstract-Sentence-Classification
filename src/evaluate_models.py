"""Evaluation helpers for medical abstract sentence classification models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate the main classification metrics used in this project."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def save_metrics(metrics: dict[str, float], output_path: str | Path, model_name: str) -> pd.DataFrame:
    """Save one model's metric dictionary as a one-row CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([{ "model": model_name, **metrics }])
    metrics_df.to_csv(output_path, index=False)
    return metrics_df


def save_classification_report(
    y_true,
    y_pred,
    labels: list[str],
    output_path: str | Path,
) -> pd.DataFrame:
    """Save a scikit-learn classification report as a CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
    report_df.to_csv(output_path, index=False)
    return report_df


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    figure_path: str | Path,
    table_path: str | Path | None = None,
) -> pd.DataFrame:
    """Create and save a confusion matrix table and figure."""

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "true_label"

    if table_path is not None:
        table_path = Path(table_path)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        cm_df.to_csv(table_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Best Model Confusion Matrix")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return cm_df


def save_model_comparison(metrics_by_model: list[dict], output_path: str | Path) -> pd.DataFrame:
    """Save model comparison metrics sorted by validation macro F1."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_df = pd.DataFrame(metrics_by_model)
    comparison_df = comparison_df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    comparison_df.to_csv(output_path, index=False)
    return comparison_df

