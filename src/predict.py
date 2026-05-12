"""Command-line prediction utility for medical abstract sentence roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
TEXT_FEATURE = "text"
STRUCTURAL_FEATURES = ["line_number", "total_lines", "relative_position"]


def load_best_model(model_dir: str | Path = DEFAULT_MODEL_DIR):
    """Load the saved best model and its metadata."""

    model_dir = Path(model_dir)
    metadata_path = model_dir / "best_model_metadata.json"

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_path = model_dir / metadata.get("model_filename", "best_model.joblib")
        input_mode = metadata.get("input_mode", "text")
    else:
        metadata = {"input_mode": "text"}
        input_mode = "text"
        model_path = model_dir / "best_model.joblib"

    if not model_path.exists():
        fallback_paths = [
            model_dir / "tfidf_position_logreg.joblib",
            model_dir / "tfidf_linearsvc.joblib",
            model_dir / "tfidf_logreg.joblib",
        ]
        for candidate in fallback_paths:
            if candidate.exists():
                model_path = candidate
                input_mode = "position" if "position" in candidate.name else "text"
                break
        else:
            raise FileNotFoundError(
                "No trained model was found. Run `python src/train_baseline_models.py` first."
            )

    model = joblib.load(model_path)
    return model, {"model_path": str(model_path), "input_mode": input_mode, **metadata}


def build_prediction_input(
    sentence: str,
    input_mode: str,
    line_number: int | None = None,
    total_lines: int | None = None,
):
    """Build the input object expected by the saved model."""

    if input_mode != "position":
        return [sentence]

    safe_line_number = 0 if line_number is None else line_number
    safe_total_lines = 1 if total_lines in (None, 0) else total_lines
    relative_position = safe_line_number / safe_total_lines

    return pd.DataFrame(
        [
            {
                TEXT_FEATURE: sentence,
                "line_number": safe_line_number,
                "total_lines": safe_total_lines,
                "relative_position": relative_position,
            }
        ]
    )[[TEXT_FEATURE, *STRUCTURAL_FEATURES]]


def predict_sentence_role(
    sentence: str,
    line_number: int | None = None,
    total_lines: int | None = None,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> str:
    """Predict the PubMed RCT sentence role for one sentence."""

    model, metadata = load_best_model(model_dir)
    model_input = build_prediction_input(
        sentence,
        input_mode=metadata.get("input_mode", "text"),
        line_number=line_number,
        total_lines=total_lines,
    )
    return str(model.predict(model_input)[0])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Predict a medical abstract sentence role.")
    parser.add_argument("--text", required=True, help="Medical abstract sentence to classify.")
    parser.add_argument("--line-number", type=int, default=None, help="Optional zero-based sentence number.")
    parser.add_argument("--total-lines", type=int, default=None, help="Optional number of sentences in the abstract.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory containing trained models.")
    return parser.parse_args()


def main() -> None:
    """Run command-line prediction."""

    args = parse_args()
    prediction = predict_sentence_role(
        sentence=args.text,
        line_number=args.line_number,
        total_lines=args.total_lines,
        model_dir=args.model_dir,
    )
    print(prediction)


if __name__ == "__main__":
    main()

