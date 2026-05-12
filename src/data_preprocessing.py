"""Utilities for loading and preprocessing PubMed RCT abstract files.

The raw PubMed RCT files store one abstract at a time. Abstract IDs start with
`###`, sentence rows contain a label and text separated by a tab, and blank
lines mark the end of an abstract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


LABEL_ORDER = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]


def get_lines(filepath: str | Path) -> list[str]:
    """Read a text file and return its lines.

    Args:
        filepath: Path to a PubMed RCT split file.

    Returns:
        A list of raw lines from the file.
    """

    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8") as file:
        return file.readlines()


def _flush_abstract(abstract_lines: list[str], samples: list[dict]) -> None:
    """Convert the collected lines for one abstract into row dictionaries."""

    total_lines = len(abstract_lines)
    if total_lines == 0:
        return

    for line_number, abstract_line in enumerate(abstract_lines):
        if "\t" not in abstract_line:
            continue

        target, text = abstract_line.split("\t", maxsplit=1)
        relative_position = line_number / total_lines if total_lines else 0.0

        samples.append(
            {
                "target": target.strip(),
                "text": text.strip(),
                "line_number": line_number,
                "total_lines": total_lines,
                "relative_position": relative_position,
            }
        )


def preprocess_text_with_line_numbers(filepath: str | Path) -> list[dict]:
    """Parse a PubMed RCT split file into sentence-level examples.

    Each returned dictionary contains the label, sentence text, sentence index
    within the abstract, total number of abstract sentences, and relative
    sentence position.

    Args:
        filepath: Path to `train.txt`, `dev.txt`, or `test.txt`.

    Returns:
        A list of dictionaries suitable for creating a pandas DataFrame.
    """

    input_lines = get_lines(filepath)
    abstract_lines: list[str] = []
    samples: list[dict] = []

    for line in input_lines:
        stripped_line = line.strip()

        if stripped_line.startswith("###"):
            _flush_abstract(abstract_lines, samples)
            abstract_lines = []
            continue

        if stripped_line == "":
            _flush_abstract(abstract_lines, samples)
            abstract_lines = []
            continue

        abstract_lines.append(stripped_line)

    _flush_abstract(abstract_lines, samples)
    return samples


def load_pubmed_dataset(raw_data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test splits from a PubMed RCT raw folder.

    Args:
        raw_data_dir: Folder containing `train.txt`, `dev.txt`, and `test.txt`.

    Returns:
        Three DataFrames in the order train, validation, test.
    """

    raw_data_dir = Path(raw_data_dir)
    split_files = {
        "train": raw_data_dir / "train.txt",
        "val": raw_data_dir / "dev.txt",
        "test": raw_data_dir / "test.txt",
    }

    missing_files = [str(path) for path in split_files.values() if not path.exists()]
    if missing_files:
        missing_text = "\n".join(missing_files)
        raise FileNotFoundError(
            "Missing PubMed RCT data files. Expected these paths:\n"
            f"{missing_text}\n\n"
            "Place the 20k split under data/raw/20k_abstracts/."
        )

    train_df = pd.DataFrame(preprocess_text_with_line_numbers(split_files["train"]))
    val_df = pd.DataFrame(preprocess_text_with_line_numbers(split_files["val"]))
    test_df = pd.DataFrame(preprocess_text_with_line_numbers(split_files["test"]))

    expected_columns = ["target", "text", "line_number", "total_lines", "relative_position"]
    return train_df[expected_columns], val_df[expected_columns], test_df[expected_columns]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    train, val, test = load_pubmed_dataset(project_root / "data" / "raw" / "20k_abstracts")
    print(f"Train rows: {len(train):,}")
    print(f"Validation rows: {len(val):,}")
    print(f"Test rows: {len(test):,}")

