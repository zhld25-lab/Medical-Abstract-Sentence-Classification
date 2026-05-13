"""Professional Streamlit app for medical abstract sentence classification."""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

LABEL_ORDER = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
STRUCTURAL_FEATURES = ["line_number", "total_lines", "relative_position"]

LABEL_EXPLANATIONS = {
    "BACKGROUND": "This sentence provides context, motivation, or prior knowledge for the research topic.",
    "OBJECTIVE": "This sentence describes the study aim, research question, or hypothesis.",
    "METHODS": "This sentence explains the study design, participants, intervention, measurements, or analysis.",
    "RESULTS": "This sentence reports findings, statistical outcomes, or observed effects.",
    "CONCLUSIONS": "This sentence summarizes interpretation, implication, or final takeaways from the study.",
}

LABEL_STYLES = {
    "BACKGROUND": {"bg": "#E8EEF5", "fg": "#314A63", "border": "#6B7F95"},
    "OBJECTIVE": {"bg": "#EFE7F8", "fg": "#5B2B82", "border": "#7E57C2"},
    "METHODS": {"bg": "#FFF1E5", "fg": "#8A4B08", "border": "#F28C28"},
    "RESULTS": {"bg": "#E7F6ED", "fg": "#1E6B3A", "border": "#2E9D57"},
    "CONCLUSIONS": {"bg": "#E6EDF7", "fg": "#0B2E5B", "border": "#173B73"},
}

SAMPLE_SENTENCES = [
    "The aim of this study was to evaluate the effect of treatment.",
    "Participants were randomly assigned to receive placebo or active medication.",
    "The intervention group showed a statistically significant reduction in symptoms.",
    "These findings suggest that the treatment may improve patient outcomes.",
]

SAMPLE_ABSTRACT = """Emotional eating is associated with overeating and the development of obesity.
The aim of this study was to test whether attention bias for food moderates food intake during sad mood.
Participants were randomly assigned to one of two experimental mood induction conditions.
Attentional biases for high caloric foods were measured using eye tracking.
Attention maintenance on food cues was significantly related to increased intake.
These findings suggest that attention maintenance may contribute to eating motivation."""


st.set_page_config(
    page_title="Medical Abstract Sentence Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    """Add custom CSS for a polished Healthcare AI product interface."""

    st.markdown(
        """
        <style>
        :root {
            --deep-blue: #0B2E5B;
            --medical-blue: #1C74D9;
            --soft-blue: #EEF6FF;
            --light-gray: #F6F8FB;
            --text-main: #1E293B;
            --text-muted: #64748B;
            --border: #DCE6F1;
        }

        .main .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }

        h1, h2, h3 {
            color: var(--deep-blue);
            letter-spacing: 0;
        }

        .hero {
            background: linear-gradient(135deg, #FFFFFF 0%, #EEF6FF 55%, #E7F0FA 100%);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 2rem 2.2rem;
            box-shadow: 0 14px 34px rgba(11, 46, 91, 0.08);
            margin-bottom: 1.3rem;
        }

        .hero-kicker {
            color: var(--medical-blue);
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.08rem;
            margin-bottom: 0.45rem;
        }

        .hero-title {
            font-size: 2.5rem;
            line-height: 1.08;
            font-weight: 780;
            color: var(--deep-blue);
            margin: 0 0 0.5rem 0;
        }

        .hero-subtitle {
            color: #315A86;
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }

        .hero-copy {
            color: var(--text-muted);
            font-size: 1rem;
            margin-bottom: 1.2rem;
            max-width: 760px;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 20px rgba(11, 46, 91, 0.06);
            min-height: 92px;
        }

        .metric-value {
            color: var(--deep-blue);
            font-size: 1.35rem;
            font-weight: 780;
            margin-bottom: 0.2rem;
        }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.87rem;
            line-height: 1.3;
        }

        .section-card {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
        }

        .section-title {
            color: var(--deep-blue);
            font-size: 1.08rem;
            font-weight: 750;
            margin-bottom: 0.25rem;
        }

        .section-copy {
            color: var(--text-muted);
            font-size: 0.92rem;
            margin-bottom: 0.65rem;
        }

        .workflow-step {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-left: 5px solid var(--medical-blue);
            border-radius: 14px;
            padding: 1rem 1.05rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
            min-height: 145px;
            margin-bottom: 0.85rem;
        }

        .workflow-index {
            color: var(--medical-blue);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06rem;
            margin-bottom: 0.35rem;
        }

        .workflow-title {
            color: var(--deep-blue);
            font-size: 1rem;
            font-weight: 760;
            margin-bottom: 0.35rem;
        }

        .workflow-copy {
            color: var(--text-muted);
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .artifact-card {
            background: #F8FBFF;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.85rem;
            margin-bottom: 0.65rem;
        }

        .artifact-name {
            color: var(--deep-blue);
            font-weight: 760;
            font-size: 0.92rem;
        }

        .artifact-path {
            color: var(--text-muted);
            font-size: 0.82rem;
            font-family: monospace;
            margin-top: 0.2rem;
        }

        .label-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.3rem 0.72rem;
            font-size: 0.78rem;
            font-weight: 760;
            border: 1px solid transparent;
            letter-spacing: 0.02rem;
        }

        .result-card {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-left: 7px solid var(--medical-blue);
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 12px 28px rgba(11, 46, 91, 0.08);
            margin-top: 0.9rem;
        }

        .result-title {
            color: var(--text-muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 760;
            letter-spacing: 0.07rem;
            margin-bottom: 0.45rem;
        }

        .result-role {
            color: var(--deep-blue);
            font-size: 1.75rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }

        .result-meta {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 0.4rem;
        }

        .sentence-card {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.65rem;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
        }

        .empty-state {
            background: var(--light-gray);
            border: 1px dashed #B8C7D9;
            border-radius: 14px;
            padding: 1.2rem;
            color: var(--text-muted);
            text-align: center;
        }

        .sidebar-block {
            background: #F8FBFF;
            border: 1px solid #DCE6F1;
            border-radius: 12px;
            padding: 0.8rem;
            color: #315A86;
            font-size: 0.9rem;
        }

        .disclaimer {
            background: #F8FBFF;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.86rem;
            padding: 1rem 0 0 0;
            margin-top: 1.2rem;
        }

        div.stButton > button {
            border-radius: 10px;
            border: 1px solid #1C74D9;
            background: #1C74D9;
            color: white;
            font-weight: 700;
            padding: 0.5rem 1rem;
        }

        div.stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid #0B2E5B;
            color: #0B2E5B;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_and_metadata() -> tuple[object, dict]:
    """Load the best saved model and metadata."""

    metadata_path = MODEL_DIR / "best_model_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_path = MODEL_DIR / metadata.get("model_filename", "best_model.joblib")
    else:
        metadata = {
            "best_model_name": "Unknown saved model",
            "model_filename": "best_model.joblib",
            "input_mode": "text",
            "labels": LABEL_ORDER,
        }
        model_path = MODEL_DIR / "best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model was found. Run `python src/train_baseline_models.py` first."
        )

    return joblib.load(model_path), metadata


@st.cache_data
def load_csv_if_exists(path: Path) -> pd.DataFrame:
    """Load a CSV file if it exists, otherwise return an empty DataFrame."""

    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def get_model_classes(model: object) -> list[str]:
    """Return class labels from a sklearn Pipeline or estimator."""

    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "classes_"):
            return list(classifier.classes_)
    return LABEL_ORDER


def softmax(values: np.ndarray) -> np.ndarray:
    """Convert decision scores to a confidence-like distribution."""

    values = np.asarray(values, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    total = exp_values.sum()
    if total == 0:
        return np.zeros_like(exp_values)
    return exp_values / total


def build_model_input(
    sentence: str,
    input_mode: str,
    line_number: int | None = None,
    total_lines: int | None = None,
):
    """Build the input format expected by the saved sklearn model."""

    if input_mode != "position":
        return [sentence]

    safe_line_number = 0 if line_number is None else int(line_number)
    safe_total_lines = 1 if total_lines in (None, 0) else int(total_lines)
    relative_position = safe_line_number / safe_total_lines

    return pd.DataFrame(
        [
            {
                "text": sentence,
                "line_number": safe_line_number,
                "total_lines": safe_total_lines,
                "relative_position": relative_position,
            }
        ]
    )[["text", *STRUCTURAL_FEATURES]]


def predict_sentence(
    model: object,
    metadata: dict,
    sentence: str,
    line_number: int | None = None,
    total_lines: int | None = None,
) -> dict:
    """Predict one sentence and return label, score, confidence, and position details."""

    input_mode = metadata.get("input_mode", "text")
    model_input = build_model_input(sentence, input_mode, line_number, total_lines)
    predicted_label = str(model.predict(model_input)[0])
    classes = get_model_classes(model)

    confidence = np.nan
    decision_score = np.nan

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(model_input))[0]
        if predicted_label in classes:
            class_index = classes.index(predicted_label)
            confidence = float(probabilities[class_index])
            decision_score = confidence
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(model_input))
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        scores = scores[0]
        if predicted_label in classes:
            class_index = classes.index(predicted_label)
            decision_score = float(scores[class_index])
            confidence = float(softmax(scores)[class_index])

    safe_total_lines = 1 if total_lines in (None, 0) else int(total_lines)
    safe_line_number = 0 if line_number is None else int(line_number)
    relative_position = safe_line_number / safe_total_lines

    return {
        "sentence": sentence,
        "line_number": safe_line_number,
        "relative_position": relative_position,
        "predicted_label": predicted_label,
        "confidence_or_score": confidence if not np.isnan(confidence) else decision_score,
        "decision_score": decision_score,
        "label_explanation": LABEL_EXPLANATIONS.get(predicted_label, "No explanation available."),
    }


def split_abstract_sentences(abstract_text: str) -> list[str]:
    """Split a pasted abstract into sentence-like units."""

    cleaned = abstract_text.strip()
    if not cleaned:
        return []

    line_candidates = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(line_candidates) > 1:
        return line_candidates

    sentence_candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [sentence.strip() for sentence in sentence_candidates if sentence.strip()]


def analyze_sentences(model: object, metadata: dict, sentences: list[str]) -> pd.DataFrame:
    """Predict roles for a list of abstract sentences."""

    total_lines = max(len(sentences), 1)
    records = [
        predict_sentence(
            model=model,
            metadata=metadata,
            sentence=sentence,
            line_number=index,
            total_lines=total_lines,
        )
        for index, sentence in enumerate(sentences)
    ]
    return pd.DataFrame(records)


def label_badge(label: str) -> str:
    """Return HTML for a colored label badge."""

    style = LABEL_STYLES.get(label, LABEL_STYLES["BACKGROUND"])
    return (
        f"<span class='label-badge' "
        f"style='background:{style['bg']}; color:{style['fg']}; border-color:{style['border']};'>"
        f"{html.escape(label)}</span>"
    )


def render_section_intro(title: str, copy: str) -> None:
    """Render a section title and short description."""

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{html.escape(title)}</div>
            <div class="section-copy">{html.escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(message: str) -> None:
    """Render a friendly empty state."""

    st.markdown(f"<div class='empty-state'>{html.escape(message)}</div>", unsafe_allow_html=True)


def render_prediction_card(result: dict, total_lines: int | None = None) -> None:
    """Render the single-sentence prediction result card."""

    label = result["predicted_label"]
    style = LABEL_STYLES.get(label, LABEL_STYLES["BACKGROUND"])
    confidence = result.get("confidence_or_score", np.nan)
    confidence_text = "Not available" if pd.isna(confidence) else f"{confidence:.3f}"
    relative_position = result.get("relative_position", np.nan)
    position_text = "Not provided" if pd.isna(relative_position) else f"{relative_position:.3f}"
    escaped_explanation = html.escape(result.get("label_explanation", ""))

    st.markdown(
        f"""
        <div class="result-card" style="border-left-color:{style['border']};">
            <div class="result-title">Predicted Role</div>
            <div class="result-role">{label_badge(label)} &nbsp; {html.escape(label)}</div>
            <div>{escaped_explanation}</div>
            <div class="result-meta">
                Confidence / decision estimate: <strong>{confidence_text}</strong><br>
                Relative Position: <strong>{position_text}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not pd.isna(confidence):
        st.progress(float(np.clip(confidence, 0, 1)))


def style_prediction_table(df: pd.DataFrame):
    """Apply light styling to prediction tables."""

    display_df = df.copy()
    if "confidence_or_score" in display_df.columns:
        display_df["confidence_or_score"] = display_df["confidence_or_score"].astype(float)

    def color_label(value: str) -> str:
        style = LABEL_STYLES.get(str(value), LABEL_STYLES["BACKGROUND"])
        return (
            f"background-color: {style['bg']}; color: {style['fg']}; "
            f"font-weight: 700; border-left: 4px solid {style['border']};"
        )

    styler = display_df.style.format(
        {
            "relative_position": "{:.3f}",
            "confidence_or_score": "{:.3f}",
            "decision_score": "{:.3f}",
        },
        na_rep="",
    )
    if "predicted_label" in display_df.columns:
        if hasattr(styler, "map"):
            styler = styler.map(color_label, subset=["predicted_label"])
        else:
            styler = styler.applymap(color_label, subset=["predicted_label"])
    return styler


def render_structured_abstract(prediction_df: pd.DataFrame) -> None:
    """Render predictions grouped as a structured abstract."""

    for label in LABEL_ORDER:
        group_df = prediction_df[prediction_df["predicted_label"] == label]
        style = LABEL_STYLES[label]
        st.markdown(label_badge(label), unsafe_allow_html=True)

        if group_df.empty:
            st.markdown(
                "<div class='sentence-card' style='color:#64748B;'>No sentences assigned to this role.</div>",
                unsafe_allow_html=True,
            )
            continue

        for _, row in group_df.iterrows():
            sentence = html.escape(str(row["sentence"]))
            line_number = int(row["line_number"])
            confidence = row.get("confidence_or_score", np.nan)
            confidence_text = "" if pd.isna(confidence) else f"Score: {confidence:.3f}"
            st.markdown(
                f"""
                <div class="sentence-card" style="border-left: 6px solid {style['border']};">
                    <div style="color:#64748B; font-size:0.82rem; margin-bottom:0.25rem;">
                        Line {line_number} &nbsp; {html.escape(confidence_text)}
                    </div>
                    <div style="color:#1E293B;">{sentence}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def create_dashboard_charts(prediction_df: pd.DataFrame) -> None:
    """Create interactive Plotly dashboard charts."""

    chart_df = prediction_df.copy()
    chart_df["label_rank"] = chart_df["predicted_label"].apply(lambda label: LABEL_ORDER.index(label))

    label_counts = (
        chart_df["predicted_label"]
        .value_counts()
        .reindex(LABEL_ORDER, fill_value=0)
        .reset_index()
    )
    label_counts.columns = ["predicted_label", "count"]

    color_map = {label: LABEL_STYLES[label]["border"] for label in LABEL_ORDER}

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            label_counts,
            x="predicted_label",
            y="count",
            color="predicted_label",
            color_discrete_map=color_map,
            title="Label Distribution",
            labels={"predicted_label": "Predicted Label", "count": "Sentence Count"},
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            chart_df,
            x="line_number",
            y="label_rank",
            markers=True,
            color="predicted_label",
            color_discrete_map=color_map,
            title="Sentence Role Timeline",
            labels={"line_number": "Line Number", "label_rank": "Sentence Role"},
            hover_data=["sentence", "confidence_or_score", "relative_position"],
        )
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(LABEL_ORDER))), ticktext=LABEL_ORDER)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.scatter(
            chart_df,
            x="relative_position",
            y="predicted_label",
            color="predicted_label",
            color_discrete_map=color_map,
            title="Relative Position vs Label",
            labels={"relative_position": "Relative Position", "predicted_label": "Predicted Label"},
            hover_data=["line_number", "sentence", "confidence_or_score"],
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        score_df = chart_df.dropna(subset=["confidence_or_score"])
        if score_df.empty:
            render_empty_state("Confidence or decision scores are not available for this model.")
        else:
            fig = px.histogram(
                score_df,
                x="confidence_or_score",
                color="predicted_label",
                color_discrete_map=color_map,
                nbins=12,
                title="Confidence Distribution",
                labels={"confidence_or_score": "Confidence / Decision Estimate"},
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)


def render_workflow_step(index: int, title: str, copy: str) -> None:
    """Render one compact project workflow card."""

    st.markdown(
        f"""
        <div class="workflow-step">
            <div class="workflow-index">Step {index}</div>
            <div class="workflow-title">{html.escape(title)}</div>
            <div class="workflow-copy">{html.escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_project_workflow() -> None:
    """Show how the project was built from raw data to deployed app."""

    st.markdown("#### How I Built This Project")
    workflow_steps = [
        (
            "Dataset selection",
            "Used the PubMed 20k RCT split first instead of the much larger 200k split so the full experiment is reproducible in Colab.",
        ),
        (
            "Raw text parsing",
            "Parsed train.txt, dev.txt, and test.txt by abstract boundaries, label prefixes, and sentence text.",
        ),
        (
            "Sentence-level preprocessing",
            "Converted each sentence into target, text, line_number, total_lines, and relative_position columns.",
        ),
        (
            "EDA outputs",
            "Generated raw line counts, label distributions, sentence length summaries, and saved figures instead of relying only on notebook output.",
        ),
        (
            "Baseline modeling",
            "Trained TF-IDF + Logistic Regression and TF-IDF + LinearSVC as text-only baselines.",
        ),
        (
            "My feature idea",
            "Added abstract-structure features because medical abstracts usually follow Background, Objective, Methods, Results, Conclusions order.",
        ),
        (
            "Evaluation and error analysis",
            "Compared validation metrics, selected the best macro F1 model, evaluated on test data, saved confusion matrix and misclassified examples.",
        ),
        (
            "Application layer",
            "Built this Streamlit interface for single sentence prediction, full abstract analysis, batch prediction, dashboards, and project storytelling.",
        ),
    ]

    for start in range(0, len(workflow_steps), 2):
        col1, col2 = st.columns(2)
        with col1:
            title, copy = workflow_steps[start]
            render_workflow_step(start + 1, title, copy)
        if start + 1 < len(workflow_steps):
            with col2:
                title, copy = workflow_steps[start + 1]
                render_workflow_step(start + 2, title, copy)


def render_artifact_card(name: str, path: Path, note: str) -> None:
    """Render a saved file artifact card."""

    status = "Available" if path.exists() else "Missing"
    st.markdown(
        f"""
        <div class="artifact-card">
            <div class="artifact-name">{html.escape(name)} · {status}</div>
            <div class="workflow-copy">{html.escape(note)}</div>
            <div class="artifact-path">{html.escape(str(path.relative_to(PROJECT_ROOT)))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_table_output(title: str, path: Path, height: int = 260, max_rows: int | None = None) -> None:
    """Render a saved CSV output with graceful empty handling."""

    st.markdown(f"#### {title}")
    df = load_csv_if_exists(path)
    if df.empty:
        render_empty_state(f"{path.name} is not available yet.")
        return
    if max_rows is not None:
        df = df.head(max_rows)
    st.dataframe(df, use_container_width=True, height=height)


def render_figure_output(title: str, path: Path, caption: str) -> None:
    """Render a saved figure output if present."""

    st.markdown(f"#### {title}")
    if not path.exists():
        render_empty_state(f"{path.name} is not available yet.")
        return
    st.image(str(path), caption=caption, use_container_width=True)


def render_saved_outputs() -> None:
    """Display the main saved outputs produced by the project pipeline."""

    st.markdown("#### Saved Output Gallery")
    col1, col2 = st.columns(2)
    with col1:
        render_figure_output(
            "Label Distribution",
            FIGURE_DIR / "label_distribution.png",
            "Class balance across train, validation, and test splits.",
        )
        render_figure_output(
            "Model Comparison",
            FIGURE_DIR / "model_comparison.png",
            "Validation accuracy, macro F1, and weighted F1 across baseline models.",
        )
    with col2:
        render_figure_output(
            "Sentence Length Histogram",
            FIGURE_DIR / "text_length_histogram.png",
            "Distribution of sentence lengths in the PubMed 20k splits.",
        )
        render_figure_output(
            "Confusion Matrix",
            FIGURE_DIR / "confusion_matrix.png",
            "Best model mistakes and correct predictions by class.",
        )


def render_output_tables() -> None:
    """Render important saved tables from the experiment."""

    table_tabs = st.tabs(
        [
            "Data Checks",
            "Model Metrics",
            "Reports",
            "Interpretation",
            "Predictions",
        ]
    )

    with table_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            render_table_output("Raw Line Counts", TABLE_DIR / "raw_line_counts.csv", height=160)
            render_table_output("Label Mapping", TABLE_DIR / "label_mapping.csv", height=170)
        with col2:
            render_table_output("Text Length Summary", TABLE_DIR / "text_length_summary.csv", height=220)
            render_table_output("Processed Data Summary", PROCESSED_DATA_DIR / "processed_data_summary.csv", height=160)

    with table_tabs[1]:
        render_table_output("Model Comparison", TABLE_DIR / "model_comparison.csv", height=180)
        render_table_output("Best Model Test Metrics", TABLE_DIR / "best_model_test_metrics.csv", height=140)

    with table_tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            render_table_output(
                "Best Model Test Classification Report",
                TABLE_DIR / "best_model_test_classification_report.csv",
                height=300,
            )
        with col2:
            render_table_output("Confusion Matrix Table", TABLE_DIR / "confusion_matrix.csv", height=260)

    with table_tabs[3]:
        render_table_output("Top TF-IDF Words by Class", TABLE_DIR / "top_words_by_class.csv", height=430)

    with table_tabs[4]:
        col1, col2 = st.columns(2)
        with col1:
            render_table_output(
                "Misclassified Examples",
                PREDICTION_DIR / "misclassified_examples.csv",
                height=430,
                max_rows=30,
            )
        with col2:
            render_table_output("Demo Predictions", PREDICTION_DIR / "demo_predictions.csv", height=250)


def render_artifact_overview() -> None:
    """Show the repository files that make the project reproducible."""

    st.markdown("#### Reproducible Project Artifacts")
    artifacts = [
        ("Raw training data", PROJECT_ROOT / "data/raw/20k_abstracts/train.txt", "Original PubMed 20k RCT training text file."),
        ("Processed training CSV", PROCESSED_DATA_DIR / "train_processed.csv", "Cleaned sentence-level training data with structural features."),
        ("Main notebook", PROJECT_ROOT / "notebooks/01_medical_abstract_sentence_classification.ipynb", "Step-by-step notebook version of the experiment."),
        ("Training script", PROJECT_ROOT / "src/train_baseline_models.py", "Reusable script that trains models and regenerates outputs."),
        ("Best model", MODEL_DIR / "best_model.joblib", "Saved model used by this Streamlit application."),
        ("Model metadata", MODEL_DIR / "best_model_metadata.json", "Metadata describing the selected best model and input mode."),
    ]

    for start in range(0, len(artifacts), 2):
        col1, col2 = st.columns(2)
        with col1:
            render_artifact_card(*artifacts[start])
        if start + 1 < len(artifacts):
            with col2:
                render_artifact_card(*artifacts[start + 1])


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to downloadable CSV bytes."""

    return df.to_csv(index=False).encode("utf-8")


def parse_uploaded_file(uploaded_file) -> list[str] | pd.DataFrame:
    """Read a TXT or CSV upload for batch prediction."""

    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    text = uploaded_file.read().decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_batch_predictions(model: object, metadata: dict, uploaded_data) -> pd.DataFrame:
    """Create predictions for uploaded TXT lines or CSV rows."""

    if isinstance(uploaded_data, pd.DataFrame):
        df = uploaded_data.copy()
        text_column = None
        for candidate in ["sentence", "text", "abstract_sentence"]:
            if candidate in df.columns:
                text_column = candidate
                break
        if text_column is None:
            raise ValueError("CSV must include a `sentence` or `text` column.")

        total_lines = int(df["total_lines"].iloc[0]) if "total_lines" in df.columns else len(df)
        records = []
        for index, row in df.iterrows():
            line_number = int(row["line_number"]) if "line_number" in df.columns else index
            row_total_lines = int(row["total_lines"]) if "total_lines" in df.columns else total_lines
            records.append(
                predict_sentence(
                    model=model,
                    metadata=metadata,
                    sentence=str(row[text_column]),
                    line_number=line_number,
                    total_lines=row_total_lines,
                )
            )
        return pd.DataFrame(records)

    sentences = list(uploaded_data)
    return analyze_sentences(model, metadata, sentences)


def render_hero() -> None:
    """Render the product-style hero section."""

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Healthcare AI / Medical NLP</div>
            <div class="hero-title">Medical Abstract Sentence Classifier</div>
            <div class="hero-subtitle">A Healthcare NLP tool for sentence-role classification and evidence extraction</div>
            <div class="hero-copy">
                Transform unstructured PubMed-style abstracts into structured sentence-level evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    metrics = [
        ("5", "Sentence Roles"),
        ("Position-aware", "ML Pipeline"),
        ("Interactive", "Streamlit App"),
    ]
    for col, (value, label) in zip([col1, col2, col3], metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{html.escape(value)}</div>
                    <div class="metric-label">{html.escape(label)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_sidebar(metadata: dict) -> None:
    """Render organized sidebar project information."""

    st.sidebar.title("Project Control Panel")

    with st.sidebar.expander("Current Model", expanded=True):
        st.markdown(
            f"""
            <div class="sidebar-block">
                <strong>{html.escape(metadata.get("best_model_name", "Unknown model"))}</strong><br>
                Model file: <code>{html.escape(metadata.get("model_filename", "best_model.joblib"))}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar.expander("Input Mode", expanded=True):
        input_mode = metadata.get("input_mode", "text")
        if input_mode == "position":
            st.write("Text plus structural abstract features")
            st.caption("Uses line_number, total_lines, and relative_position.")
        else:
            st.write("Text-only sentence classification")

    with st.sidebar.expander("Labels", expanded=True):
        for label in LABEL_ORDER:
            st.markdown(label_badge(label), unsafe_allow_html=True)

    with st.sidebar.expander("Project Disclaimer", expanded=False):
        st.warning(
            "This tool is for research and educational use only. It classifies sentence roles "
            "in medical abstracts and does not provide clinical diagnosis or medical advice."
        )

    with st.sidebar.expander("GitHub / Project Info", expanded=False):
        st.markdown(
            """
            Repository placeholder:

            `zhld25-lab/Medical-Abstract-Sentence-Classification`

            Suggested use: RA interview demo, Healthcare AI portfolio, NLP project walkthrough.
            """
        )


def render_single_sentence_tab(model: object, metadata: dict) -> None:
    """Render single sentence prediction UI."""

    render_section_intro(
        "Single Sentence Prediction",
        "Classify one PubMed-style abstract sentence and inspect its predicted sentence role.",
    )

    with st.container(border=True):
        col1, col2 = st.columns([2.2, 1])
        with col1:
            sample_sentence = st.selectbox("Sample sentences", SAMPLE_SENTENCES)
            sentence = st.text_area(
                "Medical sentence",
                value=sample_sentence,
                height=130,
                placeholder="Paste a medical abstract sentence to get started.",
            )
        with col2:
            st.markdown("#### Position Features")
            line_number = st.number_input("Line number (0-based)", min_value=0, value=1, step=1)
            total_lines = st.number_input("Total lines in abstract", min_value=1, value=10, step=1)
            st.caption("Position is optional, but improves this project's best model.")

        if st.button("Predict Sentence", type="primary"):
            if not sentence.strip():
                st.warning("Paste a medical abstract sentence to get started.")
            else:
                result = predict_sentence(
                    model=model,
                    metadata=metadata,
                    sentence=sentence.strip(),
                    line_number=int(line_number),
                    total_lines=int(total_lines),
                )
                st.session_state["latest_predictions"] = pd.DataFrame([result])
                render_prediction_card(result, total_lines=int(total_lines))

    if "latest_predictions" not in st.session_state:
        render_empty_state("Paste a medical abstract sentence to get started.")


def render_full_abstract_tab(model: object, metadata: dict) -> None:
    """Render full abstract analysis UI."""

    render_section_intro(
        "Full Abstract Analysis",
        "Paste a full abstract to classify every sentence, then switch between table and structured abstract views.",
    )

    with st.container(border=True):
        abstract_text = st.text_area(
            "Medical abstract",
            value=SAMPLE_ABSTRACT,
            height=230,
            placeholder="Paste a full PubMed-style abstract here.",
        )

        if st.button("Analyze Abstract", type="primary"):
            sentences = split_abstract_sentences(abstract_text)
            if not sentences:
                st.warning("Paste a medical abstract to analyze.")
            else:
                prediction_df = analyze_sentences(model, metadata, sentences)
                st.session_state["abstract_predictions"] = prediction_df
                st.session_state["latest_predictions"] = prediction_df
                st.success(f"Analyzed {len(prediction_df)} sentences.")

    prediction_df = st.session_state.get("abstract_predictions")
    if prediction_df is None or prediction_df.empty:
        render_empty_state("Paste a medical abstract and select Analyze Abstract to see structured results.")
        return

    table_view, structured_view = st.tabs(["Table View", "Structured Abstract View"])
    with table_view:
        display_cols = ["line_number", "sentence", "relative_position", "predicted_label", "confidence_or_score"]
        st.dataframe(style_prediction_table(prediction_df[display_cols]), use_container_width=True, height=420)
        st.download_button(
            "Download Results as CSV",
            data=dataframe_to_csv_bytes(prediction_df[display_cols]),
            file_name="abstract_sentence_predictions.csv",
            mime="text/csv",
        )

    with structured_view:
        render_structured_abstract(prediction_df)


def render_batch_tab(model: object, metadata: dict) -> None:
    """Render batch file prediction UI."""

    render_section_intro(
        "Batch File Prediction",
        "Upload a TXT file with one sentence per line or a CSV file with a sentence/text column.",
    )

    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
        st.caption("CSV files can optionally include line_number and total_lines columns.")

        if st.button("Run Batch Prediction", type="primary"):
            if uploaded_file is None:
                st.warning("Upload a CSV or TXT file to run batch prediction.")
            else:
                try:
                    uploaded_data = parse_uploaded_file(uploaded_file)
                    batch_df = build_batch_predictions(model, metadata, uploaded_data)
                    st.session_state["batch_predictions"] = batch_df
                    st.session_state["latest_predictions"] = batch_df
                    st.success(f"Generated predictions for {len(batch_df)} sentences.")
                except Exception as exc:
                    st.error(f"Could not process the file: {exc}")

    batch_df = st.session_state.get("batch_predictions")
    if batch_df is None or batch_df.empty:
        render_empty_state("Upload a CSV or TXT file to run batch prediction.")
        return

    display_cols = ["line_number", "sentence", "relative_position", "predicted_label", "confidence_or_score"]
    st.dataframe(style_prediction_table(batch_df[display_cols]), use_container_width=True, height=460)
    st.download_button(
        "Download Results as CSV",
        data=dataframe_to_csv_bytes(batch_df[display_cols]),
        file_name="batch_sentence_predictions.csv",
        mime="text/csv",
    )


def render_dashboard_tab(model: object, metadata: dict) -> None:
    """Render analytics dashboard UI."""

    render_section_intro(
        "Dashboard",
        "Explore label distribution, sentence-role timeline, relative position patterns, and confidence estimates.",
    )

    prediction_df = st.session_state.get("latest_predictions")
    if prediction_df is None or prediction_df.empty or len(prediction_df) < 2:
        sample_predictions = analyze_sentences(model, metadata, split_abstract_sentences(SAMPLE_ABSTRACT))
        prediction_df = sample_predictions
        st.info("Showing dashboard with the sample abstract. Analyze your own abstract or batch file to update it.")

    create_dashboard_charts(prediction_df)

    with st.container(border=True):
        st.markdown("#### Active Prediction Data")
        display_cols = ["line_number", "sentence", "relative_position", "predicted_label", "confidence_or_score"]
        st.dataframe(style_prediction_table(prediction_df[display_cols]), use_container_width=True, height=300)


def render_model_details_tab(metadata: dict) -> None:
    """Render project process, saved outputs, metadata, metrics, and limitations."""

    render_section_intro(
        "Model Details",
        "Review how the project was built, what outputs were generated, and how the model performed.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{html.escape(metadata.get("best_model_name", "Unknown"))}</div>
                <div class="metric-label">Current Model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{html.escape(metadata.get("input_mode", "text"))}</div>
                <div class="metric-label">Input Mode</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">PubMed 20k RCT</div>
                <div class="metric-label">Dataset Split</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    project_tabs = st.tabs(
        [
            "Project Workflow",
            "Saved Figures",
            "Saved Tables",
            "Artifacts",
            "Limitations",
        ]
    )

    with project_tabs[0]:
        with st.container(border=True):
            render_project_workflow()

        with st.container(border=True):
            st.markdown("#### My Original Contribution")
            st.markdown(
                """
                The project is not just a text classifier. The core experiment compares
                pure text-based classification with text plus abstract-structure features.
                In medical abstracts, sentence position is meaningful, so I engineered
                `line_number`, `total_lines`, and `relative_position` and compared whether
                they improved performance over plain TF-IDF baselines.
                """
            )

    with project_tabs[1]:
        with st.container(border=True):
            render_saved_outputs()

    with project_tabs[2]:
        with st.container(border=True):
            render_output_tables()

    with project_tabs[3]:
        with st.container(border=True):
            render_artifact_overview()

    with project_tabs[4]:
        with st.container(border=True):
            st.markdown("#### Limitations and Future Work")
            st.markdown(
                """
                - This model classifies sentences independently and does not fully model abstract-level sequence dependency.
                - TF-IDF models cannot deeply understand medical semantics.
                - PubMed 20k RCT results may not fully generalize to other clinical text.
                - Future work should include SciBERT or BioBERT fine-tuning, abstract-level sequence modeling, and larger PubMed 200k RCT experiments.
                """
            )


def main() -> None:
    """Run the Streamlit application."""

    inject_css()

    try:
        model, metadata = load_model_and_metadata()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar(metadata)
    render_hero()

    tabs = st.tabs(
        [
            "Single Sentence Prediction",
            "Full Abstract Analysis",
            "Batch File Prediction",
            "Dashboard",
            "Model Details",
        ]
    )

    with tabs[0]:
        render_single_sentence_tab(model, metadata)
    with tabs[1]:
        render_full_abstract_tab(model, metadata)
    with tabs[2]:
        render_batch_tab(model, metadata)
    with tabs[3]:
        render_dashboard_tab(model, metadata)
    with tabs[4]:
        render_model_details_tab(metadata)

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Disclaimer:</strong> This tool is for research and educational use only.
            It classifies sentence roles in medical abstracts and does not provide clinical diagnosis or medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
