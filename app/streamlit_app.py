"""Streamlit interface for the medical abstract sentence classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import predict_sentence_role


LABEL_EXPLANATIONS = {
    "BACKGROUND": "This sentence provides context or explains why the research area matters.",
    "OBJECTIVE": "This sentence states the aim, question, or hypothesis of the study.",
    "METHODS": "This sentence describes the study design, participants, intervention, or measurement process.",
    "RESULTS": "This sentence reports findings, measurements, statistical results, or observed outcomes.",
    "CONCLUSIONS": "This sentence summarizes the meaning, implication, or recommendation from the study.",
}

SAMPLE_SENTENCES = [
    "The aim of this study was to evaluate the effect of treatment.",
    "Participants were randomly assigned to receive placebo or active medication.",
    "The intervention group showed a statistically significant reduction in symptoms.",
    "These findings suggest that the treatment may improve patient outcomes.",
]


st.set_page_config(page_title="Medical Abstract Sentence Classifier")
st.title("Medical Abstract Sentence Classifier")

sample_sentence = st.selectbox("Sample sentences", SAMPLE_SENTENCES)
sentence = st.text_area("Medical sentence", value=sample_sentence, height=120)

col1, col2 = st.columns(2)
with col1:
    line_number = st.number_input("Line number", min_value=0, value=0, step=1)
with col2:
    total_lines = st.number_input("Total lines in abstract", min_value=1, value=10, step=1)

if st.button("Predict sentence role", type="primary"):
    if not sentence.strip():
        st.warning("Enter a sentence to classify.")
    else:
        try:
            label = predict_sentence_role(
                sentence=sentence.strip(),
                line_number=int(line_number),
                total_lines=int(total_lines),
            )
            st.subheader(label)
            st.write(LABEL_EXPLANATIONS.get(label, "No explanation is available for this label."))
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("Train the models first with: `python src/train_baseline_models.py`")
