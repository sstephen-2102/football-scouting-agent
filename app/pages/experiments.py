"""Member B — experiment board."""

import streamlit as st

from app.components.results import artifact_gate, show_warnings


def render() -> None:
    st.title("Experiment Board")
    st.caption(
        "Member B. R0 ridge, R1 gradient boosting, C1 undervalued classifier, S1 LSTM/1D-CNN. "
        "Locked metrics: MAE/RMSE/Spearman and PR AUC / ROC AUC / Brier — not accuracy."
    )
    ready = artifact_gate("scores")
    if ready:
        st.write("R0 vs R1 vs S1 comparison tables will render here from `outputs/experiments/`.")
    else:
        st.write(
            "When `python -m src.pipeline score` has written comparison tables, this page will show "
            "the locked test-season comparison."
        )
    show_warnings(
        [
            "Threshold for “undervalued” is frozen on validation residuals, never retuned on test.",
            "Portrait CNNs are out of scope.",
        ]
    )
