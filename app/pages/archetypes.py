"""Member A — clustering lab."""

import streamlit as st

from app.components.results import artifact_gate, show_warnings


def render() -> None:
    st.title("Archetype Lab")
    st.caption(
        "Member A. Cluster train players from rolling stats without market value in the fit, "
        "then profile clusters by later value and position."
    )
    ready = artifact_gate("archetypes")
    if ready:
        st.write("Cluster profiles will render here from `outputs/clustering/archetype_labels.parquet`.")
    else:
        st.write(
            "When `python -m src.pipeline cluster` has written labels, this page will show "
            "goals/90, assists/90, cards/90, minutes, age, and later median value by archetype."
        )
    show_warnings(
        [
            "Fit on train rolling stats only.",
            "Value is used after the fact to profile clusters, never inside k-means / GMM.",
        ]
    )
