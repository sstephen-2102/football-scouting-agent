"""Member B — player score lookup."""

import streamlit as st

from app.components.results import artifact_gate
from src.common.exceptions import NotBuiltError
from src.modeling.value import predict_player_scores


def render() -> None:
    st.title("Player Scores")
    st.caption("Look up a predicted next value and P(undervalued) after Member B’s export exists.")
    ready = artifact_gate("scores")
    player_id = st.number_input("Player ID", min_value=1, value=418560, step=1)
    if st.button("Score player"):
        if not ready:
            st.stop()
        try:
            predict_player_scores(model=None, player_seasons=None, feature_columns=[])  # type: ignore[arg-type]
        except NotBuiltError as exc:
            st.warning(str(exc))
            st.caption(f"Lookup for player {int(player_id)} is wired; Member B fills the score table next.")
