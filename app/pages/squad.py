"""Member C — MIP squad builder."""

import streamlit as st

from app.components.results import artifact_gate, show_warnings
from src.common.exceptions import NotBuiltError
from src.optimize.squad import export_squad


def render() -> None:
    st.title("Squad Builder")
    st.caption(
        "Member C. MIP under budget *B* versus greedy “buy the top predicted values until broke.”"
    )
    ready = artifact_gate("squad")
    club_id = st.number_input("Club ID", min_value=1, value=11, step=1)
    season = st.number_input("Season start year", min_value=2018, value=2024, step=1)
    budget = st.number_input("Budget EUR", min_value=1.0, value=80_000_000.0, step=1_000_000.0)
    if st.button("Allocate squad") and ready:
        try:
            export_squad(int(club_id), int(season), float(budget))
        except NotBuiltError as exc:
            st.warning(str(exc))
    show_warnings(
        [
            "Same *B* for MIP and greedy. Negative results stay in scope.",
        ]
    )
