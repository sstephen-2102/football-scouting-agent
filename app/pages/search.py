"""Member C — XI search lab."""

import streamlit as st

from app.components.results import artifact_gate, show_warnings
from src.common.exceptions import NotBuiltError
from src.search.xi import export_best_xi


def render() -> None:
    st.title("Search Lab")
    st.caption(
        "Member C. Uninformed vs informed XI search. "
        "State = filled slots; heuristic = predicted XI quality."
    )
    ready = artifact_gate("search")
    club_id = st.number_input("Club ID", min_value=1, value=11, step=1)
    season = st.number_input("Season start year", min_value=2018, value=2024, step=1)
    raw = st.text_area(
        "Shortlist player IDs (comma-separated, 11–40)",
        "418560,371998,38253,88755,35747,69633,148455,8198,105470,4512,22367",
    )
    if st.button("Search XI") and ready:
        ids = [int(item) for item in raw.replace(",", " ").split() if item.strip().isdigit()]
        try:
            export_best_xi(int(club_id), int(season), ids)
        except NotBuiltError as exc:
            st.warning(str(exc))
    show_warnings(["Compare BFS/random against greedy/A* at a node expansion cap."])
