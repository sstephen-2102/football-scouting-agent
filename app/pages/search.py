"""Member C — XI search lab."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.results import artifact_gate, show_warnings
from src.artifacts import artifact_path
from src.common.exceptions import InvalidParameterError, NotBuiltError
from src.search.xi import export_best_xi


def render() -> None:
    st.title("Search Lab")
    st.caption(
        "Member C. Uninformed vs informed XI search. "
        "State = filled slots; heuristic = predicted XI quality."
    )
    ready = artifact_gate("search")

    if ready:
        payload = json.loads(artifact_path("search").read_text(encoding="utf-8"))
        uninformed = payload.get("uninformed", {})
        informed = payload.get("informed", {})
        comparison = payload.get("comparison", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BFS Filled Slots", f"{len(uninformed.get('xi', {}))}/11")
        c2.metric("A* Filled Slots", f"{len(informed.get('xi', {}))}/11")
        c3.metric("A* Quality", f"€{informed.get('quality', 0.0) / 1e6:.2f}M")
        c4.metric(
            "A* − BFS Gain",
            f"€{comparison.get('informed_minus_uninformed', 0.0) / 1e6:.2f}M",
        )

        st.subheader("Best XI (A*)")
        xi = informed.get("xi", {})
        if xi:
            st.dataframe(
                pd.DataFrame(
                    [{"slot": slot, "player_id": pid} for slot, pid in xi.items()]
                ),
                use_container_width=True,
                hide_index=True,
            )

        fig_path = Path(comparison.get("figure_path", ""))
        if fig_path.is_file():
            st.image(str(fig_path), caption="Uninformed (BFS) vs Informed (A*) search")

    with st.expander("Re-run search with custom shortlist"):
        club_id = st.number_input("Club ID", min_value=1, value=11, step=1)
        season = st.number_input("Season start year", min_value=2018, value=2024, step=1)
        raw = st.text_area(
            "Shortlist player IDs (comma-separated, 11–40; leave blank for default test shortlist)",
            "",
        )
        if st.button("Search XI"):
            ids = [
                int(item)
                for item in raw.replace(",", " ").split()
                if item.strip().isdigit()
            ]
            try:
                export_best_xi(int(club_id), int(season), ids or None)
                st.rerun()
            except (NotBuiltError, InvalidParameterError, FileNotFoundError) as exc:
                st.warning(str(exc))

    show_warnings(["Compare BFS/random against greedy/A* at a node expansion cap."])
