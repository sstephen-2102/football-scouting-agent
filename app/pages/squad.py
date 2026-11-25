"""Member C — MIP squad builder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.results import artifact_gate, show_warnings
from src.artifacts import artifact_path
from src.common.exceptions import InvalidParameterError, NotBuiltError
from src.optimize.squad import export_squad


def render() -> None:
    st.title("Squad Builder")
    st.caption(
        "Member C. MIP under budget *B* versus greedy “buy the top predicted values until broke.”"
    )
    ready = artifact_gate("squad")

    default_budget = 290_000_000.0
    if ready:
        csv_path = artifact_path("squad")
        manifest_path = csv_path.parent / "allocation_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            default_budget = float(manifest.get("budget_eur", default_budget))
            mip = manifest.get("mip", {})
            greedy = manifest.get("greedy", {})
            comparison = manifest.get("comparison", {})

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Budget B", f"€{default_budget / 1e6:.1f}M")
            c2.metric(
                "MIP Quality",
                f"€{mip.get('quality', 0.0) / 1e6:.2f}M",
                f"{len(mip.get('player_ids', []))}/11 ({mip.get('status', '')})",
            )
            c3.metric(
                "Greedy Quality",
                f"€{greedy.get('quality', 0.0) / 1e6:.2f}M",
                f"{len(greedy.get('player_ids', []))}/11 ({greedy.get('status', '')})",
            )
            c4.metric(
                "MIP − Greedy Gap",
                f"€{comparison.get('quality_gap', 0.0) / 1e6:.2f}M",
            )

            fig_path = Path(comparison.get("figure_path", ""))
            if fig_path.is_file():
                st.image(str(fig_path), caption="MIP vs Greedy squad quality and spend")

        st.subheader("Selected Players (`squad.csv`)")
        st.dataframe(pd.read_csv(csv_path), use_container_width=True, hide_index=True)

    with st.expander("Re-run allocation"):
        club_id = st.number_input("Club ID", min_value=1, value=11, step=1)
        season = st.number_input("Season start year", min_value=2018, value=2024, step=1)
        budget = st.number_input(
            "Budget EUR", min_value=1.0, value=default_budget, step=5_000_000.0
        )
        if st.button("Allocate squad"):
            try:
                export_squad(int(club_id), int(season), float(budget))
                st.rerun()
            except (NotBuiltError, InvalidParameterError, FileNotFoundError) as exc:
                st.warning(str(exc))

    show_warnings(["Same *B* for MIP and greedy. Negative results stay in scope."])
