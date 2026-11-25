"""Member C — bid/pass policy."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.results import artifact_gate, show_warnings
from src.artifacts import artifact_path
from src.common.config import OUTPUTS
from src.common.exceptions import InvalidParameterError, NotBuiltError
from src.policy.bid import decide_bid


def render() -> None:
    st.title("Bid Policy")
    st.caption(
        "Member C. Bid or pass under remaining funds. "
        "Compare ε-greedy, UCB, Q-learning, and greedy-by-undervalued-score."
    )
    ready = artifact_gate("policy")

    if ready:
        comparison_path = artifact_path("policy").parent / "policy_comparison.csv"
        if comparison_path.is_file():
            st.subheader("Policy Comparison (held-out test shortlist)")
            st.dataframe(
                pd.read_csv(comparison_path), use_container_width=True, hide_index=True
            )
        fig_path = OUTPUTS / "figures" / "policy_regret.png"
        if fig_path.is_file():
            st.image(str(fig_path), caption="Policy mean return vs greedy")

    st.subheader("Query trained Q-table")
    player_id = st.number_input("Player ID", min_value=1, value=418560, step=1)
    remaining = st.number_input(
        "Remaining budget EUR", min_value=0.0, value=25_000_000.0, step=1_000_000.0
    )
    hole = st.selectbox("Position hole", ["FWD", "MID", "DEF", "GK", "ST", "CM", "CB"])
    if st.button("Decide bid/pass") and ready:
        try:
            st.json(decide_bid(int(player_id), float(remaining), str(hole)))
        except (NotBuiltError, InvalidParameterError) as exc:
            st.warning(str(exc))

    show_warnings(
        [
            "Reward is predicted residual (undervaluation) if signed, not a betting outcome.",
        ]
    )
