"""Member C — bid/pass policy."""

import streamlit as st

from app.components.results import artifact_gate, show_warnings
from src.common.exceptions import NotBuiltError
from src.policy.bid import decide_bid


def render() -> None:
    st.title("Bid Policy")
    st.caption(
        "Member C. Bid or pass under remaining funds. "
        "Compare ε-greedy, UCB, Q-learning, and greedy-by-undervalued-score."
    )
    ready = artifact_gate("policy")
    player_id = st.number_input("Player ID", min_value=1, value=418560, step=1)
    remaining = st.number_input("Remaining budget EUR", min_value=0.0, value=25_000_000.0, step=1_000_000.0)
    hole = st.selectbox("Position hole", ["ST", "CM", "CB", "GK", "RW", "LW"])
    if st.button("Decide bid/pass") and ready:
        try:
            st.json(decide_bid(int(player_id), float(remaining), str(hole)))
        except NotBuiltError as exc:
            st.warning(str(exc))
    show_warnings(
        [
            "Reward is predicted residual (undervaluation) if signed, not a betting outcome.",
        ]
    )
