"""Executive dashboard — PEAS headline, artifact board, research questions."""

import pandas as pd
import streamlit as st

from app.artifacts import status_table
from app.components.results import metric_row, show_warnings
from src.agent.peas import DISCLAIMER
from src.common.config import (
    BIG5_COMPETITION_IDS,
    MIN_APPEARANCES,
    SNAPSHOT_DATE,
)

RESEARCH_QUESTIONS = [
    ("Archetype Lab", "Do archetypes differ in value and playing style?"),
    ("Search Lab", "Does informed search match or beat uninformed search for XI quality at a node budget?"),
    ("Experiment Board", "Does a tuned GB beat linear regression on next value?"),
    ("Experiment Board", "Does a 6–8 feature scorecard match the full model?"),
    ("Squad Builder", "Does MIP under B beat “buy the top predicted values until broke”?"),
    ("Bid Policy", "Do bandits/Q-learning improve cumulative undervalued signings vs greedy, same B?"),
]


def render() -> None:
    st.title("Executive Dashboard")
    st.markdown(
        "An interactive research lab on Transfermarkt player-scores: archetypes, "
        "market-value models, XI search, MIP allocation, and a bid/pass policy. "
        "Educational research, not transfer advice."
    )
    rows = status_table()
    ready = sum(row["state"] == "ready" for row in rows)
    metric_row(
        {
            "Snapshot": SNAPSHOT_DATE,
            "Artifacts ready": f"{ready} / {len(rows)}",
            "Min appearances": MIN_APPEARANCES,
            "Leagues": " · ".join(BIG5_COMPETITION_IDS),
        }
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Questions that must have graphs")
        for page, question in RESEARCH_QUESTIONS:
            st.markdown(f"- {question} *({page})*")
    with right:
        st.subheader("Pipeline artifacts")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    show_warnings(
        [
            DISCLAIMER,
            "Negative results are in scope. This lab does not recommend buying or selling players.",
            "The FastAPI service (`python -m service.main`) is the product API; this Streamlit app is the instructor-facing lab.",
        ]
    )
