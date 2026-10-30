"""Overview — PEAS, data freeze, links to the spec."""

from pathlib import Path

import streamlit as st

from src.common.config import SNAPSHOT_DATE


def render() -> None:
    st.title("Football Scouting Agent")
    st.caption(f"AAI-501 · Transfermarkt snapshot {SNAPSHOT_DATE} · not transfer advice")
    st.markdown(
        """
This app will help a sporting director **shortlist undervalued players**,
**fill a legal starting XI**, and **spend a finite transfer budget**.

Work is split three ways:

- **A** — player archetypes and scoring/value forecasts
- **B** — market-value models and sequence networks
- **C** — XI search, MIP allocation, bid/pass policy
        """
    )
    spec = Path(__file__).resolve().parents[2] / "docs" / "REQUIREMENTS_AND_IMPLEMENTATION_PLAN.md"
    st.info(f"Full specification: `{spec.name}` in the docs folder.")
    st.warning(
        "Transfermarkt values are estimates, not transaction prices. "
        "Predictions are course research, not recommendations to buy or sell players."
    )
