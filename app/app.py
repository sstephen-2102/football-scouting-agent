"""Navigation shell. Launch with `streamlit run streamlit_app.py` from the repository root.

Running this file directly breaks imports: Streamlit would put app/ on sys.path
and auto-register app/pages/*.py.
"""

import streamlit as st

from app.pages import (
    agent,
    archetypes,
    dashboard,
    data,
    experiments,
    forecast,
    policy,
    scores,
    search,
    squad,
)

PAGES = {
    "Overview": [
        (dashboard, "Executive Dashboard"),
        (agent, "Agent (PEAS)"),
        (data, "Data Freeze"),
    ],
    "Scouting": [
        (archetypes, "Archetype Lab"),
        (forecast, "Forecast Lab"),
    ],
    "Modeling": [
        (experiments, "Experiment Board"),
        (scores, "Player Scores"),
    ],
    "Decisions": [
        (search, "Search Lab"),
        (squad, "Squad Builder"),
        (policy, "Bid Policy"),
    ],
}


def main() -> None:
    st.set_page_config(page_title="Football Scouting Agent", layout="wide")
    navigation = {
        section: [
            st.Page(
                module.render,
                title=title,
                url_path=module.__name__.rsplit(".", 1)[-1],
                default=module is dashboard,
            )
            for module, title in pages
        ]
        for section, pages in PAGES.items()
    }
    st.navigation(navigation).run()


if __name__ == "__main__":
    main()
