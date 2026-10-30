"""Streamlit lab navigation is registered and the dashboard renders."""

from app.app import PAGES
from app.artifacts import missing_message, status_table


def test_navigation_covers_the_labs() -> None:
    titles = [title for pages in PAGES.values() for _, title in pages]
    assert titles == [
        "Executive Dashboard",
        "Agent (PEAS)",
        "Data Freeze",
        "Archetype Lab",
        "Forecast Lab",
        "Experiment Board",
        "Player Scores",
        "Search Lab",
        "Squad Builder",
        "Bid Policy",
    ]


def test_artifact_status_feeds_the_dashboard() -> None:
    rows = status_table()
    names = {row["artifact"] for row in rows}
    assert names >= {"archetypes", "forecast", "scores", "search", "squad", "policy"}
    assert "python -m src.pipeline cluster" in missing_message("archetypes")
