"""PEAS description of the scouting agent (Module 1)."""

from src.common.config import SNAPSHOT_DATE

DISCLAIMER = (
    "Transfermarkt market values are crowd-informed estimates, not clearing prices. "
    "Outputs are research scores, not transfer advice."
)

PEAS = {
    "performance": (
        "Test-set Spearman of predicted vs actual next value; "
        "PR AUC for undervalued; expected residual captured under budget B"
    ),
    "environment": (
        f"Transfermarkt snapshot to {SNAPSHOT_DATE}; Big-5 player-seasons; "
        "other clubs and agents unobserved"
    ),
    "actuators": (
        "Assign archetype; emit value score; propose XI; allocate budget; bid or pass"
    ),
    "sensors": (
        "Appearances, games, historical valuations, transfers, club and player "
        "attributes dated on or before the cutoff"
    ),
}

ENVIRONMENT_NOTES = {
    "observability": "partial",
    "determinism": "stochastic",
    "episode": "sequential",
    "dynamics": "frozen snapshot of a dynamic market",
    "values": "mixed discrete slots and continuous EUR",
    "agents": "single-agent (competing clubs are not modeled)",
}

# Course Module 1: ten environment dimensions (Russell & Norvig task-environment axes).
ENVIRONMENT_DIMENSIONS = {
    "observability": "partially observable — wages, injuries, and true bids are hidden",
    "agents": "single-agent for the course; rival clubs are not modeled as agents",
    "determinism": "stochastic — form, injuries, and Transfermarkt revisions are noisy",
    "episode": "sequential — budget spent on one signing is gone for the next",
    "dynamics": "dynamic in live football; this product simulates a frozen snapshot",
    "discreteness": "mixed — discrete XI slots and archetypes, continuous EUR values",
    "known": "known rules for search/MIP; unknown true data-generating process for value",
    "competitive": "competitive market in reality; treated as a planning problem here",
    "horizon": "finite transfer window under remaining budget B",
    "feedback": "delayed and noisy — valuations lag on-pitch output",
}


def ten_environment_dimensions() -> dict[str, str]:
    """Return the Module 1 ten-dimension score for this scouting agent."""
    return dict(ENVIRONMENT_DIMENSIONS)
