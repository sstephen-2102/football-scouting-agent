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
