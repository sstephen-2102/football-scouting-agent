"""Member C — starting-XI state-space search (uninformed vs informed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import XI_SLOTS
from src.common.exceptions import not_built


def xi_slots() -> tuple[str, ...]:
    """The 11 named slots. Change only by team PR."""
    return XI_SLOTS


def initial_xi_state(
    shortlist_player_ids: list[int],
    slots: tuple[str, ...] | None = None,
) -> Any:
    """Empty XI plus the remaining shortlist. State must be hashable."""
    not_built("member_c", "initial_xi_state")


def legal_actions(state: Any, players: pd.DataFrame) -> list[tuple[str, int]]:
    """Assign a remaining player to a vacant legal position. Empty at a goal state."""
    not_built("member_c", "legal_actions")


def apply_action(state: Any, action: tuple[str, int]) -> Any:
    """Return a successor state (filled slot, player removed from remaining)."""
    not_built("member_c", "apply_action")


def is_goal(state: Any) -> bool:
    """True when all 11 slots are filled."""
    not_built("member_c", "is_goal")


def xi_quality(state: Any, scores: pd.DataFrame) -> float:
    """Heuristic / objective: sum of predicted values (or R1 quality) of assigned players."""
    not_built("member_c", "xi_quality")


def uninformed_search(
    state: Any,
    players: pd.DataFrame,
    scores: pd.DataFrame,
    algorithm: str = "bfs",
    expansion_cap: int = 5000,
) -> dict:
    """BFS or random search. Return best XI, nodes expanded, quality, algorithm."""
    not_built("member_c", "uninformed_search")


def informed_search(
    state: Any,
    players: pd.DataFrame,
    scores: pd.DataFrame,
    algorithm: str = "astar",
    expansion_cap: int = 5000,
) -> dict:
    """Greedy or A* with `xi_quality` as the heuristic. Same return contract as uninformed."""
    not_built("member_c", "informed_search")


def compare_search(uninformed_result: dict, informed_result: dict) -> dict:
    """Nodes vs XI quality — the Module 2 comparison graph."""
    not_built("member_c", "compare_search")


def export_best_xi(
    club_id: int | None = None,
    season: int | None = None,
    shortlist_player_ids: list[int] | None = None,
) -> Path:
    """Write `outputs/search/best_xi.json` (pipeline: search-xi)."""
    not_built("member_c", "export_best_xi")
