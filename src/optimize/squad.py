"""Member C — MIP squad construction under budget B, plus greedy baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import POSITION_MINIMA
from src.common.exceptions import not_built


def position_minima() -> dict[str, int]:
    """Minimum starters by group. Change only by team PR."""
    return dict(POSITION_MINIMA)


def formulate_squad_mip(
    candidates: pd.DataFrame,
    budget_eur: float,
    position_minima: dict[str, int] | None = None,
    n_starters: int = 11,
) -> Any:
    """PuLP problem: max predicted quality s.t. B, one player per slot, position minima."""
    not_built("member_c", "formulate_squad_mip")


def solve_squad_mip(problem: Any) -> dict:
    """Solve the MIP. Return status, selected player_ids, spend, quality."""
    not_built("member_c", "solve_squad_mip")


def greedy_spend(
    candidates: pd.DataFrame,
    budget_eur: float,
    n_starters: int = 11,
) -> dict:
    """Buy the highest predicted values until broke. Comparison baseline, not the product."""
    not_built("member_c", "greedy_spend")


def compare_mip_vs_greedy(mip: dict, greedy: dict) -> dict:
    """Quality and spend under the same B. Negative results stay in scope."""
    not_built("member_c", "compare_mip_vs_greedy")


def export_squad(
    club_id: int | None = None,
    season: int | None = None,
    budget_eur: float | None = None,
) -> Path:
    """Write `outputs/optimize/squad.csv` (pipeline: allocate)."""
    not_built("member_c", "export_squad")
