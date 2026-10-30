"""Member C — educational Simplex on a 2–3 variable budget LP."""

from __future__ import annotations

from src.common.exceptions import not_built


def simplex(
    objective: list[float],
    constraints_a: list[list[float]],
    constraints_b: list[float],
    maximize: bool = True,
) -> dict:
    """Hand Simplex (not a solver dump).

    Return status, x, objective_value, and tableau_history. Toy size only
    (two or three position budgets). Full XI allocation is `solve_squad_mip`.
    """
    not_built("member_c", "simplex")
