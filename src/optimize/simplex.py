"""Member C — educational Simplex on a 2–3 variable budget LP."""

from __future__ import annotations

from src.common.exceptions import InvalidParameterError


def simplex(
    objective: list[float],
    constraints_a: list[list[float]],
    constraints_b: list[float],
    maximize: bool = True,
) -> dict:
    """Hand Simplex (not a solver dump).

    Return status, x, objective_value, and tableau_history. Synthetic size only
    (two or three position budgets). Full XI allocation is `solve_squad_mip`.
    """
    if not objective or not isinstance(objective, list):
        raise InvalidParameterError("objective must be a non-empty list.")
    n = len(objective)
    if n < 1 or n > 3:
        raise InvalidParameterError("Use 1–3 decision variables for this educational Simplex.")
    if len(constraints_a) != len(constraints_b):
        raise InvalidParameterError("constraints_a and constraints_b length must match.")
    if len(constraints_a) > 5:
        raise InvalidParameterError("At most 5 constraints for this educational Simplex.")
    for row in constraints_a:
        if len(row) != n:
            raise InvalidParameterError("Each constraint row must match len(objective).")

    m = len(constraints_a)
    # Tableau: m+1 rows, n+m+1 cols (slacks + RHS). Last row is objective.
    # Maximize: bottom row starts as -c. Minimize: use +c then negate z at the end.
    cols = n + m + 1
    tableau = [[0.0 for _ in range(cols)] for _ in range(m + 1)]
    for i, row in enumerate(constraints_a):
        for j, value in enumerate(row):
            tableau[i][j] = float(value)
        tableau[i][n + i] = 1.0  # slack
        tableau[i][-1] = float(constraints_b[i])
        if tableau[i][-1] < 0:
            return {
                "status": "infeasible",
                "x": [0.0] * n,
                "objective_value": 0.0,
                "tableau_history": [],
            }

    for j, coef in enumerate(objective):
        tableau[m][j] = -float(coef) if maximize else float(coef)

    basic = list(range(n, n + m))  # slack basics
    history: list[list[list[float]]] = []

    def _snapshot() -> None:
        if len(history) < 20:
            history.append([row[:] for row in tableau])

    _snapshot()

    for _ in range(100):  # hard stop against cycling
        # Entering variable (Bland: smallest index with negative reduced cost)
        entering = None
        for j in range(cols - 1):
            if tableau[m][j] < -1e-12:
                entering = j
                break
        if entering is None:
            # Optimal
            x = [0.0] * n
            for i, var in enumerate(basic):
                if var < n:
                    x[var] = tableau[i][-1]
            z = tableau[m][-1]
            if not maximize:
                z = -z
            return {
                "status": "optimal",
                "x": x,
                "objective_value": float(z),
                "tableau_history": history,
            }

        # Leaving: min ratio RHS / pivot (Bland: smallest row index on tie)
        leaving = None
        best_ratio = None
        for i in range(m):
            pivot = tableau[i][entering]
            if pivot <= 1e-12:
                continue
            ratio = tableau[i][-1] / pivot
            if ratio < -1e-12:
                continue
            if leaving is None or ratio < best_ratio - 1e-15 or (
                abs(ratio - (best_ratio or 0.0)) <= 1e-15 and i < leaving
            ):
                leaving = i
                best_ratio = ratio
        if leaving is None:
            return {
                "status": "unbounded",
                "x": [0.0] * n,
                "objective_value": float("inf") if maximize else float("-inf"),
                "tableau_history": history,
            }

        # Pivot
        pivot = tableau[leaving][entering]
        tableau[leaving] = [value / pivot for value in tableau[leaving]]
        for i in range(m + 1):
            if i == leaving:
                continue
            factor = tableau[i][entering]
            tableau[i] = [
                tableau[i][j] - factor * tableau[leaving][j] for j in range(cols)
            ]
        basic[leaving] = entering
        _snapshot()

    return {
        "status": "infeasible",
        "x": [0.0] * n,
        "objective_value": 0.0,
        "tableau_history": history,
    }