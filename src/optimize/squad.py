"""Member C — MIP squad construction under budget B, plus greedy baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pulp

from src.common.config import OUTPUTS, POSITION_MINIMA
from src.common.exceptions import InvalidParameterError
from src.common.scored_players import (
    SHORTLIST_PER_GROUP,
    load_scored_players,
    top_per_group,
    with_ask_cost,
)
from src.search.positions import position_group


def position_minima() -> dict[str, int]:
    """Minimum starters by group. Change only by team PR."""
    return dict(POSITION_MINIMA)


def _with_group_and_cost(candidates: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(candidates, pd.DataFrame):
        raise InvalidParameterError("candidates must be a DataFrame.")
    required = {"player_id", "pred_value"}
    if missing := required - set(candidates.columns):
        raise InvalidParameterError(f"candidates missing {sorted(missing)}")

    frame = candidates.copy()
    if "group" not in frame.columns:
        if not {"position", "sub_position"} <= set(frame.columns):
            raise InvalidParameterError(
                "candidates need 'group' or both 'position' and 'sub_position'."
            )
        frame["group"] = [
            position_group(
                row["position"] if pd.notna(row["position"]) else None,
                row["sub_position"] if pd.notna(row["sub_position"]) else None,
            )
            for _, row in frame.iterrows()
        ]
    if frame["group"].isna().any():
        raise InvalidParameterError("Every candidate needs a mappable position group.")
    if "cost" not in frame.columns:
        frame["cost"] = frame["pred_value"]  # documented: ask ≈ predicted value
    frame["player_id"] = frame["player_id"].astype(int)
    frame["pred_value"] = frame["pred_value"].astype(float)
    frame["cost"] = frame["cost"].astype(float)
    return frame.reset_index(drop=True)


def formulate_squad_mip(
    candidates: pd.DataFrame,
    budget_eur: float,
    position_minima: dict[str, int] | None = None,
    n_starters: int = 11,
) -> Any:
    """PuLP problem: max predicted quality s.t. B, n starters, position minima."""
    frame = _with_group_and_cost(candidates)
    minima = dict(POSITION_MINIMA if position_minima is None else position_minima)
    if float(budget_eur) < 0:
        raise InvalidParameterError("budget_eur must be non-negative.")

    problem = pulp.LpProblem("squad_allocate", pulp.LpMaximize)
    choose = {
        int(row.player_id): pulp.LpVariable(f"x_{int(row.player_id)}", cat="Binary")
        for row in frame.itertuples(index=False)
    }
    problem += pulp.lpSum(
        float(row.pred_value) * choose[int(row.player_id)]
        for row in frame.itertuples(index=False)
    )
    problem += pulp.lpSum(choose.values()) == int(n_starters), "n_starters"
    problem += (
        pulp.lpSum(
            float(row.cost) * choose[int(row.player_id)]
            for row in frame.itertuples(index=False)
        )
        <= float(budget_eur),
        "budget",
    )
    for group, minimum in minima.items():
        problem += (
            pulp.lpSum(
                choose[int(row.player_id)]
                for row in frame.itertuples(index=False)
                if row.group == group
            )
            >= int(minimum),
            f"min_{group}",
        )
    # Stash frame for the solver helper
    problem._scouting_candidates = frame  # type: ignore[attr-defined]
    problem._scouting_choose = choose  # type: ignore[attr-defined]
    return problem


def solve_squad_mip(problem: Any) -> dict:
    """Solve the MIP. Return status, selected player_ids, spend, quality."""
    if problem is None:
        raise InvalidParameterError("problem is required.")
    # Bundled PULP_CBC_CMD is x86_64-only; COIN_CMD uses the arm64/PATH cbc.
    solver = pulp.COIN_CMD(msg=False)
    if not solver.available():
        solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = problem.solve(solver)
    status_name = pulp.LpStatus.get(status_code, "unknown").lower()
    frame = getattr(problem, "_scouting_candidates", None)
    choose = getattr(problem, "_scouting_choose", None)
    if frame is None or choose is None:
        raise InvalidParameterError("problem must come from formulate_squad_mip.")

    if status_name != "optimal":
        return {
            "status": "infeasible" if status_name == "infeasible" else status_name,
            "player_ids": [],
            "spend": 0.0,
            "quality": 0.0,
        }

    selected = [pid for pid, var in choose.items() if var.value() and var.value() > 0.5]
    picked = frame[frame["player_id"].isin(selected)]
    return {
        "status": "optimal",
        "player_ids": sorted(int(pid) for pid in selected),
        "spend": float(picked["cost"].sum()),
        "quality": float(picked["pred_value"].sum()),
    }


def greedy_spend(
    candidates: pd.DataFrame,
    budget_eur: float,
    n_starters: int = 11,
) -> dict:
    """Buy highest pred_value while budget and group quotas allow."""
    frame = _with_group_and_cost(candidates)
    remaining_quota = dict(POSITION_MINIMA)
    # If minima sum to n_starters, quotas are exact group slots.
    budget_left = float(budget_eur)
    chosen: list[int] = []
    spend = 0.0
    quality = 0.0

    ordered = frame.sort_values("pred_value", ascending=False)
    for row in ordered.itertuples(index=False):
        if len(chosen) >= int(n_starters):
            break
        group = row.group
        if remaining_quota.get(group, 0) <= 0:
            continue
        cost = float(row.cost)
        if cost > budget_left:
            continue
        chosen.append(int(row.player_id))
        remaining_quota[group] -= 1
        budget_left -= cost
        spend += cost
        quality += float(row.pred_value)

    status = "optimal" if len(chosen) == int(n_starters) else "partial"
    return {
        "status": status,
        "player_ids": sorted(chosen),
        "spend": float(spend),
        "quality": float(quality),
    }


def compare_mip_vs_greedy(mip: dict, greedy: dict) -> dict:
    """Quality and spend under the same B. Negative gaps stay in scope."""
    for name, result in (("mip", mip), ("greedy", greedy)):
        if not isinstance(result, dict):
            raise InvalidParameterError(f"{name} must be a dict.")
        for key in ("quality", "spend", "player_ids"):
            if key not in result:
                raise InvalidParameterError(f"{name} missing {key!r}.")

    summary = {
        "mip_quality": float(mip["quality"]),
        "greedy_quality": float(greedy["quality"]),
        "mip_spend": float(mip["spend"]),
        "greedy_spend": float(greedy["spend"]),
        "quality_gap": float(mip["quality"]) - float(greedy["quality"]),
        "mip_player_ids": list(mip["player_ids"]),
        "greedy_player_ids": list(greedy["player_ids"]),
    }

    figures = OUTPUTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "mip_vs_greedy.png"
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(["mip", "greedy"], [summary["mip_quality"], summary["greedy_quality"]])
    axes[0].set_title("Squad quality")
    axes[1].bar(["mip", "greedy"], [summary["mip_spend"], summary["greedy_spend"]])
    axes[1].set_title("Spend")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    summary["figure_path"] = str(path)
    return summary


DEFAULT_BUDGET_EUR = 80_000_000
COST_RULE = (
    "cost = prev_value_eur (last valuation on or before the cutoff; "
    "pred_value where missing). An ask proxy, not a real bid."
)


def _budget_from_forecast(root: Path) -> tuple[float, str]:
    """Spec rule: budget.json budget_eur when unstable is false, else the default."""
    path = root / "forecast" / "budget.json"
    if not path.is_file():
        return float(DEFAULT_BUDGET_EUR), "default (forecast/budget.json not written yet)"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("unstable", True) and "budget_eur" in payload:
        return float(payload["budget_eur"]), "forecast/budget.json"
    return float(DEFAULT_BUDGET_EUR), "default (forecast unstable)"


def _min_feasible_xi_budget(frame: pd.DataFrame) -> float | None:
    """Cost of the cheapest legal starting XI meeting POSITION_MINIMA, or None."""
    total = 0.0
    for group, minimum in POSITION_MINIMA.items():
        subset = frame[frame["group"] == group].nsmallest(int(minimum), "cost")
        if len(subset) < int(minimum):
            return None
        total += float(subset["cost"].sum())
    return total


def export_squad(
    club_id: int | None = None,
    season: int | None = None,
    budget_eur: float | None = None,
) -> Path:
    """Write `outputs/optimize/squad.csv` and `allocation_manifest.json` (pipeline: allocate).

    Shortlist = test players with per-group quotas (SHORTLIST_PER_GROUP). Cost is
    the cutoff-dated `prev_value_eur`. When `budget_eur` is None and no stable
    forecast budget exists, B defaults to the cheapest legal XI on the shortlist
    so the MIP is feasible.
    """
    root = OUTPUTS
    pool = load_scored_players("test", season, root=root)
    shortlist = with_ask_cost(top_per_group(pool))
    candidates = shortlist[["player_id", "pred_value", "position", "sub_position", "cost"]]
    frame = _with_group_and_cost(candidates).set_index("player_id", drop=False)

    if budget_eur is not None:
        budget, budget_source = float(budget_eur), "argument"
    else:
        forecast_budget, forecast_source = _budget_from_forecast(root)
        if forecast_source == "forecast/budget.json":
            budget, budget_source = forecast_budget, forecast_source
        else:
            min_budget = _min_feasible_xi_budget(frame)
            if min_budget is not None:
                budget = min_budget
                budget_source = "min_feasible_xi (cheapest legal XI on shortlist)"
            else:
                budget, budget_source = forecast_budget, forecast_source

    mip = solve_squad_mip(formulate_squad_mip(candidates, budget))
    greedy = greedy_spend(candidates, budget)
    comparison = compare_mip_vs_greedy(mip, greedy)

    rows = []
    for source, result in (("mip", mip), ("greedy", greedy)):
        for player_id in result["player_ids"]:
            row = frame.loc[int(player_id)]
            rows.append(
                {
                    "player_id": int(player_id),
                    "slot_or_group": row["group"],
                    "pred_value": float(row["pred_value"]),
                    "cost": float(row["cost"]),
                    "source": source,
                }
            )
    table = pd.DataFrame(
        rows, columns=["player_id", "slot_or_group", "pred_value", "cost", "source"]
    )

    out_dir = root / "optimize"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "squad.csv"
    table.to_csv(csv_path, index=False)

    manifest = {
        "club_id": club_id,
        "season": season,
        "budget_eur": budget,
        "budget_source": budget_source,
        "shortlist_size": int(len(candidates)),
        "shortlist_player_ids": [int(pid) for pid in candidates["player_id"]],
        "cost_rule": COST_RULE,
        "shortlist_quotas": dict(SHORTLIST_PER_GROUP),
        "n_starters": 11,
        "position_minima": position_minima(),
        "mip": {k: mip[k] for k in ("status", "player_ids", "spend", "quality")},
        "greedy": {k: greedy[k] for k in ("status", "player_ids", "spend", "quality")},
        "comparison": {
            k: v for k, v in comparison.items() if k not in {"mip_player_ids", "greedy_player_ids"}
        },
    }
    (out_dir / "allocation_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return csv_path
