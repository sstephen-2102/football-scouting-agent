"""Member C — starting-XI state-space search (uninformed vs informed)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import XI_SLOTS
from src.common.exceptions import InvalidParameterError, not_built
from src.search.positions import player_legal_for_slot
from collections import deque
from heapq import heappop, heappush
from random import Random

from src.common.config import RANDOM_STATE, XI_SLOTS
import matplotlib.pyplot as plt

from src.common.config import OUTPUTS, RANDOM_STATE, XI_SLOTS

@dataclass(frozen=True, slots=True)
class XiState:
    """Hashable XI search state.

    `assignment` is length-11, parallel to `slots`: player_id or None.
    `remaining` is the pool of unused shortlist ids.
    """

    slots: tuple[str, ...]
    assignment: tuple[int | None, ...]
    remaining: frozenset[int]

    def as_xi_dict(self) -> dict[str, int]:
        """Filled slots only — useful later for the search return payload."""
        return {
            slot: player_id
            for slot, player_id in zip(self.slots, self.assignment, strict=True)
            if player_id is not None
        }


def xi_slots() -> tuple[str, ...]:
    """The 11 named slots. Change only by team PR."""
    return XI_SLOTS


def initial_xi_state(
    shortlist_player_ids: list[int],
    slots: tuple[str, ...] | None = None,
) -> XiState:
    """Empty XI plus the remaining shortlist. State must be hashable."""
    if not isinstance(shortlist_player_ids, list):
        raise InvalidParameterError("shortlist_player_ids must be a list of ints.")

    ids = [int(player_id) for player_id in shortlist_player_ids]
    unique = frozenset(ids)
    if len(unique) != len(ids):
        raise InvalidParameterError("shortlist_player_ids must be unique.")
    if not 11 <= len(unique) <= 40:
        raise InvalidParameterError(
            f"shortlist length must be 11–40; got {len(unique)}."
        )

    use_slots = XI_SLOTS if slots is None else tuple(slots)
    if len(use_slots) != 11:
        raise InvalidParameterError("slots must contain exactly 11 names.")
    if len(set(use_slots)) != 11:
        raise InvalidParameterError("slots must be unique.")

    return XiState(
        slots=use_slots,
        assignment=tuple(None for _ in use_slots),
        remaining=unique,
    )


def is_goal(state: Any) -> bool:
    """True when all 11 slots are filled."""
    if not isinstance(state, XiState):
        raise InvalidParameterError("state must be an XiState.")
    return all(player_id is not None for player_id in state.assignment)


def legal_actions(state: Any, players: pd.DataFrame) -> list[tuple[str, int]]:
    """Assign a remaining player to the next vacant slot only (fixed order)."""
    if not isinstance(state, XiState):
        raise InvalidParameterError("state must be an XiState.")
    if not isinstance(players, pd.DataFrame):
        raise InvalidParameterError("players must be a DataFrame.")
    if is_goal(state):
        return []

    required = {"player_id", "position", "sub_position"}
    if missing := required - set(players.columns):
        raise InvalidParameterError(f"players missing columns: {sorted(missing)}")

    # Fixed order: only the first empty slot may be filled.
    try:
        slot_index = next(
            i for i, player_id in enumerate(state.assignment) if player_id is None
        )
    except StopIteration:
        return []
    slot = state.slots[slot_index]

    by_id = players.drop_duplicates(subset=["player_id"]).set_index("player_id")
    actions: list[tuple[str, int]] = []
    for player_id in sorted(state.remaining):
        if player_id not in by_id.index:
            continue
        row = by_id.loc[player_id]
        position = row["position"] if pd.notna(row["position"]) else None
        sub_position = row["sub_position"] if pd.notna(row["sub_position"]) else None
        if player_legal_for_slot(position, sub_position, slot, slots=state.slots):
            actions.append((slot, int(player_id)))
    return actions


def apply_action(state: Any, action: tuple[str, int]) -> XiState:
    """Return a successor state (filled slot, player removed from remaining)."""
    if not isinstance(state, XiState):
        raise InvalidParameterError("state must be an XiState.")
    if not (isinstance(action, tuple) and len(action) == 2):
        raise InvalidParameterError("action must be (slot, player_id).")

    slot, player_id = action
    player_id = int(player_id)
    if slot not in state.slots:
        raise InvalidParameterError(f"Unknown slot: {slot!r}")
    if player_id not in state.remaining:
        raise InvalidParameterError(f"Player {player_id} is not remaining.")

    slot_index = state.slots.index(slot)
    if state.assignment[slot_index] is not None:
        raise InvalidParameterError(f"Slot {slot!r} is already filled.")

    new_assignment = list(state.assignment)
    new_assignment[slot_index] = player_id
    return XiState(
        slots=state.slots,
        assignment=tuple(new_assignment),
        remaining=state.remaining - {player_id},
    )


def _pred_by_player(scores: pd.DataFrame) -> dict[int, float]:
    if not isinstance(scores, pd.DataFrame):
        raise InvalidParameterError("scores must be a DataFrame.")
    required = {"player_id", "pred_value"}
    if missing := required - set(scores.columns):
        raise InvalidParameterError(f"scores missing columns: {sorted(missing)}")
    frame = scores.drop_duplicates(subset=["player_id"])
    return {
        int(row.player_id): float(row.pred_value)
        for row in frame.itertuples(index=False)
    }


def xi_quality(state: Any, scores: pd.DataFrame) -> float:
    """Sum of predicted values of assigned players (empty slots add 0)."""
    if not isinstance(state, XiState):
        raise InvalidParameterError("state must be an XiState.")
    pred = _pred_by_player(scores)
    total = 0.0
    for player_id in state.assignment:
        if player_id is not None:
            total += pred.get(int(player_id), 0.0)
    return total


def _remaining_heuristic(
    state: XiState,
    players: pd.DataFrame,
    pred: dict[int, float],
) -> float:
    """Optimistic value still available for vacant slots (no player reuse)."""
    remaining = set(state.remaining)
    total = 0.0
    for slot, filled in zip(state.slots, state.assignment, strict=True):
        if filled is not None:
            continue
        best = 0.0
        best_id: int | None = None
        for player_id in remaining:
            row = players.loc[players["player_id"] == player_id]
            if row.empty:
                continue
            position = row.iloc[0]["position"]
            sub_position = row.iloc[0]["sub_position"]
            position = position if pd.notna(position) else None
            sub_position = sub_position if pd.notna(sub_position) else None
            if not player_legal_for_slot(position, sub_position, slot, slots=state.slots):
                continue
            value = pred.get(int(player_id), 0.0)
            if value > best or best_id is None:
                best = value
                best_id = int(player_id)
        if best_id is not None:
            total += best
            remaining.remove(best_id)
    return total


def _pack_result(algorithm: str, state: XiState, scores: pd.DataFrame, nodes: int, hit_cap: bool) -> dict:
    return {
        "algorithm": algorithm,
        "xi": state.as_xi_dict(),
        "quality": float(xi_quality(state, scores)),
        "nodes_expanded": int(nodes),
        "hit_cap": bool(hit_cap),
    }


def uninformed_search(
    state: Any,
    players: pd.DataFrame,
    scores: pd.DataFrame,
    algorithm: str = "bfs",
    expansion_cap: int = 5000,
) -> dict:
    """BFS or random search. Return best XI, nodes expanded, quality, algorithm."""
    if algorithm not in {"bfs", "random"}:
        raise InvalidParameterError("algorithm must be 'bfs' or 'random'.")
    if not isinstance(state, XiState):
        raise InvalidParameterError("state must be an XiState.")

    rng = Random(RANDOM_STATE)
    frontier: deque[XiState] = deque([state])
    visited: set[XiState] = {state}
    best = state
    best_quality = xi_quality(state, scores)
    nodes = 0
    hit_cap = False

    while frontier:
        if nodes >= expansion_cap:
            hit_cap = True
            break
        current = frontier.popleft()
        nodes += 1
        quality = xi_quality(current, scores)
        if is_goal(current):
            if quality > best_quality or not is_goal(best):
                best, best_quality = current, quality
            continue
        if quality > best_quality and not is_goal(best):
            best, best_quality = current, quality

        actions = legal_actions(current, players)
        if algorithm == "random":
            actions = list(actions)
            rng.shuffle(actions)
        for action in actions:
            nxt = apply_action(current, action)
            if nxt in visited:
                continue
            visited.add(nxt)
            frontier.append(nxt)

    return _pack_result(algorithm, best, scores, nodes, hit_cap)


def informed_search(
    state: Any,
    players: pd.DataFrame,
    scores: pd.DataFrame,
    algorithm: str = "astar",
    expansion_cap: int = 5000,
) -> dict:
    """Greedy or A* with remaining-value heuristic. Same return contract as uninformed."""
    if algorithm not in {"greedy", "astar"}:
        raise InvalidParameterError("algorithm must be 'greedy' or 'astar'.")
    if not isinstance(state, XiState):
        raise InvalidParameterError("state must be an XiState.")

    pred = _pred_by_player(scores)
    # heap entries: (-priority, tie, state)
    tie = 0
    g0 = xi_quality(state, scores)
    h0 = _remaining_heuristic(state, players, pred)
    priority0 = g0 + h0 if algorithm == "astar" else (g0 if algorithm == "greedy" else h0)
    heap: list[tuple[float, int, XiState]] = [(-priority0, tie, state)]
    visited: set[XiState] = set()
    best = state
    best_quality = g0
    nodes = 0
    hit_cap = False

    while heap:
        if nodes >= expansion_cap:
            hit_cap = True
            break
        _, _, current = heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        nodes += 1

        quality = xi_quality(current, scores)
        if is_goal(current):
            if quality > best_quality or not is_goal(best):
                best, best_quality = current, quality
            continue
        if quality > best_quality and not is_goal(best):
            best, best_quality = current, quality

        for action in legal_actions(current, players):
            nxt = apply_action(current, action)
            if nxt in visited:
                continue
            g = xi_quality(nxt, scores)
            h = _remaining_heuristic(nxt, players, pred)
            priority = g + h if algorithm == "astar" else g
            tie += 1
            heappush(heap, (-priority, tie, nxt))

    return _pack_result(algorithm, best, scores, nodes, hit_cap)

def compare_search(uninformed_result: dict, informed_result: dict) -> dict:
    """Nodes vs XI quality — the Module 2 comparison graph."""
    for name, result in (
        ("uninformed_result", uninformed_result),
        ("informed_result", informed_result),
    ):
        if not isinstance(result, dict):
            raise InvalidParameterError(f"{name} must be a dict.")
        for key in ("algorithm", "quality", "nodes_expanded"):
            if key not in result:
                raise InvalidParameterError(f"{name} missing key {key!r}.")

    u_quality = float(uninformed_result["quality"])
    i_quality = float(informed_result["quality"])
    u_nodes = int(uninformed_result["nodes_expanded"])
    i_nodes = int(informed_result["nodes_expanded"])

    summary = {
        "uninformed_algorithm": uninformed_result["algorithm"],
        "informed_algorithm": informed_result["algorithm"],
        "uninformed_quality": u_quality,
        "informed_quality": i_quality,
        "uninformed_nodes": u_nodes,
        "informed_nodes": i_nodes,
        "informed_minus_uninformed": i_quality - u_quality,
        "nodes_ratio": (i_nodes / u_nodes) if u_nodes else None,
    }

    figures = OUTPUTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "search_nodes_vs_quality.png"

    labels = [
        str(uninformed_result["algorithm"]),
        str(informed_result["algorithm"]),
    ]
    qualities = [u_quality, i_quality]
    nodes = [u_nodes, i_nodes]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(labels, qualities, color=["#4C78A8", "#F58518"])
    axes[0].set_title("XI quality")
    axes[0].set_ylabel("sum pred_value")
    axes[1].bar(labels, nodes, color=["#4C78A8", "#F58518"])
    axes[1].set_title("Nodes expanded")
    axes[1].set_ylabel("nodes")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    summary["figure_path"] = str(path)
    return summary


def export_best_xi(
    club_id: int | None = None,
    season: int | None = None,
    shortlist_player_ids: list[int] | None = None,
) -> Path:
    """Write `outputs/search/best_xi.json` (pipeline: search-xi)."""
    not_built("member_c", "export_best_xi")
