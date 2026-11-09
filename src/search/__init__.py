"""Member C: starting-XI state-space search."""

from src.search.positions import (
    legal_slots_for_player,
    player_legal_for_slot,
    position_group,
    slot_group,
)
from src.search.xi import (
    apply_action,
    compare_search,
    export_best_xi,
    informed_search,
    initial_xi_state,
    is_goal,
    legal_actions,
    uninformed_search,
    xi_quality,
    xi_slots,
)

__all__ = [
    "apply_action",
    "compare_search",
    "export_best_xi",
    "informed_search",
    "initial_xi_state",
    "is_goal",
    "legal_actions",
    "legal_slots_for_player",
    "player_legal_for_slot",
    "position_group",
    "slot_group",
    "uninformed_search",
    "xi_quality",
    "xi_slots",
]
