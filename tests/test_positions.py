"""Phase 0 — Transfermarkt position → XI slot / group mapping."""

import pytest

from src.common.config import XI_SLOTS
from src.common.exceptions import InvalidParameterError
from src.search.positions import (
    GROUP_SLOTS,
    legal_slots_for_player,
    player_legal_for_slot,
    position_group,
    slot_group,
)


def test_every_xi_slot_has_a_group() -> None:
    for slot in XI_SLOTS:
        assert slot_group(slot) in {"GK", "DEF", "MID", "FWD"}
    assert sum(len(slots) for slots in GROUP_SLOTS.values()) == 11


def test_sub_position_preferred_over_coarse_position() -> None:
    assert position_group("Defender", "Centre-Back") == "DEF"
    assert legal_slots_for_player("Defender", "Centre-Back") == ("RCB", "LCB")
    assert legal_slots_for_player("Attack", "Left Winger") == ("LW",)
    assert legal_slots_for_player("Midfield", "Defensive Midfield") == ("CDM",)


def test_coarse_position_fills_whole_group() -> None:
    assert legal_slots_for_player("Goalkeeper", None) == ("GK",)
    assert legal_slots_for_player("Defender", None) == ("RB", "RCB", "LCB", "LB")
    assert legal_slots_for_player("Midfield", None) == ("CDM", "CM", "CAM")
    assert legal_slots_for_player("Attack", None) == ("RW", "ST", "LW")


def test_player_legal_for_slot_examples() -> None:
    assert player_legal_for_slot("Defender", "Centre-Back", "RCB")
    assert player_legal_for_slot("Defender", "Centre-Back", "LCB")
    assert not player_legal_for_slot("Defender", "Centre-Back", "RB")
    assert player_legal_for_slot("Goalkeeper", "Goalkeeper", "GK")
    assert not player_legal_for_slot("Attack", "Centre-Forward", "CAM")


def test_missing_and_unknown_labels() -> None:
    assert position_group("Missing", None) is None
    assert legal_slots_for_player("Missing", None) == ()
    assert legal_slots_for_player(None, None) == ()
    with pytest.raises(InvalidParameterError):
        slot_group("QB")
    with pytest.raises(InvalidParameterError):
        player_legal_for_slot("Attack", "Centre-Forward", "QB")
