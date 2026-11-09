"""Map Transfermarkt position / sub_position onto XI slots and groups.

Prefer `sub_position` when present. A player is legal for a slot when the
slot is in their legal-slot set (same group; CB may fill RCB or LCB, etc.).
"""

from __future__ import annotations

from src.common.config import XI_SLOTS
from src.common.exceptions import InvalidParameterError

SLOT_GROUP: dict[str, str] = {
    "GK": "GK",
    "RB": "DEF",
    "RCB": "DEF",
    "LCB": "DEF",
    "LB": "DEF",
    "CDM": "MID",
    "CM": "MID",
    "CAM": "MID",
    "RW": "FWD",
    "ST": "FWD",
    "LW": "FWD",
}

GROUP_SLOTS: dict[str, tuple[str, ...]] = {
    "GK": ("GK",),
    "DEF": ("RB", "RCB", "LCB", "LB"),
    "MID": ("CDM", "CM", "CAM"),
    "FWD": ("RW", "ST", "LW"),
}

# Prefer these when sub_position is known (Transfermarkt labels).
_SUB_POSITION_SLOTS: dict[str, tuple[str, ...]] = {
    "Goalkeeper": ("GK",),
    "Centre-Back": ("RCB", "LCB"),
    "Right-Back": ("RB",),
    "Left-Back": ("LB",),
    "Defensive Midfield": ("CDM",),
    "Central Midfield": ("CM",),
    "Attacking Midfield": ("CAM",),
    "Left Midfield": ("CM", "CAM"),
    "Right Midfield": ("CM", "CAM"),
    "Left Winger": ("LW",),
    "Right Winger": ("RW",),
    "Centre-Forward": ("ST",),
    "Second Striker": ("ST",),
}

_POSITION_GROUP: dict[str, str] = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfield": "MID",
    "Attack": "FWD",
}


def slot_group(slot: str) -> str:
    """Return GK / DEF / MID / FWD for an XI slot name."""
    if slot not in SLOT_GROUP:
        raise InvalidParameterError(f"Unknown XI slot: {slot!r}")
    return SLOT_GROUP[slot]


def position_group(position: str | None, sub_position: str | None = None) -> str | None:
    """Map Transfermarkt labels to {GK, DEF, MID, FWD}. Prefer sub_position."""
    if sub_position is not None and not isinstance(sub_position, str):
        raise InvalidParameterError("sub_position must be a string or None.")
    if position is not None and not isinstance(position, str):
        raise InvalidParameterError("position must be a string or None.")

    if isinstance(sub_position, str) and sub_position.strip():
        key = sub_position.strip()
        if key in _SUB_POSITION_SLOTS:
            return SLOT_GROUP[_SUB_POSITION_SLOTS[key][0]]

    if isinstance(position, str) and position.strip():
        return _POSITION_GROUP.get(position.strip())
    return None


def legal_slots_for_player(
    position: str | None,
    sub_position: str | None = None,
    slots: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """XI slots this player may fill. Empty if position cannot be mapped."""
    allowed = slots if slots is not None else XI_SLOTS
    for name in allowed:
        if name not in SLOT_GROUP:
            raise InvalidParameterError(f"Unknown XI slot: {name!r}")

    if isinstance(sub_position, str) and sub_position.strip():
        key = sub_position.strip()
        if key in _SUB_POSITION_SLOTS:
            preferred = _SUB_POSITION_SLOTS[key]
            return tuple(slot for slot in preferred if slot in allowed)

    group = position_group(position, sub_position)
    if group is None:
        return ()
    return tuple(slot for slot in GROUP_SLOTS[group] if slot in allowed)


def player_legal_for_slot(
    position: str | None,
    sub_position: str | None,
    slot: str,
    slots: tuple[str, ...] | None = None,
) -> bool:
    """True when the player may be assigned to `slot`."""
    if slot not in SLOT_GROUP:
        raise InvalidParameterError(f"Unknown XI slot: {slot!r}")
    return slot in legal_slots_for_player(position, sub_position, slots=slots)
