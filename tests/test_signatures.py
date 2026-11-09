"""Every planned feature has a frozen signature that still raises NotBuiltError."""

from __future__ import annotations

import inspect

import pytest

from src.agent.peas import ten_environment_dimensions
from src.cluster.archetypes import cluster_feature_columns
from src.common.exceptions import NotBuiltError
from src.common.signatures import STUBS
from src.data.leakage import forbidden_feature_columns
from src.data.seasons import season_end_date
from src.data.tables import required_tables
from src.optimize.squad import position_minima
from src.policy.bid import bid_actions
from src.search.xi import xi_slots


def _call_with_placeholders(function):
    signature = inspect.signature(function)
    arguments = {}
    for name, parameter in signature.parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        arguments[name] = None
    return function(**arguments)


@pytest.mark.parametrize(
    "function, owner, required",
    STUBS,
    ids=[function.__name__ for function, _, _ in STUBS],
)
def test_stub_signature_and_not_built(function, owner, required) -> None:
    parameters = inspect.signature(function).parameters
    for name in required:
        assert name in parameters, f"{function.__name__} is missing parameter {name}"
    with pytest.raises(NotBuiltError) as error:
        _call_with_placeholders(function)
    assert error.value.owner == owner
    assert error.value.name == function.__name__


def test_registry_covers_unique_callables() -> None:
    names = [function.__name__ for function, _, _ in STUBS]
    assert len(names) == len(set(names))
    assert len(STUBS) >= 40
    assert "load_table" not in names
    assert "build_player_season_table" not in names


def test_implemented_contracts() -> None:
    assert season_end_date(2022).isoformat() == "2023-06-30"
    assert "appearances" in required_tables()
    assert "market_value_in_eur" in forbidden_feature_columns()
    assert "goals_per90" in cluster_feature_columns()
    assert len(xi_slots()) == 11
    assert position_minima()["GK"] == 1
    assert bid_actions() == ("bid", "pass")
    dimensions = ten_environment_dimensions()
    assert len(dimensions) == 10
    assert dimensions["observability"].startswith("partially")
