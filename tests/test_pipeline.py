"""Pipeline CLI status does not require the Kaggle download."""

import sys

import pytest

from src.artifacts import artifact_status
from src.pipeline import EXPORTS, main


def test_artifact_owners_and_member_stubs_missing() -> None:
    status = artifact_status()
    assert status["split"]["owner"] == "shared"
    assert status["archetypes"]["owner"] == "member_a"
    assert status["scores"]["owner"] == "member_b"
    assert status["policy"]["owner"] == "member_c"
    # Member A/B/C jobs stay stubs until each owner lands their export.
    assert status["archetypes"]["ready"] is False
    assert status["scores"]["ready"] is False
    assert status["policy"]["ready"] is False


def test_export_commands_are_wired() -> None:
    assert set(EXPORTS) == {
        "build-seasons",
        "cluster",
        "forecast",
        "score",
        "search-xi",
        "allocate",
        "policy",
    }


def test_cluster_command_is_still_a_stub(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["pipeline", "cluster"])
    with pytest.raises(SystemExit) as error:
        main()
    assert "signature stub" in str(error.value)
    assert "member_a" in str(error.value)
