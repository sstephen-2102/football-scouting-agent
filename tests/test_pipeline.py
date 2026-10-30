"""Pipeline CLI status does not require the Kaggle download."""

from src.artifacts import artifact_status


def test_all_artifacts_start_missing() -> None:
    status = artifact_status()
    assert status["archetypes"]["owner"] == "member_a"
    assert status["scores"]["owner"] == "member_b"
    assert status["policy"]["owner"] == "member_c"
    assert status["split"]["ready"] is False
