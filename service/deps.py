"""Shared FastAPI helpers."""

from fastapi import HTTPException

from src.artifacts import ARTIFACTS, artifact_path, artifact_status


def require_artifact(name: str) -> None:
    """Raise 503 when a contracted output file has not been produced yet."""
    spec = ARTIFACTS[name]
    path = artifact_path(name)
    if path.is_file():
        return
    raise HTTPException(
        status_code=503,
        detail={
            "status": "not_ready",
            "artifact": name,
            "owner": spec["owner"],
            "producer": spec["producer"],
            "detail": f"Missing {path}. Run the producer command, then retry.",
        },
    )


def current_status() -> dict[str, dict[str, str | bool]]:
    return artifact_status()
