"""Read pipeline outputs. The Streamlit process never trains models."""

from src.artifacts import ARTIFACTS, artifact_path, artifact_status
from src.common.config import TEAM


def is_ready(name: str) -> bool:
    return artifact_path(name).is_file()


def missing_message(name: str) -> str:
    spec = ARTIFACTS[name]
    path = artifact_path(name)
    owner = TEAM.get(spec["owner"], spec["owner"])
    return (
        f"Missing `{path}`. Owner: **{owner}**. "
        f"Run `{spec['producer']}`, then reload this page."
    )


def status_table() -> list[dict[str, str]]:
    rows = []
    for name, row in artifact_status().items():
        rows.append(
            {
                "artifact": name,
                "state": "ready" if row["ready"] else "missing",
                "owner": TEAM.get(str(row["owner"]), str(row["owner"])),
                "producer": str(row["producer"]),
            }
        )
    return rows
