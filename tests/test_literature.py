"""Reference writer writes Markdown, not notebooks."""

from src.agent.literature import write_references


def test_write_references(tmp_path) -> None:
    path = write_references(tmp_path / "references.md")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# References")
    assert "Transfermarkt" in text
    assert "Sutton" in text
