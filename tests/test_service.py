"""HTTP service contract tests (no Transfermarkt download)."""

from fastapi.testclient import TestClient

from service.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_meta_and_agent() -> None:
    meta = client.get("/v1/meta").json()
    assert meta["dataset"] == "davidcariboo/player-scores"
    assert meta["big5"] == ["GB1", "ES1", "IT1", "L1", "FR1"]
    assert meta["team"]["member_c"] == "Sanjay Varghese Stephen"
    assert meta["team"]["member_a"] == "Arunava Banerjee"
    assert meta["team"]["member_b"] == "Binish Hamdani"
    agent = client.get("/v1/agent").json()
    assert "performance" in agent["peas"]
    assert "not transfer advice" in agent["disclaimer"]
    assert len(agent["dimensions"]) == 10


def test_status_lists_all_artifacts() -> None:
    payload = client.get("/v1/status").json()
    assert set(payload["artifacts"]) >= {
        "split",
        "archetypes",
        "forecast",
        "scores",
        "search",
        "squad",
        "policy",
    }


def test_scores_are_unavailable_until_pipeline_runs() -> None:
    response = client.get("/v1/scores/418560")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["owner"] == "member_b"
    assert detail["artifact"] == "scores"


def test_index_page_is_served() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert b"Football Scouting Agent" in response.content
    for label in (
        b"Executive Dashboard",
        b"Archetype Lab",
        b"Experiment Board",
        b"Search Lab",
        b"Squad Builder",
        b"Bid Policy",
    ):
        assert label in response.content


def test_lab_assets_are_served() -> None:
    css = client.get("/static/app.css")
    script = client.get("/static/app.js")
    assert css.status_code == 200
    assert "sidebar" in css.text
    assert script.status_code == 200
    assert "ARTIFACT_FOR_PAGE" in script.text
