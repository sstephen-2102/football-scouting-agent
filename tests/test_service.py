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


def test_scores_are_unavailable_until_pipeline_runs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("service.deps.artifact_path", lambda name: tmp_path / "missing.file")
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


def test_member_c_routes_serve_artifacts_and_decisions(monkeypatch, tmp_path) -> None:
    import json

    import pandas as pd

    scores_file = tmp_path / "player_scores.parquet"
    scores_file.write_bytes(b"placeholder")

    search_file = tmp_path / "best_xi.json"
    search_file.write_text(
        json.dumps(
            {
                "shortlist_source": "per_slot_test_by_pred_value",
                "expansion_cap": 5000,
                "uninformed": {"algorithm": "bfs", "xi": {"GK": 1}, "quality": 10.0, "nodes_expanded": 5, "hit_cap": False},
                "informed": {"algorithm": "astar", "xi": {"GK": 1}, "quality": 20.0, "nodes_expanded": 3, "hit_cap": False},
                "comparison": {"informed_minus_uninformed": 10.0},
            }
        ),
        encoding="utf-8",
    )

    squad_file = tmp_path / "squad.csv"
    pd.DataFrame(
        [{"player_id": 1, "slot_or_group": "GK", "pred_value": 10.0, "cost": 8.0, "source": "mip"}]
    ).to_csv(squad_file, index=False)
    (tmp_path / "allocation_manifest.json").write_text(
        json.dumps({"budget_eur": 290_000_000.0, "mip": {"status": "optimal"}}),
        encoding="utf-8",
    )

    policy_file = tmp_path / "q_table.json"
    policy_file.write_text(
        json.dumps(
            {
                "format": "q_table_v1",
                "budget_eur": 80_000_000.0,
                "budget_bin_eur": 5_000_000,
                "actions": ["bid", "pass"],
                "entries": {"FWD|5||bid": 12.0, "FWD|5||pass": 1.0},
            }
        ),
        encoding="utf-8",
    )

    mapping = {
        "scores": scores_file,
        "search": search_file,
        "squad": squad_file,
        "policy": policy_file,
    }
    monkeypatch.setattr("service.deps.artifact_path", lambda name: mapping[name])
    monkeypatch.setattr("service.routes.capabilities.artifact_path", lambda name: mapping[name])
    monkeypatch.setattr("src.policy.bid.artifact_path", lambda name: mapping[name])

    xi_resp = client.post(
        "/v1/search/xi",
        json={"club_id": 11, "season": 2024, "shortlist_player_ids": list(range(1, 12))},
    )
    assert xi_resp.status_code == 200
    assert xi_resp.json()["informed"]["quality"] == 20.0

    squad_resp = client.post(
        "/v1/squad/allocate",
        json={"club_id": 11, "season": 2024, "budget_eur": 290_000_000.0},
    )
    assert squad_resp.status_code == 200
    assert squad_resp.json()["manifest"]["mip"]["status"] == "optimal"
    assert len(squad_resp.json()["squad"]) == 1

    bid_resp = client.post(
        "/v1/policy/bid",
        json={"player_id": 418560, "remaining_budget_eur": 25_000_000.0, "position_hole": "FWD"},
    )
    assert bid_resp.status_code == 200
    assert bid_resp.json()["action"] == "bid"
