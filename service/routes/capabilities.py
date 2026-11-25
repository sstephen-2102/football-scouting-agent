"""Domain routes. They load pipeline artifacts; they never fit models."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from fastapi import APIRouter

from service.deps import require_artifact
from service.schemas import AllocateRequest, BidRequest, BidResponse, XiSearchRequest
from src.artifacts import artifact_path
from src.policy.bid import decide_bid as policy_decide_bid

router = APIRouter(prefix="/v1")


@router.get("/archetypes")
def archetypes() -> dict[str, str]:
    require_artifact("archetypes")
    return {"status": "ok", "detail": "Archetype labels are available on disk."}


@router.get("/forecast")
def forecast() -> dict[str, str]:
    require_artifact("forecast")
    return {"status": "ok", "detail": "Forecast budget artifact is available on disk."}


@router.get("/scores/{player_id}")
def score_player(player_id: int) -> dict[str, int | str]:
    require_artifact("scores")
    return {"status": "ok", "player_id": player_id}


@router.post("/search/xi")
def search_xi(body: XiSearchRequest) -> dict[str, Any]:
    require_artifact("scores")
    require_artifact("search")
    payload = json.loads(artifact_path("search").read_text(encoding="utf-8"))
    return {
        "status": "ok",
        "club_id": body.club_id,
        "season": body.season,
        "shortlist_source": payload.get("shortlist_source"),
        "expansion_cap": payload.get("expansion_cap"),
        "uninformed": payload.get("uninformed"),
        "informed": payload.get("informed"),
        "comparison": payload.get("comparison"),
    }


@router.post("/squad/allocate")
def allocate_squad(body: AllocateRequest) -> dict[str, Any]:
    require_artifact("squad")
    csv_path = artifact_path("squad")
    squad_rows = pd.read_csv(csv_path).to_dict(orient="records")
    manifest_path = csv_path.parent / "allocation_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    return {
        "status": "ok",
        "club_id": body.club_id,
        "season": body.season,
        "budget_eur": body.budget_eur,
        "manifest": manifest,
        "squad": squad_rows,
    }


@router.post("/policy/bid", response_model=BidResponse)
def decide_bid(body: BidRequest) -> BidResponse:
    require_artifact("policy")
    require_artifact("scores")
    decision = policy_decide_bid(
        player_id=body.player_id,
        remaining_budget_eur=body.remaining_budget_eur,
        position_hole=body.position_hole,
    )
    return BidResponse(**decision)
