"""Domain routes. They load pipeline artifacts; they never fit models."""

from fastapi import APIRouter

from service.deps import require_artifact
from service.schemas import AllocateRequest, BidRequest, BidResponse, XiSearchRequest

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
def search_xi(body: XiSearchRequest) -> dict[str, int | str]:
    require_artifact("scores")
    require_artifact("search")
    return {"status": "ok", "club_id": body.club_id, "season": body.season}


@router.post("/squad/allocate")
def allocate_squad(body: AllocateRequest) -> dict[str, float | int | str]:
    require_artifact("forecast")
    require_artifact("squad")
    return {
        "status": "ok",
        "club_id": body.club_id,
        "budget_eur": body.budget_eur,
    }


@router.post("/policy/bid", response_model=BidResponse)
def decide_bid(body: BidRequest) -> BidResponse:
    require_artifact("policy")
    require_artifact("scores")
    return BidResponse(
        action="undecided",
        player_id=body.player_id,
        rationale="Policy artifact is present; decision wiring lands with Member C.",
    )
