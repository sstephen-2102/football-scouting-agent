"""Health, agent PEAS, dataset meta, and artifact status."""

from fastapi import APIRouter

from service.deps import current_status
from service.schemas import AgentResponse, HealthResponse, MetaResponse, StatusResponse
from src.agent.peas import DISCLAIMER, ENVIRONMENT_NOTES, PEAS
from src.common.config import (
    BIG5_COMPETITION_IDS,
    KAGGLE_DATASET,
    MIN_APPEARANCES,
    SNAPSHOT_DATE,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="football-scouting-agent")


@router.get("/v1/agent", response_model=AgentResponse)
def agent() -> AgentResponse:
    return AgentResponse(
        peas=PEAS,
        environment=ENVIRONMENT_NOTES,
        disclaimer=DISCLAIMER,
        snapshot_date=SNAPSHOT_DATE,
    )


@router.get("/v1/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
    return MetaResponse(
        dataset=KAGGLE_DATASET,
        snapshot_date=SNAPSHOT_DATE,
        big5=list(BIG5_COMPETITION_IDS),
        train_seasons=list(TRAIN_SEASONS),
        val_seasons=list(VAL_SEASONS),
        test_seasons=list(TEST_SEASONS),
        min_appearances=MIN_APPEARANCES,
    )


@router.get("/v1/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(artifacts=current_status())
