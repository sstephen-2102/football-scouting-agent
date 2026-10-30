"""Pydantic models for the HTTP API."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class AgentResponse(BaseModel):
    peas: dict[str, str]
    environment: dict[str, str]
    disclaimer: str
    snapshot_date: str


class MetaResponse(BaseModel):
    dataset: str
    snapshot_date: str
    big5: list[str]
    train_seasons: list[int]
    val_seasons: list[int]
    test_seasons: list[int]
    min_appearances: int


class ArtifactRow(BaseModel):
    ready: bool
    path: str
    owner: str
    producer: str


class StatusResponse(BaseModel):
    artifacts: dict[str, ArtifactRow]


class NotReadyResponse(BaseModel):
    status: str = "not_ready"
    artifact: str
    owner: str
    producer: str
    detail: str


class XiSearchRequest(BaseModel):
    club_id: int
    season: int
    shortlist_player_ids: list[int] = Field(min_length=11, max_length=40)


class AllocateRequest(BaseModel):
    club_id: int
    season: int
    budget_eur: float = Field(gt=0)


class BidRequest(BaseModel):
    player_id: int
    remaining_budget_eur: float = Field(ge=0)
    position_hole: str


class BidResponse(BaseModel):
    action: str
    player_id: int
    rationale: str
    extras: dict[str, Any] = Field(default_factory=dict)
