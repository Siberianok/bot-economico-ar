"""Projection and validation routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas.envelope import envelope
from backend.app.schemas.projections import ProjectionCreate
from backend.app.services.projection_service import create_projection, list_projections, list_validations
from backend.app.utils.time import utc_now_iso


projections_router = APIRouter(prefix="/projections", tags=["projections"])
validations_router = APIRouter(prefix="/validations", tags=["validations"])


@projections_router.get("")
def projections(db: Session = Depends(get_db)) -> dict:
    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="projection_service",
        freshness="current",
        data={"items": list_projections(db)},
        warnings=["Proyecciones live legacy pendientes de adaptar"],
    )


@projections_router.post("")
def projections_create(payload: ProjectionCreate, db: Session = Depends(get_db)) -> dict:
    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="projection_service",
        freshness="current",
        data=create_projection(db, payload),
        warnings=[],
    )


@validations_router.get("")
def validations(db: Session = Depends(get_db)) -> dict:
    return envelope(
        status="ok",
        timestamp=utc_now_iso(),
        source="projection_service",
        freshness="current",
        data={"items": list_validations(db)},
        warnings=["Evaluacion futura preparada; sin validaciones maduras todavia"],
    )

