from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.schemas.projections import ProjectionCreate
from backend.app.services.projection_service import create_projection, list_projections, list_validations


def _db():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, future=True)()


def test_projection_create_stores_scenarios():
    db = _db()
    created = create_projection(
        db,
        ProjectionCreate(
            symbol="YPFD.BA",
            horizon="3m",
            base_price=1000,
            pessimistic=-5,
            base_case=8,
            optimistic=18,
            confidence=62,
        ),
    )
    assert created["symbol"] == "YPFD.BA"
    assert created["scenarios"]["base"] == 8
    assert created["validation_status"] == "pending"
    assert list_projections(db)


def test_validations_initially_empty():
    db = _db()
    assert list_validations(db) == []

