from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.services.signal_service import emit_signal, list_signals


def _db():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, future=True)()


def test_signal_emit_deduplicates_by_key():
    db = _db()
    first, created_first = emit_signal(
        db,
        dedup_key="test:key",
        title="Test",
        description="Signal",
        score=82,
        cooldown_minutes=60,
    )
    second, created_second = emit_signal(
        db,
        dedup_key="test:key",
        title="Test",
        description="Signal",
        score=82,
        cooldown_minutes=60,
    )
    assert created_first is True
    assert created_second is False
    assert first["id"] == second["id"]


def test_list_signals_contains_system_signal():
    db = _db()
    signals = list_signals(db)
    assert signals
    assert signals[0]["score"] >= 70

