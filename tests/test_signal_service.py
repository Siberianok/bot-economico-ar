from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.services.signal_service import (
    derive_market_signal_candidates,
    emit_signal,
    list_signals,
)


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


def test_list_signals_does_not_invent_system_signal():
    db = _db()
    signals = list_signals(db)
    assert signals == []


def test_retired_synthetic_system_signal_is_not_listed():
    db = _db()
    emit_signal(
        db,
        dedup_key="system:api-online",
        title="API backend operativa",
        description="Registro sintético retirado",
        score=70,
    )

    assert list_signals(db) == []


def test_market_signal_candidates_use_observed_values_and_transparent_thresholds():
    pulse = {
        "status": "partial",
        "source": "legacy_adapter",
        "data": {
            "brechas": {"status": "ok", "data": {"ccl_pct": 4.5, "blue_pct": 1.2}},
            "dolar_ccl": {"status": "ok", "data": {"variation": -2.1}},
            "riesgo_pais": {"status": "ok", "data": {"value": 512}},
            "inflacion": {"status": "ok", "data": {"value": 2.1}},
        },
    }

    candidates = derive_market_signal_candidates(pulse)

    assert {candidate["dedup_key"] for candidate in candidates} == {
        "market:fx-gap:ccl",
        "market:fx-daily-move:ccl",
    }
    gap = next(candidate for candidate in candidates if candidate["dedup_key"] == "market:fx-gap:ccl")
    assert gap["current_value"] == 4.5
    assert gap["source"] == "legacy_adapter"
    assert gap["score"] >= 70


def test_market_signal_candidates_do_not_invent_normal_state_signal():
    pulse = {
        "status": "partial",
        "source": "legacy_adapter",
        "data": {
            "brechas": {"status": "ok", "data": {"ccl_pct": 1.0, "blue_pct": 1.0}},
            "dolar_ccl": {"status": "ok", "data": {"variation": 0.2}},
            "riesgo_pais": {"status": "ok", "data": {"value": 500}},
            "inflacion": {"status": "ok", "data": {"value": 2.0}},
        },
    }

    assert derive_market_signal_candidates(pulse) == []
