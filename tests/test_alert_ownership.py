from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.schemas.alerts import AlertCreate, AlertUpdate
from backend.app.services.alert_service import (
    create_alert,
    delete_alert,
    list_alerts,
    update_alert,
)


def _db():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, future=True)()


def test_alerts_are_scoped_to_authenticated_user_and_created_as_drafts():
    db = _db()
    created = create_alert(
        db,
        AlertCreate(asset="dolar_cripto", target_value=1500),
        "telegram-1",
    )

    assert created["status"] == "draft"
    assert created["delivery"]["status"] == "not_connected"
    assert len(list_alerts(db, "telegram-1")) == 1
    assert list_alerts(db, "telegram-2") == []
    assert update_alert(db, created["id"], AlertUpdate(status="paused"), "telegram-2") is None
    assert delete_alert(db, created["id"], "telegram-2") is False
