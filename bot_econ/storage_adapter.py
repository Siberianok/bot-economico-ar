from __future__ import annotations

from . import storage
from .config import AppConfig


def configure_storage(config: AppConfig) -> None:
    storage.configure(redis_url=config.redis_url or None, prefix=config.redis_prefix)


alerts_add = storage.alerts_add
alerts_list = storage.alerts_list
alerts_del_all = storage.alerts_del_all
alerts_del_by_index = storage.alerts_del_by_index
alert_chats_all = storage.alert_chats_all
alerts_pause_indef = storage.alerts_pause_indef
alerts_pause_hours = storage.alerts_pause_hours
alerts_pause_until = storage.alerts_pause_until
alerts_resume = storage.alerts_resume
alerts_pause_status = storage.alerts_pause_status

pf_set = storage.pf_set
pf_get = storage.pf_get
pf_del = storage.pf_del
pf_chats_all = storage.pf_chats_all

subs_set = storage.subs_set
subs_get = storage.subs_get
subs_del = storage.subs_del
subs_chats_all = storage.subs_chats_all

redis_ping = storage.redis_ping
