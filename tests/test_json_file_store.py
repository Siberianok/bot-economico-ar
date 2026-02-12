import os
import pathlib
import sys

import asyncio
import json

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")

from bot.persistence.state import JsonFileStore


def test_json_file_store_save_writes_payload(tmp_path):
    path = tmp_path / "state.json"
    store = JsonFileStore(str(path))
    payload = {"mensaje": "mañana", "valor": 1}

    ok = asyncio.run(store.save(payload))

    assert ok is True
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert "mañana" in path.read_text(encoding="utf-8")


def test_json_file_store_save_works_when_target_does_not_exist(tmp_path):
    path = tmp_path / "state.json"
    store = JsonFileStore(str(path))

    assert not path.exists()

    ok = asyncio.run(store.save({"x": 1}))

    assert ok is True
    assert json.loads(path.read_text(encoding="utf-8")) == {"x": 1}


def test_json_file_store_save_failure_keeps_existing_file_valid(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    temp_path = tmp_path / "state.json.tmp"
    original_payload = {"ok": True, "version": 1}
    path.write_text(json.dumps(original_payload), encoding="utf-8")
    store = JsonFileStore(str(path))

    def broken_dump(payload, fh, ensure_ascii):
        fh.write("{")
        raise RuntimeError("boom")

    monkeypatch.setattr("bot.persistence.state.json.dump", broken_dump)

    ok = asyncio.run(store.save({"ok": False}))

    assert ok is False
    assert json.loads(path.read_text(encoding="utf-8")) == original_payload
    assert not temp_path.exists()
