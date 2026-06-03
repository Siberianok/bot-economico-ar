# Observatorio Economico - Architecture

This phase adds a real modular product skeleton without replacing the current Telegram bot.

## Layers

- `bot_econ_full_plus_rank_alerts.py` remains the legacy Telegram bot runtime.
- `backend/app` exposes a FastAPI API for the future Mini App.
- `backend/app/services/legacy_adapter.py` imports legacy code lazily and never starts polling, webhook, schedulers or Telegram sends.
- `frontend` is a Telegram Mini App shell built with React, TypeScript, Vite, Tailwind and ECharts.
- SQLite is the MVP persistence layer for new entities only.

## API

All API responses use the envelope:

```json
{
  "status": "ok",
  "timestamp": "2026-06-02T00:00:00Z",
  "source": "service",
  "freshness": "current",
  "data": {},
  "warnings": []
}
```

## Legacy Boundary

The backend does not migrate `ALERTS`, `SUBS` or `PF`. Any feature that cannot safely read legacy data returns `not_available`.

