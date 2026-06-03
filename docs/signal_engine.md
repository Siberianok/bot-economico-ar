# Signal Engine

Implemented in this phase:

- Signal entity persisted in SQLite.
- Score, severity, status and timestamp.
- Deduplication by `dedup_key` and cooldown window.
- A structural system signal proving the backend signal layer is online.

Pending:

- Live market signal rules.
- Portfolio-aware signals.
- User-specific silence windows.
- Telegram notification delivery.

