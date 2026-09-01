# Signal Engine

Implemented in this phase:

- Signal entity persisted in SQLite.
- Score, severity, status and timestamp.
- Deduplication by `dedup_key` and cooldown window.
- Rules based on observed Market Pulse values for:
  - CCL gap from 3% and blue gap from 5%.
  - Daily FX moves from 2%.
  - Country risk from 600 basis points.
  - Monthly inflation from 4%.
- Scores derived from each rule's magnitude, capped at 100.
- Empty state with `not_available` when no real rule has produced a signal.
- Historical `system:api-online` rows are ignored without deleting the local database.
- The API refreshes candidates before listing and cooldown prevents duplicate rows.

Signals are informational readings. They are not investment recommendations and are not sent to
Telegram in this phase.

Pending:

- Portfolio-aware signals.
- User-specific silence windows.
- Telegram notification delivery.
- Historical validation and configurable thresholds per user.
