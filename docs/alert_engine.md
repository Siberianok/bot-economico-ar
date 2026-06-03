# Alert Engine

Implemented in this phase:

- Exact alerts with operators `>=`, `<=`, `>`, `<`, `==`.
- Approximation alerts with percentage tolerance.
- Cooldown enforcement.
- Active and paused states.
- Trigger history persisted in SQLite.
- Telegram channel field prepared, but no Telegram delivery is enabled yet.

Pending:

- Cross alerts.
- Range alerts.
- Spread and gap alerts.
- Multi-read confirmation runner.
- Visible Telegram buttons for pause/edit/open dashboard.

