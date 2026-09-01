# Alert Engine

Implemented in this phase:

- Exact alerts with operators `>=`, `<=`, `>`, `<`, `==`.
- Approximation alerts with percentage tolerance.
- Cooldown enforcement.
- The evaluation engine supports active and paused rules internally.
- Mini App creation is intentionally restricted to user-scoped `draft` and `paused` states.
- Trigger history is persisted in local SQLite for development.
- Telegram delivery is explicitly `not_connected`; drafts are not sent or presented as active.
- `ALERTS` and `SUBS` are now exposed to the authenticated owner through a
  deep-copied, read-only view.
- Operational Telegram rules and SQLite drafts remain separate in both the API
  contract and the Mini App.

Pending:

- Cross alerts.
- Range alerts.
- Spread and gap alerts.
- Multi-read confirmation runner.
- Promotion of a reviewed SQLite draft into an operational Telegram rule.
- Append-only operational trigger history; legacy currently retains only the
  last trigger per rule.
- Persistent external storage and a delivery runner isolated per Telegram user.
