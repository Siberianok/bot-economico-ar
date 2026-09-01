# ObeconBot Codex Project Team

This configuration is local to ObeconBot.

## Components

- `../AGENTS.md`: durable project boundaries and the required architecture, implementation, and audit workflow.
- `agents/architect_obeconbot.toml`: Plato, read-only architecture and infrastructure specialist.
- `agents/developer_obeconbot.toml`: James, project-scoped implementation specialist.
- `agents/auditor_obeconbot.toml`: Darwin, independent read-only final auditor.
- `hooks.json`: project-local `Stop` hook.
- `hooks/quality-gate.ps1`: deterministic import, full test suite, V2 lint, dependency, contamination,
  and frontend checks.

The script keeps `.codex/.tmp` as an empty runtime container and removes each per-run subdirectory when validation finishes.

## Activation

Open `C:\Users\ariel\Downloads\IA\ObeconBot` as the Codex project or working directory. Project-local agents and hooks are discovered from the active trusted `.codex` layer; opening only the parent `C:\Users\ariel\Downloads\IA` does not activate this project configuration.

Review and trust the new hook definition from `/hooks` before expecting it to run. Codex stores trust against the hook definition hash, so edits require a new review.

## Team Workflow

For substantial implementations, the coordinated order is:

1. Plato inspects the base architecture and produces the implementation brief.
2. James implements the approved work and runs focused validation.
3. Darwin independently checks operability, regressions, acceptance criteria, tests, builds, and release readiness.

For small implementations, Plato may be skipped when no architectural decision is involved, but Darwin remains the final independent reviewer. The main Codex agent coordinates all three roles and owns the final response.

## Modes

- `Informational`: reports failures and warnings but exits successfully.
- `Blocking`: returns a failing exit code when a required step fails.

The configured `Stop` hook uses `Blocking` mode because the full Python suite, V2 lint, frontend build,
strict typecheck, dependency consistency check, and project-boundary checks are green. A failed required
step prevents the implementation from being treated as complete.

The command path in `hooks.json` intentionally points to the canonical local ObeconBot folder. Update it if the project is moved or renamed.
