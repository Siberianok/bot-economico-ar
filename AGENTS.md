# ObeconBot Project Instructions

## Scope

- Work only inside `C:\Users\ariel\Downloads\IA\ObeconBot`.
- Never read, edit, move, delete, or rename files in sibling projects from this project thread.
- If a request appears to belong to another project, stop and ask for explicit confirmation before taking action.
- Do not change global operating-system, shell, Python, Node, npm, Git, editor, service, or scheduled-task configuration.

## Compatibility Boundaries

- Preserve `bot_econ_full_plus_rank_alerts.py` unless the requested change explicitly requires a small, reviewed legacy edit.
- Do not remove existing Telegram commands or migrate `ALERTS`, `SUBS`, or `PF` without explicit approval.
- Do not hardcode tokens, Render URLs, or other secrets.
- Use `not_available` when reliable data or a safe legacy adaptation is unavailable.
- Prefer project-local dependencies and reversible changes.

## Required Implementation Workflow

For every task that changes project files:

1. Convert the user request into an explicit acceptance checklist.
2. For substantial changes, delegate architecture analysis to `architect_obeconbot` before editing. It defines boundaries, contracts, risks, and the implementation brief but never edits files.
3. Delegate implementation to `developer_obeconbot`, or implement under the same restrictions when delegation is unnecessary. Inspect affected code and neighboring contracts before editing.
4. Make small, reviewable changes and preserve unrelated user work.
5. Run the project quality gate and the focused tests for the changed area.
6. Delegate the final read-only review to `auditor_obeconbot` for every completed implementation. It must verify acceptance criteria and operability independently and must not edit files.
7. Return actionable findings to the developer, resolve them, and repeat the audit when needed before claiming completion. Otherwise report the exact blocker and affected checklist item.
8. In the final response, report the checklist result, files changed, tests executed, failures, warnings, and remaining limitations.

Do not invoke these agents for conceptual questions or read-only explanations. Never allow the architect or auditor to edit files.

## Project Agent Responsibilities

- `architect_obeconbot` / Plato: base layer. Owns architecture, infrastructure, module boundaries, functional contracts, persistence design, security, compatibility, and implementation planning.
- `developer_obeconbot` / James: development layer. Implements approved backend, frontend, Telegram, data, tests, and documentation work without self-approving the result.
- `auditor_obeconbot` / Darwin: final layer. Verifies the operable result, acceptance criteria, regressions, tests, builds, deployment readiness, and residual risk independently.

The sequence for substantial work is Plato -> James -> Darwin. The main Codex agent coordinates the sequence, resolves conflicts, and remains responsible for the final answer.

## Validation

Run the local quality gate from the project root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\.codex\hooks\quality-gate.ps1 -Mode Blocking
```

The `Stop` hook provides an automatic final run in blocking mode. Do not claim completion when a required gate step fails.
