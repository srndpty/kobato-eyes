# Quality Roadmap

## Current Gates

- Local default: `.\scripts\check.ps1 -NoCoverage`
- Full local gate: `.\scripts\check.ps1`
- GUI/integration follow-up: `python -m pytest -m "gui or integration" -p no:cov`
- DB stress follow-up: `python -m pytest -m db_stress -p no:cov`
- CI runs lint/type checks, unit tests with coverage, integration tests, GUI smoke tests, and package compilation on Windows/Python 3.10.

## Known Risk Areas

- DB writes can enter unsafe fast mode; quiesce must always be released before reopening normal UI connections.
- Search runs in a background worker; cancellation must leave the UI non-busy and must not emit query errors for user-initiated cancellation.
- Search ordering must stay on an allowlist because `ORDER BY` cannot be parameterized safely.
- ONNX/GPU paths are environment-sensitive and should remain isolated behind mocks or `gpu` marker tests.

## Priority Contracts

- **A: DB recovery**
  - `quiesced()` must release global quiesce state after nested scopes, exceptions, and defensive extra cleanup.
  - Unsafe fast writes must fall back to WAL mode when an exclusive lock is unavailable.
  - Writer shutdown must best-effort restore normal SQLite mode before regular UI/search connections reopen.
- **A: Search worker cancellation**
  - User cancellation must finish as `(ok=False, cancelled=True)` without emitting `error`.
  - SQLite `interrupt()` and `OperationalError("interrupted")` must be treated as cancellation when cancellation was requested.
  - Deleted Qt objects must not leave the worker thread running.
- **B: Query safety**
  - Search ordering must be selected from the repository allowlist only.
  - Relevance ordering must only activate when positive tag terms are available.
  - Danbooru tag parsing should follow the project assumption that tag names do not start with `-`, `(`, or `)`, and do not contain whitespace.
- **B: Gate separation**
  - GPU/ONNX coverage remains opt-in behind mocks or `gpu` marker tests.
  - DB stress tests are explicit follow-up gates, not part of the fast unit default.

## Next Expansion Order

1. Keep `db.connection`, `db.repository`, `services.db_writing`, and `core.pipeline.stages.write_stage` in mypy coverage.
2. Add focused tests around DB lock recovery, worker cancellation, and query parsing before changing related behavior.
3. Expand GUI smoke coverage only for stable state transitions: start, cancel, finish, and reload.
4. Raise coverage only after GUI and model-provider modules have practical seams for deterministic tests.

## Acceptance Checklist

- [ ] DB unsafe-fast failure, writer exceptions, and user cancellation do not leave normal DB connections blocked.
- [ ] Search cancellation clears busy UI state and does not show query errors for expected cancellation.
- [ ] Search SQL uses only allowlisted `ORDER BY` clauses.
- [ ] Mypy still covers `db.connection`, `db.repository`, `services.db_writing`, and `core.pipeline.stages.write_stage`.
- [ ] GPU/ONNX paths stay outside normal CI unless mocked.
- [ ] New quality work includes a focused regression test before behavior changes.
