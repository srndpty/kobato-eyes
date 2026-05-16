# Quality Roadmap

## Current Gates

- Local default: `.\scripts\check.ps1 -NoCoverage`
- Full local gate: `.\scripts\check.ps1`
- GUI/integration follow-up: `python -m pytest -m "gui or integration" -p no:cov`
- CI runs lint/type checks, unit tests with coverage, integration tests, GUI smoke tests, and package compilation on Windows/Python 3.10.

## Known Risk Areas

- DB writes can enter unsafe fast mode; quiesce must always be released before reopening normal UI connections.
- Search runs in a background worker; cancellation must leave the UI non-busy and must not emit query errors for user-initiated cancellation.
- Search ordering must stay on an allowlist because `ORDER BY` cannot be parameterized safely.
- ONNX/GPU paths are environment-sensitive and should remain isolated behind mocks or `gpu` marker tests.

## Next Expansion Order

1. Keep `db.connection`, `db.repository`, `services.db_writing`, and `core.pipeline.stages.write_stage` in mypy coverage.
2. Add focused tests around DB lock recovery, worker cancellation, and query parsing before changing related behavior.
3. Expand GUI smoke coverage only for stable state transitions: start, cancel, finish, and reload.
4. Raise coverage only after GUI and model-provider modules have practical seams for deterministic tests.
