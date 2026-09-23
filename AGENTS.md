# Repository Guidelines

## Project Structure & Module Organization

Taktiny is an experimental JAX neural-network library. Python packages live under `src/taktiny/`. Put automated tests in `tests/`, hardware-specific cases in `tests/accelerators/{gpu,tpu}/`, benchmarks in `benchmarks/`, and exploratory notebooks in `notebook/`. Project metadata is defined by `pyproject.toml` and `uv.lock`.

## Build, Test, and Development Commands

- `uv sync --frozen --group dev` installs the Python 3.12 development environment from `uv.lock`.
- `uv run --frozen pytest` runs the CPU-oriented suite exactly as CI does.
- `uv run pytest tests/linear_test.py -q` runs one focused test module.
- `make test-fast [path]` runs tests in parallel with `pytest-xdist`.

Keep `JAX_PLATFORMS=cpu` for portable local/CI checks unless a test explicitly targets an accelerator.

## Coding Style & Naming Conventions

Use four-space indentation, modern Python type annotations, and concise docstrings for public APIs. Follow existing module conventions: `snake_case` for files, functions, and variables; `PascalCase` for classes; and uppercase names for constants. Keep imports grouped as standard library, third-party, then local packages. No repository-wide Python formatter is configured, so match nearby code and keep changes focused.

## Testing Guidelines

Tests use `pytest` and are named `tests/*_test.py`, with functions named `test_<behavior>`. Add deterministic, small cases alongside the affected feature. Avoid downloads and large checkpoints in the default suite. Mark device-only tests with `@pytest.mark.gpu` or `@pytest.mark.tpu`; no coverage threshold is configured. Run the focused test first, then the full suite before opening a PR.

## Commit & Pull Request Guidelines

Recent history follows Conventional Commit-style subjects such as `feat(trainer): ...`, `fix(overture): ...`, and `refactor: ...`. Use an imperative, concise subject with an optional scope. PRs should explain the motivation and behavioral impact, link relevant issues, list verification commands, and note device/backend assumptions. Include screenshots or logs only when they clarify user-visible or performance changes.

## Agent-Specific Instructions

- NEVER execute destructive Git commands under any circumstance:
    - FORBIDDEN: `git checkout <path>`
    - FORBIDDEN: `git restore`
    - FORBIDDEN: `git reset --hard`
    - FORBIDDEN: `git clean`
- NEVER discard, overwrite, or revert uncommitted changes in the user's working tree.
- Only stage specific target files explicitly (e.g., `git add src/...`).
