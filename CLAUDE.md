# pydantic-ai-tool-budget

Per-tool budget reminders for Pydantic AI agents.

## Development

```bash
uv sync                    # install deps
uv run pytest -v           # run tests
uv run ruff check          # lint
uv run ruff format --check # format check
uv run pyright             # type check
```

## Releasing a new version

1. Update the version in `pyproject.toml`
2. Commit: `git commit -am "release: vX.Y.Z"`
3. Tag: `git tag vX.Y.Z`
4. Push: `git push origin main --tags`
5. Create a GitHub Release from the tag (this triggers PyPI publish via trusted publishing)

The publish workflow (`.github/workflows/publish.yml`) handles building and uploading to PyPI automatically when a GitHub Release is created.

## Project structure

- `src/pydantic_ai_tool_budget/` — package source
  - `_budgeted.py` — `budgeted()` decorator (primary API)
  - `_budget.py` — `ToolBudget` class (shared budget pools)
  - `__init__.py` — public exports
- `tests/` — pytest test suite with VCR cassettes for Gemini integration tests
- `.github/workflows/ci.yml` — CI: lint, typecheck, test across Python 3.10–3.13
- `.github/workflows/publish.yml` — publish to PyPI on GitHub Release

## Conventions

- Build backend: hatchling
- Type checking: pyright (strict mode, Python 3.10 target)
- Linting: ruff (E, F, I, UP, W rules)
- Line length: 120
- Test runner: pytest with asyncio auto mode
- Minimum Python: 3.10
- Only dependency: pydantic-ai-slim>=0.100.0
