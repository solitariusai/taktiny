# If the first goal is a test or lint target, treat subsequent goals as target paths/arguments
ifneq ($(filter test test-fast test-single lint lint-fix,$(firstword $(MAKECMDGOALS))),)
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  .PHONY: $(RUN_ARGS)
  $(RUN_ARGS):
	@:
endif

LINT_PATHS := $(if $(RUN_ARGS),$(RUN_ARGS),.)

.PHONY: help test test-fast test-single docs docs-clean lint lint-fix

help:
	@echo "Available targets:"
	@echo "  make test [path]         Run tests in parallel with pytest-xdist (-n auto)"
	@echo "  make test-fast [path]    Run tests in parallel with concise output (-n auto -q)"
	@echo "  make test-single [path]  Run tests sequentially"
	@echo "  make docs                Build HTML documentation"
	@echo "  make docs-clean          Clean documentation build directory"
	@echo "  make lint [path]         Run Ruff linter (defaults to all files)"
	@echo "  make lint-fix [path]     Run Ruff linter and automatically apply fixes"

test:
	uv run --group dev pytest -n auto $(RUN_ARGS) $(ARGS)

test-fast:
	uv run --group dev pytest -n auto -q $(RUN_ARGS) $(ARGS)

test-single:
	uv run --group dev pytest $(RUN_ARGS) $(ARGS)

docs:
	uv run --group docs make -C docs html

docs-clean:
	uv run --group docs make -C docs clean

lint:
	uv run --group dev ruff check $(LINT_PATHS) $(ARGS)

lint-fix:
	uv run --group dev ruff check --fix $(LINT_PATHS) $(ARGS)
