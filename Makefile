# usage: make help

.PHONY: help test test-cpu test-gpu test-fast test-flakefinder format autoflake
.DEFAULT_GOAL := help

# number of times test-flakefinder repeats every test; the session timeout scales with it (see below)
FLAKE_RUNS ?= 10

help: ## this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[0-9a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	echo $(MAKEFILE_LIST)

test: ## run cpu and gpu tests
	pytest --disable-warnings --instafail ./tests/

test-cpu: ## run cpu-only tests
	CUDA_VISIBLE_DEVICES= pytest --disable-warnings --instafail ./tests/

test-fast: ## run tests in parallel if there are large gpus
	pytest -n4 --disable-warnings --instafail ./tests/

# repeats each test FLAKE_RUNS times to surface flakes; since the whole suite runs ~FLAKE_RUNS times, scale the
# whole-suite cap (default session_timeout=3600s, set in pyproject.toml) by FLAKE_RUNS so it doesn't abort the run
test-flakefinder: ## run the suite with flakefinder (FLAKE_RUNS=10 repeats), scaling the session timeout
	pytest --flake-finder --flake-runs=$(FLAKE_RUNS) --session-timeout=$$(( 3600 * $(FLAKE_RUNS) )) --disable-warnings --instafail ./tests/

# pre-commit here runs on all modified files of the current branch, even if already pushed
format: ## fix formatting
	@if [ ! -d "venv" ]; then \
		sudo apt update; \
		sudo apt-get install -y python3-venv; \
		python -m venv venv; \
		. venv/bin/activate; \
		pip install pre-commit -U; \
		pre-commit clean; \
		pre-commit uninstall; \
		pre-commit install; \
		deactivate; \
	fi
	. venv/bin/activate && pre-commit run --files $$(git diff --name-only $$(git merge-base main HEAD)...) && deactivate

# this tool is optional not to be run automatically as it could have unexpected side-effects, but is useful when
# needing to remove a bulk of unused imports
autoflake: ## autoremove unused imports (careful!)
	@read -p "Running autoflake will remove unused imports and modify files in place. This could have unexpected side-effects. Do you want to continue? [y/n] " ans; \
	if [ "$$ans" != "y" ]; then \
		echo "Aborted."; \
		exit 1; \
	fi; \
	autoflake --verbose --in-place --remove-all-unused-imports --ignore-init-module-imports --ignore-pass-after-docstring -r arctic_platform
