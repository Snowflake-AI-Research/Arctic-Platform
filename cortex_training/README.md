# Cortex Training Client

Python SDK, command-line tools, runnable recipes, and documentation for Cortex
Training through the Cortex Training SNOWAPI.

## Start Here

- [Follow the getting-started path](docs/getting-started/README.md)
- [Run the first supervised fine-tuning job](docs/getting-started/first-sft-run.md)
- [Browse runnable recipes](recipes/README.md)
- [Check model and training-method compatibility](docs/reference/model-compatibility.md)
- [Use the CLI and Python client](docs/reference/cli.md)
- [Read the REST API reference](docs/reference/rest-api.md)

## Install

Requires Python 3.8 or later.

```bash
pip install "cortex-training @ git+https://github.com/Snowflake-AI-Research/Arctic-Platform.git#subdirectory=cortex_training"
```

For local development:

```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform.git
cd Arctic-Platform/cortex_training
pip install -e .
```

The package installs:

- `cortex-training`, for submitting and managing jobs
- `cortex-training tui`, for viewing job logs
- `cortex_training`, the Python SDK

Verify the command entry points:

```bash
cortex-training --help
cortex-training tui --help
```

## Usage

```bash
cortex-training list
cortex-training submit examples/api/training.json
cortex-training tui JOB_ID
```

```python
from cortex_training import CortexTrainingClient, CortexTrainingEngine
```

Connection settings use `CORTEX_TRAINING_*` and `SNOWFLAKE_*` environment
variables. Login state is stored under `~/.config/cortex-training/`, and TUI
cache state is stored under `~/.cache/cortex-training/`, unless their existing
override variables are used.

See the [CLI reference](docs/reference/cli.md) for commands and configuration.

## Repository Map

| Path | Purpose |
|---|---|
| `docs/` | Getting started material, concepts, guides, and reference |
| `recipes/` | End-to-end training, sampling, and evaluation workflows |
| `examples/api/` | Small JSON examples for individual API operations |
| `examples/config/` | Connection configuration templates |
| `cortex_training/` | Installable Python client |
| `tests/` | Client and CLI tests |

The current onboarding work is tracked in
[docs/internal/onboarding-roadmap.md](docs/internal/onboarding-roadmap.md).
