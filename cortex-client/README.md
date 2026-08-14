# Cortex Training Client

Python SDK, command-line tools, runnable recipes, and documentation for Cortex
Training through the Neutrino SNOWAPI.

## Start Here

- [Set up the client](docs/getting-started/setup.md)
- [Run the first supervised fine-tuning job](docs/getting-started/first-sft-run.md)
- [Browse runnable recipes](recipes/README.md)
- [Check model and training-method compatibility](docs/reference/model-compatibility.md)
- [Use the CLI and Python client](docs/reference/cli.md)
- [Read the REST API reference](docs/reference/rest-api.md)

## Install

Requires Python 3.8 or later.

```bash
pip install "dss-client @ git+https://github.com/Snowflake-AI-Research/Arctic-Platform.git#subdirectory=cortex-client"
```

For local development:

```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform.git
cd Arctic-Platform/cortex-client
pip install -e .
```

The package installs:

- `dss-neutrino`, for submitting and managing jobs
- `neutrino-tui`, for viewing job logs
- `dss_client`, the Python SDK

## Repository Map

| Path | Purpose |
|---|---|
| `docs/` | Getting started material, concepts, guides, and reference |
| `recipes/` | End-to-end training, sampling, and evaluation workflows |
| `examples/api/` | Small JSON examples for individual API operations |
| `examples/config/` | Connection configuration templates |
| `dss_client/` | Installable Python client |
| `tests/` | Client and CLI tests |

The current onboarding work is tracked in
[docs/internal/onboarding-roadmap.md](docs/internal/onboarding-roadmap.md).
