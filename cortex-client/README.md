# Cortex Training Client

Python SDK, command-line tools, runnable recipes, and documentation for Cortex
Training through the Neutrino SNOWAPI.

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
pip install "dss-client @ git+https://github.com/Snowflake-AI-Research/Arctic-Platform.git#subdirectory=cortex-client"
```

For local development:

```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform.git
cd Arctic-Platform/cortex-client
pip install -e .
```

The package installs:

- `cortex-training`, the canonical command for submitting and managing jobs
- `cortex-training tui`, the canonical command for viewing job logs
- `cortex_training`, the canonical Python namespace
- `dss-neutrino`, `neutrino-tui`, and `dss_client`, the existing compatible
  command and Python names

Verify the command entry points:

```bash
cortex-training --help
cortex-training tui --help
dss-neutrino --help
neutrino-tui --help
```

## Cortex Training Aliases (Phase 1)

New code can use the canonical command and Python names:

```bash
cortex-training list
cortex-training submit examples/api/training.json
cortex-training tui JOB_ID
```

```python
from cortex_training import CortexTrainingClient, CortexTrainingEngine
```

This phase is deliberately additive:

- `dss_client` remains the implementation package.
- `dss-neutrino`, `neutrino-tui`, and all existing Python imports remain valid.
- Authentication behavior and the `NEUTRINO_*` / `SNOWFLAKE_*` environment
  variables are unchanged.
- Login state remains under `~/.config/dss-neutrino/` and TUI cache state
  remains under `~/.cache/neutrino-tui/` unless existing override variables are
  used.
- SNOWAPI request paths, REST payloads, and the DSSST1 wire format are unchanged.

`cortex-training` delegates to the same command implementation as
`dss-neutrino`; `cortex-training tui` delegates to `neutrino-tui`. The
[CLI reference](docs/reference/cli.md) therefore applies to both command names.

## Repository Map

| Path | Purpose |
|---|---|
| `docs/` | Getting started material, concepts, guides, and reference |
| `recipes/` | End-to-end training, sampling, and evaluation workflows |
| `examples/api/` | Small JSON examples for individual API operations |
| `examples/config/` | Connection configuration templates |
| `dss_client/` | Installable Python client |
| `cortex_training/` | Canonical compatibility facade |
| `tests/` | Client and CLI tests |

The current onboarding work is tracked in
[docs/internal/onboarding-roadmap.md](docs/internal/onboarding-roadmap.md).
