# Contributing
ArcticPlatform welcomes your contributions!

You may want to start with [good first issue label](https://github.com/Snowflake-AI-Research/Arctic-Platform/issues?q=state%3Aopen%20label%3A%22good%20first%20issue%22) issues.

## Prerequisites
ArcticPlatform uses [pre-commit](https://pre-commit.com/) to ensure that
formatting is consistent across the library and
[pytest](https://docs.pytest.org/en/stable/) to verify library functionality and
correctness.  Installing with the extra dev dependencies will ensure you have
both of these packages installed:

If you are installing from a pre-built wheel hosted on PyPI:
```bash
pip install "arctic_platform[dev]"
```

If you are installing from the cloned repository:
```bash
pip install ".[dev]"
```

Next, install the pre-commit hooks so that our suite of formatting workflows will
run automatically before each `git commit`:
```bash
pre-commit install
```

## Formatting
Our formatting pre-commit hooks will run automatically if you ran the above
command to install the pre-commit hooks. However, you can also run the
formatting manually using:
```bash
pre-commit run --all-files
```

It's better to use:
```
make format
```
as it'll install all the required venv things automatically for you and will do the right thing.

If a formatting test fails, some pre-commit hooks will attempt to modify the
code in place. Other formatting checks, like the [mypy type
checker](https://mypy-lang.org/) will require you to make modifications to the
code yourself. In either event, if any of the pre-commit hooks fail, the `git
commit` is aborted. After reviewing automatic changes or making your own changes
to fix formatting issues, you can `git add <modified files>` and then repeat the
previous `git commit` command.

## Unit Tests
A collection of unit tests can be found in the `tests/` directory. Broadly,
there are two types of tests: CPU-based and GPU-based. GPU-based tests are
guarded by `@require_torch_gpu` and automatically skip when no GPU is
available, so the full suite is safe to run anywhere.

To run all tests (CPU and GPU):
```bash
make test
```
or:
```bash
pytest tests
```

To run only the CPU-based tests (the GPU tests self-skip when GPUs are hidden):
```bash
make test-cpu
```
or:
```bash
CUDA_VISIBLE_DEVICES= pytest tests
```

If you have multiple GPUs, you can run the suite in parallel with:
```bash
make test-fast
```
