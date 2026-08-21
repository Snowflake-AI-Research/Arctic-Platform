"""Module entry point for ``python -m cortex_training.tui``."""

from dss_client.tui.__main__ import run as _run


def run(argv=None) -> int:
    return _run(argv, prog="cortex-training tui")


if __name__ == "__main__":
    raise SystemExit(run())
