import os
from pathlib import Path
from arctic_platform.rl.utils.debug import print_rank0 as pr0
import torch


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "")
    if not val:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _replay_generation_enabled() -> bool:
    # Read at call time: Ray TaskRunner may import this module before env is set.
    return _env_flag("REPLAY_GENERATION")


def _replay_generation_dir() -> Path:
    gen_dir = os.environ.get("REPLAY_GENERATION_DIR", None)
    if gen_dir is None:
        return None
    return Path(gen_dir)


def _replay_generation_mode() -> str:
    mode = os.environ.get("REPLAY_GENERATION_MODE", "record").lower()
    if mode not in ("record", "load"):
        raise ValueError(
            f"Invalid REPLAY_GENERATION_MODE={mode!r}; expected 'record' or 'load'"
        )
    return mode


def _replay_start_record() -> int:
    # Number of leading corpus records to skip in load mode.
    try:
        return max(0, int(os.environ.get("REPLAY_START_RECORD", "0") or 0))
    except ValueError:
        return 0


class RecordReplay:
    def __init__(self):
        self.enabled = _replay_generation_enabled()
        self.dir = _replay_generation_dir()
        self.mode = _replay_generation_mode()
        self.start_record = _replay_start_record()

        if self.is_replay_mode() and self.start_record > 0:
            pr0(
                f"[REPLAY_GENERATION] REPLAY_START_RECORD={self.start_record}: "
                f"will skip records 1..{self.start_record} and begin replay at "
                f"record {self.start_record + 1}",
            )
        if self.enabled:
            if self.is_record_mode():
                self.dir.mkdir(parents=True, exist_ok=True)
            elif self.is_replay_mode():
                if not self.dir.exists():
                    raise FileNotFoundError(
                        f"[REPLAY_GENERATION] load mode requires {self.dir}"
                    )

    def is_enabled(self) -> bool:
        return self.enabled and self.dir is not None

    def is_record_mode(self) -> bool:
        return self.is_enabled() and self.mode == "record"

    def is_replay_mode(self) -> bool:
        return self.is_enabled() and self.mode == "load"

    def skip_record(self, record_index: int) -> bool:
        return self.is_enabled() and self.is_replay_mode() and record_index <= self.start_record

    def save_record(self, record_index: int, batch_dict: dict):
        if self.is_record_mode():
            replay_path = self.dir / f"generate-{record_index}.pickle"
            pr0(f"[REPLAY_GENERATION] saving {replay_path}")
            torch.save(batch_dict, replay_path)

    def load_record(self, record_index: int) -> dict:
        if self.is_replay_mode():
            replay_path = self.dir / f"generate-{record_index}.pickle"
            if not replay_path.exists():
                raise FileNotFoundError(
                    f"[REPLAY_GENERATION] load mode requires {replay_path}"
                )
            pr0(f"[REPLAY_GENERATION] loading {replay_path}")
            return torch.load(replay_path, weights_only=False)