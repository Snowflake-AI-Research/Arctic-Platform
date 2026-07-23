"""NeutrinoTrainingEngine -- torch.nn.Module wrapping NeutrinoClient.

Drop-in replacement for :class:`DSSTrainingEngine` that targets the Neutrino
GS API instead of the raw dss-platform.  All async data-plane calls
(forward-backward, step, save) are made synchronous by polling
``NeutrinoClient.poll_request`` internally.
"""

import io
import json

import torch
from torch.nn.modules import Module

from .neutrino_client import NeutrinoClient, SubJobConfig


class InputBatch:
    def __init__(self, args, kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.loss = None


class LRScheduler:
    def __init__(self) -> None:
        self.last_lr = [-1.0]

    def get_last_lr(self):
        return self.last_lr


class NeutrinoTrainingEngine(Module):
    """Training engine backed by the Neutrino GS API.

    Mirrors the interface of :class:`DSSTrainingEngine` so the ArcticTraining
    trainer can use it without branching.

    Parameters
    ----------
    client:
        An already-authenticated :class:`NeutrinoClient` instance.
    sub_job:
        A validated :class:`SubJobConfig` describing the training sub-job.
    """

    def __init__(self, client: NeutrinoClient, sub_job: SubJobConfig) -> None:
        super().__init__()
        self._client = client
        self._input_batch: InputBatch | None = None
        self.global_steps: int = 0

        # Compat shims expected by the trainer
        self.module = None
        self.lr_scheduler = LRScheduler()
        self.optimizer = None
        self.training_dataloader = None

        # Create the job and wait until it's running
        print(json.dumps(sub_job.to_wire(), indent=2, sort_keys=True))
        self.job_id: str = client.create_job(sub_jobs=[sub_job])
        print(f"Created Neutrino job: {self.job_id}")
        client.wait_for_job(self.job_id)
        print(f"Neutrino job {self.job_id} is RUNNING")

    def forward(self, *args, **kwargs):
        """Stash the input batch for the next :meth:`backward` call."""
        assert self._input_batch is None, "Input batch was already set"
        self._input_batch = InputBatch(args, kwargs)
        return self._input_batch

    def backward(self, *args, **kwargs):
        """Forward-backward pass: serialize the stashed batch, send to server, poll for result."""
        assert self._input_batch is not None, "Input batch was not set"
        input_args = {"args": self._input_batch.args, "kwargs": self._input_batch.kwargs}
        buf = io.BytesIO()
        torch.save(input_args, buf)

        request_id = self._client.forward_backward(self.job_id, buf.getvalue())
        result = self._client.poll_request(self.job_id, request_id)
        loss = result["avg_loss"]

        self._input_batch = None
        return torch.tensor(loss)

    def step(self, *args, **kwargs):
        """Optimizer step on the server."""
        request_id = self._client.step(self.job_id)
        result = self._client.poll_request(self.job_id, request_id)
        self.global_steps = result.get("global_steps", self.global_steps + 1)
        last_lr = result.get("last_lr")
        if last_lr is not None:
            # Normalize to list (zone manager may return scalar or list)
            self.lr_scheduler.last_lr = last_lr if isinstance(last_lr, list) else [last_lr]

    def save_checkpoint(
        self,
        *args,
        checkpoint_type: str | None = None,
        **kwargs,
    ):
        """Trigger a server-side checkpoint save.

        Extra arguments remain accepted for training-engine compatibility.
        """
        del args, kwargs
        request_id = self._client.save(
            self.job_id,
            checkpoint_type=checkpoint_type,
        )
        result = self._client.poll_request(self.job_id, request_id)
        checkpoint_id = result.get("checkpoint_id")
        if checkpoint_id is not None:
            print(f"Checkpoint saved: {checkpoint_id}")

    def load_checkpoint(
        self,
        checkpoint_id: str,
        *,
        source_job_id: str | None = None,
    ):
        """Load a server-side checkpoint into this already-created job."""
        request_id = self._client.load(
            self.job_id,
            checkpoint_id=checkpoint_id,
            source_job_id=source_job_id,
        )
        result = self._client.poll_request(self.job_id, request_id)
        loaded = result.get("checkpoint_id", checkpoint_id)
        if loaded is not None:
            print(f"Checkpoint loaded: {loaded}")
        return result

    def destroy(self):
        """Cancel the Neutrino job."""
        try:
            self._client.cancel_job(self.job_id)
        except Exception:
            pass  # Job may already be cancelled/done

    def is_gradient_accumulation_boundary(self):
        return True
