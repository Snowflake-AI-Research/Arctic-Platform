import json
import torch
import requests
from torch.nn.modules import Module

from . import wire
from .common import JobType


class DSSTrainingEngine(Module):
    job_type = JobType.TRAINING

    def __init__(self, model, ds_config, training_config, dss_server_url, lr_scheduler=None):
        super(DSSTrainingEngine, self).__init__()
        self.ds_config = ds_config
        self.training_config = training_config
        self.server_url = dss_server_url
        self.global_steps = 0

        self.model_name = model.config.name_or_path

        # Backward compatibility with DeepSpeedEngine, this way DSSTrainingEngine
        # acts the same as a DeepSpeedEngine and is a drop-in replacement client side.
        self.module = model
        self.optimizer = None
        self.lr_scheduler = lr_scheduler
        self.training_dataloader = None

        json_payload = {
            "model_name": self.model_name,
            "ds_config": self.ds_config,
            "training_config": self.training_config,
            "job_type": self.job_type
        }
        print(f"DSS job initialization payload for {self.model_name} at {dss_server_url}:")
        print(json.dumps(json_payload, indent=2, sort_keys=True))

        r = requests.post(
            f"{self.server_url}/initialize",
            json=json_payload,
        )
        response = r.json()
        print(json.dumps(response, indent=2, sort_keys=True))
        self.job_id = response["job_id"]
        print(f"Initialized job with job_id: {self.job_id}")

    def forward_backward(self, batch: dict):
        """ forward and backward pass the input batch together """
        payload = wire.dumps({"args": (), "kwargs": batch})
        print(f"fwd-bwd payload total bytes: {len(payload)}")

        r = requests.post(
            f"{self.server_url}/fwd-bwd",
            data=payload,
            headers={"Content-Type": "application/octet-stream"},
            params={"job_id": self.job_id},
        )
        try:
            response = r.json()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Response: {r.text}")
            raise e
        loss = response["avg_loss"]
        return torch.tensor(loss)

    def step(self, learning_rate: float):
        if isinstance(learning_rate, list):
            raise ValueError("learning_rate must be a float, multiple param group learning rates are not yet supported")
        assert isinstance(learning_rate, float), "learning_rate must be a float"
        r = requests.post(
            f"{self.server_url}/step",
            params={"job_id": self.job_id, "learning_rate": learning_rate},
        )
        response = r.json()
        self.global_steps = response["global_steps"]

    def destroy(self):
        r = requests.post(f"{self.server_url}/destroy", params={"job_id": self.job_id}, json={"job_type": "training"})
        response = r.json()
        return response["job_id"]

    def is_gradient_accumulation_boundary(self):
        # Dummy function, remove when we rebase on latest AT. See: https://github.com/snowflakedb/ArcticTraining/pull/351
        return True

    def save_checkpoint(self, *args, **kwargs):
        r = requests.post(f"{self.server_url}/save-checkpoint", params={"job_id": self.job_id})
        response = r.json()
        checkpoint_path = response.get("checkpoint_path")
        if checkpoint_path is not None:
            print(f"Checkpoint saved to {checkpoint_path}")


class ArcticInferenceEngine:
    """Client-side handle for a sampling job using dss-platform's inference API."""

    def __init__(
        self,
        server_url: str,
        model_name: str,
        vllm_config: dict | None = None,
    ):
        self.server_url = server_url
        payload = {"model_name": model_name, "job_type": JobType.SAMPLING.value}
        if vllm_config is not None:
            payload["vllm_config"] = vllm_config
        r = requests.post(f"{self.server_url}/initialize", json=payload)
        r.raise_for_status()
        self.job_id = r.json()["job_id"]

    def generate(
        self,
        prompts: list,
        sampling_params: dict | list[dict | None] | None = None,
    ) -> list:
        """Generate completions for the given prompts."""
        r = requests.post(
            f"{self.server_url}/generate",
            params={"job_id": self.job_id},
            json={
                "prompts": prompts,
                "sampling_params": sampling_params if sampling_params is not None else {},
            },
        )
        r.raise_for_status()
        return r.json()["results"]

    def destroy(self) -> int:
        """Destroy the sampling job and return the job_id."""
        r = requests.post(f"{self.server_url}/destroy", params={"job_id": self.job_id}, json={"job_type": "sampling"})
        r.raise_for_status()
        return r.json()["job_id"]
