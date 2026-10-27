# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Actual backward participation in the automatically owned correctness gateway."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
from copy import deepcopy
from pathlib import Path

CAPTURE_ENV = "ARCTIC_CORRECTNESS_PARTICIPATION_CAPTURE"
GATEWAY_MODULE = "arctic_platform.debug.correctness_participation"


def maybe_spawn_participation_coordinator(manager_module, actor_key, original_actor, spawn, training, info):
    if os.environ.get(CAPTURE_ENV) != "1":
        return spawn(info)
    selected = participation_coordinator_class(training)
    manager_module._ACTOR_CLS[actor_key] = selected
    try:
        return spawn(info)
    finally:
        manager_module._ACTOR_CLS[actor_key] = original_actor


def participation_coordinator_class(training):
    import ray

    from arctic_platform.correctness.harness.native_lora import _source
    from arctic_platform.correctness.harness.native_lora import _spawn_worker_binding

    sources = {"training": _source(training._DeepSpeedWorker.__init__)}
    worker_base = training._DeepSpeedWorker
    coordinator_base = training.GPUTrainingJob.__ray_metadata__.modified_class

    class ParticipationWorker(worker_base):
        def fwd_bwd(self, batch):
            if not hasattr(self, "_correctness_participation"):
                self._correctness_participation = WorkerParticipation(self)
            try:
                return super().fwd_bwd(batch)
            except BaseException:
                self._correctness_participation.close()
                raise

        def step(self, learning_rate=None):
            if not hasattr(self, "_correctness_participation"):
                raise ValueError("participation capture requires backward before step")
            try:
                result = super().step(learning_rate)
                self._correctness_participation_receipt = self._correctness_participation.finish()
                return result
            finally:
                self._correctness_participation.close()

        def correctness_participation_receipt(self):
            return deepcopy(self._correctness_participation_receipt)

        def destroy(self, remove_checkpoint_scratch=False):
            capture = getattr(self, "_correctness_participation", None)
            if capture is not None:
                capture.close()
            return super().destroy(remove_checkpoint_scratch=remove_checkpoint_scratch)

    class ParticipationCoordinator(coordinator_base):
        async def _spawn_workers(self, cfg, ranks, master_addr, master_port, source_stage_info):
            from dss.ray_dss.jobs.gpu import training as native_training

            native_spawn = super()._spawn_workers
            namespace, original_worker = _spawn_worker_binding(native_spawn, native_training, sources)
            namespace["_DeepSpeedWorker"] = ParticipationWorker
            try:
                await native_spawn(cfg, ranks, master_addr, master_port, source_stage_info)
                if namespace.get("_DeepSpeedWorker") is not ParticipationWorker:
                    raise ValueError("participation worker binding changed during creation")
            finally:
                namespace["_DeepSpeedWorker"] = original_worker

        async def step(self, learning_rate=None):
            response = await super().step(learning_rate)
            if response.get("global_steps") != 1:
                raise ValueError("participation capture requires exactly the first optimizer step")
            receipts = await asyncio.wait_for(
                asyncio.gather(*(worker.correctness_participation_receipt.remote() for worker in self._workers)),
                timeout=60,
            )
            response["participation"] = merge_worker_receipts(
                receipts, job_id=self._info.job_id, world_size=len(self._workers)
            )
            return response

    return ray.remote(num_cpus=1, num_gpus=0)(ParticipationCoordinator)


class WorkerParticipation:
    def __init__(self, worker):
        from arctic_platform.correctness.harness.participation import ParticipationCapture

        self.worker = worker
        self.engine = worker._engine
        if self.engine.global_steps != 0:
            raise ValueError("participation must be armed before the first backward")
        model = self.engine.module
        decoder_getter = getattr(model, "get_decoder", None)
        head_getter = getattr(model, "get_output_embeddings", None)
        decoder = decoder_getter() if callable(decoder_getter) else None
        if decoder is None and hasattr(model, "model"):
            from arctic_platform.model.implementations.moe.vlm import get_language_model

            decoder = get_language_model(model)
        head = head_getter() if callable(head_getter) else getattr(model, "lm_head", None)
        sources = {Path(__file__), Path(inspect.getsourcefile(ParticipationCapture))}
        self.sources = {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources}
        self.capture = ParticipationCapture(model, decoder=decoder, head=head)
        try:
            self.before = self.evidence()
        except BaseException:
            self.close()
            raise

    def evidence(self):
        from arctic_platform.correctness.harness.participation import weight_identity

        for filename, digest in self.sources.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise ValueError("participation capture source drift")
        return {
            "participation": self.capture.result(),
            "candidate_weights": {
                name: weight_identity(self.capture.parameters[name].detach()) for name in self.capture.candidates
            },
            "participation_sources": deepcopy(self.sources),
        }

    def finish(self):
        if self.worker._engine is not self.engine or self.engine.global_steps != 1:
            raise ValueError("participation engine changed or did not complete one step")
        return {
            "rank": int(self.worker.rank),
            "job_id": self.worker.job_id,
            "before": deepcopy(self.before),
            "after": self.evidence(),
        }

    def close(self):
        self.capture.close()


def merge_worker_receipts(receipts, *, job_id, world_size):
    from arctic_platform.correctness.harness.participation import merge_rank_participation

    if (
        type(world_size) is not int
        or world_size < 1
        or not isinstance(job_id, str)
        or not job_id
        or not isinstance(receipts, list)
        or len(receipts) != world_size
    ):
        raise ValueError("participation requires every actual training worker")
    for rank, receipt in enumerate(receipts):
        if (
            not isinstance(receipt, dict)
            or type(receipt.get("rank")) is not int
            or receipt["rank"] != rank
            or receipt.get("job_id") != job_id
        ):
            raise ValueError("participation worker rank or training child differs")
        before, after = receipt.get("before"), receipt.get("after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError("participation requires before/after worker evidence")
        sources = before.get("participation_sources")
        if not isinstance(sources, dict) or not sources or sources != after.get("participation_sources"):
            raise ValueError("participation source identity differs before and after")
        for filename, digest in sources.items():
            if (
                not isinstance(filename, str)
                or not Path(filename).is_absolute()
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("participation source identity is invalid")
        for evidence in (before, after):
            inventory = evidence.get("participation")
            if (
                not isinstance(inventory, dict)
                or not isinstance(inventory.get("parameters"), dict)
                or not inventory["parameters"]
                or any(
                    not isinstance(entry, dict) or not isinstance(entry.get("shape"), list)
                    for entry in inventory["parameters"].values()
                )
            ):
                raise ValueError("participation worker inventory is incomplete")
    shapes = {name: entry["shape"] for name, entry in receipts[0]["before"]["participation"]["parameters"].items()}
    participation, _ = merge_rank_participation(
        [receipt["before"] for receipt in receipts],
        [receipt["after"] for receipt in receipts],
        shapes,
        {name: [0] for name in shapes},
    )
    return {**participation, "training_child_id": job_id, "rank_receipts": deepcopy(receipts)}


def main():
    from dss.ray_dss import gateway_server

    from arctic_platform.correctness.harness.native_lora import gateway_scope

    os.environ[CAPTURE_ENV] = "1"
    with gateway_scope(gateway_server):
        gateway_server.main()


if __name__ == "__main__":
    main()
