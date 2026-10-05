import asyncio
from types import SimpleNamespace

import pytest

from arctic_platform.inference.server import multi_model as multi_model_mod
from arctic_platform.inference.server.multi_model import Driver


class _FakePool:
    def __init__(self, worker_cls=None) -> None:
        self.uses_placement_group = False
        self.world_size = 1
        self.num_replicas = 0
        self.initialize_calls = []
        self.scale_calls = []

    async def initialize(self, config, num_replicas=None, placement_group=None):
        self.uses_placement_group = placement_group is not None
        self.world_size = config.tensor_parallel_size * getattr(
            config, "pipeline_parallel_size", 1
        )
        self.num_replicas = 1 if placement_group is not None else num_replicas
        self.initialize_calls.append((config, num_replicas, placement_group))
        return self.num_replicas

    async def scale_up(self, target):
        self.scale_calls.append(target)


def _config(tp=2, pp=2):
    return SimpleNamespace(
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
    )


def test_driver_bypasses_even_share_for_placement_group(monkeypatch):
    monkeypatch.setattr(multi_model_mod, "ensure_ray", lambda: 16)
    monkeypatch.setattr(multi_model_mod, "ReplicaPool", _FakePool)
    driver = Driver()
    driver._compute_even_share = lambda *args, **kwargs: pytest.fail(
        "placement-group initialization must bypass even-share planning"
    )
    pg = object()

    replicas = asyncio.run(
        driver.initialize(_config(), model_id="pg-model", placement_group=pg)
    )

    assert replicas == 1
    pool = driver._pools["pg-model"]
    assert pool.initialize_calls[0][1:] == (1, pg)


def test_driver_rejects_mixed_placement_and_ordinary_pools(monkeypatch):
    monkeypatch.setattr(multi_model_mod, "ensure_ray", lambda: 16)
    monkeypatch.setattr(multi_model_mod, "ReplicaPool", _FakePool)
    driver = Driver()

    asyncio.run(
        driver.initialize(
            _config(),
            model_id="pg-model",
            placement_group=object(),
        )
    )

    with pytest.raises(RuntimeError, match="ordinary pool"):
        asyncio.run(driver.initialize(_config(), model_id="ordinary-model"))


def test_driver_rejects_multiple_placement_group_pools(monkeypatch):
    monkeypatch.setattr(multi_model_mod, "ensure_ray", lambda: 16)
    monkeypatch.setattr(multi_model_mod, "ReplicaPool", _FakePool)
    driver = Driver()

    asyncio.run(
        driver.initialize(
            _config(),
            model_id="first",
            placement_group=object(),
        )
    )

    with pytest.raises(RuntimeError, match="exclusive use"):
        asyncio.run(
            driver.initialize(
                _config(),
                model_id="second",
                placement_group=object(),
            )
        )


def test_driver_rejects_multiple_replicas_for_placement_group(monkeypatch):
    monkeypatch.setattr(multi_model_mod, "ensure_ray", lambda: 16)
    driver = Driver()

    with pytest.raises(ValueError, match="num_replicas=1"):
        asyncio.run(
            driver.initialize(
                _config(),
                model_id="pg-model",
                num_replicas=2,
                placement_group=object(),
            )
        )
