"""``EngineStartupError`` keeps its fields across pickling and Ray actor boundaries.

dss-platform reads the fields duck-typed, by the names in
``engine_startup_error.FIELDS``, from the exception ``ray.get`` raises and
from the ``.cause`` chain under it, so both have to carry them.
"""

import pickle

import pytest

from arctic_platform.inference import engine_startup_error
from arctic_platform.inference.engine_startup_error import EngineStartupError

_MESSAGE = (
    "Engine core initialization failed. See root cause above. Failed core proc(s): {}\n"
    "Root cause in Worker_TP0: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB."
)
_FIELDS = {
    "root_type": "torch.OutOfMemoryError",
    "root_message": "CUDA out of memory. Tried to allocate 2.00 GiB.",
    "process": "Worker_TP0",
    "origin": "torch.nn.modules.linear.Linear.forward",
}


def _error():
    exc = EngineStartupError(_MESSAGE, **_FIELDS)
    exc.add_note("Engine console tail (stdout, stderr):\nNCCL INFO ... nvmlDev 3")
    return exc


def _fields(exc):
    return {name: getattr(exc, name) for name in engine_startup_error.FIELDS}


def test_contract_names_the_fields():
    assert engine_startup_error.FIELDS == ("root_type", "root_message", "process", "origin")
    assert _fields(_error()) == _FIELDS


def test_message_and_type_are_what_older_classifiers_match():
    exc = _error()
    assert isinstance(exc, RuntimeError)
    assert str(exc) == _MESSAGE


@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
def test_survives_pickling(protocol):
    restored = pickle.loads(pickle.dumps(_error(), protocol=protocol))

    assert type(restored) is EngineStartupError
    assert str(restored) == _MESSAGE
    assert _fields(restored) == _FIELDS
    assert restored.__notes__ == _error().__notes__


def test_origin_is_optional():
    restored = pickle.loads(pickle.dumps(EngineStartupError("m", "ValueError", "x", "EngineCore")))
    assert restored.origin is None


@pytest.fixture(scope="module")
def ray_cluster():
    ray = pytest.importorskip("ray")
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    yield ray
    ray.shutdown()


def _remote_classes(ray):
    # Workers cannot import this test module, so nothing here may refer to its globals.
    @ray.remote(num_cpus=0)
    class FailingWorker:
        def initialize(self, message, fields, note):
            from arctic_platform.inference.engine_startup_error import EngineStartupError

            exc = EngineStartupError(message, **fields)
            exc.add_note(note)
            raise exc

    @ray.remote(num_cpus=0)
    class Driver:
        """Calls the worker the way the dss sampling job calls InferenceWorker."""

        def __init__(self):
            self.worker = FailingWorker.remote()

        def initialize(self, *args):
            ray.get(self.worker.initialize.remote(*args))

    return FailingWorker, Driver


_ARGS = (_MESSAGE, _FIELDS, _error().__notes__[0])


def _cause_chain(exc):
    seen = []
    while exc is not None and all(exc is not other for other in seen):
        seen.append(exc)
        exc = getattr(exc, "cause", None) or exc.__cause__
    return seen


def test_survives_a_ray_actor_round_trip(ray_cluster):
    ray = ray_cluster
    worker, _ = _remote_classes(ray)

    with pytest.raises(ray.exceptions.RayTaskError) as info:
        ray.get(worker.remote().initialize.remote(*_ARGS), timeout=120)

    # ``ray.get`` raises the as_instanceof_cause subclass; it reads the fields from the cause.
    assert isinstance(info.value, EngineStartupError)
    assert _fields(info.value) == _FIELDS
    cause = info.value.cause
    assert type(cause) is EngineStartupError
    assert str(cause) == _MESSAGE
    assert _fields(cause) == _FIELDS
    assert any("nvmlDev" in note for note in cause.__notes__)


def test_survives_nested_ray_actor_calls(ray_cluster):
    ray = ray_cluster
    _, driver = _remote_classes(ray)

    with pytest.raises(ray.exceptions.RayTaskError) as info:
        ray.get(driver.remote().initialize.remote(*_ARGS), timeout=120)

    typed = [exc for exc in _cause_chain(info.value) if type(exc) is EngineStartupError]
    assert len(typed) == 1
    assert _fields(typed[0]) == _FIELDS
    assert str(typed[0]) == _MESSAGE
