"""What a restored single-node rank takes from the restoring job at reinit_nccl.

``vllm_child`` imports torch and CUDA at module scope, so the two helpers are
lifted out of the source, as ``test_double_buffer`` does, and run against a
fake ``nccl_bootstrap`` and a routing table written to a temporary file.
"""
from __future__ import annotations

import ast
import os
import sys
import types

import pytest

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NAMES = ("_default_route_ifname", "_single_node_job_nccl")

_ROUTES = (
    "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\n"
    "eth0\t0010040A\t00000000\t0001\t0\t0\t0\t00F0FFFF\n"
    "eth0\t00000000\t0110040A\t0003\t0\t0\t0\t00000000\n"
)


@pytest.fixture
def child(monkeypatch, tmp_path):
    with open(os.path.join(_PKG, "vllm_child.py")) as handle:
        tree = ast.parse(handle.read())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _NAMES]
    assert {f.name for f in funcs} == set(_NAMES)
    ns = {"os": os}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "vllm_child.py", "exec"), ns)
    resets = []

    class _Bootstrap:
        fail = None

        def reset(self):
            if self.fail is not None:
                raise self.fail
            resets.append(os.environ["NCCL_SOCKET_IFNAME"])
            return {"if": os.environ["NCCL_SOCKET_IFNAME"]}

    bootstrap = _Bootstrap()
    monkeypatch.setitem(sys.modules, "nccl_bootstrap", bootstrap)
    routes = tmp_path / "route"
    routes.write_text(_ROUTES)
    default = ns["_default_route_ifname"]
    ns["_default_route_ifname"] = lambda: default(str(routes))
    # setenv first, so monkeypatch puts back whatever the helper sets.
    monkeypatch.setenv("OFI_NCCL_FORCE_NUM_RAILS", "")
    monkeypatch.delenv("OFI_NCCL_FORCE_NUM_RAILS")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "lo")
    return types.SimpleNamespace(bootstrap=bootstrap, resets=resets, routes=routes,
                                 default=default, **ns)


def _worker():
    return types.SimpleNamespace(rank=0)


def test_the_default_route_names_the_interface(child, tmp_path):
    assert child.default(str(child.routes)) == "eth0"
    lonely = tmp_path / "lonely"
    lonely.write_text(_ROUTES.splitlines(keepends=True)[0])
    assert child.default(str(lonely)) is None
    assert child.default(str(tmp_path / "missing")) is None


def test_a_rank_bootstraps_on_the_default_route_with_the_jobs_settings(child):
    report = child._single_node_job_nccl(_worker(), {"OFI_NCCL_FORCE_NUM_RAILS": "2"})
    assert report == {"if": "eth0"} and child.resets == ["eth0"]
    assert os.environ["NCCL_SOCKET_IFNAME"] == "eth0"
    assert os.environ["OFI_NCCL_FORCE_NUM_RAILS"] == "2"


def test_the_jobs_own_interface_wins(child):
    child._single_node_job_nccl(_worker(), {"NCCL_SOCKET_IFNAME": "^lo,docker"})
    assert child.resets == ["^lo,docker"]


def test_a_failed_reset_leaves_the_rank_on_loopback(child, capsys):
    child.bootstrap.fail = RuntimeError("no libnccl is mapped into this process")
    assert child._single_node_job_nccl(_worker(), None) is None
    assert os.environ["NCCL_SOCKET_IFNAME"] == "lo"
    assert "stays on loopback" in capsys.readouterr().out


def test_no_default_route_leaves_the_rank_on_loopback(child, capsys):
    child.routes.write_text(_ROUTES.splitlines(keepends=True)[0])
    assert child._single_node_job_nccl(_worker(), {}) is None
    assert os.environ["NCCL_SOCKET_IFNAME"] == "lo" and child.resets == []
    assert "no default route" in capsys.readouterr().out
