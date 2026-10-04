"""Node-local replica slots: what a semi-p replica's image directory keys on.

Lifted out of the source so this runs without ray or torch installed.
"""
import ast
import os

_POOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                     "arctic_platform", "inference", "server", "replica_pool.py")


def _lift(*names):
    with open(_POOL) as handle:
        tree = ast.parse(handle.read(), filename=_POOL)
    funcs = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {f.name for f in funcs} == set(names), names
    namespace = {"Any": object}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), _POOL, "exec"),
         namespace)
    return [namespace[n] for n in names]


_node_slots, _free_slot = _lift("_node_slots", "_free_slot")


def test_one_node_numbers_like_the_worker_indices():
    assert _node_slots(["a"] * 4) == [(0, 4), (1, 4), (2, 4), (3, 4)]


def test_slots_restart_on_every_node_however_ray_interleaves():
    """Ray places actors out of index order; a global index mod 8 would clash."""
    nodes = ["n0", "n1", "n0", "n0", "n1", "n1", "n0", "n1"]
    assert _node_slots(nodes) == [
        (0, 4), (0, 4), (1, 4), (2, 4), (1, 4), (2, 4), (3, 4), (3, 4)]
    per_node = {}
    for node, (slot, _) in zip(nodes, _node_slots(nodes)):
        per_node.setdefault(node, []).append(slot)
    assert per_node == {"n0": [0, 1, 2, 3], "n1": [0, 1, 2, 3]}


def test_an_unknown_node_counts_as_one_node():
    assert _node_slots([None, None]) == [(0, 2), (1, 2)]


def test_a_replacement_keeps_its_slot_when_it_is_free():
    assert _free_slot({0, 2}, preferred=1) == 1
    assert _free_slot({0, 1, 2}, preferred=1) == 3
    assert _free_slot({1}) == 0
