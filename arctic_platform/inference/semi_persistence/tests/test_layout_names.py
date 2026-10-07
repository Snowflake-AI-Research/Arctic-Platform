"""The four directory names, and the four files that have to agree on them.

``image``, ``compilation``, ``weight`` and ``skeleton`` name the same
directories everywhere -- under ``model_dir`` locally and under the model's
prefix in the bucket -- and the code that composes them cannot share a
constant. ``instance.py`` creates the local tree; ``vllm_child.py`` derives the
compile cache inside it, spawned into a deliberately clean address space;
``server/semip_engine.py`` composes both trees and is what imports ``Instance``;
``scripts/semip_publish.py`` uploads them, copied into a pod on its own and run
with a bare ``python3``. Four files, one convention, no import that could carry
it between them.

So the convention is enforced here, by reading the spellings out of the source
instead of importing any of it -- which is also why this runs on a laptop with
no vLLM, no ray and no torch. It is the same treatment
``sanitize_label_segment`` gets against the DaemonSet's copy in
``test_publish_layout.py``, for the same reason: duplication that cannot be
removed can still be held in place.

**What this file is for.** A published weight manifest was once written with its
paths still scoped to the dump's own directory, so every node asked S3 for
``weight/<wt12>/weights/rank0/shard_0000.bin``, got a 404, and cold-started for
a day while the shards sat correctly beside the manifest. It took a day to find
partly because every one of these names was a bare literal in every one of
these files -- ``image`` appeared fourteen times -- so there was nothing to read
to learn what the layout was, and nowhere a mistake in it would show up.

Note what this does **not** check. Sharing a spelling does not make the local
prefix meaningful in the bucket: the publisher still re-scopes each manifest's
paths against the directory that will hold it, and that property belongs to
``test_publish_layout.py``. Unifying the names removed a misleading trail, not
the failure mode.
"""
from __future__ import annotations

import ast
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../semi_persistence
_INFERENCE = os.path.dirname(_PKG)                 # .../arctic_platform/inference

_ENGINE = os.path.join(_INFERENCE, "server", "semip_engine.py")
_PUBLISH = os.path.join(_PKG, "scripts", "semip_publish.py")
_INSTANCE = os.path.join(_PKG, "instance.py")
_CHILD = os.path.join(_PKG, "vllm_child.py")

# Every spelling that may only ever appear as a constant's definition. The
# retired plural is in here too: a bare "weights" is now always a mistake, and
# the point of listing it is that the mistake fails rather than lingering.
_LAYOUT_NAMES = frozenset({"image", "compilation", "weight", "weights",
                           "skeleton"})


def _parse(path):
    with open(path) as handle:
        return ast.parse(handle.read(), filename=path)


def _module_constants(path):
    """Module-level ``NAME = "str"`` and ``NAME = (NAME, ...)`` assignments.

    Tuples resolve against the constants already seen, so a derived grouping
    (``SKELETON_DIRS``, ``_COPY_SUBDIRS``) comes back as the strings it will be
    joined with rather than as the names it was written with.
    """
    out: dict[str, object] = {}
    for node in _parse(path).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            out[target.id] = value.value
        elif isinstance(value, ast.Tuple):
            items = []
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    items.append(elt.value)
                elif isinstance(elt, ast.Name) and elt.id in out:
                    items.append(out[elt.id])
                else:
                    items = None
                    break
            if items is not None:
                out[target.id] = tuple(items)
    return out


def _function(path, name):
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{os.path.basename(path)} has no {name}()")


def _names_used_in(path, func):
    return {n.id for n in ast.walk(_function(path, func))
            if isinstance(n, ast.Name)}


_ENGINE_CONSTS = _module_constants(_ENGINE)
_PUBLISH_CONSTS = _module_constants(_PUBLISH)
_INSTANCE_CONSTS = _module_constants(_INSTANCE)


# --------------------------------------------------------------------------
# All four files, one spelling each
# --------------------------------------------------------------------------

# (engine, publisher, instance). ``None`` where that file has no opinion:
# ``skeleton`` exists only once published, and the compile cache is the child's.
_NAMES = [
    ("_SKELETON_DIR", "SKELETON_DIR", None),
    ("_WEIGHT_DIR", "WEIGHT_DIR", "_WEIGHT_DIR"),
    ("_IMAGE_DIR", "IMAGE_DIR", "_IMAGE_DIR"),
    ("_COMPILATION_DIR", "COMPILATION_DIR", None),
]


def test_every_file_spells_every_name_the_same_way():
    """A disagreement here is silent, which is why it is worth a test.

    Publish writes where the engine reads and the engine reads what the dump
    wrote. Any mismatch resolves a directory nothing was published to, reports a
    miss, and cold-starts forever with nothing above WARNING.
    """
    for eng, pub, inst in _NAMES:
        assert eng in _ENGINE_CONSTS, f"engine lost {eng}"
        assert pub in _PUBLISH_CONSTS, f"publisher lost {pub}"
        assert _ENGINE_CONSTS[eng] == _PUBLISH_CONSTS[pub], (
            f"{eng}={_ENGINE_CONSTS[eng]!r} but {pub}={_PUBLISH_CONSTS[pub]!r}")
        if inst is not None:
            assert inst in _INSTANCE_CONSTS, f"instance.py lost {inst}"
            assert _ENGINE_CONSTS[eng] == _INSTANCE_CONSTS[inst], (
                f"{eng}={_ENGINE_CONSTS[eng]!r} but "
                f"instance.{inst}={_INSTANCE_CONSTS[inst]!r}")


def test_the_local_and_published_weight_directories_share_one_name():
    """The rename this convention exists to make true.

    ``weight`` is both ``<model_dir>/weight/`` and ``<model>/weight/<wt12>/``.
    It was ``weights`` locally, which cost a re-dump of every model to unify --
    ``weights_hash`` folds this prefix in, so every directory published under
    the plural has a hash no dump produces again. Reverting would cost the same
    again, which is the argument for pinning it.
    """
    assert _INSTANCE_CONSTS["_WEIGHT_DIR"] == "weight"
    assert _PUBLISH_CONSTS["WEIGHT_DIR"] == "weight"


def test_the_skeleton_holds_the_same_two_directories_on_both_sides():
    """Same pair, deliberately different order, so compare as sets.

    The publisher uploads in its own order; the restore copies ``image`` last
    because it holds the hit predicate. Pinning the order would pin a detail
    each side is entitled to choose -- membership is the property.
    """
    assert (set(_PUBLISH_CONSTS["SKELETON_DIRS"])
            == set(_ENGINE_CONSTS["_COPY_SUBDIRS"]))


# --------------------------------------------------------------------------
# The constants are the ones actually used
# --------------------------------------------------------------------------

def test_instance_derives_its_paths_from_its_own_constants():
    """Guards the seam the engine depends on rather than controls.

    ``dump_and_wrap`` passes no ``filename`` or ``weights_dir`` override --
    deliberately, since an override could only disagree with the restore side --
    so what ``Instance`` derives here *is* the local layout. A constant that
    existed but went unused would leave the engine agreeing with a decoration.
    """
    assert "_IMAGE_DIR" in _names_used_in(_INSTANCE, "_resolve_image_dir")
    assert "_WEIGHT_DIR" in _names_used_in(_INSTANCE, "_resolve_weights_dir")


def test_the_engine_names_the_compile_cache_the_child_writes():
    """``compilation`` is the one directory whose absolute paths CRIU baked.

    The child derives it from ``model_dir`` before vLLM is imported, in a
    process that starts with a clean address space, so nothing can be passed in
    and nothing can be imported: the engine can only agree. Disagreeing would
    materialize an image without the compile cache CRIU recorded, which fails a
    restore rather than cold-starting.
    """
    roots = set()
    for node in ast.walk(_parse(_CHILD)):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_compile_root"
                and isinstance(node.value, ast.Call)):
            roots |= {a.value for a in node.value.args
                      if isinstance(a, ast.Constant)
                      and isinstance(a.value, str)}
    assert roots, "vllm_child.py no longer assigns _compile_root from a join"
    assert roots == {_ENGINE_CONSTS["_COMPILATION_DIR"]}, roots


def test_the_engine_names_the_shard_index_the_child_writes():
    """``weights_meta.json`` keeps its plural, and that is not an oversight.

    It is a filename rather than one of the four directory names, and it sits
    inside every weight directory already published -- the restore tests a
    directory's usability by opening it -- so renaming it would make those
    unreadable rather than merely inconsistent.
    """
    for func in ("_semip_save_weights", "_semip_load_weights"):
        strings = {n.value for n in ast.walk(_function(_CHILD, func))
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert _ENGINE_CONSTS["_WEIGHTS_MANIFEST"] in strings, func


# --------------------------------------------------------------------------
# Nothing spells a layout name inline
# --------------------------------------------------------------------------

def _bare_literals(path, defined_names):
    """Layout-name literals that are not part of a constant's definition.

    Walks expressions, so a name inside a comment, a docstring or a longer
    string is not a finding -- only a literal some expression will use as a
    path component.
    """
    tree = _parse(path)
    defining = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in defined_names:
                defining |= {id(n) for n in ast.walk(node.value)}
    return sorted((node.lineno, node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Constant)
                  and isinstance(node.value, str)
                  and node.value in _LAYOUT_NAMES
                  and id(node) not in defining)


def test_no_bare_layout_literals_in_the_engine():
    """What makes the constants load-bearing rather than decorative.

    Without this the next ``os.path.join(model_dir, "image")`` reintroduces the
    duplication silently, and the constants become documentation of a convention
    the code has stopped following.
    """
    found = _bare_literals(_ENGINE, {e for e, _, _ in _NAMES})
    assert not found, f"semip_engine.py spells layout names inline: {found}"


def test_no_bare_layout_literals_in_the_publisher():
    found = _bare_literals(_PUBLISH, {p for _, p, _ in _NAMES})
    assert not found, f"semip_publish.py spells layout names inline: {found}"


def test_no_bare_layout_literals_in_instance():
    found = _bare_literals(_INSTANCE, {"_IMAGE_DIR", "_WEIGHT_DIR"})
    assert not found, f"instance.py spells layout names inline: {found}"


def test_engine_and_publisher_spell_the_replica_level_alike():
    """The engine resolves replica<K>/ inside a skeleton the publisher wrote.

    A disagreement is a permanent miss for every multi-replica pod, and a silent
    one: ``_replica_source`` reads a skeleton with no replica<K>/ as a cold start.
    """
    assert (_ENGINE_CONSTS["_REPLICA_DIR_PREFIX"]
            == _PUBLISH_CONSTS["REPLICA_DIR_PREFIX"] == "replica")


def test_engine_and_publisher_spell_the_node_level_alike():
    """Each pod of a node-spanning engine copies ``node<k>/`` out of the
    skeleton the publisher wrote; ``_node_source`` names it from this prefix."""
    assert (_ENGINE_CONSTS["_NODE_DIR_PREFIX"]
            == _PUBLISH_CONSTS["NODE_DIR_PREFIX"] == "node")


if __name__ == "__main__":
    fails = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  [ok  ] {name}")
        except AssertionError as exc:
            fails.append(f"{name}: {exc}")
            print(f"  [FAIL] {name}: {exc}")
    print()
    print("FAILURES:", fails if fails else "none")
    raise SystemExit(1 if fails else 0)
