---
name: good-code
description: >-
  Write and review code without defensive noise: no redundant library
  re-validation, no type-coercion wrappers, no getattr defaults for
  attributes set in init, no values computed then thrown away on the
  default path, one default per variable (resolve once, do not re-default
  the same key later). Use when writing, editing, or reviewing Python
  (especially training.py / DeepSpeed / Ray DSS), when adding helpers or
  guards, when the user asks if a check is necessary, or when cleaning a
  PR diff.
---

# Good code

Trust the invariants that already exist. Do not add a second copy of a check, a cast, or a default "just in case." A given variable gets its default in **one** place; later reads do not invent another.

This is not the refactoring skill (behavior-preserving structural pass). This skill is about what not to write in the first place, and what to delete when a review calls it noise.

## Before adding a guard

Ask, in order:

1. Does a library we already call assert this? (DeepSpeed `_batch_assertion` already requires `micro_batch > 0`.) Then do not wrap it again.
2. Is this attribute assigned in `__init__` / `create` / engine setup? Then read `self.x`, not `getattr(self, "x", default)`.
3. Is the value only used in some branches? Compute it inside those branches. Do not hoist it onto the default path and throw it away.
4. Is the source already the right type? Do not `int(ds_config["train_micro_batch_size_per_gpu"])` when the default is `1` and DeepSpeed does not require an int.
5. Are you only adding this so `object.__new__` test stubs stay quiet? Set the attribute on the stub. Do not make production code defensive for skipped constructors.
6. Is this a default for a name that is already defaulted at the config / ctor / call boundary? Pass the resolved value through. Do not `.get(..., other)` again.

If the answer to any of those is "the extra code is only for a case that cannot happen on this path," delete it.

## Concrete don'ts (from the mbs / pack-meta pass)

```python
# BAD — DeepSpeed already asserts micro_batch > 0
def _train_micro_batch_size(ds_config):
    value = ds_config.get("train_micro_batch_size_per_gpu")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(...)
    return value

# GOOD
micro_batch_size = ds_config["train_micro_batch_size_per_gpu"]
```

```python
# BAD — init always sets this; getattr hides missing test stubs
steps = build(..., getattr(self, "_micro_batch_size", 1))

# GOOD
steps = build(..., self._micro_batch_size)
# tests that use object.__new__ must set worker._micro_batch_size
```

```python
# BAD — default token-mean path never uses this
valid_sequences = _rl_valid_sequence_mask(microbatch)
if loss_agg_mode in SEQUENCE_MODES:
    ...
if loss_agg_mode == "prompt-mean":
    ...
return int(loss_mask.sum())

# GOOD — compute only in the two branches that read it
if loss_agg_mode in SEQUENCE_MODES:
    valid_sequences = _rl_valid_sequence_mask(microbatch)
    ...
```

```python
# BAD — same condition already decided the branch
if not sp_enabled:
    mb_sync_group = self.sp_group if self.sp_size > 1 else None

# GOOD
if not sp_enabled:
    mb_list = split(..., group=None)
```

## What to leave alone

- `getattr(self, "_engine", None)` in `destroy` / teardown: init can fail before the object exists.
- `getattr(module, "optional_hook", None)` on **other** objects (HF config, DeepSpeed engine APIs, mixins that run before those fields are assigned).
- Checks that enforce **this service's** schedule contract (`model_call_count`, DSS `micro_batch_size` for row grouping). Those are not DeepSpeed's `train_micro_batch_size_per_gpu`.
- A real early-return that applies to **every** branch (e.g. `real_batch_size == 0` before loss-mode dispatch). That is not "thrown away on the default path."

## Upstream already set it

Before writing `getattr(self, "foo", 4)` (or any other fallback), check whether the **upstream owner already always assigns** that attribute. If `__init__` / `create` / a parent setup / a required config fill always sets `foo`, later code must use `self.foo`. A second default is a lie: it hides a broken stub and can disagree with the real init value.

```python
class Worker:
    def __init__(self, foo=4):
        self.foo = foo

    def step(self):
        # BAD — init always set foo; this default is a second, drifting one
        foo = getattr(self, "foo", 4)
        # GOOD
        foo = self.foo
```

Same bug, other spellings:

- `hasattr(self, "foo")` then a default branch
- `self.__dict__.get("foo", 4)` / `vars(self).get("foo", 4)`
- `try: self.foo` / `except AttributeError: foo = 4` on a path that ran init
- `foo = self.foo if self.foo is not None else 4` when init never leaves it `None`

`object.__new__` tests that skip init must **set the attribute on the stub**. Do not weaken production reads for them.

The exception is teardown / failed-init (`destroy` reading `_engine`) and attributes on **other** objects you do not construct — see What to leave alone.

## One default per variable

The same conceptual value must not pick up a default in two places. If it does, those literals drift (`4` vs `8`) and later code cannot tell which one is in force.

Resolve **once**, at the boundary that owns the setting (config parse, `__init__`, request model, CLI). After that, pass the value. Downstream code reads it; it does not default it again.

```python
# BAD — same key, two defaults; they will diverge
foo = cfg.get("foo", 4)
...
foo = batch.get("foo", 8)

# BAD — signature default plus a second default in the body
def train(foo=4):
    foo = cfg.get("foo", 16)

# BAD — "missing" spelled three ways, three literals
foo = cfg.get("foo") or 4
foo = foo if foo is not None else 4
foo = getattr(self, "foo", 4)

# GOOD — one named default, one resolve, then use
DEFAULT_FOO = 4
foo = cfg.get("foo", DEFAULT_FOO)
# later: use foo, or cfg["foo"] after a required fill, not get-with-default again
```

Other shapes of the same bug:

- `dict.get` / `os.environ.get` / `os.getenv` / `setdefault` / `getattr(..., default)` for one key in more than one module
- `x or default` (also treats `0` / `""` / `[]` as missing — usually wrong for counts and paths)
- `x if x is not None else default` next to a `.get` of the same name
- kwarg default on `def f(foo=4)` **and** `cfg.get("foo", 4)` or `Field(default=4)` for the same field
- dataclass / Pydantic / YAML schema default plus a second fallback in the caller or the worker
- `kwargs.pop("foo", 4)` in two helpers that both run on the same request
- copying the literal `4` into every call site instead of one constant or one resolved local

Fix by centralizing, not by making every site use the same magic number. If the key is required after init, use `cfg["foo"]` / `self.foo` and let a missing key fail.

`getattr(self, "x", default)` for an attribute assigned in `__init__` is this bug **and** the getattr rule above.

## Comments, docstrings, markdown

Docstrings and comments are **terse**. Explain only what names do not. Do not narrate the next line.

**Write for a reader who was not in the chat** that produced the change. Product code, tests, and shipped docs must stand alone. Do not leak process labels (`#N`, `k` / `kA`, `G01`, attack ids), nicknames, clone folder names, “as discussed”, “the operator said”, or other conversation leftovers. If a fact is worth keeping, state the **contract or reason** in full — `save_total_limit` prunes the job dir, not its parent — not `#3: …`. Process notes stay in chat and process files (`PROBE_*.md`), not in what ships.

**Code comments** only when a special nuance needs saying. Otherwise the variable / function / method name is the explanation. Delete comments that restate `reshape(-1, V)` or `label_shape = labels.shape`.

**Wrap at 119** for `#` comments, `"""` / `'''` docstrings, and `Field(description=...)`. Count the full line (indent + `# ` / quotes + text). Fill the line; do not wrap early at 80/100. Break at a word boundary on or before column 119.

**Markdown** (`.md`, `.mdc`, `PR_DESCRIPTION.md`): do not hard-wrap paragraphs or list items. One paragraph = one line. One list item = one line. Keep newlines for fences, tables, headings, and YAML frontmatter.

## When reviewing or porting

- If a requested move lands on a checkout that does not have the crash path, say so and stop. Do not invent a no-op shim so the patch "applies."
- When you delete a redundant guard, delete the tests that existed only to pin that guard. Keep tests that pin remaining real behavior.
- When two sites default the same name, do not "harmonize" by pasting one literal into both. Collapse to one resolve; grep the key (`get("foo"`, `foo=`, `Field(.*foo`) and delete the extras.
