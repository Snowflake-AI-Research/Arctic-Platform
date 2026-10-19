#!/usr/bin/env python3
"""Check a CRIU image's recorded task ids against the ones live on this node.

    python3 scripts/pidcheck.py /data-fast/image-cache/qwen_27b/image
    python3 scripts/pidcheck.py <image> --burn        # make room, then restore
    python3 scripts/pidcheck.py --burn-to 200000      # dump side, before init

Use this before a restore in reduced-capability mode (``SEMIP_UNPRIVILEGED=1``),
which has no private PID namespace: CRIU recreates every task at its recorded
id with ``clone3(set_tid)``, so every one of them must be free on the host.

A PID is only the id of a thread group's leader, and leader ids and thread ids
are allocated from a single per-namespace counter.  So a TP2 image needs its
three leaders *and* the ~900 thread ids they recorded, and an unrelated
200-thread process sitting anywhere in that range blocks the restore just as a
same-PID process would.  ``ps`` and a plain ``/proc`` scan both hide this,
because threads appear only under ``/proc/<pid>/task``.

Worse, an id with **no task at all** can be unallocatable.  A pid is a
refcounted ``struct pid``, and every process using it as a process-group or
session id holds a reference on it -- a zombie included, since a corpse keeps
those links until it is reaped.  So this script scans ``pgrp`` and ``session``
(fields 5 and 6 of ``/proc/*/stat``) as well as ``/proc/<pid>/task``.  Without
that it reported "No collisions" through the dev-cluster restore failure, which was a
dumped leader's id pinned by two unreaped members of its own session.

CRIU reports the cases through different code paths and names neither the id
nor the occupant:

    Can't fork for 47619: File exists                 # a leader, by either route
    pie: 2103: Unable to create a thread: -17         # any other thread

``--burn`` advances the namespace's PID counter past the image's highest
recorded id by forking throwaway children -- it consumes fresh numbers, it does
not free occupied ones, and it cannot move a process that is already running.
So it fixes "my launcher landed in the image's range" and nothing else; ids
held by a live process have to be freed by that process exiting.  The counter
is namespace-global, so burning in this script carries over to the restore
launched after it.  ``/proc/sys/kernel/ns_last_pid`` would do the same in one
write, but it is read-only on the nodes this mode exists for.

The durable fix is dump-side: ``--burn-to N`` before the cold start, so the
image records ids far above anything a container hands out in normal operation.
That form needs no image and no root -- it only advances the counter.

The serving adapter now does this on its own: ``_raise_pid_floor`` in
``server/semip_engine.py`` runs before the ``Instance`` that spawns the tree,
so images it dumps record ids above ``SEMIP_PID_FLOOR`` (100000 by default) and
note the floor reached in ``meta.json``.  It burns in parallel, since the
counter is namespace-global and a single fork costs ~540us on these nodes.
This script stays the manual form, for an image dumped elsewhere or a node
being investigated by hand.

Like ``imgdiff.py`` this imports nothing from the package and needs no GPU --
just ``crit`` (from the CRIU install) and read access to the image, which
means running as the uid that dumped it.
"""

import json
import os
import subprocess
import sys

BURN_MARGIN = 2000  # headroom for the restore driver and its threads


def decode_pstree(path):
    res = subprocess.run(["crit", "decode", "-i", path], capture_output=True)
    if res.returncode != 0:
        sys.exit("crit decode failed (rc=%d); the image must be readable by "
                 "this user (uid %d):\n%s"
                 % (res.returncode, os.getuid(),
                    res.stderr.decode("utf-8", "replace")))
    return json.loads(res.stdout.decode("utf-8", "replace"))


def recorded_tasks(doc):
    """(leaders, tids) recorded in pstree.img."""
    leaders, tids = [], set()
    for ent in doc.get("entries", []):
        if ent.get("pid") is None:
            continue
        leaders.append(int(ent["pid"]))
        tids.add(int(ent["pid"]))
        tids.update(int(t) for t in (ent.get("threads") or []))
    return leaders, tids


def live_task_ids():
    """{task id: leader pid} for every task live in this PID namespace."""
    live = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            for tid in os.listdir("/proc/%s/task" % entry):
                live[int(tid)] = int(entry)
        except OSError:
            continue                     # exited while we walked it
    return live


def group_references(wanted):
    """{id: [(pid, comm, state, kind), ...]} for ids held with no task.

    A task scan is not the whole picture, and the gap is what made this script
    report "No collisions" through a restore that then failed EEXIST on the
    first id it tried.  A pid is a refcounted ``struct pid``, and references
    come from the task itself *and* from every process using it as a process
    group or session id -- a group is named by its leader's pid, and an
    unreaped zombie keeps those links.  So an id with no task can still be
    unallocatable, and ``clone3(set_tid)`` fails on it while ``/proc`` shows
    nothing there, because ``/proc`` lists tasks.

    Self-references are skipped: an id held by its own live leader is a task
    collision, already reported with its occupant named.
    """
    refs = {}
    if not wanted:
        return refs
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        # comm is parenthesised and may contain parens and spaces, so split
        # after the last ')': index 0 is then field 3 (state), which puts
        # pgrp and session -- fields 5 and 6 -- at 2 and 3.
        try:
            with open("/proc/%s/stat" % entry) as fh:
                head, _, tail = fh.read().rpartition(")")
            fields = tail.split()
            comm = head.partition("(")[2] or "?"
            state, pgrp, sid = fields[0], int(fields[2]), int(fields[3])
        except (OSError, IndexError, ValueError):
            continue                     # exited while we walked it
        pid = int(entry)
        for rid in set([pgrp, sid]):
            if rid not in wanted or rid == pid:
                continue
            kind = "+".join(k for k, v in (("pgid", pgrp), ("sid", sid))
                            if v == rid)
            refs.setdefault(rid, []).append((pid, comm, state, kind))
    return refs


def proc_field(pid, name):
    try:
        with open("/proc/%d/%s" % (pid, name)) as fh:
            return fh.read().strip()
    except OSError:
        return "?"


def ranges(ids):
    out = []
    for v in sorted(ids):
        if out and v == out[-1][1] + 1:
            out[-1][1] = v
        else:
            out.append([v, v])
    return ", ".join("%d-%d" % (a, b) if a != b else str(a) for a, b in out)


def read_sysctl(name):
    try:
        with open("/proc/sys/kernel/%s" % name) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def read_counter():
    return read_sysctl("ns_last_pid")


def burn_past(target):
    """Fork throwaway children until the counter passes ``target``."""
    pid_max = read_sysctl("pid_max") or 32768
    if target >= pid_max:
        sys.exit("target %d exceeds pid_max %d, so no range on this node is "
                 "safe from collisions" % (target, pid_max))
    last = read_counter()
    print("counter before : %s" % last)
    burned = 0
    while True:
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        burned += 1
        if pid >= target:
            break
        if pid < (last or 0) and burned > pid_max:
            sys.exit("counter wrapped without reaching %d" % target)
    print("counter after  : %d  (%d id(s) burned)" % (pid, burned))
    print("\nStart the next process now -- anything spawned after this lands "
          "above %d.  Already-running processes did NOT move, so an occupant "
          "reported above still has to exit." % target)


USAGE = ("usage: pidcheck.py <image-dir-or-model-dir> [--burn]\n"
         "       pidcheck.py --burn-to <id>")


def parse_argv(argv):
    """(positional, burn, burn_to) -- rejects anything else."""
    positional, burn, burn_to = [], False, None
    rest = list(argv)
    while rest:
        arg = rest.pop(0)
        if arg == "--burn":
            burn = True
        elif arg == "--burn-to" or arg.startswith("--burn-to="):
            value = (arg.split("=", 1)[1] if "=" in arg
                     else (rest.pop(0) if rest else ""))
            if not value.isdigit():
                sys.exit("--burn-to needs a task id, e.g. --burn-to 200000")
            burn_to = int(value)
        elif arg.startswith("-"):
            sys.exit(USAGE)
        else:
            positional.append(arg)
    return positional, burn, burn_to


def main():
    args, burn, burn_to = parse_argv(sys.argv[1:])

    # Dump-side form: no image exists yet, so there is nothing to compare
    # against -- just move the counter so the tree about to be created records
    # ids nothing will squat on later.
    if burn_to is not None and not args:
        print("BURN  (target %d)" % burn_to)
        burn_past(burn_to)
        sys.exit(0)
    if len(args) != 1:
        sys.exit(USAGE)

    target = args[0]
    if target.endswith(".img"):
        pstree = target
    else:
        pstree = os.path.join(target, "pstree.img")
        if not os.path.exists(pstree):
            pstree = os.path.join(target, "image", "pstree.img")
    if not os.path.exists(pstree):
        sys.exit("no pstree.img under %s" % target)

    leaders, tids = recorded_tasks(decode_pstree(pstree))
    if not tids:
        sys.exit("pstree.img recorded no tasks (unexpected image layout)")

    print("=" * 78)
    print("CRIU image: %s" % pstree)
    print("=" * 78)
    print("thread group leaders : %s" % leaders)
    print("task ids required    : %d  (range %d-%d)"
          % (len(tids), min(tids), max(tids)))
    print("recorded id ranges   : %s" % ranges(tids)[:500])
    print("ns_last_pid now      : %s   (pid_max %s)"
          % (read_counter(), read_sysctl("pid_max")))
    print()

    live = live_task_ids()
    taken_by = {}
    for tid in sorted(tids.intersection(live)):
        taken_by.setdefault(live[tid], []).append(tid)

    refs = group_references(tids.difference(live))

    if taken_by:
        print("-" * 78)
        print("OCCUPIED BY A TASK  (%d id(s) across %d process(es))"
              % (sum(len(v) for v in taken_by.values()), len(taken_by)))
        print("-" * 78)
        for owner, taken in sorted(taken_by.items()):
            state = proc_field(owner, "stat").rsplit(")", 1)[-1].split()
            print("  pid %-7d %-20s state=%-2s threads=%-4s" %
                  (owner, proc_field(owner, "comm"),
                   state[0] if state else "?",
                   len(os.listdir("/proc/%d/task" % owner))
                   if os.path.isdir("/proc/%d/task" % owner) else "?"))
            print("      takes %d recorded id(s): %s"
                  % (len(taken), ranges(taken)[:200]))
        print()
        print("A restore will fail with EEXIST on the first of these.")
        print("Zombies clear themselves once reaped; a live process must exit,")
        print("or the image must be re-dumped with its ids placed higher.")

    if refs:
        print("-" * 78)
        print("HELD WITH NO TASK  (%d id(s) referenced as a pgid or sid)"
              % len(refs))
        print("-" * 78)
        for rid, holders in sorted(refs.items()):
            print("  id %-7d referenced by %d process(es):" % (rid,
                                                               len(holders)))
            for hpid, hcomm, hstate, hkind in holders:
                print("      pid %-7d %-20s state=%-2s as its %s"
                      % (hpid, hcomm, hstate, hkind))
        print()
        print("These ids have no task, so a /proc or ps scan calls them free,")
        print("and a restore still fails EEXIST on them: a process group or")
        print("session id stays allocated until every member is reaped, and a")
        print("zombie is a member.  Reap the holders above -- if their PPid is")
        print("1 and PID 1 does not wait(), only its exit will -- or re-dump")
        print("with the ids placed higher.")

    if not taken_by and not refs:
        print("No collisions: every recorded task id is free on this node,")
        print("as a task and as a process group / session reference.")

    if burn or burn_to is not None:
        goal = burn_to if burn_to is not None else max(tids) + BURN_MARGIN
        print()
        print("-" * 78)
        print("BURN  (target %d%s)"
              % (goal, "" if burn_to is not None
                 else " = max recorded id + %d" % BURN_MARGIN))
        print("-" * 78)
        burn_past(goal)

    sys.exit(1 if (taken_by or refs) else 0)


if __name__ == "__main__":
    main()
