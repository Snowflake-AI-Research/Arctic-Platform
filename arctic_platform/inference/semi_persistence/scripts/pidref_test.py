#!/usr/bin/env python3
"""Check, against the running kernel, the three facts the restore depends on.

    python3 scripts/pidref_test.py

1. A pid with no task at all can still be allocated, because a member of the
   process group or session it names holds a reference on it.
2. A task-based occupancy scan cannot see that and the pgid/sid scan can --
   the difference between `pidcheck.py` reporting "No collisions" and the
   truth.
3. `PR_SET_CHILD_SUBREAPER` plus a `waitpid` sweep collects the holder and
   frees the id.  This is the dump-side fix.

All three were established by a dev-cluster failure where a dumped leader's id was
pinned by two unreaped members of its own session, so this is a regression
test for that: run it on any Linux box and it reproduces the state in about a
second, unprivileged, with no GPU and no criu.  Note that it passes on a pod
whose PID 1 *does* reap -- the point of the subreaper is that the fix no
longer depends on PID 1 at all.

Exercises the real `worker.py` and `pidcheck.py` functions, so it needs the
package's own imports (pynvml, torch) on `PYTHONPATH`, unlike `pidcheck.py`.
"""

import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                        # pidcheck
sys.path.insert(0, os.path.dirname(_HERE))       # worker + its flat imports

import pidcheck
import worker

logging.basicConfig(level=logging.INFO, format="      [log] %(message)s")
log = logging.getLogger("pidref_test")

FAILURES = []


def check(label, got, want):
    ok = got == want
    print("    %-58s %-14s %s" % (label, repr(got), "OK" if ok else
                                  "FAIL (wanted %r)" % (want,)))
    if not ok:
        FAILURES.append(label)


def task_exists(pid):
    """What /proc, ps and a task-only scan all resolve: is there a task here."""
    return os.path.exists("/proc/%d" % pid)


def group_allocated(pgid):
    """killpg(pgid, 0) finds the group by its struct pid, so it succeeds only
    while that id is still allocated -- with or without a task at it."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def ppid_of(pid):
    try:
        with open("/proc/%d/stat" % pid) as fh:
            return int(fh.read().rpartition(")")[2].split()[1])
    except OSError:
        return None


def state_of(pid):
    try:
        with open("/proc/%d/stat" % pid) as fh:
            return fh.read().rpartition(")")[2].split()[0]
    except OSError:
        return None


def spawn_leader(hold_alive):
    """Fork a session leader that spawns one member, then exits.

    Mirrors the vLLM child, which setsid()s at startup: the leader's pid names
    both its process group and its session, so the member it forks holds a
    reference on that pid.  Returns ``(leader_pid, member_pid)`` with the
    leader already reaped, so its id has no task of its own.

    ``hold_alive`` picks the holder: a live member (the reference with no
    zombie involved anywhere) or an unreaped corpse (what a destructive dump
    actually leaves behind).
    """
    read_fd, write_fd = os.pipe()
    leader = os.fork()
    if leader == 0:                                  # the leader
        os.close(read_fd)
        os.setsid()
        member = os.fork()
        if member == 0:                              # the member
            os.close(write_fd)
            if hold_alive:
                time.sleep(120)
            os._exit(0)                       # a corpse: the leader won't reap
        os.write(write_fd, b"%d" % member)
        os.close(write_fd)
        time.sleep(0.2)                              # let the member settle
        os._exit(0)
    os.close(write_fd)
    member = int(os.read(read_fd, 32))
    os.close(read_fd)
    os.waitpid(leader, 0)                      # reap: no task left at this id
    return leader, member


def main():
    print(__doc__)
    print("=" * 78)
    print("1. A live member pins its session leader's id, which has no task")
    print("=" * 78)

    leader, member = spawn_leader(hold_alive=True)
    print("    leader %d (reaped), member %d (alive, in its group+session)"
          % (leader, member))
    check("task at the leader's id (/proc/<pid>)", task_exists(leader), False)
    check("leader's id still allocated (killpg)", group_allocated(leader),
          True)
    check("member's pgid == leader's id", os.getpgid(member) == leader, True)

    check("task scan alone sees the leader's id",
          leader in pidcheck.live_task_ids(), False)
    refs = pidcheck.group_references(set([leader]))
    check("pgid/sid scan sees it", leader in refs, True)
    if leader in refs:
        check("  ... and names the member",
              [h[0] for h in refs[leader]], [member])
        check("  ... with the reference kind", refs[leader][0][3], "pgid+sid")
        check("  ... holder alive, not a zombie", refs[leader][0][2], "S")

    print()
    print("    Freeing the member must free the id:")
    os.kill(member, 9)
    for _ in range(100):
        if not group_allocated(leader):
            break
        time.sleep(0.02)
    check("leader's id released once the member is gone",
          group_allocated(leader), False)
    check("scan now reports nothing",
          pidcheck.group_references(set([leader])), {})

    print()
    print("=" * 78)
    print("2. The fix: subreaper + sweep collects an unreaped member")
    print("=" * 78)

    check("_set_child_subreaper()", worker._set_child_subreaper(log), True)

    leader2, member2 = spawn_leader(hold_alive=False)
    print("    leader %d (reaped), member %d (exited, unreaped)"
          % (leader2, member2))
    check("task at the leader's id", task_exists(leader2), False)
    check("leader's id still allocated", group_allocated(leader2), True)
    check("member is a zombie", state_of(member2), "Z")
    check("member reparented to us, not PID 1", ppid_of(member2), os.getpid())
    check("pgid/sid scan sees the held id",
          leader2 in pidcheck.group_references(set([leader2])), True)

    print()
    print("    Sweeping:")
    reaped = worker._reap_orphaned_descendants(log)
    check("sweep reaped the member", member2 in reaped, True)
    check("leader's id released", group_allocated(leader2), False)
    check("scan now reports nothing",
          pidcheck.group_references(set([leader2])), {})

    print()
    print("=" * 78)
    if FAILURES:
        print("FAILED: %s" % "; ".join(FAILURES))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
