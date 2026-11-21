"""Stall watchdog: tell a slow run from a hung one.

Long-running scripts call ``install()`` once. From then on:

* If nothing calls ``heartbeat()`` for ``stall_s`` seconds, every thread's Python stack is
  printed to stderr (and again every ``stall_s`` while the stall lasts). The run is not
  stopped: a long legitimate step (first-time kernel or torch.compile compilation) just
  leaves a trace of where the time went.
* ``kill -USR1 <pid>`` prints the stacks on demand.
* A crash inside native code (segfault, abort) prints the Python stack as well.

Two timers. A Python thread does the normal stall report: it holds the interpreter lock
while it reads the stacks, which is safe for a run that is merely slow. If that thread
itself cannot run for ``GIL_S`` seconds, the main thread is stuck inside native code
holding the lock; then faulthandler's C timer, which needs no lock, prints the stacks.

    QUADRUPED_STALL_S=600 python scripts/train.py ...     # default 300; 0 disables

Ending a stalled run (scripts/train.py --stall-timeout): ``supervise(kill_s)`` re-runs the
script as a child process and watches a heartbeat file the child touches on progress. A
child silent for ``kill_s`` seconds is asked for its stacks (SIGUSR1), killed, and the
supervisor exits with code 75, so a wrapper can continue the run from its last checkpoint
instead of waiting out a cluster time limit (hpc/lib.sh). Being a separate process, the
supervisor also catches hangs that hold the interpreter lock, which no thread inside the
stuck process can do (seen in practice: torch.compile stuck in sympy for many minutes).

No timeouts at all, for runs watched by hand (importing packages and first-time kernel
compilation can be silent for many minutes and look like a stall):

    QUADRUPED_NO_TIMEOUT=1 python scripts/train.py ...    # or: scripts/train.py --no-timeout

Then ``supervise()`` returns at once (nothing is ever killed) and ``install()`` starts
neither the stall report nor faulthandler's timer; crash stacks and ``kill -USR1`` still
work. Being an environment variable, it reaches every child process (pipeline stages,
evaluate.py, record.py). scripts/dev/run_stages.sh NO_TIMEOUT=1 sets it, and hpc/lib.sh
then passes no --stall-timeout.
"""

from __future__ import annotations

import faulthandler
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

GIL_S = 120.0       # the monitor thread could not run for this long: stuck in native code
STALL_EXIT_CODE = 75  # EX_TEMPFAIL: "try again"
HEARTBEAT_ENV = "QUADRUPED_HEARTBEAT_FILE"
NO_TIMEOUT_ENV = "QUADRUPED_NO_TIMEOUT"
_state = {"stall_s": 0.0, "last": 0.0, "touched": 0.0, "installed": False, "what": "start",
          "file": os.environ.get(HEARTBEAT_ENV), "noted": False}


def timeouts_disabled() -> bool:
    """True when QUADRUPED_NO_TIMEOUT is set to 1 (or true/yes/on)."""
    return os.environ.get(NO_TIMEOUT_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def disable_timeouts() -> None:
    """Turn every timeout off for this process and its children (sets QUADRUPED_NO_TIMEOUT=1)."""
    os.environ[NO_TIMEOUT_ENV] = "1"


def _note_disabled() -> None:
    if not _state["noted"]:
        _state["noted"] = True
        print(f"[watchdog] timeouts are off ({NO_TIMEOUT_ENV}=1): no stall kill, no stall stack dumps",
              file=sys.stderr, flush=True)


def supervise(kill_s: float | None) -> None:
    """Run the rest of this script in a child process and end it if it stalls.

    Returns immediately in the child (and when ``kill_s`` is unset); in the supervising
    parent it never returns: it exits with the child's exit code, or 75 after a stall.
    """
    if timeouts_disabled():
        _note_disabled()
        return
    if not kill_s or kill_s <= 0 or os.environ.get(HEARTBEAT_ENV):
        return
    fd, beat = tempfile.mkstemp(prefix="quadruped_heartbeat_")
    os.close(fd)
    child = subprocess.Popen(sys.orig_argv, env=dict(os.environ, **{HEARTBEAT_ENV: beat}))
    for sig in (signal.SIGTERM, signal.SIGINT):      # Slurm's time limit, scancel, Ctrl-C
        signal.signal(sig, lambda signum, _frame: child.send_signal(signum))
    print(f"[watchdog] supervising pid {child.pid}: ended with exit code {STALL_EXIT_CODE} "
          f"after {kill_s:.0f} s without progress", file=sys.stderr, flush=True)
    code = None
    while code is None:
        try:
            code = child.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            idle = time.time() - os.path.getmtime(beat)
            if idle > kill_s:
                print(f"\n[watchdog] pid {child.pid} made no progress for {idle:.0f} s: asking "
                      "it for its stacks, then ending it", file=sys.stderr, flush=True)
                if hasattr(signal, "SIGUSR1"):
                    child.send_signal(signal.SIGUSR1)     # faulthandler prints from C
                    time.sleep(3.0)
                child.kill()
                child.wait()
                code = STALL_EXIT_CODE
    try:
        os.unlink(beat)
    except OSError:
        pass
    sys.exit(code if code >= 0 else 128 - code)


def install(stall_s: float | None = None) -> None:
    if _state["installed"]:
        return
    if stall_s is None:
        stall_s = float(os.environ.get("QUADRUPED_STALL_S", "300"))
    faulthandler.enable(file=sys.stderr, all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    if timeouts_disabled():
        _note_disabled()
        stall_s = 0.0
    _state.update(stall_s=float(stall_s), installed=True, last=time.monotonic())
    if stall_s > 0:
        print(f"[watchdog] pid {os.getpid()}: stacks are printed after {stall_s:.0f} s without "
              f"progress, or on  kill -USR1 {os.getpid()}", file=sys.stderr, flush=True)
        threading.Thread(target=_monitor, name="stall-watchdog", daemon=True).start()


def heartbeat(what: str | None = None) -> None:
    """Progress was made: restart the stall clock (cheap; safe to call every step)."""
    now = time.monotonic()
    _state["last"] = now
    if what is not None:
        _state["what"] = what
    if _state["file"] and now - _state["touched"] > 2.0:     # tell the supervisor, if any
        _state["touched"] = now
        try:
            os.utime(_state["file"])
        except OSError:
            pass


def _monitor() -> None:
    reported = 0.0
    while _state["stall_s"] > 0:
        # Re-armed every few seconds, so it only fires when this thread is starved.
        faulthandler.dump_traceback_later(GIL_S, repeat=True, file=sys.stderr)
        time.sleep(min(5.0, _state["stall_s"] / 4))
        idle = time.monotonic() - _state["last"]
        if idle < _state["stall_s"]:
            reported = 0.0
        elif idle - reported >= _state["stall_s"]:
            reported = idle
            print(f"\n[watchdog] no progress for {idle:.0f} s (last: {_state['what']}). "
                  "Still running; stacks of every thread:", file=sys.stderr, flush=True)
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            print("[watchdog] end of stacks\n", file=sys.stderr, flush=True)


def watch_env(env):
    """Count every ``env.step`` as progress. Returns ``env`` (patched in place)."""
    if not _state["installed"] or getattr(env, "_watchdog_wrapped", False):
        return env
    step = env.step

    def stepped(*args, **kwargs):
        heartbeat("env.step")
        return step(*args, **kwargs)

    env.step = stepped
    env._watchdog_wrapped = True
    return env


def disable() -> None:
    _state["stall_s"] = 0.0
    faulthandler.cancel_dump_traceback_later()
