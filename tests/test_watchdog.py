"""QUADRUPED_NO_TIMEOUT turns the stall watchdog's timeouts off (utils/watchdog.py)."""

import threading

import pytest

from quadruped_rl.utils import watchdog


def _monitors() -> set[int]:
    return {t.ident for t in threading.enumerate() if t.name == "stall-watchdog"}


@pytest.fixture
def fresh(monkeypatch):
    """A watchdog not yet installed; its state is put back afterwards."""
    saved = dict(watchdog._state)
    monkeypatch.delenv(watchdog.HEARTBEAT_ENV, raising=False)
    monkeypatch.delenv(watchdog.NO_TIMEOUT_ENV, raising=False)
    watchdog._state.update(installed=False, noted=False, file=None)
    yield
    watchdog.disable()                     # ends a monitor thread started here, cancels the C timer
    watchdog._state.clear()
    watchdog._state.update(saved)


def test_no_timeout_skips_supervisor_and_monitor(fresh, monkeypatch, capfd):
    monkeypatch.setenv(watchdog.NO_TIMEOUT_ENV, "1")

    def no_child(*args, **kwargs):
        raise AssertionError("supervise() started a child process")

    monkeypatch.setattr(watchdog.subprocess, "Popen", no_child)
    watchdog.supervise(5)                  # returns: no supervisor, nothing to kill
    before = _monitors()
    watchdog.install(300)
    assert not (_monitors() - before)      # no stall monitor thread
    assert watchdog._state["installed"] and watchdog._state["stall_s"] == 0.0
    err = capfd.readouterr().err
    assert err.count("timeouts are off") == 1


def test_disable_timeouts_sets_the_variable(fresh, monkeypatch):
    monkeypatch.setenv(watchdog.NO_TIMEOUT_ENV, "0")     # so the change is undone afterwards
    assert not watchdog.timeouts_disabled()
    watchdog.disable_timeouts()
    assert watchdog.timeouts_disabled()


def test_install_starts_the_monitor_by_default(fresh):
    before = _monitors()
    watchdog.install(300)
    assert _monitors() - before
