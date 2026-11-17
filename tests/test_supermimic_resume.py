"""Where ``--continue-run`` resumes: the newest numbered checkpoint, or SuperMimic's
ckpt_latest.pt when that is further along (it is overwritten every latest_interval
iterations, so a run ended by the stall watchdog loses little progress).

CPU only, no simulator; the SuperMimic reference repository is not needed.
"""

import pytest

from quadruped_rl.algorithms import get_algorithm
from quadruped_rl.algorithms.supermimic import backend
from quadruped_rl.algorithms.supermimic.agent import SuperMimicAgent

torch = pytest.importorskip("torch")


def _ckpt(path, iteration):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"iteration": iteration}, path)
    return path


# ---- latest_interval ----------------------------------------------------------------

def test_latest_interval_uses_the_configured_value():
    assert backend.latest_interval({"latest_interval": 250, "save_interval": 2000}) == 250


@pytest.mark.parametrize("save_interval,expected", [(2000, 100), (50, 50), (100, 100)])
def test_latest_interval_defaults_for_configs_written_before_the_key(save_interval, expected):
    assert backend.latest_interval({"save_interval": save_interval}) == expected
    assert backend.latest_interval({"latest_interval": None, "save_interval": save_interval}) == expected


@pytest.mark.parametrize("value", [0, -1, -500])
def test_latest_interval_zero_or_negative_turns_it_off(value):
    assert backend.latest_interval({"latest_interval": value, "save_interval": 2000}) == 0


# ---- SuperMimicAgent.resume_checkpoint ----------------------------------------------

def test_no_checkpoint_resumes_from_nothing(tmp_path):
    assert SuperMimicAgent.resume_checkpoint(tmp_path) is None


def test_newest_numbered_checkpoint_without_latest(tmp_path):
    _ckpt(tmp_path / "ckpt_0002000.pt", 2000)
    newest = _ckpt(tmp_path / "ckpt_0004000.pt", 4000)
    assert SuperMimicAgent.resume_checkpoint(tmp_path) == (newest, 4000)


def test_latest_alone_is_resumed(tmp_path):
    latest = _ckpt(tmp_path / "ckpt_latest.pt", 300)
    assert SuperMimicAgent.resume_checkpoint(tmp_path) == (latest, 300)


def test_latest_wins_when_further_along(tmp_path):
    _ckpt(tmp_path / "ckpt_0002000.pt", 2000)
    latest = _ckpt(tmp_path / "ckpt_latest.pt", 2700)
    assert SuperMimicAgent.resume_checkpoint(tmp_path) == (latest, 2700)


def test_latest_loses_when_behind(tmp_path):
    newest = _ckpt(tmp_path / "ckpt_0004000.pt", 4000)
    _ckpt(tmp_path / "ckpt_latest.pt", 3900)
    assert SuperMimicAgent.resume_checkpoint(tmp_path) == (newest, 4000)


def test_latest_at_the_same_iteration_keeps_the_numbered_one(tmp_path):
    newest = _ckpt(tmp_path / "ckpt_0004000.pt", 4000)
    _ckpt(tmp_path / "ckpt_latest.pt", 4000)
    assert SuperMimicAgent.resume_checkpoint(tmp_path) == (newest, 4000)


def test_unreadable_latest_is_ignored(tmp_path, capsys):
    newest = _ckpt(tmp_path / "ckpt_0002000.pt", 2000)
    (tmp_path / "ckpt_latest.pt").write_bytes(b"\x00not a checkpoint\xff" * 16)
    assert SuperMimicAgent.resume_checkpoint(tmp_path) == (newest, 2000)
    assert "ignoring unreadable" in capsys.readouterr().out


def test_registry_resolves_to_the_supermimic_override(tmp_path):
    latest = _ckpt(tmp_path / "ckpt_latest.pt", 150)
    assert get_algorithm("supermimic").resume_checkpoint(tmp_path) == (latest, 150)


# ---- base Agent.resume_checkpoint ---------------------------------------------------

@pytest.mark.parametrize("algo,rel,last", [("ppo", "checkpoints/model_{}.pt", 1500),
                                           ("ppo_rslrl", "model_{}.pt", 1500)])
def test_ppo_resumes_from_the_newest_numbered_checkpoint(tmp_path, algo, rel, last):
    cls = get_algorithm(algo)
    assert cls.resume_checkpoint(tmp_path) is None
    for n in (100, last, 300):
        p = tmp_path / rel.format(n)
        p.parent.mkdir(exist_ok=True)
        p.touch()                               # the base class only reads file names
    (tmp_path / "ckpt_latest.pt").touch()       # SuperMimic's file is not a PPO resume point
    assert cls.resume_checkpoint(tmp_path) == (tmp_path / rel.format(last), last)
