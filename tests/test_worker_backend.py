"""Real spawn/IPC tests; the scripted backend exists only in this test module."""

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import pytest

from mcsociety.craftground_backend import CraftGroundBackend
from mcsociety.models import Action
from mcsociety.worker_backend import SupervisedBackend


class WorkerTestBackend:
    def __init__(self, _config):
        self.tick = 0
        self.capabilities = {"backend": "spawned-test-only", "rgb": True}
        self.child = None

    def _observation(self):
        return {
            "rgb": np.full((2, 3, 3), self.tick, np.uint8),
            "state": {"tick": self.tick, "worker_pid": os.getpid()},
        }

    def reset(self, *, commands=(), **_):
        self.tick = 0
        if "hang-reset" in commands:
            time.sleep(60)
        return self._observation()

    def step(self, action, *, first_tick=True, commands=()):
        if "error" in commands:
            raise ValueError("scripted failure")
        if "crash" in commands:
            os._exit(23)
        if "hang" in commands:
            time.sleep(60)
        if commands and commands[0] in {"spawn-and-hang", "spawn-and-crash"}:
            self.child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
            )
            Path(commands[1]).write_text(str(self.child.pid))
            if commands[0] == "spawn-and-crash":
                # Give the supervisor a chance to observe the owned descendant
                # before the parent disappears and the child gets reparented.
                time.sleep(0.2)
                os._exit(24)
            time.sleep(60)
        self.tick += 1
        result = self._observation()
        result["state"].update(
            forward=action.forward, first_tick=first_tick, commands=list(commands)
        )
        return result

    def close(self):
        if self.child is not None:
            with contextlib.suppress(ProcessLookupError):
                self.child.terminate()
            self.child.wait(timeout=1)


def failing_factory(_config):
    raise RuntimeError("constructor failure")


def is_live(pid):
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def backend(**kwargs):
    return SupervisedBackend(backend_factory=WorkerTestBackend, startup_timeout=5, **kwargs)


def test_default_capabilities_are_honest_without_starting_worker():
    with SupervisedBackend() as supervised:
        assert supervised.worker_pid is None
        assert supervised.capabilities == CraftGroundBackend().capabilities
        assert supervised.worker_pid is None


def test_spawned_worker_preserves_observations_and_reset_starts_new_process():
    with backend(step_timeout=2) as supervised:
        initial = supervised.reset(seed=1, biome="forest", commands=[])
        first_pid = supervised.worker_pid
        assert initial["state"]["worker_pid"] == first_pid != os.getpid()
        assert supervised.capabilities == {"backend": "spawned-test-only", "rgb": True}
        after = supervised.step(Action(forward=True), first_tick=False, commands=["test"])
        assert after["state"] == {
            "tick": 1,
            "worker_pid": first_pid,
            "forward": True,
            "first_tick": False,
            "commands": ["test"],
        }
        np.testing.assert_array_equal(after["rgb"], np.ones((2, 3, 3), np.uint8))
        reset = supervised.reset(seed=2, biome="desert", commands=[])
        second_pid = supervised.worker_pid
        assert second_pid != first_pid and not is_live(first_pid)
        assert reset["state"]["tick"] == 0
    assert not is_live(second_pid)
    assert supervised.worker_pid is None
    supervised.close()


@pytest.mark.parametrize(
    "command,error,match",
    [
        ("error", RuntimeError, "ValueError: scripted failure"),
        ("crash", RuntimeError, "exit code 23"),
        ("hang", TimeoutError, "deadline"),
    ],
)
def test_failure_disposes_worker_and_requires_clean_reset(command, error, match):
    with backend(step_timeout=0.15) as supervised:
        supervised.reset(seed=1, biome="forest", commands=[])
        failed_pid = supervised.worker_pid
        started = time.monotonic()
        with pytest.raises(error, match=match):
            supervised.step(Action(), commands=[command])
        assert time.monotonic() - started < 5
        assert not is_live(failed_pid)
        assert supervised.worker_pid is None
        with pytest.raises(RuntimeError, match="reset"):
            supervised.step(Action())
        recovered = supervised.reset(seed=2, biome="forest", commands=[])
        assert recovered["state"]["tick"] == 0
        assert supervised.step(Action())["state"]["tick"] == 1


def test_startup_failure_is_structured_and_bounded():
    with SupervisedBackend(backend_factory=failing_factory, startup_timeout=3) as supervised:
        with pytest.raises(RuntimeError, match="RuntimeError: constructor failure"):
            supervised.reset(seed=0, biome="forest", commands=[])
        assert supervised.worker_pid is None


def test_reset_hang_deadline_includes_worker_startup():
    with SupervisedBackend(backend_factory=WorkerTestBackend, startup_timeout=0.8) as supervised:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="deadline"):
            supervised.reset(seed=0, biome="forest", commands=["hang-reset"])
        assert time.monotonic() - started < 5
        assert supervised.worker_pid is None


@pytest.mark.parametrize(
    "command,error",
    [("spawn-and-hang", TimeoutError), ("spawn-and-crash", RuntimeError)],
)
def test_failure_cleans_owned_new_session_descendant_without_touching_unrelated_process(
    tmp_path,
    command,
    error,
):
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    marker = tmp_path / "owned-child.pid"
    try:
        with backend(step_timeout=0.5) as supervised:
            supervised.reset(seed=0, biome="forest", commands=[])
            worker_pid = supervised.worker_pid
            with pytest.raises(error):
                supervised.step(Action(), commands=[command, str(marker)])
            child_pid = int(marker.read_text())
            assert not is_live(worker_pid)
            assert not is_live(child_pid)
            assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_reused_pid_is_not_treated_as_owned_process():
    with backend() as supervised:
        current = psutil.Process()
        supervised._owned_children[current.pid] = current.create_time() - 10
        assert supervised._owned_processes() == []


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_timeouts_must_be_finite_positive(value):
    with pytest.raises(ValueError):
        SupervisedBackend(startup_timeout=value)
    with pytest.raises(ValueError):
        SupervisedBackend(step_timeout=value)
