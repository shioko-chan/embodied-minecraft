import pytest
from test_environment import ScriptedBackend, scenario

from mcsociety.controller import Controller, navigation_action
from mcsociety.environment import Simulation
from mcsociety.memory import MemoryStore


def test_skill_does_not_claim_unfinished_action_completed():
    with MemoryStore(":memory:") as memory, Simulation(ScriptedBackend(), record_dir=None) as env:
        memory.save_skill("long_walk", "Walk", [{"forward": True, "ticks": 200}])
        env.reset(scenario(max_steps=2))
        result = Controller(env).run_skill(memory, "long_walk")
        assert result["actions_executed"] == 1
        assert result["plan_completed"] is False


def test_skill_validates_all_actions_before_execution():
    backend = ScriptedBackend()
    with MemoryStore(":memory:") as memory, Simulation(backend, record_dir=None) as env:
        memory.save_skill("invalid", "Bad plan", [{"forward": True}, {"teleport": True}])
        env.reset(scenario())
        with pytest.raises(ValueError):
            Controller(env).run_skill(memory, "invalid")
        assert not backend.calls


def test_navigation_heading_uses_minecraft_axes_and_wrap():
    state = {"position": [0, 64, 0], "yaw": 170, "pitch": 10}
    action = navigation_action(state, [0, 64, -10])
    assert action.yaw == 10 and action.pitch == -10
    state["yaw"] = 0
    assert navigation_action(state, [10, 64, 0]).yaw == -90
