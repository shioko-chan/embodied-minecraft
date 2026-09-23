import json

import numpy as np
import pytest

from mcsociety.dataset import TrajectoryDataset
from mcsociety.environment import Simulation
from mcsociety.models import Action, HighLevelAction, WorldEdit
from mcsociety.scenarios import generate_scenario
from mcsociety.world import edit_commands


class ScriptedBackend:
    """Test double for causal sequencing, never exported as a Minecraft backend."""

    def __init__(self):
        self.calls = []
        self.closed = 0
        self.tick = 0

    def observation(self):
        return {
            "rgb": np.full((64, 64, 3), self.tick, dtype=np.uint8),
            "state": {"position": [0, 64, self.tick], "health": 20, "tick": self.tick},
        }

    def reset(self, **kwargs):
        self.tick = 0
        self.reset_options = kwargs
        return self.observation()

    def step(self, action, *, first_tick=True, commands=()):
        self.calls.append((action.controls(first_tick=first_tick), commands))
        self.tick += 1
        return self.observation()

    def close(self):
        self.closed += 1


def scenario(max_steps=3):
    result = generate_scenario("exploration", 123)
    result.goal["target"] = [0, 64, 20]
    result.max_steps = max_steps
    return result


def test_macro_records_each_tick_and_stops_at_terminal(tmp_path):
    backend = ScriptedBackend()
    simulation = Simulation(backend, record_dir=tmp_path)
    simulation.reset(scenario())
    _, _, terminated, truncated, info = simulation.step(Action(forward=True, yaw=90, ticks=7))
    assert not terminated and truncated
    assert info["executed_ticks"] == 3
    assert [call[0]["camera_yaw"] for call in backend.calls] == [90, 0, 0]
    records = list(TrajectoryDataset(tmp_path))
    assert len(records) == 3
    assert [row["action"]["yaw"] for row in records] == [90, 0, 0]
    for i, row in enumerate(records):
        assert row["observation"]["state"]["tick"] == i
        assert row["next_observation"]["state"]["tick"] == i + 1
    with pytest.raises(RuntimeError, match="reset"):
        simulation.step(Action())
    simulation.close()


def test_reset_separates_episode_and_preserves_seed(tmp_path):
    backend = ScriptedBackend()
    with Simulation(backend, record_dir=tmp_path) as simulation:
        for _ in range(2):
            simulation.reset(scenario())
            simulation.step(Action())
    rows = list(TrajectoryDataset(tmp_path))
    assert len({row["episode_id"] for row in rows}) == 2
    assert backend.reset_options["seed"] == 123
    assert rows[0]["observation"]["state"]["tick"] == 0


def test_failure_requires_reset_and_keeps_valid_prefix(tmp_path):
    backend = ScriptedBackend()
    simulation = Simulation(backend, record_dir=tmp_path)
    simulation.reset(scenario())
    simulation.step(Action())

    def fail(*args, **kwargs):
        raise TimeoutError("lost engine")

    backend.step = fail
    with pytest.raises(TimeoutError):
        simulation.step(Action())
    with pytest.raises(RuntimeError, match="reset"):
        simulation.step(Action())
    assert len(list(TrajectoryDataset(tmp_path))) == 1
    assert backend.closed == 1


def test_intervention_is_separate_cause_and_costs_tick(tmp_path):
    backend = ScriptedBackend()
    simulation = Simulation(backend, record_dir=tmp_path)
    simulation.reset(scenario())
    simulation.intervene(WorldEdit(operation="weather", value="rain"))
    simulation.close()
    assert backend.calls[0][1] == ["weather rain"]
    row = next(iter(TrajectoryDataset(tmp_path)))
    assert row["action"]["intervention"]["value"] == "rain"
    assert row["info"]["metrics"]["assisted"] is True


def test_peer_skill_is_recorded_as_external_action_not_privileged_edit(tmp_path):
    backend = ScriptedBackend()
    with Simulation(backend, record_dir=tmp_path) as simulation:
        simulation.reset(scenario(max_steps=5))
        action = HighLevelAction(kind="navigate", agent="AgentA", position=(4, 64, 0))
        simulation.step_peer(action)
        simulation.step_peer(action, phase="completed")
    records = list(TrajectoryDataset(tmp_path))
    assert [row["action"]["peer_action"]["phase"] for row in records] == [
        "active", "completed"
    ]
    assert records[0]["action"]["peer_action"]["agent"] == "AgentA"
    assert records[0]["action"]["forward"] is False
    assert all(row["info"]["metrics"]["assisted"] is False for row in records)


def test_inventory_inspection_uses_server_count_and_records_read_only_cause(tmp_path):
    class QueryBackend(ScriptedBackend):
        def step(self, action, *, first_tick=True, commands=()):
            observation = super().step(action, first_tick=first_tick, commands=commands)
            if commands:
                marker = json.loads(commands[0].split(" ", 2)[2])["text"]
                observation["state"]["messages"] = [
                    {"content": marker, "time": self.tick},
                    {"content": "Found 1 matching item(s) on player AgentB", "time": self.tick},
                ]
            return observation

    backend = QueryBackend()
    with Simulation(backend, record_dir=tmp_path) as simulation:
        simulation.reset(scenario(max_steps=5))
        assert simulation.inspect_item_count("AgentB", "shears") == 1
    assert backend.calls[0][1][1] == "clear AgentB minecraft:shears 0"
    record = next(iter(TrajectoryDataset(tmp_path)))
    assert record["action"]["inspection"]["kind"] == "inventory_count"
    assert record["info"]["metrics"]["assisted"] is False


def test_inventory_inspection_does_not_use_stale_or_missing_chat(tmp_path):
    class NoReplyBackend(ScriptedBackend):
        def observation(self):
            observation = super().observation()
            observation["state"]["messages"] = [
                {"content": "Found 99 matching item(s) on player AgentB", "time": 0}
            ]
            return observation

    with Simulation(NoReplyBackend(), record_dir=tmp_path) as simulation:
        simulation.reset(scenario(max_steps=5))
        assert simulation.inspect_item_count("AgentB", "shears", wait_ticks=1) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forward": True, "back": True},
        {"yaw": float("nan")},
        {"ticks": 0},
        {"ticks": 2.5},
        {"hotbar": 10},
        {"teleport": [0, 1, 2]},
    ],
)
def test_invalid_actions(kwargs):
    with pytest.raises(ValueError):
        Action(**kwargs)


def test_world_edit_validation_and_no_command_injection():
    assert edit_commands(WorldEdit(operation="place_block", position=(0, 64, 0), name="stone")) == [
        "setblock 0 64 0 minecraft:stone"
    ]
    assert edit_commands(WorldEdit(operation="rule", name="doWeatherCycle", value=False)) == [
        "gamerule doWeatherCycle false"
    ]
    assert edit_commands(
        WorldEdit(operation="teleport_player", target="AgentA", position=(0, 64, 0))
    ) == ["tp AgentA 0 64 0"]
    assert edit_commands(
        WorldEdit(operation="give_item", target="AgentA", name="oak_log", value=2)
    ) == ["give AgentA minecraft:oak_log 2"]
    assert edit_commands(WorldEdit(operation="clear_inventory", target="AgentA")) == [
        "clear AgentA"
    ]
    for edit in [
        WorldEdit(operation="spawn", position=(0, 64, 0), name="wolf\nsay injected"),
        WorldEdit(operation="remove_entity", name="@e"),
        WorldEdit(operation="place_block", position=(0, 400, 0), name="stone"),
        WorldEdit(operation="rule", name="arbitrary", value=True),
        WorldEdit(operation="teleport_player", target="@a", position=(0, 64, 0)),
        WorldEdit(operation="give_item", target="AgentA", name="oak_log", value=0),
        WorldEdit(operation="clear_inventory", target="@a"),
    ]:
        with pytest.raises(ValueError):
            edit_commands(edit)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "craft", "agent": "AgentA"},
        {"kind": "navigate", "agent": "AgentA", "position": [float("nan"), 64, 0]},
        {"kind": "place_block", "agent": "AgentA", "position": [0.5, 64, 0], "block": "stone"},
        {"kind": "say", "agent": "AgentA", "message": "/op AgentA"},
        {"kind": "say", "agent": "AgentA", "message": "hello\nworld"},
        {"kind": "craft", "agent": "@p", "item": "oak_planks"},
        {"kind": "deposit", "agent": "AgentA"},
    ],
)
def test_high_level_action_rejects_invalid_or_privileged_requests(payload):
    with pytest.raises(ValueError):
        HighLevelAction.model_validate(payload)
