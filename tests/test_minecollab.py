"""Boundaries around loading tasks from the pinned upstream data set."""

import json

import pytest

from mcsociety.minecollab import MineCollabTask
from mcsociety.world import edit_commands


@pytest.fixture
def task_root(tmp_path):
    directory = tmp_path / ".runtime/upstream/mindcraft/tasks"
    directory.mkdir(parents=True)
    task = {
        "type": "techtree",
        "goal": "Make shears together",
        "agent_count": 2,
        "initial_inventory": {
            "0": {"iron_ingot": 1},
            "1": {"iron_ingot": 1},
        },
        "blocked_actions": {"0": ["!collectBlocks"], "1": ["!collectBlocks"]},
        "target": "shears",
        "number_of_target": 1,
    }
    (directory / "crafting.json").write_text(json.dumps({"shears": task}))
    return tmp_path


def test_minecollab_loader_uses_upstream_task_and_validated_setup(task_root):
    task = MineCollabTask.load(
        "crafting.json", "shears", ("AgentA", "AgentB"), source_root=task_root
    )
    assert task.summary()["target"] == "shears"
    assert task.data["task_id"] == "shears"
    assert [edit_commands(edit)[0] for edit in task.setup_edits] == [
        "clear AgentA", "give AgentA minecraft:iron_ingot 1",
        "clear AgentB", "give AgentB minecraft:iron_ingot 1",
    ]


def test_minecollab_loader_rejects_unsafe_or_unavailable_tasks(task_root):
    agents = ("AgentA", "AgentB")
    with pytest.raises(ValueError, match="inside"):
        MineCollabTask.load("../escape.json", "shears", agents, source_root=task_root)
    with pytest.raises(ValueError, match="agent_count"):
        MineCollabTask.load("crafting.json", "shears", ("AgentA",), source_root=task_root)
    path = task_root / ".runtime/upstream/mindcraft/tasks/crafting.json"
    document = json.loads(path.read_text())
    document["shears"]["blocked_actions"]["0"] = ["!craftRecipe"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="blocks a skill"):
        MineCollabTask.load("crafting.json", "shears", agents, source_root=task_root)
