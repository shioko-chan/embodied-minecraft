"""Load pinned MineCollab task data without copying its task evaluator.

The initial integration supports techtree tasks whose action restrictions are
enforceable by the current Mindcraft skill bridge. Cooking and construction
require their upstream world setup and are rejected until that is connected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import WorldEdit
from .world import edit_commands

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXPOSED_BLOCKED_ACTIONS = {
    "!craftRecipe", "!goToCoordinates", "!placeHere", "!putInChest", "!takeFromChest",
}


@dataclass(frozen=True)
class MineCollabTask:
    source: str
    task_id: str
    data: dict[str, Any]
    setup_edits: tuple[WorldEdit, ...]

    @classmethod
    def load(
        cls,
        source: str,
        task_id: str,
        agents: tuple[str, ...],
        *,
        source_root: str | Path = _PROJECT_ROOT,
    ) -> MineCollabTask:
        tasks_root = (Path(source_root) / ".runtime/upstream/mindcraft/tasks").resolve()
        if not isinstance(source, str) or not source or Path(source).is_absolute():
            raise ValueError("source must be a relative Mindcraft task JSON path")
        path = (tasks_root / source).resolve()
        if not path.is_relative_to(tasks_root) or path.suffix != ".json":
            raise ValueError("source must stay inside the pinned Mindcraft tasks directory")
        if not path.is_file():
            raise FileNotFoundError(f"Mindcraft task file is unavailable: {source}")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id is required")
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or task_id not in document:
            raise ValueError(f"MineCollab task {task_id!r} was not found")
        data = document[task_id]
        if not isinstance(data, dict) or data.get("type") != "techtree":
            raise ValueError("Only MineCollab techtree tasks are currently supported")
        count = data.get("agent_count", 1)
        if type(count) is not int or count != len(agents):
            raise ValueError("MineCollab agent_count must equal active Mindcraft agents")
        target = data.get("target")
        quantity = data.get("number_of_target", 1)
        if not isinstance(target, str) or not target or type(quantity) is not int or quantity < 1:
            raise ValueError("This techtree task needs one target item and a positive count")
        if quantity > 2304:
            raise ValueError("Target quantity exceeds the supported inventory limit")

        inventory = data.get("initial_inventory", {})
        if not isinstance(inventory, dict) or any(
            key not in {str(index) for index in range(count)} for key in inventory
        ):
            raise ValueError("initial_inventory must be indexed by the participating agents")
        edits: list[WorldEdit] = []
        for index, agent in enumerate(agents):
            edits.append(WorldEdit(operation="clear_inventory", target=agent))
            items = inventory.get(str(index), {})
            if not isinstance(items, dict):
                raise TypeError("each agent's initial_inventory must be an item mapping")
            for item, amount in items.items():
                edit = WorldEdit(
                    operation="give_item", target=agent, name=item, value=amount
                )
                edit_commands(edit)
                edits.append(edit)
        for edit in edits:
            edit_commands(edit)
        restrictions = data.get("blocked_actions", {})
        if not isinstance(restrictions, dict):
            raise TypeError("blocked_actions must map agent indexes to commands")
        for index in range(count):
            blocked = restrictions.get(str(index), [])
            if not isinstance(blocked, list) or any(
                name in _EXPOSED_BLOCKED_ACTIONS for name in blocked
            ):
                raise ValueError("Task blocks a skill currently exposed by the bridge")
        return cls(
            source=source,
            task_id=task_id,
            data={**data, "task_id": task_id},
            setup_edits=tuple(edits),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "task_id": self.task_id,
            "type": self.data["type"],
            "goal": self.data.get("goal"),
            "target": self.data["target"],
            "number_of_target": self.data.get("number_of_target", 1),
            "agent_count": self.data.get("agent_count", 1),
        }
