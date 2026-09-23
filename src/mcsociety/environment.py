"""Episode lifecycle, causal recording, and task evaluation around a simulator."""

import json
import re
import uuid
from pathlib import Path

from .backend import Backend
from .dataset import TrajectoryWriter
from .evaluation import TaskEvaluator
from .models import Action, HighLevelAction, WorldEdit
from .scenarios import Scenario, generate_scenario
from .world import edit_commands, inventory_count_command

_COUNT_FOUND = re.compile(r"Found (\d+) matching item\(s\) on player ([A-Za-z0-9_]+)")
_COUNT_NONE = re.compile(r"No items were found on player ([A-Za-z0-9_]+)")


class Simulation:
    """Synchronous environment: one owner, one episode, one tick at a time.

    Each primitive tick is recorded even when the caller requests a multi-tick
    action. Intervention records remain distinguishable from policy transitions.
    """

    def __init__(self, backend: Backend, *, record_dir: str | Path | None = "runs"):
        self.backend = backend
        self.record_dir = record_dir
        self.observation = None
        self.scenario = None
        self.evaluator = None
        self.writer = None
        self.done = True
        self.steps = 0
        self.assisted = False

    def reset(self, scenario: Scenario | None = None, *, seed: int | None = None) -> tuple[dict, dict]:
        if scenario is not None and seed is not None and seed != scenario.seed:
            raise ValueError("seed must match an explicitly supplied scenario.seed")
        self._close_writer()
        self.done = True
        self.observation = None
        self.scenario = scenario or generate_scenario("exploration", seed or 0)
        self.evaluator = TaskEvaluator(self.scenario)
        self.steps = 0
        self.assisted = False
        try:
            self.observation = self.backend.reset(
                seed=self.scenario.seed,
                biome=self.scenario.biome,
                commands=self.scenario.setup_commands(agent_selector="@p"),
            )
            if self.record_dir is not None:
                self.writer = TrajectoryWriter(
                    self.record_dir,
                    {
                        "scenario": self.scenario.model_dump(mode="json"),
                        "backend": getattr(self.backend, "capabilities", {}),
                    },
                )
                self.writer.record_initial(self.observation)
            self.done = False
            return self.observation, {"scenario": self.scenario.model_dump(mode="json")}
        except BaseException:
            self.backend.close()
            self._close_writer()
            raise

    def step(self, action: Action | dict) -> tuple[dict, float, bool, bool, dict]:
        action = action if isinstance(action, Action) else Action.model_validate(action)
        self._require_episode()
        reward = 0.0
        info = {}
        terminated = truncated = False
        for index in range(action.ticks):
            observation, value, terminated, truncated, info = self._tick(action, index == 0)
            reward += value
            if terminated or truncated:
                break
        info["executed_ticks"] = index + 1
        return observation, reward, terminated, truncated, info

    def intervene(self, edit: WorldEdit | dict) -> tuple[dict, float, bool, bool, dict]:
        """Apply an experiment edit with a no-op physics tick and record its cause."""
        edit = edit if isinstance(edit, WorldEdit) else WorldEdit.model_validate(edit)
        commands = edit_commands(edit)
        self._require_episode()
        return self._tick(Action(), True, commands, edit.model_dump(mode="json"))

    def step_peer(
        self, action: HighLevelAction | dict, *, phase: str = "active"
    ) -> tuple[dict, float, bool, bool, dict]:
        """Advance one game tick while a Mindcraft peer action is in flight.

        The recorded action describes a real-time skill spanning one or more
        ticks; it is not represented as a CraftGround primitive control.
        """
        action = (
            action if isinstance(action, HighLevelAction) else HighLevelAction.model_validate(action)
        )
        if phase not in {"active", "completed"}:
            raise ValueError("peer action phase must be active or completed")
        self._require_episode()
        return self._tick(
            Action(), True,
            peer_action={**action.model_dump(mode="json", exclude_none=True), "phase": phase},
        )

    def inspect_item_count(self, player: str, item: str, *, wait_ticks: int = 8) -> int | None:
        """Ask the game server for an inventory count; return None without a reply.

        The vanilla count-only command is run by the primary experiment client.
        A unique chat marker separates its reply from earlier command feedback.
        Each inspection tick is recorded and never counted as an Agent skill.
        """
        command = inventory_count_command(player, item)
        if not 0 <= wait_ticks <= 40:
            raise ValueError("wait_ticks must be between 0 and 40")
        self._require_episode()
        marker = f"MCSOCIETY_QUERY_{uuid.uuid4().hex}"
        inspection = {"kind": "inventory_count", "player": player, "item": item}
        commands = [f"tellraw @s {json.dumps({'text': marker})}", command]
        marker_seen = False
        seen_messages: set[tuple] = set()
        for index in range(wait_ticks + 1):
            observation, _, terminated, truncated, _ = self._tick(
                Action(), True,
                commands=commands if index == 0 else (),
                inspection={**inspection, "phase": "query" if index == 0 else "wait"},
            )
            for message in observation["state"].get("messages", []):
                content = str(message.get("content", ""))
                key = (message.get("time"), content)
                if key in seen_messages:
                    continue
                seen_messages.add(key)
                if marker in content:
                    marker_seen = True
                    continue
                if not marker_seen:
                    continue
                found = _COUNT_FOUND.search(content)
                if found and found.group(2) == player:
                    return int(found.group(1))
                none = _COUNT_NONE.search(content)
                if none and none.group(1) == player:
                    return 0
            if terminated or truncated:
                break
        return None

    def _tick(
        self, action: Action, first_tick: bool, commands=(), intervention=None,
        peer_action=None, inspection=None,
    ):
        before = self.observation
        scheduled = self.scenario.scheduled_commands(self.steps + 1)
        record_action = action.model_dump(mode="json")
        record_action.update(
            ticks=1, yaw=action.yaw if first_tick else 0, pitch=action.pitch if first_tick else 0
        )
        if not first_tick:
            record_action.update(drop=False, inventory=False, hotbar=None)
        if intervention is not None:
            record_action["intervention"] = intervention
            self.assisted = True
        if peer_action is not None:
            record_action["peer_action"] = peer_action
        if inspection is not None:
            record_action["inspection"] = inspection
        if scheduled:
            record_action["scheduled_commands"] = scheduled
        try:
            after = self.backend.step(
                action, first_tick=first_tick, commands=[*commands, *scheduled]
            )
            self.steps += 1
            result = self.evaluator.update(before["state"], after["state"], record_action)
            result["metrics"]["assisted"] = self.assisted
            info = {"metrics": result["metrics"], "success": result["success"]}
            if intervention is not None:
                info["intervention"] = intervention
            if peer_action is not None:
                info["peer_action"] = peer_action
            if inspection is not None:
                info["inspection"] = inspection
            if scheduled:
                # A submitted command is not proof that the game applied it.
                info["scheduled_commands_submitted"] = scheduled
            if self.writer is not None:
                self.writer.record(
                    record_action,
                    after,
                    result["reward"],
                    result["terminated"],
                    result["truncated"],
                    info,
                )
            self.observation = after
            self.done = result["terminated"] or result["truncated"]
            if self.done:
                self._close_writer()
            return after, result["reward"], result["terminated"], result["truncated"], info
        except BaseException:
            # An uncertain step cannot safely be retried or attached to another
            # transition. Force reset and preserve the valid recorded prefix.
            self.done = True
            self._close_writer()
            self.backend.close()
            raise

    def _require_episode(self):
        if self.done or self.observation is None:
            raise RuntimeError("Call reset before acting, including after episode end or failure")

    def _close_writer(self):
        writer, self.writer = self.writer, None
        if writer is not None:
            writer.close()

    def close(self):
        self.done = True
        try:
            self._close_writer()
        finally:
            self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
