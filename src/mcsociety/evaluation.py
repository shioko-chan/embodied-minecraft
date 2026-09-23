"""Evidence-based task evaluation and transparent benchmark aggregation.

Missing sensors never count as evidence of success. The environment must include
current ``goal_blocks`` to verify a structure beyond the local block sensor, and
verified ``transfers`` to establish social item exchange. Merely requesting an
action does not establish its outcome.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from .scenarios import Scenario


def _position(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    try:
        converted = tuple(float(x) for x in value)
        return converted if all(math.isfinite(x) for x in converted) else None
    except (TypeError, ValueError):
        return None


def _identifier(value: Any) -> str:
    return str(value).removeprefix("minecraft:")


class TaskEvaluator:
    """One evaluator per episode. ``update`` counts one environment action step."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.steps = 0
        self.start_tick: int | None = None
        self.initial_position: tuple[float, float, float] | None = None
        self.path_length = 0.0
        self.path_complete = True
        self.message_seen = False
        self.transferred_count = 0
        self._transfer_ids: set[str] = set()
        self._previous_progress = 0.0
        self._terminal_result: dict[str, Any] | None = None
        self._adaptation_samples: dict[str, list[float]] = {"before": [], "after": []}
        self._adaptation_applied = False

    def update(
        self, before: Mapping[str, Any], after: Mapping[str, Any], action: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return reward, termination and metrics from observed outcomes.

        Transfer evidence has ``id``, ``sender``, ``recipient``, ``item``, ``count``
        and ``success: true``. Evidence can appear in ``after.transfers`` or a
        backend-populated ``after.action_result.transfers``. The ``action`` input
        is deliberately never trusted as evidence of completed actions.
        """
        if self._terminal_result is not None:
            raise RuntimeError("episode already ended; create a new evaluator after reset")
        self.steps += 1
        first_position, next_position = (
            _position(before.get("position")),
            _position(after.get("position")),
        )
        if self.steps == 1:
            self.initial_position = first_position
            tick = before.get("tick")
            self.start_tick = (
                int(tick) if isinstance(tick, (int, float)) and math.isfinite(tick) else None
            )
        if first_position is None or next_position is None:
            self.path_complete = False
        else:
            self.path_length += math.dist(first_position, next_position)

        goal = self.scenario.goal
        metrics: dict[str, Any] = {
            "scenario_id": self.scenario.id,
            "kind": self.scenario.kind,
            "biome": self.scenario.biome,
            "split": self.scenario.split,
            "steps": self.steps,
            "path_length": self.path_length if self.path_complete else None,
            "planning_efficiency": None,
            "planning_efficiency_reason": "No certified action-space optimum supplied",
        }
        success, progress, evidence = False, 0.0, False
        if self.scenario.kind == "exploration":
            target = _position(goal["target"])
            distance = math.dist(next_position, target) if next_position and target else None
            initial_distance = (
                math.dist(self.initial_position, target)
                if self.initial_position and target
                else None
            )
            evidence = distance is not None
            success = evidence and distance <= float(goal.get("radius", 1.5))
            progress = (
                max(0.0, 1.0 - distance / initial_distance)
                if distance is not None and initial_distance
                else float(success)
            )
            metrics.update(
                target_distance=distance, navigation_success=bool(success) if evidence else None
            )
        elif self.scenario.kind == "survival":
            tick, health = after.get("tick"), after.get("health")
            evidence = (
                self.start_tick is not None
                and isinstance(tick, (int, float))
                and math.isfinite(tick)
                and isinstance(health, (int, float))
                and math.isfinite(health)
            )
            elapsed = max(0, int(tick) - self.start_tick) if evidence else None
            progress = min(1.0, elapsed / goal["duration_ticks"]) if elapsed is not None else 0.0
            success = (
                evidence
                and elapsed >= goal["duration_ticks"]
                and health >= goal.get("minimum_health", 1)
            )
            metrics.update(survived_ticks=elapsed)
        elif self.scenario.kind == "building":
            blocks: dict[tuple[float, float, float], str] = {}
            # goal_blocks overrides local readings and must be sampled this step.
            for block in list(after.get("nearby_blocks", [])) + list(after.get("goal_blocks", [])):
                if isinstance(block, Mapping):
                    position = _position(block.get("position"))
                    if position is not None and block.get("block") is not None:
                        blocks[position] = _identifier(block["block"])
            required = goal["required_blocks"]
            required_empty = goal.get("required_empty", [])
            observed = sum(tuple(block["position"]) in blocks for block in required)
            matched = sum(
                blocks.get(tuple(block["position"])) == _identifier(block["block"])
                for block in required
            )
            observed_empty = sum(tuple(position) in blocks for position in required_empty)
            matched_empty = sum(
                blocks.get(tuple(position)) in {"air", "cave_air", "void_air"}
                for position in required_empty
            )
            evidence = observed == len(required) and observed_empty == len(required_empty)
            success = evidence and matched == len(required) and matched_empty == len(required_empty)
            progress = matched / len(required)
            metrics.update(
                required_blocks=len(required),
                observed_goal_blocks=observed,
                matched_goal_blocks=matched,
                required_empty=len(required_empty),
                matched_empty=matched_empty,
            )
        else:
            recipient = goal["recipient"]
            actor = after.get("agent_id", before.get("agent_id"))
            for message in after.get("messages", []):
                if (
                    isinstance(message, Mapping)
                    and message.get("recipient") == recipient
                    and message.get("content")
                    and actor is not None
                    and message.get("sender") == actor
                ):
                    self.message_seen = True
            transfers = list(after.get("transfers", []))
            action_result = after.get("action_result", {})
            if isinstance(action_result, Mapping):
                transfers += action_result.get("transfers", [])
            else:
                action_result = {}
            for transfer in transfers:
                if (
                    not isinstance(transfer, Mapping)
                    or transfer.get("success") is not True
                    or not transfer.get("id")
                ):
                    continue
                if (
                    actor is None
                    or transfer.get("sender") != actor
                    or transfer.get("recipient") != recipient
                ):
                    continue
                if _identifier(transfer.get("item")) != _identifier(goal["item"]):
                    continue
                transfer_id = str(transfer["id"])
                count = transfer.get("count")
                if transfer_id not in self._transfer_ids and isinstance(count, int) and count > 0:
                    self._transfer_ids.add(transfer_id)
                    self.transferred_count += count
            message_ok = self.message_seen or not goal.get("require_message", True)
            evidence = (
                actor is not None
                and ("messages" in after or not goal.get("require_message", True))
                and ("transfers" in after or "transfers" in action_result)
            )
            progress = 0.5 * min(1, self.transferred_count / goal["count"]) + 0.5 * float(
                message_ok
            )
            success = self.transferred_count >= goal["count"] and message_ok
            metrics.update(
                message_delivered=self.message_seen,
                transferred_count=self.transferred_count,
                recipient=recipient,
            )

        health = after.get("health")
        dead = after.get("dead") is True or (isinstance(health, (int, float)) and health <= 0)
        success = bool(success and not dead)
        if self.scenario.kind == "exploration" and metrics["navigation_success"] is not None:
            metrics["navigation_success"] = success
        terminated = success or dead
        truncated = self.steps >= self.scenario.max_steps and not terminated
        progress_delta = progress - self._previous_progress
        reward = progress_delta + (1.0 if success else -1.0 if dead else -0.001)
        self._previous_progress = progress
        oracle_steps, oracle_source = goal.get("oracle_steps"), goal.get("oracle_source")
        if (
            success
            and isinstance(oracle_steps, int)
            and oracle_steps > 0
            and isinstance(oracle_source, str)
            and oracle_source.strip()
        ):
            metrics["planning_efficiency"] = self.steps / oracle_steps
            metrics["planning_efficiency_reason"] = None
            metrics["oracle_source"] = oracle_source

        event_steps = [event.step for event in self.scenario.events]
        phase = "after" if event_steps and self.steps >= min(event_steps) else "before"
        for event in after.get("events_applied", []):
            kind = event.get("kind") if isinstance(event, Mapping) else event
            if kind in {scheduled.kind for scheduled in self.scenario.events}:
                self._adaptation_applied = True
        # Progress deltas are a transparent task-performance signal, not proof of
        # causal adaptation. Missing observations are omitted, never imputed zero.
        if event_steps and evidence and (phase == "before" or self._adaptation_applied):
            self._adaptation_samples[phase].append(progress_delta)
        adaptation = {
            key: (sum(samples) / len(samples) if samples else None)
            for key, samples in self._adaptation_samples.items()
        }
        adaptation["difference"] = (
            adaptation["after"] - adaptation["before"]
            if all(adaptation[key] is not None for key in ("before", "after"))
            else None
        )
        metrics.update(
            success=success,
            evidence_complete=evidence,
            progress=progress,
            adaptation=adaptation if event_steps else None,
            adaptation_signal="mean_task_progress_delta" if event_steps else None,
            adaptation_event_applied=self._adaptation_applied if event_steps else None,
        )
        result = {
            "reward": float(reward),
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
            "metrics": metrics,
        }
        if terminated or truncated:
            self._terminal_result = result
        return result


def aggregate_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate final episode results; missing evidence remains explicit.

    Generalization reports domain rates and their gap. Adaptation is descriptive
    pre/post task progress and must not be described as a causal adaptation score.
    """
    all_episodes = [dict(result.get("metrics", result)) for result in results]
    episodes = [episode for episode in all_episodes if episode.get("assisted") is not True]

    def success_rate(selected: list[dict[str, Any]]) -> float | None:
        return (
            sum(episode.get("success") is True for episode in selected) / len(selected)
            if selected
            else None
        )

    navigation = [episode for episode in episodes if episode.get("kind") == "exploration"]
    navigation_observed = [
        episode for episode in navigation if episode.get("navigation_success") is not None
    ]
    splits = {
        split: [episode for episode in episodes if episode.get("split") == split]
        for split in ("train", "test")
    }
    split_rates = {split: success_rate(selected) for split, selected in splits.items()}
    efficiency = [
        episode["planning_efficiency"]
        for episode in episodes
        if episode.get("planning_efficiency") is not None
    ]
    adaptation = [
        episode["adaptation"]
        for episode in episodes
        if episode.get("adaptation") and episode["adaptation"].get("difference") is not None
    ]
    return {
        "total_episodes": len(all_episodes),
        "excluded_assisted": len(all_episodes) - len(episodes),
        "episodes": len(episodes),
        "success_rate": success_rate(episodes),
        "navigation": {
            "episodes": len(navigation),
            "evaluated": len(navigation_observed),
            "success_rate": sum(
                episode["navigation_success"] is True for episode in navigation_observed
            )
            / len(navigation_observed)
            if navigation_observed
            else None,
        },
        "planning": {
            "evaluated": len(efficiency),
            "mean_actual_over_optimal_steps": sum(efficiency) / len(efficiency)
            if efficiency
            else None,
        },
        "generalization": {
            "split_episodes": {split: len(selected) for split, selected in splits.items()},
            "success_rates": split_rates,
            "test_minus_train": split_rates["test"] - split_rates["train"]
            if all(value is not None for value in split_rates.values())
            else None,
        },
        "adaptation": {
            "evaluated": len(adaptation),
            "mean_before_progress_delta": sum(item["before"] for item in adaptation)
            / len(adaptation)
            if adaptation
            else None,
            "mean_after_progress_delta": sum(item["after"] for item in adaptation) / len(adaptation)
            if adaptation
            else None,
            "mean_difference": sum(item["difference"] for item in adaptation) / len(adaptation)
            if adaptation
            else None,
            "signal": "mean_task_progress_delta",
            "causal": False,
        },
    }
