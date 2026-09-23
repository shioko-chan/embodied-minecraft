"""Closed-loop primitive skills. No privileged world-edit commands are used."""

import math
from collections.abc import Callable

from .environment import Simulation
from .memory import MemoryStore
from .models import Action


def navigation_action(state: dict, target: list | tuple, *, ticks: int = 5) -> Action:
    """Steer toward an XYZ target; this is a baseline, not an optimal path oracle."""
    x, _, z = state["position"]
    dx, dz = target[0] - x, target[2] - z
    heading = math.degrees(math.atan2(-dx, dz))
    delta = (heading - state["yaw"] + 180) % 360 - 180
    return Action(forward=True, yaw=delta, pitch=-state["pitch"], ticks=ticks)


class Controller:
    def __init__(self, simulation: Simulation):
        self.simulation = simulation

    def move_forward(self, ticks=5):
        return self.simulation.step(Action(forward=True, ticks=ticks))

    def turn(self, angle: float):
        return self.simulation.step(Action(yaw=angle))

    def jump(self):
        return self.simulation.step(Action(jump=True))

    def attack(self, ticks=1):
        return self.simulation.step(Action(attack=True, ticks=ticks))

    def navigate(self, target, *, radius=1.5, max_ticks=200) -> dict:
        if radius <= 0 or max_ticks < 1:
            raise ValueError("radius and max_ticks must be positive")
        used = 0
        while used < max_ticks:
            self.simulation._require_episode()
            state = self.simulation.observation["state"]
            if math.dist(state["position"], target) <= radius:
                return {"success": True, "ticks": used, "reason": "reached"}
            action = navigation_action(state, target, ticks=min(5, max_ticks - used))
            observation, _, terminated, truncated, info = self.simulation.step(action)
            used += info["executed_ticks"]
            reached = math.dist(observation["state"]["position"], target) <= radius
            if reached or terminated or truncated:
                return {
                    "success": reached,
                    "ticks": used,
                    "reason": "reached" if reached else "episode_ended",
                }
        return {"success": False, "ticks": used, "reason": "budget_exhausted"}

    def run_skill(self, memory: MemoryStore, name: str) -> dict:
        skill = memory.get_skill(name)
        if skill is None:
            raise KeyError(name)
        # Validate the entire plan before changing the world.
        actions = [Action.model_validate(action) for action in skill["actions"]]
        executed = 0
        completed = True
        for action in actions:
            _, _, terminated, truncated, info = self.simulation.step(action)
            executed += 1
            completed = info["executed_ticks"] == action.ticks
            if terminated or truncated:
                break
        return {
            "skill": name,
            "actions_executed": executed,
            "plan_completed": completed and executed == len(actions),
        }


def run_episode(simulation: Simulation, policy: Callable[[dict], Action], scenario) -> dict:
    """An LLM, VLM or RL policy can implement the same observation→Action callable."""
    observation, _ = simulation.reset(scenario)
    while True:
        observation, _, terminated, truncated, info = simulation.step(policy(observation))
        if terminated or truncated:
            return info["metrics"]
