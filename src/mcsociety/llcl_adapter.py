"""Minecraft game adapter for LLCL's existing ``AgentEpisode`` contract.

The policy receives only explicitly selected sensors. Scenario goals, coordinates,
inventory, evaluator metrics and engine state stay on the experiment side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .client import SimulationClient
from .environment import Simulation
from .models import Action
from .scenarios import Scenario

ACTION_NAMES = (
    "wait",
    "forward",
    "backward",
    "left",
    "right",
    "jump",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "attack",
    "use",
)

_ACTIONS = {
    "wait": Action(),
    "forward": Action(forward=True),
    "backward": Action(back=True),
    "left": Action(left=True),
    "right": Action(right=True),
    "jump": Action(jump=True),
    "turn_left": Action(yaw=-15),
    "turn_right": Action(yaw=15),
    "look_up": Action(pitch=-15),
    "look_down": Action(pitch=15),
    "attack": Action(attack=True),
    "use": Action(use=True),
}


@dataclass(frozen=True)
class LLCLFrame:
    step: int
    sensors: dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool


class LLCLGameAdapter:
    """Translate a MCSociety session into LLCL's discrete game interface.

    The caller owns the simulation and the LLCL learner. One adapter instance
    represents one episode; construct a new LLCL ``AgentEpisode`` after reset.
    ``task_features`` must be a fixed public instruction vector, prepared by
    LLCL's task code. No task or evaluator state is copied into policy sensors.
    """

    def __init__(
        self,
        simulation: Simulation | SimulationClient,
        scenario: Scenario,
        *,
        task_features: np.ndarray | None = None,
    ):
        self.simulation = simulation
        self.scenario = scenario
        self.task_features = None
        if task_features is not None:
            task = np.asarray(task_features)
            if (
                task.dtype != np.float32
                or task.ndim != 1
                or not task.size
                or not np.isfinite(task).all()
            ):
                raise ValueError("task_features must be a finite 1D float32 instruction vector")
            self.task_features = task.copy()
        self.step_number = 0
        self.started = False
        self.ended = False
        self.last_evaluation: dict | None = None

    @staticmethod
    def _image(observation: dict) -> np.ndarray:
        rgb = observation["rgb"]
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 3
            or rgb.shape[2] != 3
        ):
            raise ValueError("Minecraft observation requires HWC uint8 RGB")
        return np.asarray(
            Image.fromarray(rgb).resize((64, 64), Image.Resampling.BILINEAR)
        ).copy()

    def _frame(
        self, observation: dict, reward: float, terminated: bool, truncated: bool
    ) -> LLCLFrame:
        sensors = {"image": self._image(observation)}
        if self.task_features is not None:
            sensors["task"] = self.task_features.copy()
        return LLCLFrame(self.step_number, sensors, reward, terminated, truncated)

    def reset(self) -> LLCLFrame:
        observation, info = self.simulation.reset(self.scenario)
        self.step_number = 0
        self.started = True
        self.ended = False
        self.last_evaluation = info
        return self._frame(observation, 0.0, False, False)

    def step(self, action: str) -> LLCLFrame:
        if not self.started or self.ended:
            raise RuntimeError("Reset before acting, and stop after episode end")
        if action not in _ACTIONS:
            raise ValueError(f"Unknown Minecraft action: {action}")
        observation, _, terminated, truncated, info = self.simulation.step(_ACTIONS[action])
        self.step_number += 1
        self.ended = bool(terminated or truncated)
        self.last_evaluation = info
        # The evaluator alone sees the goal and full game state. LLCL receives
        # a sparse outcome reward, not the evaluator's target-distance shaping.
        success = info.get("success") is True
        reward = -0.001 + (1.0 if success else -1.0 if terminated else 0.0)
        return self._frame(observation, reward, bool(terminated), bool(truncated))
