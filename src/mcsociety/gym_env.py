"""Gymnasium's fixed numeric interface over the full simulation observations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .environment import Simulation
from .models import Action, EnvironmentConfig
from .scenarios import Scenario, generate_scenario

BUTTONS = (
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "sneak",
    "sprint",
    "attack",
    "use",
    "drop",
    "inventory",
)
PROPRIOCEPTION_FIELDS = (
    "x",
    "y",
    "z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "yaw",
    "pitch",
    "health",
    "food",
    "time",
)


class MCSocietyEnv(gym.Env):
    """A real Minecraft environment with injectable simulation for testing.

    ``buttons`` follows :data:`BUTTONS`; contradictory movement pairs cancel.
    ``camera`` is ``[pitch_delta, yaw_delta]`` in degrees. ``hotbar=0`` leaves the
    selected slot unchanged, and slots 1..9 are explicit selections. ``ticks``
    holds controls for 1..200 simulation ticks, stopping early on episode end.

    RGB, optional depth and the eleven-value proprioception vector are numeric
    observations. The complete structured observation remains in ``info.state``.
    Default backend construction and Minecraft startup wait until ``reset``.
    """

    metadata: ClassVar = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: EnvironmentConfig | dict[str, Any] | None = None,
        *,
        simulation: Simulation | None = None,
        render_mode: str | None = "rgb_array",
        record_dir: str | Path | None = "runs",
    ):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be 'rgb_array' or None")
        if config is None and simulation is not None:
            config = getattr(simulation.backend, "config", None)
        self.config = (
            config
            if isinstance(config, EnvironmentConfig)
            else EnvironmentConfig.model_validate(config or {})
        )
        self.render_mode = render_mode
        self.simulation = simulation
        self.record_dir = record_dir
        self._rgb: np.ndarray | None = None
        self._has_reset = False

        observation_spaces = {
            "rgb": spaces.Box(
                0, 255, shape=(self.config.height, self.config.width, 3), dtype=np.uint8
            ),
            "proprioception": spaces.Box(
                -np.inf, np.inf, shape=(len(PROPRIOCEPTION_FIELDS),), dtype=np.float32
            ),
        }
        if self.config.depth:
            observation_spaces["depth"] = spaces.Box(
                0, 1, shape=(self.config.height, self.config.width), dtype=np.float32
            )
        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = spaces.Dict(
            {
                "buttons": spaces.MultiBinary(len(BUTTONS)),
                "camera": spaces.Box(-180, 180, shape=(2,), dtype=np.float32),
                "hotbar": spaces.Discrete(10),
                "ticks": spaces.Discrete(200, start=1),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"scenario"}:
            raise ValueError("Only 'scenario' is supported in reset options")
        supplied = options.get("scenario")
        if supplied is None:
            task_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31))
            scenario = generate_scenario("exploration", task_seed)
        else:
            scenario = (
                supplied if isinstance(supplied, Scenario) else Scenario.model_validate(supplied)
            )
            if seed is not None and scenario.seed != seed:
                raise ValueError("reset seed must match an explicitly supplied scenario.seed")
        self._has_reset = False
        self._rgb = None
        if self.simulation is None:
            from .worker_backend import SupervisedBackend

            self.simulation = Simulation(
                SupervisedBackend(self.config), record_dir=self.record_dir
            )
        raw, info = self.simulation.reset(scenario)
        observation = self._observation(raw)
        self._has_reset = True
        return observation, self._info(raw, info)

    def step(self, action):
        if not self._has_reset or self.simulation is None:
            raise RuntimeError("Call reset before step")
        if not self.action_space.contains(action):
            raise ValueError("Action is outside the declared action_space")
        controls = {
            name: bool(value) for name, value in zip(BUTTONS, action["buttons"], strict=True)
        }
        for positive, negative in (("forward", "back"), ("left", "right")):
            if controls[positive] and controls[negative]:
                controls[positive] = controls[negative] = False
        primitive = Action(
            **controls,
            pitch=float(action["camera"][0]),
            yaw=float(action["camera"][1]),
            hotbar=int(action["hotbar"]) or None,
            ticks=int(action["ticks"]),
        )
        raw, reward, terminated, truncated, info = self.simulation.step(primitive)
        observation = self._observation(raw)
        return observation, float(reward), bool(terminated), bool(truncated), self._info(raw, info)

    def _observation(self, raw: dict) -> dict[str, np.ndarray]:
        state = raw["state"]
        proprioception = np.asarray(
            [
                *state["position"],
                *state["velocity"],
                state["yaw"],
                state["pitch"],
                state["health"],
                state["food"],
                state["time"],
            ],
            dtype=np.float32,
        )
        if not np.isfinite(proprioception).all():
            raise ValueError("Proprioception contains non-finite sensor values")
        observation = {"rgb": np.asarray(raw["rgb"]).copy(), "proprioception": proprioception}
        if self.config.depth:
            observation["depth"] = np.asarray(raw["depth"], dtype=np.float32).copy()
        if not self.observation_space.contains(observation):
            raise ValueError("Backend observation does not match the configured observation_space")
        self._rgb = observation["rgb"].copy()
        return observation

    @staticmethod
    def _info(raw: dict, info: dict) -> dict:
        return {
            **deepcopy(info),
            "state": deepcopy(raw["state"]),
            "sensors": deepcopy(raw.get("sensors", {})),
        }

    def render(self):
        return None if self._rgb is None else self._rgb.copy()

    def close(self):
        self._has_reset = False
        self._rgb = None
        if self.simulation is not None:
            self.simulation.close()


if "MCSociety-v0" not in gym.registry:
    gym.register("MCSociety-v0", entry_point="mcsociety.gym_env:MCSocietyEnv")
