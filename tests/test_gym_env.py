import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from mcsociety.environment import Simulation
from mcsociety.gym_env import BUTTONS, PROPRIOCEPTION_FIELDS, MCSocietyEnv
from mcsociety.models import EnvironmentConfig
from mcsociety.scenarios import generate_scenario


class RecordingBackend:
    """Deterministic test-only sensor fixture, never a production simulator."""

    def __init__(self, *, depth=True):
        self.config = EnvironmentConfig(width=64, height=64, depth=depth)
        self.tick = 0
        self.seed = 0
        self.actions = []
        self.closed = False

    def reset(self, *, seed, commands, biome):
        self.seed = seed
        self.tick = 0
        self.actions = []
        self.closed = False
        return self.observe()

    def observe(self):
        observation = {
            "rgb": np.full((64, 64, 3), (self.seed + self.tick) % 256, dtype=np.uint8),
            "state": {
                "position": [0.0, 64.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "yaw": 0.0,
                "pitch": 0.0,
                "health": 20.0,
                "food": 20,
                "time": self.tick,
                "tick": self.tick,
                "inventory": [],
                "messages": [],
            },
            "sensors": {"rgb": "test_fixture"},
        }
        if self.config.depth:
            observation["depth"] = np.full((64, 64), 0.5, dtype=np.float32)
        return observation

    def step(self, action, *, first_tick=True, commands=()):
        self.actions.append((action, first_tick))
        self.tick += 1
        return self.observe()

    def close(self):
        self.closed = True


def make_env(*, depth=True):
    backend = RecordingBackend(depth=depth)
    return MCSocietyEnv(simulation=Simulation(backend, record_dir=None)), backend


def noop(**overrides):
    return {
        "buttons": np.zeros(len(BUTTONS), dtype=np.int8),
        "camera": np.zeros(2, dtype=np.float32),
        "hotbar": 0,
        "ticks": 1,
        **overrides,
    }


@pytest.mark.parametrize("depth", [True, False])
def test_gymnasium_environment_checker(depth):
    env, _ = make_env(depth=depth)
    check_env(env, skip_render_check=False)
    env.close()


def test_spaces_contain_real_adapter_observations_and_keep_structured_info():
    env, _ = make_env()
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert observation["proprioception"].shape == (11,)
    assert len(PROPRIOCEPTION_FIELDS) == 11
    assert observation["rgb"].dtype == np.uint8
    assert observation["depth"].dtype == np.float32
    assert info["state"]["inventory"] == []
    assert info["scenario"]["seed"] == 7
    # Returned buffers and dictionaries cannot corrupt the live simulation.
    observation["rgb"][:] = 255
    info["state"]["position"][0] = 999
    assert np.all(env.render() == 7)
    assert env.simulation.observation["state"]["position"][0] == 0


def test_actions_convert_camera_slots_ticks_and_cancel_opposing_directions():
    env, backend = make_env()
    env.reset(seed=1)
    buttons = np.ones(len(BUTTONS), dtype=np.int8)
    action = noop(
        buttons=buttons, camera=np.asarray([15, -30], dtype=np.float32), hotbar=4, ticks=3
    )
    observation, _, terminated, truncated, info = env.step(action)
    primitive, first = backend.actions[0]
    assert first and len(backend.actions) == 3
    assert (
        not primitive.forward and not primitive.back and not primitive.left and not primitive.right
    )
    assert primitive.jump and primitive.attack
    assert primitive.pitch == 15 and primitive.yaw == -30
    assert primitive.hotbar == 4 and primitive.ticks == 3
    assert info["executed_ticks"] == 3
    assert not terminated and not truncated
    assert env.observation_space.contains(observation)
    np.testing.assert_array_equal(env.render(), observation["rgb"])


def test_invalid_action_is_rejected_before_advancing_game():
    env, backend = make_env()
    env.reset(seed=1)
    for invalid in (
        noop(ticks=0),
        noop(hotbar=10),
        noop(camera=np.array([999, 0], dtype=np.float32)),
        {},
    ):
        with pytest.raises(ValueError, match="action_space"):
            env.step(invalid)
    assert backend.tick == 0


def test_reset_scenario_options_preserve_scenario_seed():
    env, backend = make_env()
    scenario = generate_scenario("survival", 12, "desert")
    _, info = env.reset(seed=12, options={"scenario": scenario.model_dump()})
    assert info["scenario"]["kind"] == "survival"
    assert backend.seed == 12
    with pytest.raises(ValueError, match="match"):
        env.reset(seed=13, options={"scenario": scenario})


def test_termination_and_truncation_preserve_simulation_contract():
    env, _ = make_env()
    scenario = generate_scenario("survival", 1)
    scenario.max_steps = 2
    env.reset(options={"scenario": scenario})
    _, _, terminated, truncated, info = env.step(noop(ticks=10))
    assert truncated and not terminated and info["executed_ticks"] == 2
    with pytest.raises(RuntimeError, match="reset"):
        env.step(noop())


def test_registered_environment_is_lazy_and_closes_injected_simulation():
    lazy = gym.make(
        "MCSociety-v0", config={"width": 64, "height": 64, "depth": False}, record_dir=None
    )
    assert lazy.unwrapped.simulation is None
    lazy.close()
    env, backend = make_env()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(noop())
    env.reset(seed=1)
    env.close()
    assert backend.closed and env.render() is None
