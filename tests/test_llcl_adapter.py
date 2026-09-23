import numpy as np
import pytest

from mcsociety.llcl_adapter import ACTION_NAMES, LLCLGameAdapter
from mcsociety.scenarios import generate_scenario


class Session:
    def __init__(self):
        self.actions = []
        self.scenario = None

    def reset(self, scenario):
        self.scenario = scenario
        return self._observation(0), {"scenario": scenario.model_dump()}

    def step(self, action):
        self.actions.append(action)
        # Dense evaluator shaping must not leak through this LLCL adapter.
        return self._observation(len(self.actions)), 99.0, len(self.actions) == 2, False, {
            "success": len(self.actions) == 2,
            "metrics": {"target_distance": 0, "hidden_map": "private"},
        }

    @staticmethod
    def _observation(step):
        return {
            "rgb": np.full((80, 100, 3), step, dtype=np.uint8),
            "state": {"position": [123, 64, -30], "inventory": ["secret"]},
        }


def test_llcl_episode_keeps_evaluator_data_private_and_aligns_sparse_reward():
    session = Session()
    scenario = generate_scenario("exploration", 7)
    task = np.array([1.0, 0.0], dtype=np.float32)
    game = LLCLGameAdapter(session, scenario, task_features=task)

    first = game.reset()
    assert first.step == 0 and first.reward == 0 and not first.terminated
    assert set(first.sensors) == {"image", "task"}
    assert first.sensors["image"].shape == (64, 64, 3)
    assert first.sensors["image"].dtype == np.uint8
    task[0] = 0
    assert first.sensors["task"].tolist() == [1.0, 0.0]
    assert game.last_evaluation["scenario"]["goal"] == scenario.goal

    with pytest.raises(ValueError, match="Unknown"):
        game.step("teleport")
    assert session.actions == []

    second = game.step("forward")
    assert second.step == 1 and second.reward == pytest.approx(-0.001)
    assert not second.terminated and not second.truncated
    assert second.sensors["image"][0, 0, 0] == 1
    assert session.actions[0].forward
    assert "state" not in second.sensors and "metrics" not in second.sensors
    assert game.last_evaluation["metrics"]["hidden_map"] == "private"

    final = game.step("use")
    assert final.step == 2 and final.terminated and not final.truncated
    assert final.reward == pytest.approx(0.999)
    assert session.actions[1].use
    with pytest.raises(RuntimeError, match="episode end"):
        game.step("wait")


def test_llcl_action_contract_and_task_vector_validation():
    assert len(ACTION_NAMES) == len(set(ACTION_NAMES)) == 12
    scenario = generate_scenario("exploration", 7)
    with pytest.raises(ValueError, match="float32"):
        LLCLGameAdapter(Session(), scenario, task_features=np.array([1.0]))
    with pytest.raises(ValueError, match="finite"):
        LLCLGameAdapter(Session(), scenario, task_features=np.array([np.nan], dtype=np.float32))
