"""Opt-in proof that the public Python Mindcraft bridge drives real Minecraft."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from mcsociety.api import create_app
from mcsociety.dataset import TrajectoryDataset
from mcsociety.environment import Simulation
from mcsociety.mindcraft import MindcraftBridge
from mcsociety.models import Action, EnvironmentConfig, HighLevelAction, WorldEdit
from mcsociety.scenarios import Difficulty, Scenario
from mcsociety.worker_backend import SupervisedBackend

pytestmark = pytest.mark.minecraft


@pytest.mark.skipif(
    os.environ.get("MCSOCIETY_RUN_MINECRAFT") != "1",
    reason="Set MCSOCIETY_RUN_MINECRAFT=1 to launch the real game",
)
def test_mindcraft_skills_through_python_api_and_causal_log(tmp_path):
    scenario = Scenario(
        id="mindcraft-live",
        name="Mindcraft shared world integration",
        kind="survival",
        seed=1729,
        max_steps=3000,
        goal={"type": "survive", "duration_ticks": 10000},
        difficulty=Difficulty(weather="clear", resources="normal", mobs="low"),
    )
    config = EnvironmentConfig(
        width=96, height=64, render_distance=2, lan_port=55916
    )
    with Simulation(SupervisedBackend(config), record_dir=tmp_path) as simulation:
        simulation.reset(scenario)
        with MindcraftBridge(lan_port=55916, agents=("AgentA", "AgentB")) as peers:
            peers.start(pump=lambda: simulation.step(Action()))
            simulation.intervene(
                WorldEdit(operation="teleport_player", target="AgentA", position=(0, 64, 3))
            )
            simulation.intervene(
                WorldEdit(operation="teleport_player", target="AgentB", position=(0, 64, 5))
            )
            simulation.intervene(
                WorldEdit(operation="give_item", target="AgentA", name="oak_log", value=1)
            )
            for _ in range(30):
                peers_state = peers.observe(pump=lambda: simulation.step(Action()))
                if peers_state["AgentA"]["inventory"].get("oak_log", 0) >= 1:
                    break
                simulation.step(Action())
            assert peers_state["AgentA"]["inventory"].get("oak_log", 0) >= 1

            crafted, _ = peers.execute_in_simulation(
                simulation, HighLevelAction(kind="craft", agent="AgentA", item="oak_planks")
            )
            assert crafted["success"]
            assert crafted["after"]["inventory"].get("oak_planks", 0) >= 4
            assert peers.block_at(
                "AgentA", (1, 64, 0), pump=lambda: simulation.step(Action())
            ) == "air"

            placed, (observation, _, _, _, _) = peers.execute_in_simulation(
                simulation,
                HighLevelAction(
                    kind="place_block", agent="AgentA", block="oak_planks",
                    position=(1, 64, 0),
                ),
            )
            assert placed["success"]
            assert placed["observed_block"] == "oak_planks"
            assert placed["observed_by_craftground"] is True
            assert any(
                block["position"] == [1, 64, 0] and block["block"] == "minecraft:oak_planks"
                for block in observation["state"]["nearby_blocks"]
            )

            moved, _ = peers.execute_in_simulation(
                simulation,
                HighLevelAction(
                    kind="navigate", agent="AgentA", position=(5, 64, 0), radius=1.5
                ),
            )
            assert moved["success"]
            assert moved["after"]["position"][0] > 2

    rows = list(TrajectoryDataset(tmp_path))
    peer_rows = [row for row in rows if "peer_action" in row["action"]]
    assert {row["action"]["peer_action"]["kind"] for row in peer_rows} == {
        "craft", "place_block", "navigate"
    }
    assert any(row["action"]["peer_action"]["phase"] == "completed" for row in peer_rows)


@pytest.mark.skipif(
    os.environ.get("MCSOCIETY_RUN_MINECRAFT") != "1",
    reason="Set MCSOCIETY_RUN_MINECRAFT=1 to launch the real game",
)
def test_mindcraft_shared_world_through_rest_api(tmp_path):
    scenario = Scenario(
        id="mindcraft-api-live",
        name="Mindcraft REST integration",
        kind="survival",
        seed=1729,
        max_steps=1000,
        goal={"type": "survive", "duration_ticks": 5000},
    )
    app = create_app(
        config=EnvironmentConfig(width=96, height=64, render_distance=2, lan_port=55916),
        record_dir=tmp_path,
    )
    with TestClient(app) as client:
        assert client.post("/reset", json={"scenario": scenario.model_dump(mode="json")}).status_code == 200
        started = client.post("/agents", json={"agents": ["AgentA", "AgentB"]})
        assert started.status_code == 200, started.json()
        assert started.json()["ready"]["agents"] == ["AgentA", "AgentB"]
        result = client.post(
            "/agent-action",
            json={"kind": "say", "agent": "AgentA", "message": "hello from AgentA"},
        )
        assert result.status_code == 200, result.json()
        assert result.json()["receipt"]["success"] is True
        assert result.json()["info"]["peer_action"]["kind"] == "say"
        for _ in range(20):
            observed = client.get("/agents").json()
            assert "agents" in observed, observed
            if any(
                message["sender"] == "AgentA" and message["content"] == "hello from AgentA"
                for message in observed["agents"]["AgentB"]["messages"]
            ):
                break
            client.post("/step", json={})
        assert any(
            message["sender"] == "AgentA" and message["content"] == "hello from AgentA"
            for message in observed["agents"]["AgentB"]["messages"]
        )


@pytest.mark.skipif(
    os.environ.get("MCSOCIETY_RUN_MINECRAFT") != "1",
    reason="Set MCSOCIETY_RUN_MINECRAFT=1 to launch the real game",
)
def test_minecollab_techtree_uses_existing_chest_and_validator(tmp_path):
    scenario = Scenario(
        id="mindcraft-give-live",
        name="MineCollab two-agent techtree integration",
        kind="survival",
        seed=1730,
        max_steps=1800,
        goal={"type": "survive", "duration_ticks": 10000},
    )
    config = EnvironmentConfig(width=96, height=64, render_distance=2, lan_port=55916)
    with Simulation(SupervisedBackend(config), record_dir=tmp_path) as simulation:
        simulation.reset(scenario)
        with MindcraftBridge(lan_port=55916, agents=("AgentA", "AgentB")) as peers:
            peers.start(pump=lambda: simulation.step(Action()))
            simulation.intervene(WorldEdit(operation="teleport_player", target="AgentA", position=(0, 64, 3)))
            simulation.intervene(WorldEdit(operation="teleport_player", target="AgentB", position=(0, 64, 8)))
            simulation.intervene(WorldEdit(operation="place_block", position=(0, 64, 6), name="chest"))
            task = peers.configure_task(
                simulation, "multiagent_crafting_tasks.json", "multiagent_techtree_1_shears"
            )
            assert task["target"] == "shears"
            assert task["assisted_setup"] is True
            before = peers.evaluate_task(simulation)
            assert before["success"] is False
            assert before["server_verified"] is True
            assert before["server_success"] is False
            deposited, _ = peers.execute_in_simulation(
                simulation,
                HighLevelAction(
                    kind="deposit", agent="AgentA", item="iron_ingot", count=1,
                ),
                timeout=90,
            )
            assert deposited["success"], deposited["log"]
            withdrawn, _ = peers.execute_in_simulation(
                simulation,
                HighLevelAction(kind="withdraw", agent="AgentB", item="iron_ingot", count=1),
                timeout=90,
            )
            assert withdrawn["success"], withdrawn["log"]
            for _ in range(30):
                state = peers.observe(pump=lambda: simulation.step(Action()))
                if state["AgentB"]["inventory"].get("iron_ingot", 0) >= 1:
                    break
                simulation.step(Action())
            assert state["AgentB"]["inventory"].get("iron_ingot", 0) >= 2, withdrawn["log"]
            crafted, _ = peers.execute_in_simulation(
                simulation,
                HighLevelAction(kind="craft", agent="AgentB", item="shears", count=1),
            )
            assert crafted["success"], crafted["log"]
            for _ in range(10):
                evaluation = peers.evaluate_task(simulation)
                if evaluation["success"] and evaluation["server_success"]:
                    break
                simulation.step(Action())
            assert evaluation["success"] is True
            assert evaluation["agents"]["AgentB"]["valid"] is True
            assert evaluation["server_verified"] is True
            assert evaluation["server_success"] is True
            assert evaluation["server_counts"]["AgentB"] >= 1
