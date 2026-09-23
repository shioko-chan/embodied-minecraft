"""Opt-in end-to-end check of the real Minecraft-backed application path."""

from __future__ import annotations

import os

import numpy as np
import pytest

from mcsociety.dataset import TrajectoryDataset
from mcsociety.environment import Simulation
from mcsociety.models import Action, EnvironmentConfig
from mcsociety.scenarios import Difficulty, Scenario
from mcsociety.worker_backend import SupervisedBackend

pytestmark = pytest.mark.minecraft


@pytest.mark.skipif(
    os.environ.get("MCSOCIETY_RUN_MINECRAFT") != "1",
    reason="Set MCSOCIETY_RUN_MINECRAFT=1 to launch the real game",
)
def test_real_rgb_depth_action_and_transition(tmp_path):
    scenario = Scenario(
        id="live-smoke",
        name="Live Minecraft integration smoke",
        kind="survival",
        seed=1729,
        max_steps=4,
        goal={"type": "survive", "duration_ticks": 4},
        difficulty=Difficulty(weather="clear", resources="normal", mobs="low"),
    )
    backend = SupervisedBackend(EnvironmentConfig(width=96, height=64, render_distance=2))
    with Simulation(backend, record_dir=tmp_path) as environment:
        first, _ = environment.reset(scenario)
        assert first["state"]["position"] == [0.5, 64.0, 0.5]
        assert first["state"]["biome"] == "minecraft:forest"
        assert any(
            block["position"] == [0, 63, 0]
            and block["block"] == "minecraft:grass_block"
            for block in first["state"]["nearby_blocks"]
        )
        second, _, _, _, _ = environment.step(Action(forward=True))
        assert first["rgb"].shape == second["rgb"].shape == (64, 96, 3)
        assert first["depth"].shape == second["depth"].shape == (64, 96)
        assert np.isfinite(second["depth"]).all()
        assert first["state"]["tick"] == 0
        assert second["state"]["tick"] == 1
        assert second["state"]["position"] != first["state"]["position"]
    records = list(TrajectoryDataset(tmp_path))
    assert len(records) == 1
    assert records[0]["observation"]["state"]["tick"] == 0
    assert records[0]["next_observation"]["state"]["tick"] == 1


@pytest.mark.skipif(
    os.environ.get("MCSOCIETY_RUN_MINECRAFT") != "1",
    reason="Set MCSOCIETY_RUN_MINECRAFT=1 to launch the real game",
)
def test_desert_domain_changes_native_biome_and_floor():
    scenario = Scenario(
        id="desert-domain-live",
        name="Desert test domain",
        kind="survival",
        seed=1729,
        biome="desert",
        max_steps=2,
        goal={"type": "survive", "duration_ticks": 4},
    )
    with Simulation(
        SupervisedBackend(EnvironmentConfig(width=96, height=64, render_distance=2)),
        record_dir=None,
    ) as environment:
        observation, _ = environment.reset(scenario)
        assert observation["state"]["biome"] == "minecraft:desert"
        assert any(
            block["position"] == [0, 63, 0]
            and block["block"] == "minecraft:sand"
            for block in observation["state"]["nearby_blocks"]
        )
