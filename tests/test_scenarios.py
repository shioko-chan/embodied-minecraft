import random
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcsociety.scenarios import Difficulty, Scenario, ScheduledEvent, generate_scenario


@pytest.mark.parametrize("kind", ["survival", "exploration", "social", "building"])
@pytest.mark.parametrize("biome", ["forest", "desert"])
def test_seeded_scenarios_generate_real_commands(kind, biome):
    scenario = generate_scenario(kind, 12, biome, adaptation=True)
    assert scenario == generate_scenario(kind, 12, biome, adaptation=True)
    assert scenario.split == ("train" if biome == "forest" else "test")
    commands = scenario.setup_commands("TestAgent")
    assert "gamemode survival TestAgent" in commands
    assert f"fillbiome -16 62 -16 16 79 16 minecraft:{biome}" in commands
    assert not any("required_blocks" in command for command in commands)
    assert "fill -3 63 5 3 63 7 minecraft:water" in commands
    assert scenario.scheduled_commands(29) == []
    assert scenario.scheduled_commands(30) == [
        "fill -3 63 5 3 63 7 minecraft:ice replace minecraft:water"
    ]
    assert scenario.scheduled_commands(31) == []


def test_generator_does_not_change_global_random():
    random.seed(42)
    state = random.getstate()
    generate_scenario("exploration", 120)
    assert random.getstate() == state


def test_setup_loads_arena_and_protects_player_before_editing_world():
    commands = generate_scenario("survival", 12).setup_commands("TestAgent")
    first_fill = next(i for i, command in enumerate(commands) if command.startswith("fill "))
    assert commands.index("forceload add -17 -17 17 17") < first_fill
    assert commands.index("gamemode spectator TestAgent") < first_fill
    assert commands.index("tp TestAgent 0 64 0 0 0") < first_fill
    assert commands.index("gamemode survival TestAgent") > first_fill
    assert "gamerule doImmediateRespawn false" in commands


def test_arena_clears_entire_column_top_down_in_legal_fill_batches():
    commands = generate_scenario("exploration", 123).setup_commands("TestAgent")
    cleared_heights = []
    top_heights = []
    for command in commands:
        parts = command.split()
        if parts[0] != "fill":
            continue
        x1, y1, z1, x2, y2, z2 = map(int, parts[1:7])
        volume = (abs(x2 - x1) + 1) * (abs(y2 - y1) + 1) * (abs(z2 - z1) + 1)
        assert volume <= 32768
        if parts[7] == "minecraft:air":
            assert (x1, z1, x2, z2) == (-17, -17, 17, 17)
            assert y1 <= y2
            cleared_heights.extend(range(y1, y2 + 1))
            top_heights.append(y2)
    assert sorted(cleared_heights) == list(range(64, 320))
    assert top_heights == sorted(top_heights, reverse=True)
    assert len(top_heights) == 10
    assert "kill @e[type=!minecraft:player,x=-17,y=62,z=-17,dx=34,dy=257,dz=34]" in commands


def test_building_is_not_completed_during_setup():
    scenario = generate_scenario("building", 0)
    commands = scenario.setup_commands("Builder")
    assert "give Builder minecraft:oak_planks 31" in commands
    assert not any(
        command.startswith(("setblock", "fill ")) and "oak_planks" in command
        for command in commands
    )
    assert len(scenario.goal["required_blocks"]) == 23
    required = {tuple(block["position"]) for block in scenario.goal["required_blocks"]}
    xs, zs = [p[0] for p in required], [p[2] for p in required]
    center_x, front_z = min(xs) + 1, min(zs)
    assert (center_x, 64, front_z) not in required
    assert (center_x, 65, front_z) not in required
    assert (center_x, 66, front_z) in required


def test_difficulty_changes_world_not_only_metadata():
    low = generate_scenario(
        "survival", 1, difficulty={"resources": "low", "mobs": "low"}
    ).setup_commands("A")
    high = generate_scenario(
        "survival", 1, difficulty={"resources": "high", "mobs": "high", "weather": "rain"}
    ).setup_commands("A")
    assert "weather rain" in high
    assert "give A minecraft:bread 2" in low
    assert "give A minecraft:bread 12" in high
    assert sum("summon minecraft:husk" in command for command in high) == 3
    assert not any("summon minecraft:husk" in command for command in low)


def test_yaml_round_trip(tmp_path):
    scenario = generate_scenario("building", 123, "desert", adaptation=True)
    path = tmp_path / "scenario.yaml"
    scenario.to_yaml(path)
    assert "task:" in path.read_text()
    assert Scenario.from_yaml(path) == scenario


def test_yaml_duplicate_fields_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("kind: survival\ntask:\n  kind: social\n")
    with pytest.raises(ValueError, match="duplicate"):
        Scenario.from_yaml(path)


def test_bundled_yaml_scenarios_are_valid():
    paths = list((Path(__file__).resolve().parents[1] / "scenarios").glob("*.yaml"))
    scenarios = [Scenario.from_yaml(path) for path in paths]
    assert {scenario.kind for scenario in scenarios} == {
        "survival",
        "exploration",
        "building",
        "social",
    }
    assert {scenario.split for scenario in scenarios} == {"train", "test"}
    assert any(scenario.events for scenario in scenarios)


def test_invalid_scenarios_fail_early():
    with pytest.raises(ValidationError):
        Difficulty(weather="snow")
    with pytest.raises(ValidationError, match="bounds"):
        ScheduledEvent(step=1, bounds=((-999, 63, 0), (0, 63, 1)))
    with pytest.raises(ValidationError, match="survival"):
        Scenario(id="bad", name="Bad", kind="survival", seed=0, goal={"type": "reach"})
    with pytest.raises(ValueError, match="unsupported"):
        generate_scenario("unknown", 1)
