"""Seeded task generation and concrete commands for a bounded Minecraft arena.

These commands are privileged *environment setup*. Agent actions run separately in
survival mode; placing a requested building in setup would invalidate the task.
"""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ScenarioKind = Literal["survival", "exploration", "social", "building"]
Biome = Literal["forest", "desert"]
Position = tuple[int, int, int]


class Difficulty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weather: Literal["clear", "rain", "thunder"] = "clear"
    resources: Literal["low", "normal", "high"] = "normal"
    mobs: Literal["low", "normal", "high"] = "low"


class ScheduledEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1)
    kind: Literal["water_freezes"] = "water_freezes"
    bounds: tuple[Position, Position] = ((-3, 63, 5), (3, 63, 7))

    @field_validator("bounds")
    @classmethod
    def bounded_event(cls, bounds: tuple[Position, Position]) -> tuple[Position, Position]:
        for x, y, z in bounds:
            if not (-16 <= x <= 16 and 62 <= y <= 79 and -16 <= z <= 16):
                raise ValueError("event bounds must stay inside the scenario arena")
        if any(a > b for a, b in zip(*bounds)):
            raise ValueError("event bounds must be ordered from minimum to maximum")
        return bounds

    def commands(self) -> list[str]:
        start, end = (_xyz(position) for position in self.bounds)
        return [f"fill {start} {end} minecraft:ice replace minecraft:water"]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: ScenarioKind
    seed: int
    biome: Biome = "forest"
    max_steps: int = Field(default=300, ge=1)
    start: Position = (0, 64, 0)
    goal: dict[str, Any]
    difficulty: Difficulty = Field(default_factory=Difficulty)
    events: list[ScheduledEvent] = Field(default_factory=list)

    @field_validator("start")
    @classmethod
    def bounded_start(cls, value: Position) -> Position:
        x, y, z = value
        if not (-15 <= x <= 15 and 64 <= y <= 76 and -15 <= z <= 15):
            raise ValueError("start must be inside the arena above its floor")
        return value

    @model_validator(mode="after")
    def valid_goal(self) -> Scenario:
        expected = {
            "exploration": "reach",
            "survival": "survive",
            "building": "build",
            "social": "cooperate",
        }[self.kind]
        if self.goal.get("type") != expected:
            raise ValueError(f"{self.kind} goal type must be {expected!r}")
        if self.kind == "exploration":
            target = self.goal.get("target")
            if (
                not isinstance(target, (list, tuple))
                or len(target) != 3
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in target)
            ):
                raise ValueError("reach goal requires a target xyz")
            x, y, z = target
            if not (-15 <= x <= 15 and 64 <= y <= 76 and -15 <= z <= 15):
                raise ValueError("reach target must stay inside the arena")
            radius = self.goal.get("radius", 1.5)
            if not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
                raise ValueError("reach radius must be positive")
        elif self.kind == "survival":
            duration = self.goal.get("duration_ticks")
            if not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
                raise ValueError("survival requires positive duration_ticks")
            minimum_health = self.goal.get("minimum_health", 1)
            if (
                not isinstance(minimum_health, (int, float))
                or not math.isfinite(minimum_health)
                or not 0 < minimum_health <= 20
            ):
                raise ValueError("minimum_health must be positive and at most 20")
        elif self.kind == "building":
            required = self.goal.get("required_blocks")
            if not isinstance(required, list) or not required:
                raise ValueError("building requires explicit required_blocks")
            positions = []
            for block in required:
                position = block.get("position", [])
                if (
                    len(position) != 3
                    or not all(isinstance(x, int) and not isinstance(x, bool) for x in position)
                    or not _valid_resource(block.get("block"))
                ):
                    raise ValueError("required_blocks must contain position xyz and block id")
                x, y, z = position
                if not (-15 <= x <= 15 and 64 <= y <= 76 and -15 <= z <= 15):
                    raise ValueError("required building blocks must be inside the arena")
                positions.append(tuple(position))
            if len(positions) != len(set(positions)):
                raise ValueError("required block coordinates must be unique")
            for position in self.goal.get("required_empty", []):
                if (
                    not isinstance(position, (list, tuple))
                    or len(position) != 3
                    or not all(isinstance(x, int) and not isinstance(x, bool) for x in position)
                ):
                    raise ValueError("required_empty must contain integer xyz positions")
                if tuple(position) in positions:
                    raise ValueError("a required block cannot also be required_empty")
        else:
            count = self.goal.get("count")
            if (
                not isinstance(self.goal.get("recipient"), str)
                or not self.goal["recipient"]
                or not _valid_resource(self.goal.get("item"))
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
            ):
                raise ValueError("social goal requires recipient, item and positive count")
        if any(event.step > self.max_steps for event in self.events):
            raise ValueError("scheduled event must occur within max_steps")
        return self

    @property
    def split(self) -> str:
        """A fixed domain split makes accidental train/test leakage visible."""
        return "train" if self.biome == "forest" else "test"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scenario:
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise TypeError("scenario YAML must be a mapping")
        # The task envelope follows the user-facing scenario authoring format.
        if "task" in data:
            task = data.pop("task")
            if not isinstance(task, dict):
                raise ValueError("task must be a mapping")
            overlap = set(task) & set(data)
            if overlap:
                raise ValueError(f"duplicate task fields: {sorted(overlap)}")
            data.update(task)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        data = self.model_dump(mode="json")
        task = {key: data.pop(key) for key in ("kind", "max_steps", "start", "goal")}
        data["task"] = task
        with Path(path).open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)

    def scheduled_commands(self, step: int) -> list[str]:
        """Return commands due on precisely this step; the caller executes once."""
        return [
            command for event in self.events if event.step == step for command in event.commands()
        ]

    def setup_commands(self, agent_selector: str = "@a[tag=mcsociety_agent]") -> list[str]:
        """Build a fresh 33×33 arena using Java Edition 1.21 commands.

        The selector must identify all participating agents, or the primary agent
        when the backend handles peer setup separately. Only arena entities are
        removed. This mutates the dedicated experiment world and its game rules.
        """
        if not agent_selector or any(c in agent_selector for c in "\r\n;"):
            raise ValueError("agent_selector must be a single command selector or player name")
        floor = "minecraft:grass_block" if self.biome == "forest" else "minecraft:sand"
        commands = [
            "gamerule doMobSpawning false",
            "gamerule doDaylightCycle false",
            "gamerule doWeatherCycle false",
            "gamerule doImmediateRespawn false",
            "gamerule keepInventory true",
            "gamerule mobGriefing false",
            "forceload add -17 -17 17 17",
            f"gamemode spectator {agent_selector}",
            f"tp {agent_selector} {_xyz(self.start)} 0 0",
            "kill @e[type=!minecraft:player,x=-17,y=62,z=-17,dx=34,dy=257,dz=34]",
            "fill -17 62 -17 17 62 17 minecraft:bedrock",
            f"fill -17 63 -17 17 63 17 {floor}",
            # Clear downward so floating natural gravel, mountains and water do
            # not remain over the arena. A 35×35×26 batch stays below Java's
            # default 32768-block fill limit; terrain outside the column stays.
            *[
                f"fill -17 {max(64, top - 25)} -17 17 {top} 17 minecraft:air"
                for top in range(319, 63, -26)
            ],
            "fill -17 64 -17 -17 70 17 minecraft:barrier",
            "fill 17 64 -17 17 70 17 minecraft:barrier",
            "fill -17 64 -17 17 70 -17 minecraft:barrier",
            "fill -17 64 17 17 70 17 minecraft:barrier",
            f"fillbiome -16 62 -16 16 79 16 minecraft:{self.biome}",
            "fill -3 63 5 3 63 7 minecraft:water",
            f"weather {self.difficulty.weather}",
            f"time set {'midnight' if self.difficulty.mobs == 'high' else 'day'}",
            f"gamemode survival {agent_selector}",
            f"clear {agent_selector}",
            f"effect clear {agent_selector}",
            f"effect give {agent_selector} minecraft:instant_health 1 10 true",
            f"effect give {agent_selector} minecraft:saturation 1 10 true",
            f"tp {agent_selector} {_xyz(self.start)} 0 0",
            f"give {agent_selector} minecraft:bread { {'low': 2, 'normal': 6, 'high': 12}[self.difficulty.resources] }",
        ]
        resource_count = {"low": 1, "normal": 3, "high": 5}[self.difficulty.resources]
        resource_sites = [(-10, -10), (10, -10), (-10, 10), (13, 3), (-13, 3)]
        for x, z in resource_sites[:resource_count]:
            if self.biome == "forest":
                commands += [
                    f"fill {x - 1} 67 {z - 1} {x + 1} 68 {z + 1} minecraft:oak_leaves[persistent=true]",
                    f"fill {x} 64 {z} {x} 67 {z} minecraft:oak_log",
                ]
            else:
                commands += [
                    f"fill {x} 64 {z} {x} 66 {z} minecraft:cactus",
                    f"setblock {x + 2} 64 {z} minecraft:dead_bush",
                    f"setblock {x - 2} 64 {z} minecraft:sandstone",
                ]
        commands += [
            'summon minecraft:cow -6 64 -5 {Tags:["mcsociety_scenario"]}',
            'summon minecraft:villager 6 64 -5 {Tags:["mcsociety_scenario"],PersistenceRequired:1b}',
        ]
        for x, z in [(13, 13), (-13, -13), (13, -13)][
            : {"low": 0, "normal": 1, "high": 3}[self.difficulty.mobs]
        ]:
            commands.append(
                f'summon minecraft:husk {x} 64 {z} {{Tags:["mcsociety_scenario"],PersistenceRequired:1b}}'
            )
        if self.kind == "exploration":
            x, y, z = self.goal["target"]
            commands.append(f"setblock {int(x)} {int(y) - 1} {int(z)} minecraft:gold_block")
        elif self.kind == "building":
            counts: dict[str, int] = {}
            for block in self.goal["required_blocks"]:
                counts[block["block"]] = counts.get(block["block"], 0) + 1
            for block, count in sorted(counts.items()):
                commands.append(f"give {agent_selector} {block} {count + 8}")
            commands.append(f"give {agent_selector} minecraft:wooden_axe 1")
        elif self.kind == "social":
            commands.append(f"give {agent_selector} {self.goal['item']} {self.goal['count']}")
        return commands


def _xyz(position: Position) -> str:
    return " ".join(str(int(coordinate)) for coordinate in position)


def _valid_resource(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"(?:[a-z0-9_.-]+:)?[a-z0-9_./-]+", value) is not None
    )


def generate_scenario(
    kind: ScenarioKind,
    seed: int,
    biome: Biome = "forest",
    difficulty: Difficulty | dict[str, Any] | None = None,
    *,
    adaptation: bool = False,
) -> Scenario:
    """Generate reproducible task parameters without touching global RNG state."""
    rng = random.Random(seed)
    if kind == "exploration":
        goal = {
            "type": "reach",
            "target": [rng.choice([-12, -8, 8, 12]), 64, rng.choice([-12, -8, 8, 12])],
            "radius": 1.5,
        }
    elif kind == "survival":
        goal = {"type": "survive", "duration_ticks": 400, "minimum_health": 1}
    elif kind == "social":
        goal = {
            "type": "cooperate",
            "recipient": "trader",
            "item": "minecraft:bread",
            "count": 2,
            "require_message": True,
        }
    elif kind == "building":
        # Standing room at y64..65, roof at y66, and a two-block doorway.
        origin_x, origin_z = rng.choice([(-6, -6), (4, -6), (4, 3)])
        blocks = []
        required_empty = []
        for y in (64, 65, 66):
            for dx in range(3):
                for dz in range(3):
                    if y < 66 and (dx == dz == 1 or (dx == 1 and dz == 0)):
                        required_empty.append([origin_x + dx, y, origin_z + dz])
                        continue
                    blocks.append(
                        {
                            "position": [origin_x + dx, y, origin_z + dz],
                            "block": "minecraft:oak_planks",
                        }
                    )
        goal = {
            "type": "build",
            "structure": "shelter",
            "required_blocks": blocks,
            "required_empty": required_empty,
        }
    else:
        raise ValueError(f"unsupported scenario kind: {kind}")
    return Scenario(
        id=f"{kind}-{biome}-{seed}{'-adaptation' if adaptation else ''}",
        name=f"{kind.title()} in {biome}",
        kind=kind,
        seed=seed,
        biome=biome,
        max_steps=600 if kind == "survival" else 300,
        goal=goal,
        difficulty=difficulty
        if isinstance(difficulty, Difficulty)
        else Difficulty.model_validate(difficulty or {}),
        events=[ScheduledEvent(step=30)] if adaptation else [],
    )
