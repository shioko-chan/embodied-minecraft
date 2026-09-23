"""Backend-neutral actions and configuration; coordinates use Minecraft's XYZ axes."""

import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Action(Model):
    """Apply controls for a bounded number of simulation ticks.

    Rotation deltas are in degrees and applied ONCE, on the first tick.
    No operating-system keyboard or mouse events are generated.
    """

    forward: bool = False
    back: bool = False
    left: bool = False
    right: bool = False
    jump: bool = False
    sneak: bool = False
    sprint: bool = False
    attack: bool = False
    use: bool = False
    drop: bool = False
    inventory: bool = False
    yaw: float = Field(default=0, ge=-180, le=180)
    pitch: float = Field(default=0, ge=-180, le=180)
    hotbar: int | None = Field(default=None, ge=1, le=9)
    ticks: int = Field(default=1, ge=1, le=200, strict=True)

    @model_validator(mode="after")
    def validate_directions(self):
        if (self.forward and self.back) or (self.left and self.right):
            raise ValueError("Opposing movement controls cannot be enabled together")
        return self

    def controls(self, *, first_tick: bool = True) -> dict:
        controls = {
            key: getattr(self, key)
            for key in (
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
        }
        # CraftGround's public converter expects dotted hotbar keys.
        controls.update({f"hotbar.{i}": first_tick and self.hotbar == i for i in range(1, 10)})
        controls["camera_yaw"] = self.yaw if first_tick else 0.0
        controls["camera_pitch"] = self.pitch if first_tick else 0.0
        # Inventory is a toggle and drop is a discrete operation.
        controls["inventory"] = self.inventory and first_tick
        controls["drop"] = self.drop and first_tick
        return controls


class EnvironmentConfig(Model):
    width: int = Field(default=320, ge=64, le=1920)
    height: int = Field(default=180, ge=64, le=1080)
    depth: bool = True
    segmentation: bool = False
    render_distance: int = Field(default=4, ge=2, le=16)
    port: int = Field(default=8000, ge=1024, le=65535)
    lan_port: int | None = Field(default=None, ge=1024, le=65535)
    runtime_path: str | None = None
    verbose: bool = False

    @model_validator(mode="after")
    def distinct_ports(self):
        if self.lan_port == self.port:
            raise ValueError("lan_port must differ from the CraftGround IPC port")
        return self


_MINECRAFT_NAME = re.compile(r"^[A-Za-z0-9_]{1,16}$")
_ITEM_NAME = re.compile(r"^[a-z0-9_]+$")


class HighLevelAction(Model):
    """A bounded request for an existing Mindcraft survival-mode skill."""

    kind: Literal["navigate", "craft", "place_block", "deposit", "withdraw", "say"]
    agent: str
    position: tuple[float, float, float] | None = None
    radius: float = Field(default=1.5, gt=0, le=16)
    item: str | None = None
    block: str | None = None
    count: int = Field(default=1, ge=1, le=64, strict=True)
    message: str | None = None

    @model_validator(mode="after")
    def valid_skill(self):
        if not _MINECRAFT_NAME.fullmatch(self.agent):
            raise ValueError("agent must be a Minecraft username")
        if self.kind in {"navigate", "place_block"}:
            if self.position is None:
                raise ValueError(f"{self.kind} requires position")
            x, y, z = self.position
            if (not all(math.isfinite(value) for value in (x, y, z))
                    or abs(x) > 29_999_984 or abs(z) > 29_999_984 or not -64 <= y < 320):
                raise ValueError("position is outside Minecraft world bounds")
            if self.kind == "place_block" and not all(
                value.is_integer() for value in (x, y, z)
            ):
                raise ValueError("place_block coordinates must be integers")
        if self.kind == "craft" and not self.item:
            raise ValueError("craft requires item")
        if self.kind in {"deposit", "withdraw"} and not self.item:
            raise ValueError(f"{self.kind} requires item")
        if self.kind == "place_block" and not self.block:
            raise ValueError("place_block requires block")
        if self.kind == "say" and (
            not self.message or len(self.message) > 256 or self.message.startswith("/")
            or any(character in self.message for character in "\r\n")
        ):
            raise ValueError("say requires a non-command message of at most 256 characters")
        for value in (self.item, self.block):
            if value is not None and not _ITEM_NAME.fullmatch(value):
                raise ValueError("item and block must be unqualified Minecraft names")
        return self


class WorldEdit(Model):
    """Privileged experiment interventions, deliberately separate from policy actions."""

    operation: Literal[
        "place_block", "remove_block", "spawn", "remove_entity", "weather", "time", "rule",
        "teleport_player", "give_item", "clear_inventory",
    ]
    position: tuple[int, int, int] | None = None
    name: str | None = None
    target: str | None = None
    value: str | int | bool | None = None
