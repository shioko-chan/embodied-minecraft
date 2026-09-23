"""Typed Minecraft experiment interventions; not survival-mode agent skills."""

import re
import uuid

from .models import WorldEdit

_RESOURCE = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$")
_USERNAME = re.compile(r"^[A-Za-z0-9_]{1,16}$")
_BOOLEAN_RULES = {
    "doDaylightCycle",
    "doWeatherCycle",
    "doMobSpawning",
    "mobGriefing",
    "keepInventory",
    "doFireTick",
    "naturalRegeneration",
    "doImmediateRespawn",
}
_INTEGER_RULES = {"randomTickSpeed", "spawnRadius", "maxEntityCramming"}


def resource_id(name: str | None) -> str:
    if not name:
        raise ValueError("A Minecraft resource id is required")
    name = name if ":" in name else f"minecraft:{name}"
    if not _RESOURCE.fullmatch(name):
        raise ValueError("Invalid resource id")
    return name


def inventory_count_command(player: str, item: str) -> str:
    """Vanilla `/clear <player> <item> 0` counts without removing items."""
    if not _USERNAME.fullmatch(player):
        raise ValueError("inventory query requires a Minecraft username")
    return f"clear {player} {resource_id(item)} 0"


def edit_commands(edit: WorldEdit) -> list[str]:
    op = edit.operation
    if op in ("place_block", "remove_block", "spawn", "teleport_player"):
        if edit.position is None:
            raise ValueError(f"{op} requires position")
        x, y, z = edit.position
        if abs(x) > 29_999_984 or abs(z) > 29_999_984 or not -64 <= y < 320:
            raise ValueError("Position is outside the overworld build bounds")
        pos = f"{x} {y} {z}"
        if op == "teleport_player":
            if not edit.target or not _USERNAME.fullmatch(edit.target):
                raise ValueError("teleport_player requires a Minecraft username target")
            return [f"tp {edit.target} {pos}"]
        if op == "spawn":
            return [f"summon {resource_id(edit.name)} {pos}"]
        block = "minecraft:air" if op == "remove_block" else resource_id(edit.name)
        return [f"setblock {pos} {block}"]
    if op == "give_item":
        if not edit.target or not _USERNAME.fullmatch(edit.target):
            raise ValueError("give_item requires a Minecraft username target")
        if type(edit.value) is not int or not 1 <= edit.value <= 2304:
            raise ValueError("give_item requires a positive count at most 2304")
        return [f"give {edit.target} {resource_id(edit.name)} {edit.value}"]
    if op == "clear_inventory":
        if not edit.target or not _USERNAME.fullmatch(edit.target):
            raise ValueError("clear_inventory requires a Minecraft username target")
        return [f"clear {edit.target}"]
    if op == "remove_entity":
        entity_id = str(uuid.UUID(edit.name or ""))
        return [f"kill {entity_id}"]
    if op == "weather":
        if edit.value not in ("clear", "rain", "thunder"):
            raise ValueError("Weather must be clear, rain or thunder")
        return [f"weather {edit.value}"]
    if op == "time":
        if type(edit.value) is not int or not 0 <= edit.value <= 2_147_483_647:
            raise ValueError("Time must be a nonnegative 32-bit tick count")
        return [f"time set {edit.value}"]
    if edit.name in _BOOLEAN_RULES and type(edit.value) is bool:
        return [f"gamerule {edit.name} {str(edit.value).lower()}"]
    if edit.name in _INTEGER_RULES and type(edit.value) is int and 0 <= edit.value <= 4096:
        return [f"gamerule {edit.name} {edit.value}"]
    raise ValueError("Unsupported rule or invalid rule value")
