"""Check whether CraftGround's integrated world accepts standard Minecraft bots."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from craftground import ActionSpaceVersion
from craftground.environment.action_space import no_op_v2
from craftground.initial_environment_config import Difficulty, InitialEnvironmentConfig, WorldType

from mcsociety.socket_runtime import make_socket_environment

LAN_PORT = 55916


def main() -> None:
    use_mindcraft = sys.argv[1:] == ["--mindcraft"]
    if sys.argv[1:] and not use_mindcraft:
        raise SystemExit("Usage: probe_shared_world.py [--mindcraft]")
    config = InitialEnvironmentConfig(
        image_width=96,
        image_height=64,
        world_type=WorldType.SUPERFLAT,
        difficulty=Difficulty.PEACEFUL,
        seed="1729",
        render_distance=2,
        simulation_distance=5,
        requires_surrounding_blocks=True,
        initial_extra_commands=[
            "gamerule doMobSpawning false",
            "gamerule doDaylightCycle false",
            "time set day",
            f"publish true survival {LAN_PORT}",
        ],
    )
    env = make_socket_environment(
        config,
        env_path=os.environ.get("MCSOCIETY_MINECRAFT_PATH"),
        port=18472,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        cleanup_world=True,
    )
    try:
        observation, _ = env.reset(seed=1729)
        for _ in range(10):
            observation, *_ = env.step(no_op_v2())
        with socket.create_connection(("127.0.0.1", LAN_PORT), timeout=2):
            pass
        print(json.dumps({
            "lan_port": LAN_PORT,
            "tcp_listening": True,
            "world_time": observation["full"].world_time,
        }), flush=True)
        bot = subprocess.Popen(
            ["node", "scripts/probe_mindcraft.mjs" if use_mindcraft else "scripts/probe_mineflayer.cjs"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert bot.stdout is not None
        os.set_blocking(bot.stdout.fileno(), False)
        chunks: list[bytes] = []
        gave_stone = False
        checked_crafting = False
        server_inventory_messages: list[str] = []
        deadline = time.monotonic() + (50 if use_mindcraft else 35)
        try:
            while bot.poll() is None and time.monotonic() < deadline:
                observation, *_ = env.step(no_op_v2())
                try:
                    chunk = os.read(bot.stdout.fileno(), 65536)
                except BlockingIOError:
                    chunk = b""
                if chunk:
                    chunks.append(chunk)
                if not gave_stone and b'mineflayer_spawned' in b''.join(chunks):
                    env.add_commands([
                        "give ProbeA minecraft:stone 1",
                        "give ProbeA minecraft:oak_log 1",
                    ])
                    gave_stone = True
                if not checked_crafting and b'mineflayer_crafting' in b''.join(chunks):
                    env.add_commands([
                        "clear ProbeA minecraft:oak_log 0",
                        "clear ProbeA minecraft:oak_planks 0",
                    ])
                    checked_crafting = True
                server_inventory_messages.extend(
                    message.message
                    for message in observation["full"].chat_messages
                    if ("matching item" in message.message
                        or "No items were found" in message.message)
                )
            if bot.poll() is None:
                raise TimeoutError("Minecraft bots did not finish before the probe deadline")
            remaining, _ = bot.communicate(timeout=2)
            output = b''.join(chunks).decode(errors="replace") + remaining
            print(output, end="", flush=True)
            if (bot.returncode != 0 or "mineflayer_spawned" not in output
                    or "mineflayer_action" not in output
                    or '"block_observed":"stone"' not in output
                    or '"plank_count":4' not in output):
                raise RuntimeError(f"Mineflayer join/action/place/craft failed (exit {bot.returncode})")
            action = next(
                json.loads(line)["mineflayer_action"]
                for line in output.splitlines()
                if line.startswith('{"mineflayer_action":')
            )
            if use_mindcraft and (action.get("reached") is not True
                                  or action.get("distance", 0) < 2):
                raise RuntimeError("Mindcraft navigation skill did not reach its goal")
            placement = next(
                json.loads(line)["mineflayer_placement"]
                for line in output.splitlines()
                if line.startswith('{"mineflayer_placement":')
            )
            x, y, z = placement["position"]
            env.add_commands([f"tp @s {x + 0.5} {y} {z - 0.5} 0 0"])
            for _ in range(5):
                observation, *_ = env.step(no_op_v2())
            same_block = any(
                (block.x, block.y, block.z) == (x, y, z)
                and block.translation_key.endswith(".stone")
                for block in observation["full"].surrounding_blocks
            )
            print(json.dumps({
                "shared_world_block_observed": same_block,
                "stone_position": [x, y, z],
            }), flush=True)
            print(json.dumps({
                "server_inventory_messages": list(dict.fromkeys(server_inventory_messages)),
            }), flush=True)
            messages = list(dict.fromkeys(server_inventory_messages))
            if (not any("No items were found on player ProbeA" in item for item in messages)
                    or not any("Found 4 matching item(s)" in item for item in messages)):
                raise RuntimeError("Server did not confirm one log consumed and four planks made")
            spawned = next(
                json.loads(line)["mineflayer_spawned"]
                for line in output.splitlines()
                if line.startswith('{"mineflayer_spawned":')
            )
            result = {
                "minecraft": "1.21",
                "mineflayer": json.loads(Path(
                    ".runtime/mindcraft-deps/node_modules/mineflayer/package.json"
                    if use_mindcraft else ".runtime/probe-node/node_modules/mineflayer/package.json"
                ).read_text())["version"],
                "bots": len(spawned),
                "framework": "Mindcraft skills" if use_mindcraft else "Mineflayer",
                "mindcraft_revision": (
                    "5f3acc87b479864124173de444f31fa5538f94a6"
                    if use_mindcraft else None
                ),
                "navigation_distance": action["distance"],
                "reused_skills": (
                    ["goToPosition", "craftRecipe", "placeBlock"] if use_mindcraft else []
                ),
                "placement_observed_by_craftground": same_block,
                "server_inventory_messages": messages,
                "protocol_partial_read_errors": sum(
                    line.startswith("PartialReadError:") for line in output.splitlines()
                ),
            }
            path = Path(
                ".runtime/evidence/mindcraft/report.json" if use_mindcraft
                else ".runtime/evidence/shared_world/report.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
            if not same_block:
                raise RuntimeError("CraftGround did not observe Mineflayer's placed block")
            print(json.dumps({
                "minecraft_ticks_after_join": observation["full"].world_time,
            }), flush=True)
        finally:
            if bot.poll() is None:
                bot.terminate()
                bot.wait(timeout=5)
    finally:
        env.close()


if __name__ == "__main__":
    main()
