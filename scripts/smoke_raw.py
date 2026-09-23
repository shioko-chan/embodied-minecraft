"""Exercise the actual CraftGround mod; save evidence without using the app adapter."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image


def smoke(output: Path, width: int = 320, height: int = 180) -> dict:
    from craftground import ActionSpaceVersion
    from craftground.environment.action_space import no_op_v2
    from craftground.initial_environment_config import (
        Difficulty,
        InitialEnvironmentConfig,
        WorldType,
    )

    from mcsociety.socket_runtime import make_socket_environment

    output.mkdir(parents=True, exist_ok=True)
    config = InitialEnvironmentConfig(
        image_width=width,
        image_height=height,
        world_type=WorldType.SUPERFLAT,
        difficulty=Difficulty.PEACEFUL,
        seed="1729",
        hud_hidden=True,
        render_distance=2,
        simulation_distance=5,
        requires_depth=True,
        requires_depth_conversion=False,
        requires_surrounding_blocks=True,
        requires_biome_info=True,
        surrounding_entity_distances=[16],
        initial_extra_commands=[
            "gamerule doMobSpawning false",
            "gamerule doDaylightCycle false",
            "time set day",
            "weather clear",
            "tp @p 0.5 -60 0.5 0 0",
            # A wall in the initial camera view makes depth non-constant.
            "fill -2 -60 5 2 -55 5 minecraft:stone",
            "summon minecraft:cow 3 -60 3",
        ],
    )
    started = time.monotonic()
    env = make_socket_environment(
        config,
        env_path=os.environ.get("MCSOCIETY_MINECRAFT_PATH"),
        port=18472,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        verbose_gradle=True,
        cleanup_world=True,
    )
    try:
        observation, _ = env.reset(seed=1729)
        for _ in range(5):
            observation, *_ = env.step(no_op_v2())
        first = observation["full"]
        first_depth = np.asarray(first.depth, dtype=np.float32)
        start = np.array([first.x, first.y, first.z])
        for _ in range(24):
            action = no_op_v2()
            action["forward"] = True
            observation, *_ = env.step(action)
        full = observation["full"]
        end = np.array([full.x, full.y, full.z])
        rgb = np.asarray(observation["pov"])
        depth = np.asarray(full.depth, dtype=np.float32)
        assert rgb.shape == (height, width, 3), rgb.shape
        assert rgb.dtype == np.uint8, rgb.dtype
        assert float(rgb.std()) > 1, "RGB frame is blank"
        assert depth.size == height * width, depth.size
        assert first_depth.size == height * width, first_depth.size
        assert np.isfinite(depth).all(), "Depth contains non-finite values"
        assert float(depth.max() - depth.min()) > 1e-5, (
            f"Depth frame is constant: {float(depth.min())}..{float(depth.max())}"
        )
        center = (height // 2) * width + (width // 2)
        start_center_depth = float(first_depth[center])
        end_center_depth = float(depth[center])
        assert end_center_depth < start_center_depth - 0.01, (
            f"Depth did not decrease while approaching the wall: "
            f"{start_center_depth} -> {end_center_depth}"
        )
        distance = float(np.linalg.norm((end - start)[[0, 2]]))
        assert distance > 0.1, f"Agent did not move: {start} -> {end}"
        Image.fromarray(rgb).save(output / "rgb.png")
        # OpenGL reads bottom row first; align raw depth with RGB for consumers.
        np.save(output / "depth_opengl.npy", depth.reshape(height, width)[::-1].copy())
        report = {
            "craftground_version": importlib.metadata.version("craftground"),
            "minecraft_version": "1.21",
            "transport": "unix_socket",
            "rgb_shape": list(rgb.shape),
            "rgb_std": float(rgb.std()),
            "depth_shape": [height, width],
            "depth_range": [float(depth.min()), float(depth.max())],
            "wall_center_depth_before_after": [start_center_depth, end_center_depth],
            "depth_encoding": "opengl_depth_0_1",
            "start_position": start.tolist(),
            "end_position": end.tolist(),
            "horizontal_distance": distance,
            "world_time": full.world_time,
            "health": full.health,
            "surrounding_blocks": len(full.surrounding_blocks),
            "elapsed_seconds": time.monotonic() - started,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".runtime/evidence/raw"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    args = parser.parse_args()
    smoke(args.output, args.width, args.height)
