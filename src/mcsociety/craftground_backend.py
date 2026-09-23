"""Adapter for the pinned CraftGround 2.7.4 Minecraft 1.21 runtime.

RGB and depth come from the real game renderer. Missing capabilities fail
explicitly; synthetic frames are never substituted for Minecraft observations.
"""

import os
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .models import Action, EnvironmentConfig


class CapabilityError(RuntimeError):
    pass


def _identifier(translation_key: str) -> str:
    parts = translation_key.split(".", 2)
    if len(parts) == 3 and parts[0] in ("block", "item", "entity"):
        return f"{parts[1]}:{parts[2]}"
    return translation_key


def _position(obj) -> list[float]:
    return [obj.x, obj.y, obj.z]


def decode_observation(raw: dict, config: EnvironmentConfig, tick: int) -> dict:
    """Normalize upstream protobuf state while preserving sensor provenance."""
    full = raw["full"]
    rgb = np.asarray(raw["pov"])
    if rgb.shape != (config.height, config.width, 3) or rgb.dtype != np.uint8:
        raise RuntimeError(f"Unexpected RGB shape/dtype: {rgb.shape}/{rgb.dtype}")
    entities = {}
    for group in full.surrounding_entities.values():
        for entity in group.entities:
            entities[entity.unique_name] = {
                "id": entity.unique_name,
                "type": _identifier(entity.translation_key),
                "position": _position(entity),
                "health": entity.health,
            }
    state = {
        "agent_id": "agent_0",
        "tick": tick,
        "position": _position(full),
        "velocity": [full.velocity_x, full.velocity_y, full.velocity_z],
        "yaw": full.yaw,
        "pitch": full.pitch,
        "health": full.health,
        "food": full.food_level,
        "saturation": full.saturation_level,
        "dead": full.is_dead,
        "on_ground": full.is_on_ground,
        "inventory": [
            {
                "slot": i,
                "item": _identifier(item.translation_key),
                "count": item.count,
                "durability": item.durability,
                "max_durability": item.max_durability,
            }
            for i, item in enumerate(full.inventory)
            if item.count
        ],
        "nearby_blocks": [
            {"position": _position(block), "block": _identifier(block.translation_key)}
            for block in full.surrounding_blocks
        ],
        "entities": list(entities.values()),
        "time": full.world_time,
        "biome": full.biome_info.biome_name,
        # The upstream protocol has no weather field. Do not misrepresent the
        # requested setup weather as an observed weather sensor.
        "weather": None,
        "heightmap": [
            {"x": item.x, "z": item.z, "height": item.height, "block": item.block_name}
            for item in full.height_info
        ],
        "messages": [
            {"content": message.message, "time": message.added_time}
            for message in full.chat_messages
        ],
        "raycast": {"type": full.raycast_result.type},
    }
    observation = {
        "rgb": rgb.copy(),
        "state": state,
        "sensors": {"rgb": "minecraft_framebuffer", "block_extent": "3x3x3"},
    }
    if config.depth:
        depth = np.asarray(full.depth, dtype=np.float32)
        if depth.size != config.width * config.height:
            raise CapabilityError(
                f"Requested depth has {depth.size} pixels; expected {config.width * config.height}. "
                "Check native framebuffer resolution and runtime depth support."
            )
        if not np.isfinite(depth).all() or np.any((depth < 0) | (depth > 1)):
            raise RuntimeError("Invalid native OpenGL depth buffer")
        observation["depth"] = np.flipud(depth.reshape(config.height, config.width)).copy()
        # A nonlinear OpenGL z buffer is not metric distance. Native conversion
        # uses an assumed far plane; retain raw depth for honest interpretation.
        observation["sensors"]["depth"] = "opengl_depth_0_1_nonlinear"
    return observation


class CraftGroundBackend:
    def __init__(self, config: EnvironmentConfig | None = None):
        self.config = config or EnvironmentConfig()
        if self.config.segmentation:
            raise CapabilityError("CraftGround 2.7.4 does not expose semantic segmentation")
        self.env = None
        self.tick = 0
        self._runtime_lock = None

    @property
    def capabilities(self) -> dict:
        return {
            "backend": "craftground",
            "minecraft": "1.21",
            "rgb": True,
            "depth": True,
            "segmentation": False,
            "shared_world_multi_agent": False,
            "weather_sensor": False,
            "world_commands": True,
            "lan_server": self.config.lan_port is not None,
        }

    def reset(self, *, seed: int, commands: list[str], biome: str) -> dict:
        self.close()
        from craftground.environment.action_space import ActionSpaceVersion
        from craftground.initial_environment_config import InitialEnvironmentConfig
        from craftground.screen_encoding_modes import ScreenEncodingMode

        from .socket_runtime import make_socket_environment

        if version("craftground") != "2.7.4":
            raise RuntimeError("Install the pinned Minecraft extra: uv sync --extra minecraft")
        config = self.config
        initial = InitialEnvironmentConfig(
            image_width=config.width,
            image_height=config.height,
            seed=str(seed),
            initial_extra_commands=[
                *commands,
                *(
                    [f"publish true survival {config.lan_port}"]
                    if config.lan_port is not None else []
                ),
            ],
            hud_hidden=True,
            no_fov_effect=True,
            render_distance=config.render_distance,
            simulation_distance=max(5, config.render_distance),
            request_raycast=True,
            requires_surrounding_blocks=True,
            surrounding_entity_distances=[32],
            requires_biome_info=True,
            requires_heightmap=True,
            requires_depth=config.depth,
            requires_depth_conversion=False,
            screen_encoding_mode=ScreenEncodingMode.RAW,
        )
        runtime = config.runtime_path or os.environ.get("MCSOCIETY_MINECRAFT_PATH")
        if runtime is None:
            raise RuntimeError(
                "Run scripts/bootstrap_minecraft.sh and source .runtime/env.sh first"
            )
        runtime = str(Path(runtime).resolve())
        from filelock import FileLock

        lock = FileLock(str(Path(runtime) / ".mcsociety.lock"))
        lock.acquire(timeout=0)
        self._runtime_lock = lock
        try:
            self.env = make_socket_environment(
                initial,
                env_path=runtime,
                port=config.port,
                action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
                cleanup_world=True,
                verbose_gradle=config.verbose,
            )
            # A full restart is required: upstream fast_reset only executes
            # commands and does not recreate the seeded world or clear entities.
            raw, _ = self.env.reset(seed=seed, options={"fast_reset": False})
            # Upstream reset can return before initial_extra_commands run and
            # before the client receives all nearby chunks. Treat that as
            # setup, not as the episode's first observation.
            initialized = False
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                full = raw["full"]
                initialized = initialized or any(
                    "Initialization Done" in message.message
                    for message in full.chat_messages
                )
                blocks = full.surrounding_blocks
                if initialized and blocks and all(
                    not block.translation_key.endswith(".void_air") for block in blocks
                ):
                    break
                raw, _, _, _, _ = self.env.step(Action().controls())
                time.sleep(0.05)
            else:
                raise RuntimeError(
                    "Minecraft setup did not finish with loaded nearby chunks within 20 seconds"
                )
            self.tick = 0
            return decode_observation(raw, config, self.tick)
        except BaseException:
            self.close()
            raise

    def step(self, action: Action, *, first_tick: bool = True, commands=()) -> dict:
        if self.env is None:
            raise RuntimeError("Call reset before step")
        if commands:
            self.env.add_commands(list(commands))
        raw, _, _, _, _ = self.env.step(action.controls(first_tick=first_tick))
        self.tick += 1
        return decode_observation(raw, self.config, self.tick)

    def close(self):
        env, self.env = self.env, None
        try:
            if env is not None:
                env.close()
        finally:
            lock, self._runtime_lock = self._runtime_lock, None
            if lock is not None:
                lock.release()
