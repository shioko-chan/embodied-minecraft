from types import SimpleNamespace as Obj

import numpy as np
import pytest

from mcsociety.craftground_backend import CapabilityError, CraftGroundBackend, decode_observation
from mcsociety.models import EnvironmentConfig


def native_frame():
    entity = Obj(
        unique_name="uuid", translation_key="entity.minecraft.cow", x=1, y=64, z=2, health=10
    )
    full = Obj(
        x=0,
        y=64,
        z=0,
        velocity_x=0,
        velocity_y=-0.1,
        velocity_z=0.2,
        yaw=30,
        pitch=10,
        health=18,
        food_level=17,
        saturation_level=3,
        is_dead=False,
        is_on_ground=True,
        inventory=[
            Obj(translation_key="item.minecraft.stick", count=3, durability=0, max_durability=0)
        ],
        surrounding_blocks=[Obj(x=0, y=63, z=0, translation_key="block.minecraft.stone")],
        surrounding_entities={3: Obj(entities=[entity]), 32: Obj(entities=[entity])},
        world_time=6000,
        biome_info=Obj(biome_name="minecraft:forest"),
        height_info=[Obj(x=0, z=0, height=64, block_name="minecraft:stone")],
        chat_messages=[Obj(message="hello", added_time=4)],
        raycast_result=Obj(type=1),
        depth=np.linspace(0, 1, 4096, dtype=np.float32),
    )
    return {"pov": np.zeros((64, 64, 3), dtype=np.uint8), "full": full}


def test_real_schema_conversion_preserves_state_and_sensor_orientation():
    raw = native_frame()
    decoded = decode_observation(raw, EnvironmentConfig(width=64, height=64), 12)
    assert decoded["state"]["tick"] == 12
    assert decoded["state"]["position"] == [0, 64, 0]
    assert decoded["state"]["inventory"][0]["item"] == "minecraft:stick"
    assert len(decoded["state"]["entities"]) == 1
    assert decoded["state"]["nearby_blocks"][0]["block"] == "minecraft:stone"
    assert decoded["state"]["weather"] is None
    assert decoded["depth"][0, 0] == raw["full"].depth[63 * 64]
    assert "nonlinear" in decoded["sensors"]["depth"]
    raw["pov"][:] = 255
    assert not decoded["rgb"].any()


def test_requested_missing_depth_fails_instead_of_fake_frame():
    raw = native_frame()
    raw["full"].depth = []
    with pytest.raises(CapabilityError, match="depth"):
        decode_observation(raw, EnvironmentConfig(width=64, height=64), 0)
    decoded = decode_observation(raw, EnvironmentConfig(width=64, height=64, depth=False), 0)
    assert "depth" not in decoded


def test_invalid_native_data_and_unsupported_sensor():
    raw = native_frame()
    raw["full"].depth[0] = np.nan
    with pytest.raises(RuntimeError, match="depth"):
        decode_observation(raw, EnvironmentConfig(width=64, height=64), 0)
    with pytest.raises(CapabilityError, match="segmentation"):
        CraftGroundBackend(EnvironmentConfig(segmentation=True))
