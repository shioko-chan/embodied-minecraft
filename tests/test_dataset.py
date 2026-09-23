import json

import numpy as np
import pytest

from mcsociety.dataset import TrajectoryDataset, TrajectoryWriter


def observation(tick, *, depth=True):
    result = {
        "rgb": np.full((3, 4, 3), tick, dtype=np.uint8),
        "state": {
            "tick": tick,
            "position": np.array([tick, 64, 0], dtype=np.float64),
            "inventory": {"wood": np.int64(tick)},
        },
        "agent_id": "explorer",
    }
    if depth:
        result["depth"] = (np.arange(12, dtype=np.float32).reshape(3, 4) / 7) + tick
        result["segmentation"] = np.full((3, 4), 65_537 + tick, dtype=np.uint32)
    return result


def record_episode(root, count, *, terminal=True):
    with TrajectoryWriter(root, {"world_id": "forest", "seed": np.int64(12)}) as writer:
        writer.record_initial(observation(0))
        for tick in range(count):
            writer.record(
                {"type": "move", "forward": np.float32(1)},
                observation(tick + 1),
                tick / 10,
                terminal and tick == count - 1,
                False,
                {"success": tick == count - 1},
            )
    return writer.episode_dir


def test_exact_transition_alignment_and_lossless_sensor_values(tmp_path):
    episode = record_episode(tmp_path, 3)
    records = list(TrajectoryDataset(tmp_path))
    assert len(records) == 3
    for index, record in enumerate(records):
        assert record["episode_id"] == episode.name
        assert record["index"] == index
        assert record["observation"]["state"]["tick"] == index
        assert record["next_observation"]["state"]["tick"] == index + 1
        assert record["action"] == {"type": "move", "forward": 1.0}
        for key in ("rgb", "depth", "segmentation"):
            before, after = observation(index)[key], observation(index + 1)[key]
            assert record["observation"][key].dtype == before.dtype
            assert record["next_observation"][key].dtype == after.dtype
            np.testing.assert_array_equal(record["observation"][key], before)
            np.testing.assert_array_equal(record["next_observation"][key], after)
        assert record["reward"] == index / 10
    assert records[-1]["terminated"]
    manifest = json.loads((episode / "episode.json").read_text())
    assert manifest["closed"] and manifest["transition_count"] == 3
    assert manifest["metadata"]["seed"] == 12


def test_initial_and_terminal_lifecycle(tmp_path):
    writer = TrajectoryWriter(tmp_path, {})
    with pytest.raises(RuntimeError, match="record_initial"):
        writer.record({}, observation(1), 0, False, False)
    writer.record_initial(observation(0, depth=False))
    with pytest.raises(RuntimeError, match="already"):
        writer.record_initial(observation(0))
    writer.record({"type": "jump"}, observation(1, depth=False), 0, False, True)
    with pytest.raises(RuntimeError, match="termination"):
        writer.record({}, observation(2), 0, False, False)
    writer.close()
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.record({}, observation(2), 0, False, False)
    records = list(TrajectoryDataset(tmp_path))
    assert len(records) == 1
    assert records[0]["truncated"]
    assert "depth" not in records[0]["observation"]


def test_windows_never_cross_episodes_and_honor_stride(tmp_path):
    record_episode(tmp_path, 4)
    record_episode(tmp_path, 3)
    dataset = TrajectoryDataset(tmp_path)
    windows = list(dataset.windows(2))
    assert len(windows) == 5
    for window in windows:
        assert len({item["episode_id"] for item in window}) == 1
        assert window[1]["index"] == window[0]["index"] + 1
        assert not window[0]["terminated"]
        assert window[0]["next_observation"]["state"] == window[1]["observation"]["state"]
    strided = list(dataset.windows(2, stride=2))
    assert len(strided) == 3
    assert all(window[0]["index"] % 2 == 0 for window in strided)
    assert list(dataset.windows(5)) == []


def test_interrupted_recording_recovers_only_committed_prefix(tmp_path):
    writer = TrajectoryWriter(tmp_path, {})
    writer.record_initial(observation(0))
    writer.record({"type": "move"}, observation(1), 1, False, False)
    with (writer.episode_dir / "transitions.jsonl").open("ab") as stream:
        stream.write(b'{"episode_id": "interrupted')
    records = list(TrajectoryDataset(tmp_path))
    assert len(records) == 1
    assert records[0]["next_observation"]["state"]["tick"] == 1
    with pytest.raises(ValueError, match="not closed"):
        list(TrajectoryDataset(tmp_path, allow_incomplete=False))


def test_incomplete_episode_still_rejects_corrupt_complete_records(tmp_path):
    writer = TrajectoryWriter(tmp_path, {})
    writer.record_initial(observation(0))
    with (writer.episode_dir / "transitions.jsonl").open("ab") as stream:
        stream.write(b'{"broken": true}\n')
    with pytest.raises(ValueError, match="invalid transition"):
        list(TrajectoryDataset(tmp_path))


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("episode_id", "another-episode", "episode boundary"),
        ("index", 2, "not contiguous"),
        ("observation", "observations/00000001.json", "preceding"),
        ("next_observation", "../outside.json", "not contiguous"),
        ("reward", True, "finite number"),
        ("terminated", 1, "booleans"),
    ],
)
def test_reader_rejects_misaligned_or_malformed_transition(tmp_path, field, value, match):
    episode = record_episode(tmp_path, 1)
    log = episode / "transitions.jsonl"
    record = json.loads(log.read_text())
    record[field] = value
    log.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match=match):
        list(TrajectoryDataset(tmp_path))


def test_reader_rejects_transition_after_terminal(tmp_path):
    episode = record_episode(tmp_path, 2)
    log = episode / "transitions.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    records[0]["terminated"] = True
    log.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="after terminal"):
        list(TrajectoryDataset(tmp_path))


def test_missing_frame_and_symlink_escape_are_rejected(tmp_path):
    episode = record_episode(tmp_path / "dataset", 1)
    image_path = episode / "observations/00000001.rgb.npy"
    contents = image_path.read_bytes()
    image_path.unlink()
    with pytest.raises(ValueError, match="missing or unreadable"):
        list(TrajectoryDataset(tmp_path / "dataset"))
    outside = tmp_path / "outside.npy"
    outside.write_bytes(contents)
    image_path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        list(TrajectoryDataset(tmp_path / "dataset"))


def test_invalid_values_do_not_commit_transitions(tmp_path):
    with TrajectoryWriter(tmp_path) as writer:
        writer.record_initial(observation(0))
        with pytest.raises(ValueError, match="finite"):
            writer.record({}, observation(1), np.nan, False, False)
        with pytest.raises(ValueError, match="finite"):
            writer.record({"value": np.inf}, observation(1), 0, False, False)
        bad_observation = observation(1)
        bad_observation["depth"][0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            writer.record({}, bad_observation, 0, False, False)
        with pytest.raises(ValueError, match="booleans"):
            writer.record({}, observation(1), 0, 0, False)
        writer.record({}, observation(1), 0, False, False)
    records = list(TrajectoryDataset(tmp_path))
    assert len(records) == 1
    assert records[0]["index"] == 0


def test_closed_episode_requires_complete_log_and_exact_count(tmp_path):
    episode = record_episode(tmp_path, 2)
    log = episode / "transitions.jsonl"
    contents = log.read_bytes()
    log.write_bytes(contents[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        list(TrajectoryDataset(tmp_path))
    log.write_bytes(contents.splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="count mismatch"):
        list(TrajectoryDataset(tmp_path))


def test_reader_rejects_pickle_arrays(tmp_path):
    episode = record_episode(tmp_path, 1)
    path = episode / "observations/00000001.rgb.npy"
    with path.open("wb") as stream:
        np.save(stream, np.full((3, 4, 3), object(), dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="Object arrays"):
        list(TrajectoryDataset(tmp_path))


def test_zero_transition_episode_is_readable(tmp_path):
    with TrajectoryWriter(tmp_path) as writer:
        writer.record_initial(observation(0))
    assert list(TrajectoryDataset(tmp_path)) == []


@pytest.mark.parametrize(
    "change,match",
    [
        ({"shape": [True, 4, 3]}, "positive integers"),
        ({"dtype": "<f4"}, "dtype or dimensions"),
        ({"path": "../../other.npy"}, "belong"),
    ],
)
def test_reader_validates_sensor_descriptors(tmp_path, change, match):
    episode = record_episode(tmp_path, 1)
    path = episode / "observations/00000000.json"
    descriptor = json.loads(path.read_text())
    descriptor["sensors"]["rgb"].update(change)
    path.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match=match):
        list(TrajectoryDataset(tmp_path))


def test_reader_rejects_duplicate_json_keys_and_overflow(tmp_path):
    episode = record_episode(tmp_path, 1)
    path = episode / "episode.json"
    original = path.read_text()
    path.write_text(original.replace('"version":1', '"version":1,"version":1'))
    with pytest.raises(ValueError, match="duplicate"):
        list(TrajectoryDataset(tmp_path))
    path.write_text(original.replace('"seed":12', '"seed":1e999'))
    with pytest.raises(ValueError, match="finite"):
        list(TrajectoryDataset(tmp_path))
