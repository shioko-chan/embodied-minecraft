"""Lossless, inspectable (observation, action, next observation) datasets.

Each writer owns one episode. Arrays use non-pickle NumPy files, structured
observations use JSON, and fsynced JSONL transitions are the commit log. Readers
can recover the committed prefix of an interrupted recording; malformed complete
records, missing committed frames, and discontinuities always raise ValueError.
"""

# Data validation uses one ValueError contract for malformed records and inputs.
# ruff: noqa: TRY004

from __future__ import annotations

import json
import math
import os
import uuid
from collections import deque
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import numpy as np

_FORMAT = "mcsociety.trajectory"
_SENSORS = ("rgb", "depth", "segmentation")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def _encode(value: Any) -> str:
    return json.dumps(
        _json_value(value), allow_nan=False, ensure_ascii=False, separators=(",", ":")
    )


def _load_json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        # The second pass also rejects overflowing JSON numbers, e.g. 1e999.
        return _json_value(json.loads(text, object_pairs_hook=pairs))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid trajectory JSON") from error


def _atomic_json(path: Path, value: Any) -> None:
    encoded = _encode(value)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_path(episode_dir: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError("dataset paths must be strings")
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in relative:
        raise ValueError("dataset path escapes its episode")
    resolved = (episode_dir / path).resolve()
    if not resolved.is_relative_to(episode_dir.resolve()):
        raise ValueError("dataset path escapes its episode")
    return resolved


def _validate_observation(observation: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observation, dict) or not isinstance(observation.get("state"), dict):
        raise ValueError("observation must contain a state object")
    rgb = observation.get("rgb")
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape:
        raise ValueError("rgb must be a nonempty H x W x 3 ndarray")
    for name in _SENSORS:
        if name not in observation:
            continue
        array = observation[name]
        if not isinstance(array, np.ndarray) or array.dtype.kind not in "buif":
            raise ValueError(f"{name} must be a numeric non-object ndarray")
        if name != "rgb" and array.shape != rgb.shape[:2]:
            raise ValueError(f"{name} dimensions must match rgb")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    return _json_value({key: value for key, value in observation.items() if key not in _SENSORS})


class TrajectoryWriter:
    """Append one episode; record_initial precedes record, terminal ends writes."""

    def __init__(self, root: str | Path, metadata: dict[str, Any] | None = None):
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        metadata = _json_value(metadata)
        self.episode_id = uuid.uuid4().hex
        self.episode_dir = Path(root).resolve() / self.episode_id
        self.episode_dir.mkdir(parents=True)
        (self.episode_dir / "observations").mkdir()
        self._manifest = {
            "format": _FORMAT,
            "version": 1,
            "episode_id": self.episode_id,
            "created_at": datetime.now(UTC).isoformat(),
            "metadata": metadata,
            "initial_observation": None,
            "closed": False,
            "transition_count": None,
        }
        self._last_observation: str | None = None
        self._count = 0
        self._terminal = False
        self._closed = False
        self._failed = False
        (self.episode_dir / "transitions.jsonl").touch(exist_ok=False)
        _atomic_json(self.episode_dir / "episode.json", self._manifest)

    def _check_open(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("trajectory writer is closed or failed")
        if self._terminal:
            raise RuntimeError(
                "cannot append after termination or truncation; create another writer"
            )

    def _save_observation(self, observation: dict[str, Any], index: int) -> str:
        fields = _validate_observation(observation)
        sensors = {}
        for name in _SENSORS:
            if name not in observation:
                continue
            array = observation[name]
            relative = f"observations/{index:08d}.{name}.npy"
            path = self.episode_dir / relative
            temporary = path.with_suffix(".tmp")
            try:
                with temporary.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            sensors[name] = {"path": relative, "dtype": array.dtype.str, "shape": list(array.shape)}
        relative = f"observations/{index:08d}.json"
        _atomic_json(
            self.episode_dir / relative, {"index": index, "fields": fields, "sensors": sensors}
        )
        return relative

    def record_initial(self, observation: dict[str, Any]) -> None:
        self._check_open()
        if self._last_observation is not None:
            raise RuntimeError("initial observation is already recorded")
        reference = self._save_observation(observation, 0)
        self._manifest["initial_observation"] = reference
        _atomic_json(self.episode_dir / "episode.json", self._manifest)
        self._last_observation = reference

    def record(
        self,
        action: dict[str, Any],
        observation: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        self._check_open()
        if self._last_observation is None:
            raise RuntimeError("record_initial must precede record")
        if info is None:
            info = {}
        if not isinstance(action, dict) or not isinstance(info, dict):
            raise ValueError("action and info must be JSON objects")
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError("terminated and truncated must be booleans")
        if isinstance(reward, (bool, np.bool_)) or not isinstance(reward, (float, int, np.number)):
            raise ValueError("reward must be a finite number")
        if not np.isrealobj(reward) or not math.isfinite(float(reward)):
            raise ValueError("reward must be a finite number")
        action, info = _json_value(action), _json_value(info)
        reference = self._save_observation(observation, self._count + 1)
        transition = {
            "episode_id": self.episode_id,
            "index": self._count,
            "observation": self._last_observation,
            "action": action,
            "next_observation": reference,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }
        try:
            with (self.episode_dir / "transitions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(_encode(transition) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            self._failed = True
            raise
        self._last_observation = reference
        self._count += 1
        self._terminal = terminated or truncated

    def close(self) -> None:
        if self._closed:
            return
        if not self._failed:
            self._manifest.update(closed=True, transition_count=self._count)
            _atomic_json(self.episode_dir / "episode.json", self._manifest)
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class TrajectoryDataset:
    """Read one episode or a directory of episodes with strict integrity checks.

    ``allow_incomplete=True`` reads committed records in an unclosed episode and
    ignores only an unterminated trailing log fragment. It never ignores a bad
    complete record. Windows contain ``length`` transitions (length + 1 states).
    """

    def __init__(self, root: str | Path, *, allow_incomplete: bool = True):
        self.root = Path(root).resolve()
        self.allow_incomplete = allow_incomplete
        if not self.root.is_dir():
            raise ValueError(f"dataset directory does not exist: {self.root}")

    def episode_paths(self) -> list[Path]:
        if (self.root / "episode.json").is_file():
            return [self.root]
        paths = []
        for path in sorted(self.root.iterdir()):
            if path.is_dir() and (path / "episode.json").is_file():
                if not path.resolve().is_relative_to(self.root):
                    raise ValueError("episode directory escapes dataset root")
                paths.append(path)
        return paths

    @staticmethod
    def _observation(episode: Path, relative: str, index: int) -> dict[str, Any]:
        expected = f"observations/{index:08d}.json"
        if relative != expected:
            raise ValueError("observation sequence is not contiguous")
        try:
            descriptor = _load_json(_safe_path(episode, relative).read_text(encoding="utf-8"))
            if not isinstance(descriptor, dict) or set(descriptor) != {
                "index",
                "fields",
                "sensors",
            }:
                raise ValueError("invalid observation descriptor")
            if type(descriptor["index"]) is not int or descriptor["index"] != index:
                raise ValueError("observation index mismatch")
            fields, sensors = descriptor["fields"], descriptor["sensors"]
            if not isinstance(fields, dict) or not isinstance(sensors, dict):
                raise ValueError("observation fields and sensors must be objects")
            if (
                set(fields) & set(_SENSORS)
                or not set(sensors) <= set(_SENSORS)
                or "rgb" not in sensors
            ):
                raise ValueError("invalid observation sensor names")
            observation = dict(fields)
            for name, spec in sensors.items():
                if not isinstance(spec, dict) or set(spec) != {"path", "dtype", "shape"}:
                    raise ValueError("invalid sensor descriptor")
                if spec["path"] != f"observations/{index:08d}.{name}.npy":
                    raise ValueError("sensor does not belong to its observation")
                if not isinstance(spec["shape"], list) or any(
                    type(size) is not int or size < 1 for size in spec["shape"]
                ):
                    raise ValueError("sensor dimensions must be positive integers")
                array = np.load(_safe_path(episode, spec["path"]), allow_pickle=False)
                if not isinstance(array, np.ndarray):
                    if isinstance(array, np.lib.npyio.NpzFile):
                        array.close()
                    raise ValueError("sensor must be a single NumPy array")
                if array.dtype.str != spec["dtype"] or list(array.shape) != spec["shape"]:
                    raise ValueError("sensor dtype or dimensions do not match descriptor")
                observation[name] = array
            _validate_observation(observation)
            return observation
        except (OSError, EOFError, UnicodeError) as error:
            raise ValueError(f"missing or unreadable observation {relative}") from error

    def _episode_transitions(self, episode: Path) -> Iterator[dict[str, Any]]:
        try:
            manifest = _load_json(_safe_path(episode, "episode.json").read_text(encoding="utf-8"))
            required = {
                "format",
                "version",
                "episode_id",
                "created_at",
                "metadata",
                "initial_observation",
                "closed",
                "transition_count",
            }
            if not isinstance(manifest, dict) or set(manifest) != required:
                raise ValueError("invalid episode manifest")
            if (
                manifest["format"] != _FORMAT
                or type(manifest["version"]) is not int
                or manifest["version"] != 1
            ):
                raise ValueError("unsupported trajectory format")
            if manifest.get("episode_id") != episode.name:
                raise ValueError("episode id does not match its directory")
            if type(manifest.get("closed")) is not bool or not isinstance(
                manifest.get("metadata"), dict
            ):
                raise ValueError("invalid episode manifest")
            closed = manifest["closed"]
            if not isinstance(manifest["created_at"], str):
                raise ValueError("invalid episode creation timestamp")
            if not closed and manifest["transition_count"] is not None:
                raise ValueError("open episode must not declare a final transition count")
            if not closed and not self.allow_incomplete:
                raise ValueError("episode was not closed")
            reference = manifest.get("initial_observation")
            if reference is None:
                observation = None
            else:
                observation = self._observation(episode, reference, 0)
            count, terminal = 0, False
            with _safe_path(episode, "transitions.jsonl").open("rb") as stream:
                for raw_line in stream:
                    if not raw_line.endswith(b"\n"):
                        if closed or not self.allow_incomplete:
                            raise ValueError("incomplete transition log record")
                        break
                    transition = _load_json(raw_line.decode("utf-8"))
                    required = {
                        "episode_id",
                        "index",
                        "observation",
                        "action",
                        "next_observation",
                        "reward",
                        "terminated",
                        "truncated",
                        "info",
                    }
                    if not isinstance(transition, dict) or set(transition) != required:
                        raise ValueError("invalid transition record")
                    if terminal:
                        raise ValueError("transition occurs after terminal state")
                    if observation is None or transition["observation"] != reference:
                        raise ValueError("transition does not match preceding observation")
                    if transition["episode_id"] != manifest["episode_id"]:
                        raise ValueError("transition crosses episode boundary")
                    if type(transition["index"]) is not int or transition["index"] != count:
                        raise ValueError("transition sequence is not contiguous")
                    if not isinstance(transition["action"], dict) or not isinstance(
                        transition["info"], dict
                    ):
                        raise ValueError("transition action and info must be objects")
                    if (
                        type(transition["terminated"]) is not bool
                        or type(transition["truncated"]) is not bool
                    ):
                        raise ValueError("transition terminal flags must be booleans")
                    if type(transition["reward"]) not in (int, float):
                        raise ValueError("transition reward must be a finite number")
                    following = self._observation(
                        episode, transition["next_observation"], count + 1
                    )
                    yield transition | {"observation": observation, "next_observation": following}
                    observation, reference = following, transition["next_observation"]
                    terminal = transition["terminated"] or transition["truncated"]
                    count += 1
            if closed and (
                type(manifest.get("transition_count")) is not int
                or manifest["transition_count"] != count
            ):
                raise ValueError("closed episode transition count mismatch")
        except (OSError, UnicodeError) as error:
            raise ValueError(f"missing or unreadable episode data: {episode}") from error

    def transitions(self) -> Iterator[dict[str, Any]]:
        for episode in self.episode_paths():
            yield from self._episode_transitions(episode)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self.transitions()

    def windows(self, length: int, stride: int = 1) -> Iterator[list[dict[str, Any]]]:
        """Yield transition windows without crossing resets or terminal states."""
        if type(length) is not int or length < 1 or type(stride) is not int or stride < 1:
            raise ValueError("length and stride must be positive integers")
        for episode in self.episode_paths():
            window: deque[dict[str, Any]] = deque(maxlen=length)
            for transition in self._episode_transitions(episode):
                window.append(transition)
                start = transition["index"] - length + 1
                if len(window) == length and start % stride == 0:
                    yield list(window)
