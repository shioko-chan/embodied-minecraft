"""Synchronous Python client and bounded, lossless observation wire codecs.

Requests are never retried automatically: an uncertain network result must not
cause an action to execute twice. One server owns one shared simulation session.
"""

# Wire-data validation deliberately has one ValueError contract for malformed data.
# ruff: noqa: TRY004

from __future__ import annotations

import base64
import binascii
import io
import json
import math
from typing import Any, Self
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, UnidentifiedImageError

from .models import Action, HighLevelAction, WorldEdit
from .scenarios import Scenario

MAX_PIXELS = 1920 * 1080
MAX_SENSOR_BYTES = 32 * 1024 * 1024
MAX_RESPONSE_BYTES = 96 * 1024 * 1024
_SENSORS = ("rgb", "depth", "segmentation")


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def encode_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Encode RGB as PNG, optional sensors as exact non-pickle NumPy arrays."""
    if not isinstance(observation.get("state"), dict):
        raise ValueError("observation requires a state object")
    rgb = observation.get("rgb")
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError("RGB must be a uint8 array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not 0 < rgb.shape[0] * rgb.shape[1] <= MAX_PIXELS:
        raise ValueError("RGB dimensions exceed protocol limits")
    result = _plain({key: value for key, value in observation.items() if key not in _SENSORS})
    for name in _SENSORS:
        if name not in observation:
            continue
        array = observation[name]
        if (
            not isinstance(array, np.ndarray)
            or array.dtype.kind not in "buif"
            or array.dtype.itemsize > 8
        ):
            raise ValueError("sensor must be a supported numeric array")
        if name != "rgb" and array.shape != rgb.shape[:2]:
            raise ValueError("sensor dimensions must match RGB")
        if array.nbytes > MAX_SENSOR_BYTES or not np.isfinite(array).all():
            raise ValueError("sensor is oversized or non-finite")
        stream = io.BytesIO()
        if name == "rgb":
            Image.fromarray(array).save(stream, format="PNG")
        else:
            np.save(stream, array, allow_pickle=False)
        result[name] = {
            "encoding": "png" if name == "rgb" else "npy",
            "data": base64.b64encode(stream.getvalue()).decode("ascii"),
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        }
    return result


def _sensor_bytes(spec: dict, dimensions: int) -> tuple[bytes, tuple[int, ...]]:
    if not isinstance(spec, dict) or set(spec) != {"encoding", "data", "shape", "dtype"}:
        raise ValueError("invalid sensor descriptor")
    shape = spec["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != dimensions
        or any(type(x) is not int or x < 1 for x in shape)
    ):
        raise ValueError("invalid sensor shape")
    if shape[0] * shape[1] > MAX_PIXELS or (dimensions == 3 and shape[2] != 3):
        raise ValueError("sensor dimensions exceed protocol limits")
    encoded = spec["data"]
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_SENSOR_BYTES + 2) // 3):
        raise ValueError("sensor payload exceeds protocol limits")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("invalid base64 sensor payload") from error
    if len(raw) > MAX_SENSOR_BYTES:
        raise ValueError("sensor payload exceeds protocol limits")
    return raw, tuple(shape)


def decode_observation(wire: dict[str, Any]) -> dict[str, Any]:
    """Validate dimensions before allocating decoded arrays; never load pickle."""
    if not isinstance(wire, dict) or not isinstance(wire.get("state"), dict) or "rgb" not in wire:
        raise ValueError("invalid observation response")
    result = _plain({key: value for key, value in wire.items() if key not in _SENSORS})
    for name in _SENSORS:
        if name not in wire:
            continue
        spec = wire[name]
        raw, shape = _sensor_bytes(spec, 3 if name == "rgb" else 2)
        if name == "rgb":
            if spec["encoding"] != "png" or spec["dtype"] != "|u1":
                raise ValueError("RGB must use PNG with uint8 pixels")
            try:
                with Image.open(io.BytesIO(raw)) as frame:
                    if (
                        frame.format != "PNG"
                        or frame.mode != "RGB"
                        or frame.size != (shape[1], shape[0])
                    ):
                        raise ValueError("PNG dimensions or mode disagree with descriptor")
                    array = np.array(frame)
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
                raise ValueError("invalid PNG sensor") from error
        else:
            if spec["encoding"] != "npy":
                raise ValueError("depth and segmentation must use non-pickle NumPy encoding")
            stream = io.BytesIO(raw)
            try:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    actual_shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    actual_shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported NumPy sensor version")
                if actual_shape != shape or dtype.str != spec["dtype"]:
                    raise ValueError("NumPy sensor disagrees with descriptor")
                if dtype.kind not in "buif" or dtype.itemsize > 8:
                    raise ValueError("NumPy sensor must contain numeric values without pickle")
                size = math.prod(shape) * dtype.itemsize
                if size > MAX_SENSOR_BYTES or len(raw) != stream.tell() + size:
                    raise ValueError("invalid NumPy sensor byte length")
                stream.seek(0)
                array = np.load(stream, allow_pickle=False)
            except (EOFError, OSError, TypeError) as error:
                raise ValueError("invalid NumPy sensor") from error
            if not np.isfinite(array).all():
                raise ValueError("non-finite sensor values")
        result[name] = array
    if any(
        result[name].shape != result["rgb"].shape[:2] for name in _SENSORS[1:] if name in result
    ):
        raise ValueError("sensor dimensions must match RGB")
    return result


class RemoteError(RuntimeError):
    """A structured error returned by the simulation API."""

    def __init__(self, status: int, error_type: str, message: str):
        super().__init__(f"{error_type}: {message}")
        self.status = status
        self.error_type = error_type


class SimulationClient:
    """Blocking HTTP client with the same reset/step tuple shapes as Simulation."""

    def __init__(self, base_url: str = "http://127.0.0.1:8765", *, timeout: float = 180):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an HTTP(S) URL")
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be positive and finite")
        self.base_url, self.timeout = base_url.rstrip("/"), timeout

    @staticmethod
    def _response(stream) -> dict:
        length = stream.headers.get("Content-Length")
        if length is not None and (int(length) < 0 or int(length) > MAX_RESPONSE_BYTES):
            raise ValueError("response exceeds protocol limits")
        raw = stream.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("response exceeds protocol limits")
        result = _plain(json.loads(raw))
        if not isinstance(result, dict):
            raise ValueError("API response must be a JSON object")
        return result

    def _request(self, method: str, path: str, data: dict | None = None) -> dict:
        payload = (
            None if data is None else json.dumps(_plain(data), allow_nan=False).encode("utf-8")
        )
        request = Request(
            self.base_url + path,
            data=payload,
            method=method,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return self._response(response)
        except HTTPError as error:
            with error:
                result = self._response(error)
            problem = result.get("error", {})
            raise RemoteError(
                error.code, problem.get("type", "HTTPError"), problem.get("message", str(result))
            ) from error

    def health(self) -> dict:
        return self._request("GET", "/health")

    def reset(self, scenario: Scenario | dict | None = None, *, seed: int | None = None) -> tuple[dict, dict]:
        if isinstance(scenario, dict):
            scenario = Scenario.model_validate(scenario)
        data = {
            "seed": seed,
            "scenario": None if scenario is None else scenario.model_dump(mode="json"),
        }
        result = self._request("POST", "/reset", data)
        return decode_observation(result["observation"]), result["info"]

    @staticmethod
    def _transition(result: dict) -> tuple[dict, float, bool, bool, dict]:
        return (
            decode_observation(result["observation"]),
            result["reward"],
            result["terminated"],
            result["truncated"],
            result["info"],
        )

    def step(self, action: Action | dict) -> tuple[dict, float, bool, bool, dict]:
        action = action if isinstance(action, Action) else Action.model_validate(action)
        return self._transition(self._request("POST", "/step", action.model_dump(mode="json")))

    def intervene(self, edit: WorldEdit | dict) -> tuple[dict, float, bool, bool, dict]:
        edit = edit if isinstance(edit, WorldEdit) else WorldEdit.model_validate(edit)
        return self._transition(self._request("POST", "/world", edit.model_dump(mode="json")))

    def observe(self) -> dict:
        return decode_observation(self._request("GET", "/observation")["observation"])

    def start_agents(self, agents: list[str] | tuple[str, ...]) -> tuple[dict, dict]:
        result = self._request("POST", "/agents", {"agents": list(agents)})
        return result["ready"], decode_observation(result["observation"])

    def observe_agents(self) -> dict:
        return self._request("GET", "/agents")["agents"]

    def agent_action(
        self, action: HighLevelAction | dict
    ) -> tuple[dict, tuple[dict, float, bool, bool, dict]]:
        action = (
            action if isinstance(action, HighLevelAction) else HighLevelAction.model_validate(action)
        )
        result = self._request("POST", "/agent-action", action.model_dump(mode="json"))
        return result["receipt"], self._transition(result)

    def start_task(self, source: str, task_id: str) -> dict:
        return self._request("POST", "/task", {"source": source, "task_id": task_id})["task"]

    def evaluate_task(self) -> dict:
        return self._request("GET", "/task")["evaluation"]

    def close(self) -> None:
        """Close the server's shared simulation; another reset may start a new one."""
        self._request("POST", "/close", {})

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
