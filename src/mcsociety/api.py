"""Local REST/WebSocket control plane for one serialized simulation session.

The server has no authentication or remote multi-tenant isolation. Bind it to
loopback. WebSocket ids are single-use per connection: duplicates are rejected
even if their original operation failed, so transport replay cannot repeat edits.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from .client import _plain, encode_observation
from .craftground_backend import CapabilityError
from .environment import Simulation
from .mindcraft import MindcraftBridge
from .minecollab import MineCollabTask
from .models import Action, EnvironmentConfig, HighLevelAction, WorldEdit
from .scenarios import Scenario
from .worker_backend import SupervisedBackend
from .world import edit_commands

logger = logging.getLogger(__name__)
MAX_WS_REQUEST_BYTES = 1024 * 1024
MAX_WS_IDS = 4096


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: Scenario | None = None
    seed: int | None = Field(default=None, strict=True)

    @model_validator(mode="after")
    def consistent_seed(self):
        if self.scenario is not None and self.seed is not None and self.seed != self.scenario.seed:
            raise ValueError("seed must match an explicitly supplied scenario.seed")
        return self


class EmptyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agents: tuple[str, ...] = Field(min_length=1, max_length=8)

    @field_validator("agents")
    @classmethod
    def valid_agents(cls, agents: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(agents)) != len(agents) or any(
            re.fullmatch(r"[A-Za-z0-9_]{1,16}", name) is None for name in agents
        ):
            raise ValueError("agents must be unique Minecraft usernames")
        return agents


class TaskStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    task_id: str = Field(min_length=1)


class SocketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Any
    method: Literal[
        "reset", "step", "world", "observe", "agents", "agent_action", "agent_observe",
        "task", "task_evaluate",
    ]
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: Any) -> str | int:
        if type(value) not in (str, int) or (isinstance(value, str) and not 0 < len(value) <= 128):
            raise ValueError("id must be an integer or a nonempty string of at most 128 characters")
        return value


class SessionStateError(RuntimeError):
    pass


class BackendUnavailable(RuntimeError):
    pass


class SimulationFailure(RuntimeError):
    """Keep backend-data errors distinct from invalid client requests."""

    def __init__(self, cause: Exception):
        super().__init__(str(cause))
        self.cause = cause


def _error(error: Exception) -> dict:
    if isinstance(error, SimulationFailure):
        error = error.cause
    return {"type": type(error).__name__, "message": str(error)}


def _backend_status(config: EnvironmentConfig) -> dict:
    # Package metadata can be inspected without importing CraftGround or starting Java.
    try:
        installed = version("craftground")
    except PackageNotFoundError:
        return {
            "available": False,
            "detail": "Install the Minecraft backend with: uv sync --extra minecraft",
        }
    if installed != "2.7.4":
        return {"available": False, "detail": f"CraftGround 2.7.4 required; found {installed}"}
    if config.segmentation:
        return {
            "available": False,
            "detail": "CraftGround 2.7.4 does not expose semantic segmentation",
        }
    return {
        "available": True,
        "detail": "Python backend installed; runtime readiness is verified on reset",
    }


class _Session:
    def __init__(self, config: EnvironmentConfig, record_dir, factory):
        self.config = config
        self.record_dir = record_dir
        self.factory = factory
        self.simulation: Simulation | None = None
        self.peers: MindcraftBridge | None = None
        self.lock = threading.RLock()
        self.last_error: dict | None = None

    def close(self) -> None:
        with self.lock:
            peers, self.peers = self.peers, None
            if peers is not None:
                peers.close()
            simulation, self.simulation = self.simulation, None
            if simulation is not None:
                simulation.close()

    def _health(self) -> dict:
        simulation = self.simulation
        if self.factory is None:
            availability = _backend_status(self.config)
            capabilities = {
                "backend": "craftground",
                "minecraft": "1.21",
                "rgb": True,
                "depth": True,
                "segmentation": False,
                "shared_world_multi_agent": False,
                "weather_sensor": False,
                "world_commands": True,
            }
        else:
            availability = {
                "available": True,
                "detail": "Injected simulation factory; reset verifies readiness",
            }
            capabilities = {"backend": "injected", "verified": False}
        if simulation is not None:
            capabilities = _plain(getattr(simulation.backend, "capabilities", {}))
        if self.config.lan_port is not None:
            capabilities["mindcraft_peer_actions"] = self.peers is not None
        return {
            "status": "ok",
            "backend": availability,
            "capabilities": capabilities,
            "session": "failed"
            if self.last_error
            else "idle"
            if simulation is None
            else "ended"
            if simulation.done
            else "active",
            "last_error": self.last_error,
            "protocol": {
                "version": 1,
                "rgb": "png/base64",
                "optional_sensors": "npy/base64",
                "websocket_ids": "single-use per connection",
                "max_websocket_ids": MAX_WS_IDS,
            },
        }

    def invoke(self, method: str, params: dict) -> dict:
        # Validate before acquiring or modifying the environment. Invalid requests
        # must never kill a healthy episode or become recorded policy actions.
        if method == "reset":
            request = ResetRequest.model_validate(params)
        elif method == "step":
            request = Action.model_validate(params)
        elif method == "world":
            request = WorldEdit.model_validate(params)
            edit_commands(request)
        elif method == "agents":
            request = AgentStartRequest.model_validate(params)
        elif method == "agent_action":
            request = HighLevelAction.model_validate(params)
        elif method == "task":
            request = TaskStartRequest.model_validate(params)
        else:
            EmptyRequest.model_validate(params)
            request = None
        with self.lock:
            if method == "health":
                return self._health()
            if method == "close":
                self.close()
                self.last_error = None
                return {"closed": True}
            if method != "reset" and (
                self.simulation is None or self.simulation.observation is None
            ):
                raise SessionStateError("Call reset before observing or acting")
            if method in {"step", "world", "agents", "agent_action", "agent_observe", "task", "task_evaluate"} and self.simulation.done:
                raise SessionStateError("Episode ended; call reset before acting")
            if method in {"agents", "agent_action", "agent_observe", "task", "task_evaluate"} and self.config.lan_port is None:
                raise CapabilityError("Set lan_port to publish the Minecraft world first")
            if method == "agents" and self.peers is not None:
                raise SessionStateError("Mindcraft peers are already active")
            if method in {"agent_action", "agent_observe", "task", "task_evaluate"} and self.peers is None:
                raise SessionStateError("Start Mindcraft peers before acting or observing")
            if method == "task_evaluate" and self.peers.task is None:
                raise SessionStateError("Start a MineCollab task before evaluating it")
            if method == "task" and self.peers.task is not None:
                raise SessionStateError("A MineCollab task is already active")
            if method == "task":
                MineCollabTask.load(request.source, request.task_id, self.peers.agents)
            try:
                if method == "reset":
                    peers, self.peers = self.peers, None
                    if peers is not None:
                        peers.close()
                    if self.simulation is None:
                        if self.factory is None:
                            status = _backend_status(self.config)
                            if not status["available"]:
                                raise BackendUnavailable(status["detail"])
                            self.simulation = Simulation(
                                SupervisedBackend(self.config), record_dir=self.record_dir
                            )
                        else:
                            self.simulation = self.factory()
                    observation, info = self.simulation.reset(request.scenario, seed=request.seed)
                    result = {"observation": encode_observation(observation), "info": _plain(info)}
                elif method == "observe":
                    result = {"observation": encode_observation(self.simulation.observation)}
                elif method == "agents":
                    self.peers = MindcraftBridge(
                        lan_port=self.config.lan_port, agents=request.agents
                    )
                    ready = self.peers.start(pump=lambda: self.simulation.step(Action()))
                    result = {
                        "ready": ready,
                        "observation": encode_observation(self.simulation.observation),
                    }
                elif method == "agent_observe":
                    result = {"agents": self.peers.observe()}
                elif method == "task":
                    result = {"task": self.peers.configure_task(
                        self.simulation, request.source, request.task_id
                    )}
                elif method == "task_evaluate":
                    result = {"evaluation": self.peers.evaluate_task(self.simulation)}
                elif method == "agent_action":
                    receipt, transition = self.peers.execute_in_simulation(
                        self.simulation, request
                    )
                    observation, reward, terminated, truncated, info = transition
                    result = {
                        "receipt": _plain(receipt),
                        "observation": encode_observation(observation),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": _plain(info),
                    }
                elif method in {"step", "world"}:
                    call = self.simulation.step if method == "step" else self.simulation.intervene
                    observation, reward, terminated, truncated, info = call(request)
                    result = {
                        "observation": encode_observation(observation),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": _plain(info),
                    }
                else:
                    raise ValueError(f"Unknown method: {method}")
                self.last_error = None
                return result
            except Exception as error:
                self.last_error = _error(error)
                try:
                    self.close()
                except Exception:
                    logger.exception("Failed to close simulation after an operation error")
                if isinstance(error, (BackendUnavailable, CapabilityError, ImportError)):
                    raise
                raise SimulationFailure(error) from error


def create_app(
    config: EnvironmentConfig | None = None,
    record_dir: str | Path | None = "runs",
    simulation_factory=None,
) -> FastAPI:
    """Create a lazy app; the optional zero-argument factory returns a Simulation."""
    session = _Session(config or EnvironmentConfig(), record_dir, simulation_factory)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await run_in_threadpool(session.close)

    app = FastAPI(title="MC Society simulation API", version="1", lifespan=lifespan)
    app.state.session = session

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, error):
        return JSONResponse(status_code=422, content={"error": _error(error)})

    async def invoke(method: str, params: dict):
        try:
            return await run_in_threadpool(session.invoke, method, params)
        except (ValidationError, ValueError, TypeError, FileNotFoundError) as error:
            return JSONResponse(status_code=422, content={"error": _error(error)})
        except SessionStateError as error:
            return JSONResponse(status_code=409, content={"error": _error(error)})
        except (BackendUnavailable, CapabilityError, ImportError) as error:
            return JSONResponse(status_code=503, content={"error": _error(error)})
        except Exception as error:
            logger.exception("Simulation API operation failed")
            return JSONResponse(status_code=502, content={"error": _error(error)})

    @app.get("/health")
    async def health():
        return await invoke("health", {})

    @app.post("/reset")
    async def reset(request: ResetRequest):
        return await invoke("reset", request.model_dump(mode="json"))

    @app.post("/step")
    async def step(action: Action):
        return await invoke("step", action.model_dump(mode="json"))

    @app.post("/world")
    async def world(edit: WorldEdit):
        return await invoke("world", edit.model_dump(mode="json"))

    @app.get("/observation")
    async def observe():
        return await invoke("observe", {})

    @app.post("/agents")
    async def start_agents(request: AgentStartRequest):
        return await invoke("agents", request.model_dump(mode="json"))

    @app.get("/agents")
    async def observe_agents():
        return await invoke("agent_observe", {})

    @app.post("/agent-action")
    async def agent_action(action: HighLevelAction):
        return await invoke("agent_action", action.model_dump(mode="json"))

    @app.post("/task")
    async def start_task(request: TaskStartRequest):
        return await invoke("task", request.model_dump(mode="json"))

    @app.get("/task")
    async def evaluate_task():
        return await invoke("task_evaluate", {})

    @app.post("/close")
    async def close(request: EmptyRequest):
        return await invoke("close", request.model_dump())

    @app.websocket("/ws")
    async def websocket(socket: WebSocket):
        await socket.accept()
        seen: set[tuple[type, str | int]] = set()
        try:
            while True:
                text = await socket.receive_text()
                if len(text.encode("utf-8")) > MAX_WS_REQUEST_BYTES:
                    await socket.close(code=1009, reason="Request exceeds protocol limits")
                    return
                request_id = None
                try:
                    raw = json.loads(text)
                    if isinstance(raw, dict) and type(raw.get("id")) in (str, int):
                        request_id = raw["id"]
                    request = SocketRequest.model_validate(raw)
                    key = (type(request.id), request.id)
                    if key in seen:
                        await socket.send_json(
                            {
                                "id": request.id,
                                "error": {
                                    "type": "DuplicateRequest",
                                    "message": "This id has already been used; no action was executed",
                                },
                            }
                        )
                        continue
                    if len(seen) >= MAX_WS_IDS:
                        await socket.send_json(
                            {
                                "id": request.id,
                                "error": {
                                    "type": "SessionLimit",
                                    "message": "Reconnect with a new connection; no action was executed",
                                },
                            }
                        )
                        await socket.close(code=1008, reason="Request id limit reached")
                        return
                    seen.add(key)
                    result = await run_in_threadpool(session.invoke, request.method, request.params)
                    await socket.send_json({"id": request.id, "result": result})
                except Exception as error:
                    if isinstance(error, WebSocketDisconnect):
                        raise
                    await socket.send_json({"id": request_id, "error": _error(error)})
        except WebSocketDisconnect:
            return

    return app
