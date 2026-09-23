import base64
import copy
import io
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

from mcsociety.api import create_app
from mcsociety.client import RemoteError, SimulationClient, decode_observation, encode_observation
from mcsociety.dataset import TrajectoryDataset
from mcsociety.environment import Simulation
from mcsociety.models import Action, EnvironmentConfig
from mcsociety.scenarios import Scenario


class ScriptedBackend:
    """Test-only, explicitly scripted sensor/action behavior; never a runtime backend."""

    def __init__(self):
        self.capabilities = {"backend": "test-script", "rgb": True, "depth": True}
        self.tick = 0
        self.x = 0
        self.reset_count = 0
        self.step_count = 0
        self.close_count = 0
        self.fail_next = False
        self.commands = []
        self.delay = 0
        self.in_step = False
        self.overlap = False
        self.entered = threading.Event()
        self.gate = None

    def _observation(self):
        return {
            "rgb": np.full((4, 6, 3), self.tick, dtype=np.uint8),
            "depth": np.full((4, 6), self.tick / 7, dtype=np.float32),
            "segmentation": np.full((4, 6), 65_538 + self.tick, dtype=np.uint32),
            "state": {
                "tick": self.tick,
                "position": [self.x, 64, 0],
                "health": 20,
                "agent_id": "agent_0",
                "inventory": [],
            },
            "sensors": {"rgb": "test-script-only", "depth": "test-script-only"},
        }

    def reset(self, **_):
        self.reset_count += 1
        self.tick = self.x = 0
        return self._observation()

    def step(self, action, *, commands=(), **_):
        self.overlap |= self.in_step
        self.in_step = True
        self.entered.set()
        try:
            if self.gate is not None:
                self.gate.wait(timeout=5)
            if self.delay:
                time.sleep(self.delay)
            self.step_count += 1
            if self.fail_next:
                self.fail_next = False
                raise OSError("scripted backend disconnect")
            self.tick += 1
            self.x += int(action.forward)
            self.commands.extend(commands)
            return self._observation()
        finally:
            self.in_step = False

    def close(self):
        self.close_count += 1


def scenario(*, target=2, max_steps=10):
    return Scenario(
        id="network",
        name="Network protocol test",
        kind="exploration",
        seed=1,
        max_steps=max_steps,
        goal={"type": "reach", "target": [target, 64, 0], "radius": 0.1},
    )


@pytest.fixture
def api(tmp_path):
    backend = ScriptedBackend()
    app = create_app(simulation_factory=lambda: Simulation(backend, record_dir=tmp_path))
    with TestClient(app) as client:
        yield client, backend, tmp_path


def test_health_does_not_create_or_start_simulation(api):
    client, backend, _ = api
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["session"] == "idle"
    assert health["capabilities"] == {"backend": "injected", "verified": False}
    assert backend.reset_count == backend.close_count == 0
    assert client.get("/observation").status_code == 409


def test_missing_backend_health_and_reset_are_explicit(monkeypatch):
    def missing(_):
        raise PackageNotFoundError("craftground")

    monkeypatch.setattr("mcsociety.api.version", missing)
    with TestClient(create_app(record_dir=None)) as client:
        health = client.get("/health").json()
        assert health["backend"]["available"] is False
        assert "uv sync" in health["backend"]["detail"]
        assert health["capabilities"]["segmentation"] is False
        assert health["capabilities"]["shared_world_multi_agent"] is False
        response = client.post("/reset", json={"seed": 1})
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "BackendUnavailable"


def test_rest_framing_terminal_reset_and_recording(api):
    client, backend, directory = api
    result = client.post("/reset", json={"scenario": scenario().model_dump(mode="json"), "seed": 1})
    assert result.status_code == 200
    first = decode_observation(result.json()["observation"])
    assert first["state"]["tick"] == 0
    assert first["rgb"].shape == (4, 6, 3)
    assert first["segmentation"][0, 0] == 65_538
    result = client.post("/step", json={"forward": True, "ticks": 2}).json()
    assert result["terminated"] is True and result["truncated"] is False
    assert result["info"]["executed_ticks"] == 2
    assert decode_observation(result["observation"])["state"]["tick"] == 2
    assert client.post("/step", json={}).status_code == 409
    assert (
        decode_observation(client.get("/observation").json()["observation"])["state"]["tick"] == 2
    )
    assert len(list(TrajectoryDataset(directory))) == 2
    assert client.post("/reset", json={"seed": 3}).status_code == 200
    assert backend.reset_count == 2
    assert client.get("/health").json()["capabilities"]["backend"] == "test-script"


@pytest.mark.parametrize(
    "action", [{"ticks": 0}, {"forward": True, "back": True}, {"key": "W"}, {"yaw": 181}]
)
def test_invalid_actions_never_mutate_simulation(api, action):
    client, backend, _ = api
    client.post("/reset", json={"seed": 0})
    response = client.post("/step", json=action)
    assert response.status_code == 422
    assert "error" in response.json()
    assert backend.step_count == 0
    assert client.get("/health").json()["session"] == "active"


def test_world_edits_are_distinct_recorded_causes(api):
    client, backend, directory = api
    client.post("/reset", json={"seed": 0})
    response = client.post("/world", json={"operation": "weather", "value": "rain"})
    assert response.status_code == 200
    assert backend.commands == ["weather rain"]
    record = next(iter(TrajectoryDataset(directory)))
    assert record["action"]["intervention"]["operation"] == "weather"
    assert record["observation"]["state"]["tick"] == 0
    assert record["next_observation"]["state"]["tick"] == 1
    invalid = client.post("/world", json={"operation": "spawn", "name": "cow\nkill @p"})
    assert invalid.status_code == 422
    assert backend.step_count == 1
    assert client.get("/health").json()["session"] == "active"


def test_agent_api_lifecycle_validation_and_peer_transition(monkeypatch, tmp_path):
    backend = ScriptedBackend()
    bridges = []

    class ScriptedPeerBridge:
        def __init__(self, *, lan_port, agents):
            self.lan_port = lan_port
            self.agents = agents
            self.closed = False
            self.task = None
            bridges.append(self)

        def start(self, *, pump):
            pump()
            return {"event": "ready", "agents": list(self.agents)}

        def observe(self):
            return {name: {"agent_id": name} for name in self.agents}

        def execute_in_simulation(self, simulation, action):
            return {"agent": action.agent, "kind": action.kind, "success": True}, simulation.step_peer(
                action, phase="completed"
            )

        def configure_task(self, simulation, source, task_id):
            self.task = (source, task_id)
            simulation.step(Action())
            return {"source": source, "task_id": task_id, "assisted_setup": True}

        def evaluate_task(self, _simulation=None):
            return {"task_id": self.task[1], "success": False, "server_verified": False}

        def close(self):
            self.closed = True

    monkeypatch.setattr("mcsociety.api.MindcraftBridge", ScriptedPeerBridge)
    monkeypatch.setattr("mcsociety.api.MineCollabTask.load", lambda *_args: object())
    app = create_app(
        config=EnvironmentConfig(lan_port=55916),
        simulation_factory=lambda: Simulation(backend, record_dir=tmp_path),
    )
    with TestClient(app) as client:
        client.post("/reset", json={"scenario": scenario(target=9).model_dump(mode="json")})
        assert client.post("/agent-action", json={"kind": "say", "agent": "AgentA", "message": "hi"}).status_code == 409
        assert client.post("/agents", json={"agents": ["@a"]}).status_code == 422
        assert backend.step_count == 0
        result = client.post("/agents", json={"agents": ["AgentA", "AgentB"]})
        assert result.status_code == 200
        assert result.json()["ready"]["agents"] == ["AgentA", "AgentB"]
        assert client.get("/health").json()["capabilities"]["mindcraft_peer_actions"] is True
        assert client.get("/agents").json()["agents"]["AgentB"]["agent_id"] == "AgentB"
        assert client.post("/agents", json={"agents": ["AgentC"]}).status_code == 409
        assert client.get("/task").status_code == 409
        result = client.post(
            "/agent-action", json={"kind": "say", "agent": "AgentA", "message": "hello"}
        )
        assert result.status_code == 200
        assert result.json()["receipt"]["success"] is True
        assert decode_observation(result.json()["observation"])["state"]["tick"] == 2
        task = client.post("/task", json={
            "source": "multiagent_crafting_tasks.json",
            "task_id": "multiagent_techtree_1_shears",
        })
        assert task.status_code == 200
        assert task.json()["task"]["task_id"] == "multiagent_techtree_1_shears"
        assert client.get("/task").json()["evaluation"]["server_verified"] is False
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"id": 1, "method": "agent_observe", "params": {}})
            assert ws.receive_json()["result"]["agents"]["AgentB"]["agent_id"] == "AgentB"
            ws.send_json({
                "id": 2,
                "method": "agent_action",
                "params": {"kind": "say", "agent": "AgentB", "message": "reply"},
            })
            response = ws.receive_json()
            assert response["result"]["receipt"]["agent"] == "AgentB"
            assert response["result"]["info"]["peer_action"]["kind"] == "say"
            ws.send_json({"id": 3, "method": "task_evaluate", "params": {}})
            assert ws.receive_json()["result"]["evaluation"]["task_id"] == "multiagent_techtree_1_shears"
        assert client.post("/reset", json={"scenario": scenario(target=9).model_dump(mode="json")}).status_code == 200
        assert bridges[0].closed is True
    rows = list(TrajectoryDataset(tmp_path))
    assert any(row["action"].get("peer_action", {}).get("kind") == "say" for row in rows)


def test_backend_exception_invalidates_session_and_requires_reset(api):
    client, backend, _ = api
    client.post("/reset", json={"seed": 0})
    backend.fail_next = True
    response = client.post("/step", json={"forward": True})
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "OSError"
    assert backend.close_count >= 1
    assert client.get("/health").json()["session"] == "failed"
    assert client.get("/observation").status_code == 409
    assert client.post("/step", json={}).status_code == 409
    assert client.post("/reset", json={"seed": 0}).status_code == 200
    assert client.get("/health").json()["session"] == "active"


def test_invalid_backend_sensor_data_is_not_reported_as_client_error(api, monkeypatch):
    client, backend, _ = api
    original = backend._observation

    def invalid_observation():
        observation = original()
        observation["depth"][0, 0] = np.nan
        return observation

    monkeypatch.setattr(backend, "_observation", invalid_observation)
    response = client.post("/reset", json={"seed": 0})
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "ValueError"
    assert client.get("/health").json()["session"] == "failed"
    assert client.get("/observation").status_code == 409


def test_websocket_envelopes_duplicates_and_errors(api):
    client, backend, _ = api
    with client.websocket_connect("/ws") as ws:
        reset = {"id": "one", "method": "reset", "params": {"seed": 0}}
        ws.send_json(reset)
        result = ws.receive_json()
        assert result["id"] == "one"
        assert decode_observation(result["result"]["observation"])["state"]["tick"] == 0
        ws.send_json(reset)
        assert ws.receive_json()["error"]["type"] == "DuplicateRequest"
        assert backend.reset_count == 1
        ws.send_json({"id": 2, "method": "step", "params": {"ticks": 0}})
        assert ws.receive_json()["error"]["type"] == "ValidationError"
        ws.send_json({"id": 2, "method": "step", "params": {"forward": True}})
        assert ws.receive_json()["error"]["type"] == "DuplicateRequest"
        assert backend.step_count == 0
        ws.send_json({"id": 3, "method": "step", "params": {"forward": True}})
        step = ws.receive_json()
        assert step["id"] == 3
        assert decode_observation(step["result"]["observation"])["state"]["tick"] == 1
        ws.send_json({"id": 4, "method": "observe", "params": {}})
        assert decode_observation(ws.receive_json()["result"]["observation"])["state"]["tick"] == 1
        ws.send_text("not JSON")
        assert ws.receive_json()["id"] is None
    # Connection lifetime does not implicitly close the shared session.
    assert client.get("/health").json()["session"] == "active"


def test_requests_serialize_without_blocking_event_loop(api):
    client, backend, _ = api
    client.post("/reset", json={"scenario": scenario(target=15).model_dump(mode="json")})
    backend.gate = threading.Event()
    with ThreadPoolExecutor(max_workers=3) as executor:
        first = executor.submit(client.post, "/step", json={"forward": True})
        assert backend.entered.wait(timeout=2)
        second = executor.submit(client.post, "/step", json={"forward": True})
        # OpenAPI requires no session lock; it must respond while a step is blocked.
        try:
            independent = executor.submit(client.get, "/openapi.json")
            assert independent.result(timeout=2).status_code == 200
        finally:
            backend.gate.set()
        results = [first.result(), second.result()]
    assert all(result.status_code == 200 for result in results)
    assert {result.json()["observation"]["state"]["tick"] for result in results} == {1, 2}
    assert backend.overlap is False


def test_lifespan_closes_backend():
    backend = ScriptedBackend()
    app = create_app(simulation_factory=lambda: Simulation(backend, record_dir=None))
    with TestClient(app) as client:
        client.post("/reset", json={"seed": 0})
    assert backend.close_count == 1


def test_codecs_preserve_all_sensor_values_and_dtypes():
    original = ScriptedBackend()._observation()
    original["depth"][:] = np.arange(24).reshape(4, 6) / 11
    decoded = decode_observation(encode_observation(original))
    for name in ("rgb", "depth", "segmentation"):
        assert decoded[name].dtype == original[name].dtype
        np.testing.assert_array_equal(decoded[name], original[name])
    assert decoded["sensors"] == original["sensors"]


def test_client_rejects_oversized_shapes_and_pickle_payloads():
    wire = encode_observation(ScriptedBackend()._observation())
    bad = copy.deepcopy(wire)
    bad["rgb"]["shape"] = [100_000, 100_000, 3]
    with pytest.raises(ValueError, match="limits"):
        decode_observation(bad)
    bad = copy.deepcopy(wire)
    stream = io.BytesIO()
    np.save(stream, np.full((4, 6), object(), dtype=object), allow_pickle=True)
    bad["depth"].update(data=base64.b64encode(stream.getvalue()).decode(), dtype="|O")
    with pytest.raises(ValueError, match="without pickle"):
        decode_observation(bad)
    bad = copy.deepcopy(wire)
    bad["rgb"]["data"] = "?bad base64"
    with pytest.raises(ValueError, match="base64"):
        decode_observation(bad)


def test_python_client_over_real_loopback_http(tmp_path):
    backend = ScriptedBackend()
    app = create_app(simulation_factory=lambda: Simulation(backend, record_dir=tmp_path))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", ws="none"))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        client = SimulationClient(f"http://127.0.0.1:{port}", timeout=5)
        assert client.health()["session"] == "idle"
        observation, info = client.reset(scenario())
        assert observation["state"]["tick"] == 0
        assert info["scenario"]["id"] == "network"
        observation, reward, terminated, truncated, info = client.step(
            Action(forward=True, ticks=2)
        )
        assert observation["state"]["tick"] == 2
        assert terminated and not truncated and reward > 0
        np.testing.assert_array_equal(client.observe()["depth"], observation["depth"])
        with pytest.raises(RemoteError) as error:
            client.step(Action())
        assert error.value.status == 409
        client.close()
        assert client.health()["session"] == "idle"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
    assert not thread.is_alive()
