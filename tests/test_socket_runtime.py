"""Pin the transport ownership boundary without starting a game."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("craftground") is None,
    reason="optional Minecraft runtime is not installed",
)


def test_transport_injection_is_scoped_and_reused_after_restart(monkeypatch, tmp_path):
    import craftground.environment.environment as upstream
    from craftground.environment.socket_ipc import SocketIPC
    from craftground.initial_environment_config import InitialEnvironmentConfig

    from mcsociety.socket_runtime import make_socket_environment

    constructor = upstream.CraftGroundEnvironment.__init__
    ensure_alive = upstream.CraftGroundEnvironment.ensure_alive

    def forbidden_scan(_):
        raise AssertionError("Never scan or terminate another experiment's Java processes")

    monkeypatch.setattr(SocketIPC, "remove_orphan_java_processes", forbidden_scan)
    gradlew = tmp_path / "gradlew"
    gradlew.write_text("#!/bin/sh\nexit 0\n")
    gradlew.chmod(0o755)
    env = make_socket_environment(
        InitialEnvironmentConfig(image_width=96, image_height=64),
        env_path=str(tmp_path),
        port=18473,
    )
    try:
        assert isinstance(env.ipc, SocketIPC)
        assert type(env.ipc) is not SocketIPC
        assert upstream.SocketIPC is SocketIPC
        assert upstream.CraftGroundEnvironment.__init__ is constructor
        assert upstream.CraftGroundEnvironment.ensure_alive is ensure_alive
        assert constructor.__globals__["SocketIPC"] is SocketIPC
        assert ensure_alive.__globals__["SocketIPC"] is SocketIPC
        first_transport = env.ipc
        starts = []
        monkeypatch.setattr(env, "start_server", lambda seed: starts.append(seed))
        env.ensure_alive(False, [], 13)
        assert starts == [13]
        assert env.ipc is not first_transport
        assert type(env.ipc) is type(first_transport)
        env.ipc.remove_orphan_java_processes()
    finally:
        env.close()
