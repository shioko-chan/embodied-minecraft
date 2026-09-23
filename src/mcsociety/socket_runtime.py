"""Inject owned socket transport into the pinned upstream environment.

CraftGround's SocketIPC constructor otherwise scans and terminates Java processes
owned by other experiments. We disable that scan only on our IPC instances and
retain upstream's existing socket protocol. No upstream module global is changed.
"""

from importlib.metadata import version
from types import FunctionType


def make_socket_environment(initial, **kwargs):
    if version("craftground") != "2.7.4":
        raise RuntimeError("Socket runtime injection requires pinned craftground==2.7.4")
    from craftground.environment.environment import CraftGroundEnvironment
    from craftground.environment.socket_ipc import SocketIPC

    class OwnedSocketIPC(SocketIPC):
        def remove_orphan_java_processes(self):
            """Other experiments' Java processes are never ours to terminate."""

    def bind_transport(method):
        namespace = {**method.__globals__, "SocketIPC": OwnedSocketIPC}
        bound = FunctionType(
            method.__code__, namespace, method.__name__, method.__defaults__, method.__closure__
        )
        bound.__kwdefaults__ = method.__kwdefaults__
        return bound

    class OwnedSocketEnvironment(CraftGroundEnvironment):
        __init__ = bind_transport(CraftGroundEnvironment.__init__)
        ensure_alive = bind_transport(CraftGroundEnvironment.ensure_alive)

    return OwnedSocketEnvironment(initial_env=initial, use_shared_memory=False, **kwargs)
