"""Bounded process supervision for a potentially blocking native game runtime.

Every reset starts a fresh spawn-process worker. A timed-out or failed command is
never retried: its outcome is uncertain, so the session is discarded. Shutdown
only addresses the owned worker and descendant identities actually observed
beneath it; it never searches for Java processes by name or scans unrelated games.
"""

from __future__ import annotations

import contextlib
import logging
import math
import multiprocessing as mp
import threading
import time
from multiprocessing.connection import Connection
from typing import Any, Self

import psutil

from .models import Action, EnvironmentConfig

logger = logging.getLogger(__name__)
_POLL_INTERVAL = 0.05
_CLOSE_TIMEOUT = 2.0


def _create_craftground(config: EnvironmentConfig):
    """Import the game adapter inside the worker, without importing its native runtime here."""
    from .craftground_backend import CraftGroundBackend

    return CraftGroundBackend(config)


def _worker(connection: Connection, config: EnvironmentConfig, backend_factory) -> None:
    backend = None
    request_id = None
    try:
        backend = backend_factory(config)
        connection.send({"kind": "ready", "capabilities": getattr(backend, "capabilities", {})})
        while True:
            request = connection.recv()
            request_id = request["id"]
            method, arguments = request["method"], request["arguments"]
            if method == "close":
                backend.close()
                backend = None
                connection.send({"kind": "result", "id": request_id, "value": None})
                return
            if method == "reset":
                result = backend.reset(**arguments)
            elif method == "step":
                result = backend.step(**arguments)
            else:
                raise ValueError(f"Unknown worker operation: {method}")
            connection.send({"kind": "result", "id": request_id, "value": result})
    except (EOFError, BrokenPipeError):
        return
    except BaseException as error:
        # Exceptions themselves may not be pickleable (including native errors).
        # Strings also preserve the original failure without importing its type.
        logger.exception("Game worker operation failed")
        with contextlib.suppress(EOFError, BrokenPipeError, OSError):
            connection.send(
                {
                    "kind": "error",
                    "id": request_id,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException:
                logger.exception("Game worker could not close its backend")
        connection.close()


class SupervisedBackend:
    """Backend protocol implemented by an isolated, deadline-bound child process.

    ``backend_factory(config)`` is a top-level pickleable callable, primarily for
    tests. Production uses CraftGround. The startup deadline includes spawn,
    construction and reset; the step deadline includes the complete response.
    Deadline expiry also spends at most two seconds on cooperative shutdown,
    followed by bounded termination of owned processes.
    """

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        *,
        startup_timeout: float = 600,
        step_timeout: float = 30,
        backend_factory=_create_craftground,
    ):
        for name, value in (("startup_timeout", startup_timeout), ("step_timeout", step_timeout)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        self.config = config or EnvironmentConfig()
        self.startup_timeout = float(startup_timeout)
        self.step_timeout = float(step_timeout)
        self.backend_factory = backend_factory
        self._context = mp.get_context("spawn")
        self._lock = threading.RLock()
        self._process: mp.Process | None = None
        self._worker_created_at: float | None = None
        self._connection: Connection | None = None
        self._owned_children: dict[int, float] = {}
        self._request_id = 0
        self._active = False
        self._capabilities: dict | None = None

    @property
    def capabilities(self) -> dict:
        if self._capabilities is not None:
            return dict(self._capabilities)
        if self.backend_factory is _create_craftground:
            # Constructing the lightweight adapter does not import CraftGround,
            # launch Java, or create a worker; only reset does those operations.
            return dict(_create_craftground(self.config).capabilities)
        return {"backend": "injected", "verified": False}

    @property
    def worker_pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    def _capture_children(self) -> None:
        process = self._process
        if process is None or process.pid is None:
            return
        # Creation times guard against PID reuse. Previously captured children
        # remain ownership roots even when a crashing worker has been reaped.
        roots = self._owned_processes()
        try:
            worker = psutil.Process(process.pid)
            if worker.create_time() == self._worker_created_at:
                roots.append(worker)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        for root in roots:
            try:
                children = root.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            for child in children:
                try:
                    self._owned_children[child.pid] = child.create_time()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

    def _receive(self, deadline: float, operation: str) -> dict:
        while True:
            self._capture_children()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Minecraft worker {operation} exceeded its deadline; reset is required"
                )
            try:
                if self._connection.poll(min(_POLL_INTERVAL, remaining)):
                    message = self._connection.recv()
                    if not isinstance(message, dict):
                        raise RuntimeError("Invalid Minecraft worker response")
                    if message.get("kind") == "error":
                        error = message.get("error", {})
                        raise RuntimeError(
                            f"Minecraft worker {error.get('type', 'Error')}: {error.get('message', '')}"
                        )
                    return message
            except (EOFError, BrokenPipeError, OSError) as error:
                self._process.join(timeout=0.1)
                raise RuntimeError(
                    f"Minecraft worker disconnected during {operation} (exit code {self._process.exitcode}); reset is required"
                ) from error
            if not self._process.is_alive():
                self._process.join(timeout=0.1)
                raise RuntimeError(
                    f"Minecraft worker exited during {operation} (exit code {self._process.exitcode}); reset is required"
                )

    def _call(self, method: str, arguments: dict[str, Any], deadline: float):
        self._request_id += 1
        request_id = self._request_id
        try:
            self._connection.send({"id": request_id, "method": method, "arguments": arguments})
        except (EOFError, BrokenPipeError, OSError) as error:
            raise RuntimeError("Minecraft worker connection closed; reset is required") from error
        response = self._receive(deadline, method)
        if response.get("kind") != "result" or response.get("id") != request_id:
            raise RuntimeError("Minecraft worker response sequence mismatch; reset is required")
        return response["value"]

    def reset(self, *, seed: int, commands: list[str], biome: str) -> dict:
        with self._lock:
            self.close()
            deadline = time.monotonic() + self.startup_timeout
            parent, child = self._context.Pipe(duplex=True)
            self._connection = parent
            self._process = self._context.Process(
                target=_worker,
                args=(child, self.config, self.backend_factory),
                name="mcsociety-minecraft",
                daemon=False,
            )
            try:
                self._process.start()
                with contextlib.suppress(psutil.NoSuchProcess):
                    self._worker_created_at = psutil.Process(self._process.pid).create_time()
                child.close()
                ready = self._receive(deadline, "startup")
                if ready.get("kind") != "ready" or not isinstance(ready.get("capabilities"), dict):
                    raise RuntimeError("Invalid Minecraft worker startup response")
                self._capabilities = ready["capabilities"]
                observation = self._call(
                    "reset", {"seed": seed, "commands": list(commands), "biome": biome}, deadline
                )
                self._active = True
                return observation
            except BaseException:
                child.close()
                self.close()
                raise

    def step(self, action: Action, *, first_tick: bool = True, commands=()) -> dict:
        with self._lock:
            if not self._active:
                raise RuntimeError("Call reset before stepping a closed or failed Minecraft worker")
            try:
                return self._call(
                    "step",
                    {"action": action, "first_tick": first_tick, "commands": list(commands)},
                    time.monotonic() + self.step_timeout,
                )
            except BaseException:
                self.close()
                raise

    def _owned_processes(self) -> list[psutil.Process]:
        result = []
        for pid, created_at in self._owned_children.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() == created_at:
                    result.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return result

    def close(self) -> None:
        with self._lock:
            self._active = False
            worker, connection = self._process, self._connection
            if worker is None:
                return
            self._capture_children()
            if worker.pid is not None and worker.is_alive():
                # A pending native call may never read close. Do not wait forever.
                with contextlib.suppress(EOFError, BrokenPipeError, OSError):
                    self._request_id += 1
                    connection.send({"id": self._request_id, "method": "close", "arguments": {}})
                deadline = time.monotonic() + _CLOSE_TIMEOUT
                while worker.is_alive() and time.monotonic() < deadline:
                    self._capture_children()
                    try:
                        # Drain a late step result so the worker can reach close.
                        if connection.poll(
                            min(_POLL_INTERVAL, max(0, deadline - time.monotonic()))
                        ):
                            connection.recv()
                    except (EOFError, BrokenPipeError, OSError):
                        break
                self._capture_children()
            children = self._owned_processes()
            for child in children:
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    child.terminate()
            if worker.pid is not None:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=0.5)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=0.5)
            _, alive = psutil.wait_procs(children, timeout=0.5)
            for child in alive:
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    child.kill()
            psutil.wait_procs(alive, timeout=0.5)
            connection.close()
            if worker.pid is not None and not worker.is_alive():
                worker.close()
            self._process = self._connection = None
            self._worker_created_at = None
            self._owned_children.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
