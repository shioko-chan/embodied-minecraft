"""Python control of pinned Mindcraft skills in a published CraftGround world.

Mineflayer bots run in real time on the integrated server. A caller that also
owns a synchronous CraftGround Simulation should pass a pump callback while
waiting, usually one no-op simulation tick per callback. Skill receipts contain
the bot's provisional inventory; benchmark item counts need server verification.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

from .minecollab import MineCollabTask
from .models import Action, HighLevelAction

_MARKER = "MCSOCIETY:"
_NAME = re.compile(r"^[A-Za-z0-9_]{1,16}$")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MindcraftBridge:
    """One local process and 1–8 Mineflayer bots; requests execute serially."""

    def __init__(
        self,
        *,
        lan_port: int = 55916,
        agents: tuple[str, ...] = ("AgentA",),
        source_root: str | Path = _PROJECT_ROOT,
    ):
        if type(lan_port) is not int or not 1024 <= lan_port <= 65535:
            raise ValueError("lan_port must be a non-privileged TCP port")
        if (
            not 1 <= len(agents) <= 8 or len(set(agents)) != len(agents)
            or not all(isinstance(name, str) and _NAME.fullmatch(name) for name in agents)
        ):
            raise ValueError("agents must be 1–8 unique Minecraft usernames")
        self.lan_port = lan_port
        self.agents = tuple(agents)
        self.source_root = Path(source_root).resolve()
        self.process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._messages: queue.Queue[dict] = queue.Queue()
        self._diagnostics: deque[str] = deque(maxlen=24)
        self._request_id = 0
        self._pending: int | None = None
        self._ready = False
        self._task: MineCollabTask | None = None

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    @property
    def task(self) -> MineCollabTask | None:
        return self._task

    def _read_output(self, output) -> None:
        for line in output:
            line = line.rstrip("\r\n")
            if line.startswith(_MARKER):
                try:
                    message = json.loads(line[len(_MARKER):])
                    if isinstance(message, dict):
                        self._messages.put(message)
                        continue
                except json.JSONDecodeError:
                    pass
            self._diagnostics.append(line)
        self._messages.put({"event": "exit"})

    def _next(self, deadline: float, pump: Callable[[], Any] | None) -> dict:
        while True:
            try:
                message = self._messages.get_nowait()
            except queue.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Mindcraft operation exceeded its deadline")
                if self.process is None or self.process.poll() is not None:
                    raise RuntimeError(
                        "Mindcraft process exited before responding: "
                        + " | ".join(self._diagnostics)[-1000:]
                    )
                if pump is not None:
                    pump()
                try:
                    message = self._messages.get(timeout=min(0.05, remaining))
                except queue.Empty:
                    continue
            if message.get("event") == "fatal":
                raise RuntimeError(f"Mindcraft bot failed: {message.get('error')}")
            if message.get("event") == "exit":
                raise RuntimeError(
                    "Mindcraft process exited: " + " | ".join(self._diagnostics)[-1000:]
                )
            return message

    def start(
        self, *, pump: Callable[[], Any] | None = None, timeout: float = 45
    ) -> dict:
        if self.process is not None:
            raise RuntimeError("Mindcraft bridge has already started")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        script = self.source_root / "integrations/mindcraft/bridge.mjs"
        upstream = self.source_root / ".runtime/upstream/mindcraft/src/agent/library/skills.js"
        if not script.is_file() or not upstream.is_file():
            raise RuntimeError("Run bash scripts/bootstrap_mindcraft.sh first")
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("Node.js is required for Mindcraft skills")
        deadline = time.monotonic() + timeout
        while True:
            try:
                with socket.create_connection(("127.0.0.1", self.lan_port), timeout=0.2):
                    break
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Minecraft LAN port {self.lan_port} did not open"
                    ) from error
                if pump is not None:
                    pump()
                else:
                    time.sleep(0.05)
        environment = os.environ.copy()
        environment.update(
            MCSOCIETY_LAN_PORT=str(self.lan_port),
            MCSOCIETY_AGENT_NAMES=json.dumps(self.agents),
        )
        self.process = subprocess.Popen(
            [node, str(script)],
            cwd=self.source_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._reader = threading.Thread(
            target=self._read_output, args=(self.process.stdout,), daemon=True
        )
        self._reader.start()
        try:
            message = self._next(deadline, pump)
            if message.get("event") != "ready" or message.get("agents") != list(self.agents):
                raise RuntimeError(f"Unexpected Mindcraft startup response: {message}")
            self._ready = True
            return message
        except BaseException:
            self.close()
            raise

    def _send(self, request: dict) -> int:
        if not self._ready or self.process is None or self.process.stdin is None:
            raise RuntimeError("Start Mindcraft bridge after publishing the Minecraft world")
        if self._pending is not None:
            raise RuntimeError("The previous Mindcraft action is still running")
        self._request_id += 1
        request["id"] = self._request_id
        try:
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.close()
            raise RuntimeError("Mindcraft process stopped before accepting action") from error
        self._pending = self._request_id
        return self._request_id

    def submit(self, action: HighLevelAction | dict) -> int:
        action = (
            action if isinstance(action, HighLevelAction) else HighLevelAction.model_validate(action)
        )
        if action.agent not in self.agents:
            raise ValueError(f"Unknown Mindcraft agent: {action.agent}")
        return self._send(action.model_dump(mode="json", exclude_none=True))

    def result(
        self, *, pump: Callable[[], Any] | None = None, timeout: float = 60
    ) -> dict:
        if self._pending is None:
            raise RuntimeError("No Mindcraft action is pending")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        expected = self._pending
        try:
            response = self._next(time.monotonic() + timeout, pump)
            if response.get("id") != expected:
                raise RuntimeError(f"Unexpected Mindcraft response id: {response}")
            if "error" in response:
                raise RuntimeError(f"Mindcraft action failed: {response['error']}")
            return response
        except BaseException:
            # An uncertain action must not be retried against the same world.
            self.close()
            raise
        finally:
            self._pending = None

    def execute(
        self,
        action: HighLevelAction | dict,
        *,
        pump: Callable[[], Any] | None = None,
        timeout: float = 60,
    ) -> dict:
        self.submit(action)
        return self.result(pump=pump, timeout=timeout)

    def execute_in_simulation(
        self, simulation, action: HighLevelAction | dict, *, timeout: float = 60
    ) -> tuple[dict, tuple]:
        """Run a peer skill while recording each intervening CraftGround tick.

        The final completed tick samples the Minecraft state after the skill
        receipt. Peer inventory in the receipt remains provisional until a
        server-authoritative query confirms it.
        """
        action = (
            action if isinstance(action, HighLevelAction) else HighLevelAction.model_validate(action)
        )
        self.submit(action)
        ticks = 0

        def advance() -> None:
            nonlocal ticks
            _, _, terminated, truncated, _ = simulation.step_peer(action)
            ticks += 1
            if terminated or truncated:
                raise RuntimeError("Simulation episode ended while Mindcraft skill was running")

        receipt = self.result(pump=advance, timeout=timeout)
        transition = simulation.step_peer(action, phase="completed")
        receipt["simulation_ticks"] = ticks + 1
        if action.kind == "place_block" and receipt.get("success"):
            target = list(action.position)
            expected = f"minecraft:{action.block}"

            def visible_block() -> str | None:
                observation = transition[0]
                return next(
                    (
                        block["block"]
                        for block in observation["state"].get("nearby_blocks", [])
                        if block["position"] == target
                    ),
                    None,
                )

            actual = visible_block()
            deadline = time.monotonic() + 3
            while (
                actual is not None and actual != expected
                and not (transition[2] or transition[3])
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
                transition = simulation.step_peer(action, phase="completed")
                receipt["simulation_ticks"] += 1
                actual = visible_block()
            # None means the target was outside CraftGround's local 3x3x3
            # block sensor; false means it was in view but not confirmed.
            receipt["observed_by_craftground"] = (
                None if actual is None else actual == expected
            )
        return receipt, transition

    def observe(self, *, pump: Callable[[], Any] | None = None) -> dict:
        self._send({"agent": self.agents[0], "kind": "observe"})
        return self.result(pump=pump)["agents"]

    def block_at(
        self, agent: str, position: tuple[int, int, int],
        *, pump: Callable[[], Any] | None = None
    ) -> str | None:
        if agent not in self.agents:
            raise ValueError(f"Unknown Mindcraft agent: {agent}")
        if len(position) != 3 or any(type(value) is not int for value in position):
            raise ValueError("position must be three integer coordinates")
        self._send({"agent": agent, "kind": "block_at", "position": position})
        return self.result(pump=pump)["block"]

    def configure_task(self, simulation, source: str, task_id: str) -> dict:
        """Prepare an upstream MineCollab techtree task in the shared world.

        Setup uses logged, privileged world edits. The Mindcraft Task validator
        remains upstream; its Mineflayer inventory view is provisional.
        """
        if self._task is not None:
            raise RuntimeError("A MineCollab task is already configured")
        task = MineCollabTask.load(
            source, task_id, self.agents, source_root=self.source_root
        )
        for edit in task.setup_edits:
            simulation.intervene(edit)
        expected = task.data.get("initial_inventory", {})
        for _ in range(40):
            states = self.observe(pump=lambda: simulation.step(Action()))
            if all(
                all(
                    states[name]["inventory"].get(item, 0) >= amount
                    for item, amount in expected.get(str(index), {}).items()
                )
                for index, name in enumerate(self.agents)
            ):
                break
            simulation.step(Action())
        else:
            raise RuntimeError("Mindcraft did not observe the MineCollab initial inventory")
        self._send({
            "agent": self.agents[0], "kind": "configure_task", "task": task.data,
        })
        self.result()
        self._task = task
        return {**task.summary(), "assisted_setup": True}

    def evaluate_task(self, simulation=None) -> dict:
        """Score with Mindcraft, optionally checking target counts on the server."""
        if self._task is None:
            raise RuntimeError("Configure a MineCollab task before evaluation")
        self._send({"agent": self.agents[0], "kind": "evaluate_task"})
        result = self.result()
        if simulation is not None:
            target = self._task.data["target"]
            required = self._task.data.get("number_of_target", 1)
            counts = {
                name: simulation.inspect_item_count(name, target)
                for name in self.agents
                if not simulation.done
            }
            verified = len(counts) == len(self.agents) and all(
                count is not None for count in counts.values()
            )
            result["server_counts"] = counts
            result["server_verified"] = verified
            result["server_success"] = (
                any(count >= required for count in counts.values()) if verified else None
            )
        return {**self._task.summary(), **result}

    def close(self) -> None:
        self._ready = False
        self._pending = None
        self._task = None
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if process.stdout is not None:
            process.stdout.close()
        if self._reader is not None:
            self._reader.join(timeout=1)
            self._reader = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
