"""Persistent spatial, episodic, and declarative skill memory.

Spatial and episodic memories are isolated by both agent and world. Skills are
global reusable JSON action recipes; this module never evaluates code.
"""

# Data validation uses one ValueError contract for malformed records and inputs.
# ruff: noqa: TRY004

from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any, Self


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _position(position: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(position, (tuple, list)) or len(position) != 3:
        raise ValueError("position must contain x, y, z")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in position):
        raise ValueError("position coordinates must be numbers")
    result = tuple(float(v) for v in position)
    if not all(math.isfinite(v) for v in result):
        raise ValueError("position coordinates must be finite")
    return result


def _json(value: Any) -> str:
    def validate(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                validate(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                validate(child)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise ValueError("memory values must be JSON data")

    validate(value)
    return json.dumps(value, allow_nan=False, separators=(",", ":"), ensure_ascii=False)


class MemoryStore:
    """SQLite memory store; all returned records are independent dictionaries."""

    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS places (
                agent_id TEXT NOT NULL, world_id TEXT NOT NULL, name TEXT NOT NULL,
                x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
                resources TEXT NOT NULL,
                PRIMARY KEY (agent_id, world_id, name)
            );
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL, world_id TEXT NOT NULL,
                event TEXT NOT NULL, tick INTEGER NOT NULL, details TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS episode_lookup
                ON episodes(agent_id, world_id, tick DESC, id DESC);
            CREATE TABLE IF NOT EXISTS skills (
                name TEXT PRIMARY KEY, description TEXT NOT NULL, actions TEXT NOT NULL
            );
            """
        )

    def remember_place(
        self,
        agent_id: str,
        world_id: str,
        name: str,
        position: tuple[float, float, float],
        resources: list[str],
    ) -> dict[str, Any]:
        """Insert or update a named place within one agent's world memory."""
        for value, label in ((agent_id, "agent_id"), (world_id, "world_id"), (name, "name")):
            _text(value, label)
        coordinates = _position(position)
        if not isinstance(resources, list):
            raise ValueError("resources must be a list of strings")
        for resource in resources:
            _text(resource, "resource")
        encoded = _json(resources)
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, world_id, name) DO UPDATE SET
                x=excluded.x, y=excluded.y, z=excluded.z, resources=excluded.resources""",
                (agent_id, world_id, name, *coordinates, encoded),
            )
        return {
            "agent_id": agent_id,
            "world_id": world_id,
            "name": name,
            "position": list(coordinates),
            "resources": list(resources),
        }

    def nearby_places(
        self,
        agent_id: str,
        world_id: str,
        position: tuple[float, float, float],
        radius: float,
    ) -> list[dict[str, Any]]:
        """Return places inside a 3-D Euclidean radius, nearest first."""
        _text(agent_id, "agent_id")
        _text(world_id, "world_id")
        x, y, z = _position(position)
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise ValueError("radius must be a nonnegative finite number")
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("radius must be a nonnegative finite number")
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM places WHERE agent_id=? AND world_id=?
                AND x BETWEEN ? AND ? AND y BETWEEN ? AND ? AND z BETWEEN ? AND ?""",
                (
                    agent_id,
                    world_id,
                    x - radius,
                    x + radius,
                    y - radius,
                    y + radius,
                    z - radius,
                    z + radius,
                ),
            ).fetchall()
        records = []
        for row in rows:
            coordinates = [row["x"], row["y"], row["z"]]
            distance = math.dist((x, y, z), coordinates)
            if distance <= radius:
                records.append(
                    {
                        "agent_id": row["agent_id"],
                        "world_id": row["world_id"],
                        "name": row["name"],
                        "position": coordinates,
                        "resources": json.loads(row["resources"]),
                        "distance": distance,
                    }
                )
        return sorted(records, key=lambda record: (record["distance"], record["name"]))

    def add_episode(
        self,
        agent_id: str,
        world_id: str,
        event: str,
        tick: int,
        details: dict[str, Any] | None = None,
    ) -> int:
        """Append an experience and return its stable database id."""
        for value, label in ((agent_id, "agent_id"), (world_id, "world_id"), (event, "event")):
            _text(value, label)
        if isinstance(tick, bool) or not isinstance(tick, int) or not 0 <= tick < 2**63:
            raise ValueError("tick must be a nonnegative 64-bit integer")
        if details is None:
            details = {}
        if not isinstance(details, dict):
            raise ValueError("details must be a JSON object")
        encoded = _json(details)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "INSERT INTO episodes(agent_id, world_id, event, tick, details) VALUES (?, ?, ?, ?, ?)",
                (agent_id, world_id, event, tick, encoded),
            )
            return int(cursor.lastrowid)

    def episodes(self, agent_id: str, world_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Return experiences by decreasing tick, with newest insert breaking ties."""
        _text(agent_id, "agent_id")
        _text(world_id, "world_id")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM episodes WHERE agent_id=? AND world_id=? ORDER BY tick DESC, id DESC LIMIT ?",
                (agent_id, world_id, limit),
            ).fetchall()
        return [dict(row) | {"details": json.loads(row["details"])} for row in rows]

    def save_skill(self, name: str, description: str, actions: list[dict[str, Any]]) -> None:
        """Store a JSON recipe. Consumers decide which declared actions are allowed."""
        _text(name, "name")
        _text(description, "description")
        if (
            not isinstance(actions, list)
            or not actions
            or any(not isinstance(a, dict) for a in actions)
        ):
            raise ValueError("actions must be a nonempty list of JSON objects")
        encoded = _json(actions)
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO skills VALUES (?, ?, ?) ON CONFLICT(name) DO UPDATE SET
                description=excluded.description, actions=excluded.actions""",
                (name, description, encoded),
            )

    def get_skill(self, name: str) -> dict[str, Any] | None:
        _text(name, "name")
        with self._lock:
            row = self._connection.execute("SELECT * FROM skills WHERE name=?", (name,)).fetchone()
        return None if row is None else dict(row) | {"actions": json.loads(row["actions"])}

    def list_skills(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM skills ORDER BY name").fetchall()
        return [dict(row) | {"actions": json.loads(row["actions"])} for row in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
