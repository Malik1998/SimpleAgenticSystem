from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import aiosqlite


@dataclass
class SessionSnapshot:
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)


class SessionStore(Protocol):
    async def save(self, snapshot: SessionSnapshot) -> None: ...
    async def load(self, session_id: str) -> SessionSnapshot | None: ...
    async def delete(self, session_id: str) -> None: ...


class SqliteSessionStore:
    """Default SessionStore: one JSON blob per session_id in a local sqlite file.
    Swap for a Postgres-backed implementation later without touching SessionStore callers."""

    def __init__(self, db_path: str = "sessions.sqlite3"):
        self._db_path = db_path
        self._schema_ready = False

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, data TEXT NOT NULL)"
            )
            await db.commit()
        self._schema_ready = True

    async def save(self, snapshot: SessionSnapshot) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO sessions (session_id, data) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET data = excluded.data",
                (snapshot.session_id, json.dumps(snapshot.data)),
            )
            await db.commit()

    async def load(self, session_id: str) -> SessionSnapshot | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT data FROM sessions WHERE session_id = ?", (session_id,))
            row = await cursor.fetchone()
        if row is None:
            return None
        return SessionSnapshot(session_id=session_id, data=json.loads(row[0]))

    async def delete(self, session_id: str) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            await db.commit()
