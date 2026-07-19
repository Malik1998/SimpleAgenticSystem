from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from docker.models.containers import Container

from .pool import DockerPool


@dataclass
class _Entry:
    container: Container
    last_used: float = field(default_factory=time.monotonic)


class SessionContainerRegistry:
    """One container per session_id, reused across exec_docker calls, reaped after
    idle_ttl_seconds. Killing a container only drops its filesystem state — the
    conversation/tool state that matters lives in ContextManager, not the container —
    so a session can keep going after its container is recycled; a fresh one is
    created lazily on next use.
    """

    def __init__(self, pool: DockerPool, idle_ttl_seconds: float = 600):
        self._pool = pool
        self._idle_ttl_seconds = idle_ttl_seconds
        self._entries: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str) -> Container:
        async with self._lock:
            entry = self._entries.get(session_id)
            if entry is not None:
                entry.last_used = time.monotonic()
                return entry.container
            container = await self._pool.acquire()
            self._entries[session_id] = _Entry(container=container)
            return container

    async def release(self, session_id: str) -> None:
        async with self._lock:
            entry = self._entries.pop(session_id, None)
        if entry is not None:
            await self._pool.release(entry.container)

    async def reap_idle(self) -> list[str]:
        """Call periodically (e.g. a background task) to kill containers idle past
        idle_ttl_seconds. Returns the session_ids that were reaped."""
        now = time.monotonic()
        async with self._lock:
            expired = [sid for sid, e in self._entries.items() if now - e.last_used > self._idle_ttl_seconds]
        for session_id in expired:
            await self.release(session_id)
        return expired
