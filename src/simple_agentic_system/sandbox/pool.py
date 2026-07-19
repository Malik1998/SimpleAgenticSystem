from __future__ import annotations

import asyncio
from dataclasses import dataclass

import docker
from docker.models.containers import Container


@dataclass
class DockerPoolConfig:
    image: str
    max_concurrent_containers: int = 4


class DockerPool:
    """Bounds how many sandbox containers run concurrently with a semaphore, and owns
    creation/teardown of individual containers. SessionContainerRegistry acquires/
    releases through this pool to give sessions a long-lived, session-affine container."""

    def __init__(self, config: DockerPoolConfig, client: docker.DockerClient | None = None):
        self._config = config
        self._client = client or docker.from_env()
        self._semaphore = asyncio.Semaphore(config.max_concurrent_containers)

    async def acquire(self) -> Container:
        await self._semaphore.acquire()
        try:
            return await asyncio.to_thread(
                self._client.containers.run,
                self._config.image,
                detach=True,
                tty=True,
                command="sleep infinity",
            )
        except Exception:
            self._semaphore.release()
            raise

    async def release(self, container: Container) -> None:
        try:
            await asyncio.to_thread(container.remove, force=True)
        finally:
            self._semaphore.release()
