"""Simple concurrency lock for the chat backend."""

import asyncio

from app.actor import ActorBusyError


class ConvRegistry:
    def __init__(self, project_dir: str = "") -> None:
        self.project_dir = project_dir
        self._busy = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self._busy = False

    async def assert_available(self) -> None:
        if self._busy:
            raise ActorBusyError("上一条消息仍在回复")

    def set_busy(self, busy: bool) -> None:
        self._busy = busy

    async def invalidate(self, conv_id: str | None = None) -> None:
        self._busy = False


registry: ConvRegistry | None = None


def configure_registry(project_dir: str) -> ConvRegistry:
    global registry
    if registry is None:
        registry = ConvRegistry(project_dir)
    return registry


def get_registry() -> ConvRegistry:
    if registry is None:
        raise RuntimeError("Chat registry is not initialized")
    return registry
