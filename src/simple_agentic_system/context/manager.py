from __future__ import annotations

from typing import Any

from ..llm.base import LLMMessage, ToolCall
from ..tools.decorators import FunctionTool, tool
from .history import History, InMemoryHistory
from .memory import MemoryStore, NaiveMemoryStore
from .session import SessionSnapshot, SessionStore
from .state import InMemoryStateStore, StateStore


class ContextManager:
    """Facade composing History + StateStore + MemoryStore + optional SessionStore.

    RAG is exposed to the agent as a normal Tool (search_memory) via as_memory_tool(),
    not as a special code path — it's a tool like any other.
    """

    def __init__(
        self,
        *,
        history: History | None = None,
        state: StateStore | None = None,
        memory: MemoryStore | None = None,
        session_store: SessionStore | None = None,
        session_id: str = "default",
    ):
        self.history = history or InMemoryHistory()
        self.state = state or InMemoryStateStore()
        self.memory = memory or NaiveMemoryStore()
        self.session_store = session_store
        self.session_id = session_id

    async def add_message(self, message: LLMMessage) -> None:
        await self.history.add(message)

    async def get_messages(self) -> list[LLMMessage]:
        return await self.history.get_messages()

    def describe_state(self) -> dict[str, str]:
        return self.state.describe()

    def as_memory_tool(self) -> FunctionTool:
        memory = self.memory

        @tool(name="search_memory", description="Search past conversation memory for relevant snippets.")
        async def search_memory(query: str, k: int = 5) -> str:
            hits = await memory.search(query, k=k)
            if not hits:
                return "No relevant memories found."
            return "\n".join(f"- ({hit.score:.2f}) {hit.text}" for hit in hits)

        return search_memory

    async def save_session(self) -> None:
        """Persist history + a state summary. State *values* aren't restored on load —
        only history is authoritative for resuming a conversation; the state snapshot
        is there for inspection/debugging."""
        if self.session_store is None:
            return
        messages = await self.get_messages()
        snapshot = SessionSnapshot(
            session_id=self.session_id,
            data={"messages": [_message_to_dict(m) for m in messages], "state": self.state.describe()},
        )
        await self.session_store.save(snapshot)

    async def load_session(self) -> bool:
        if self.session_store is None:
            return False
        snapshot = await self.session_store.load(self.session_id)
        if snapshot is None:
            return False
        await self.history.clear()
        for raw in snapshot.data.get("messages", []):
            await self.history.add(_dict_to_message(raw))
        return True


def _message_to_dict(message: LLMMessage) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_call_id": message.tool_call_id,
        "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in message.tool_calls],
    }


def _dict_to_message(raw: dict[str, Any]) -> LLMMessage:
    return LLMMessage(
        role=raw["role"],
        content=raw.get("content", ""),
        tool_call_id=raw.get("tool_call_id"),
        tool_calls=[ToolCall(**tc) for tc in raw.get("tool_calls", [])],
    )
