from simple_agentic_system.context import ContextManager, SqliteSessionStore
from simple_agentic_system.llm.base import LLMMessage


async def test_state_describe():
    cm = ContextManager()
    cm.state.set("foo", {"a": 1}, summary="foo=1")
    assert cm.describe_state() == {"foo": "foo=1"}


async def test_memory_search_and_tool():
    cm = ContextManager()
    await cm.memory.add("the sky is blue")
    await cm.memory.add("bananas are yellow")
    memory_tool = cm.as_memory_tool()

    result = await memory_tool.run(query="sky color")
    assert not result.is_error
    assert "sky is blue" in result.output


async def test_session_save_and_load(tmp_path):
    store = SqliteSessionStore(str(tmp_path / "sessions.sqlite3"))

    cm = ContextManager(session_store=store, session_id="s1")
    await cm.add_message(LLMMessage(role="user", content="hello"))
    await cm.save_session()

    cm2 = ContextManager(session_store=store, session_id="s1")
    loaded = await cm2.load_session()
    assert loaded is True
    messages = await cm2.get_messages()
    assert messages[0].content == "hello"


async def test_load_session_missing_returns_false(tmp_path):
    store = SqliteSessionStore(str(tmp_path / "sessions.sqlite3"))
    cm = ContextManager(session_store=store, session_id="does-not-exist")
    assert await cm.load_session() is False
