# SimpleAgenticSystem

A modular, from-scratch agentic framework skeleton in async Python. Every layer sits
behind a small interface with a working default implementation (in-memory / sqlite /
fake LLM), so the whole thing runs and is testable end-to-end with no API keys and no
Docker — and any single layer (LLM provider, tool transport, vector store, sandbox
image, tracer) can be swapped without touching the others.

## Quickstart

```bash
uv sync                      # install deps
cp .env.example .env         # fill in whichever API keys you actually have

uv run pytest -q             # 32 tests, no API keys / Docker required
uv run python examples/minimal_agent.py      # full agent loop, fake LLM
uv run python examples/self_repair_demo.py   # RetryingTool + agent-as-fixer
```

## Layers

```
llm/            LLMProvider protocol + LLMRouter (priority, retries, fallback).
                Providers: Anthropic, OpenAI, OpenRouter (OpenAI-compatible),
                FakeLLMProvider (scripted, for tests/examples).

tools/          Tool protocol + ToolRegistry (single source of truth for what an
                agent can call). @tool decorator wraps plain functions. Hybrid MCP
                bridge: serve_registry_as_mcp() exposes a registry as an MCP server,
                MCPToolSource pulls tools from an external MCP server into a registry
                — same registry, same tools, reachable by the main agent loop AND by
                code running in the sandbox. RetryingTool + RepairPolicy: generic
                self-repair wrapper around any Tool (sequential or parallel fixer
                attempts via an AgentAsTool "fixer").

prompt/         PromptPipeline: an ordered chain of PromptEnrichers that build the
                system prompt (static text, tool descriptions, state snapshot, ...).

context/        ContextManager composing History (conversation), StateStore
                (key -> value + short summary, injected into the prompt instead of
                dumping large values), MemoryStore (RAG — exposed to the agent as a
                normal search_memory Tool, not a special code path), and SessionStore
                (sqlite-backed save/load so a conversation can resume).

agent/          Agent: the tool-calling loop tying llm + tools + prompt + context
                together. AgentAsTool wraps any Agent as a Tool, so one agent can call
                another — including as the "fixer" for a RetryingTool.

sandbox/        DockerPool (semaphore-bounded) + SessionContainerRegistry (one
                container per session, idle-TTL reaped) + PythonExecTool ("run_python"
                tool executing LLM-authored code in the container — no agent runs
                inside it, just code). bridge/codegen.py turns the ToolRegistry into a
                typed python module (`tools.py`) so sandboxed code calls tools as
                plain functions; bridge/runtime.py (copied into the container) backs
                those calls with a real MCP call to the host's tools/server.py.

observability/  Tracer protocol wrapping every llm/tool/agent call as a span.
                NullTracer (default, zero-cost), LoggingTracer (stdlib logging, zero
                deps), LangfuseTracer (optional `[langfuse]` extra).

config.py       pydantic-settings, loads from .env.
```

## Known limitations

- `sandbox/` is fully implemented against the `docker` SDK but not run against a real
  Docker daemon in this pass (none was available in the dev environment) — no image is
  built yet either. `bridge/codegen.py`'s output is unit-tested (compiles, dispatches
  correctly) independent of Docker.
- `NaiveMemoryStore` does keyword-overlap search, not real embeddings — it exists so
  `MemoryStore` has a working default; swap in a real vector store behind the same
  interface when needed.
- `SqliteSessionStore` persists conversation history; `StateStore` values are not
  restored on load (only their text summaries, for inspection), since state values
  aren't guaranteed JSON-serializable.

## Extending

- **New LLM provider**: implement `LLMProvider.complete()` (see
  `llm/providers/openai_compatible.py` for the simplest example), add a
  `ProviderEntry` to your `LLMRouter`.
- **New tool**: `@tool` a function, or implement the `Tool` protocol directly for
  anything stateful; `registry.register(...)`.
- **New tracer backend**: implement `Tracer.span()` (see `observability/tracer.py`);
  pass it into `LLMRouter`, `ToolRegistry`, and `Agent`.
