"""Runs INSIDE the sandbox container as a sibling module to the generated tools.py.

Not part of the importable package on the host — its source is copied verbatim into
the container by sandbox/exec.py (via inspect.getsource), since it must run with only
the stdlib + `mcp` client available in the sandbox image, not the full host package.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

TOOLS_MCP_URL = os.environ.get("TOOLS_MCP_URL", "http://host.docker.internal:8765/mcp")


def call_tool(name: str, **kwargs: Any) -> Any:
    """Sync wrapper — sandboxed code is typically plain synchronous python — around an
    MCP tool call against the ToolRegistry exposed on the host (see tools/server.py)."""
    return asyncio.run(_call_tool_async(name, kwargs))


async def _call_tool_async(name: str, arguments: dict[str, Any]) -> Any:
    async with streamablehttp_client(TOOLS_MCP_URL) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=arguments)
            return "\n".join(block.text for block in result.content if getattr(block, "type", None) == "text")
