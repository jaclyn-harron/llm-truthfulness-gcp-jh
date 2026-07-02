"""Tiny MCP client helper: open a streamable-HTTP session, call a tool, return
the parsed JSON result. Used by the agents (they are MCP clients)."""

from __future__ import annotations

import json
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def call_mcp_tool(mcp_url: str, tool: str, arguments: dict[str, Any]) -> Any:
    """Call `tool` on the MCP server at `mcp_url` and return its result.

    Prefers the structured result; falls back to parsing the text content.
    """
    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)
            if getattr(result, "structuredContent", None):
                sc = result.structuredContent
                # FastMCP wraps non-dict returns under "result"; dicts pass through.
                return sc.get("result", sc) if isinstance(sc, dict) else sc
            if result.content:
                text = getattr(result.content[0], "text", None)
                if text is not None:
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
            return None


async def list_mcp_tools(mcp_url: str) -> list[str]:
    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [t.name for t in tools.tools]
