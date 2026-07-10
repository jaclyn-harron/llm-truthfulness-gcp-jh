"""Tiny MCP client helper: open a streamable-HTTP session, call a tool, return
the parsed JSON result. Used by the agents (they are MCP clients)."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlparse

import google.auth
import google.auth.transport.requests
from google.oauth2 import id_token
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger("truthfulness.mcp_client")


def fetch_id_token(url: str) -> str:
    """Fetch a Google ID token from the metadata server or ADC for the given URL's audience."""
    try:
        parsed = urlparse(url)
        audience = f"{parsed.scheme}://{parsed.netloc}"
        credentials, _ = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        return id_token.fetch_id_token(auth_req, audience)
    except Exception as e:
        logger.debug("Could not fetch ID token for audience of %s: %s", url, e)
        return ""


async def call_mcp_tool(mcp_url: str, tool: str, arguments: dict[str, Any]) -> Any:
    """Call `tool` on the MCP server at `mcp_url` and return its result.

    Prefers the structured result; falls back to parsing the text content.
    """
    headers = {}
    tok = fetch_id_token(mcp_url)
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    async with streamablehttp_client(mcp_url, headers=headers) as (read, write, _):
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
    headers = {}
    tok = fetch_id_token(mcp_url)
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    async with streamablehttp_client(mcp_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [t.name for t in tools.tools]

