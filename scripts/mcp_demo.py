"""Demonstrate that the MCP tools work from a GENERIC MCP client (not our agents).

Usage:
    MCP_URL=http://localhost:8085/mcp python scripts/mcp_demo.py

Lists the server's tools and calls a couple of them. Any MCP-compatible client
(Claude, Cursor, etc.) can consume these the same way.
"""

from __future__ import annotations

import asyncio
import json
import os

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = os.environ.get("MCP_URL", "http://localhost:8085/mcp")


async def main() -> None:
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            sample = await session.call_tool("dataset_sample", {"n": 2})
            data = json.loads(sample.content[0].text)
            print("\ndataset_sample ->", len(data["points"]), "points")

            pred = await session.call_tool(
                "predict_zero_shot",
                {"points": data["points"], "labels": data["labels"]},
            )
            print("\npredict_zero_shot ->")
            print(json.dumps(json.loads(pred.content[0].text), indent=2)[:600])


if __name__ == "__main__":
    asyncio.run(main())
