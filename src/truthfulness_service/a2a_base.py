"""Shared A2A scaffolding: an executor that speaks JSON, an agent-card builder,
a uvicorn server runner, and client helpers for capability discovery + task
delegation. Built on the official a2a-sdk (0.2.x)."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Awaitable, Callable

import httpx
import uvicorn
from a2a.client import A2ACardResolver, A2AClient
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    TextPart,
)

logger = logging.getLogger("truthfulness.a2a")

# A2A payloads carry a `trace_id` so a single request can be followed across the
# orchestrator -> predictor/explainer hops in the logs.
Handler = Callable[[dict], Awaitable[dict]]


def build_card(
    *, name: str, description: str, url: str, version: str,
    skill_id: str, skill_name: str, skill_description: str, tags: list[str],
) -> AgentCard:
    return AgentCard(
        name=name,
        description=description,
        url=url if url.endswith("/") else url + "/",
        version=version,
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text", "application/json"],
        default_output_modes=["text", "application/json"],
        skills=[
            AgentSkill(
                id=skill_id, name=skill_name, description=skill_description, tags=tags,
            )
        ],
    )


class JsonAgentExecutor(AgentExecutor):
    """Adapts a coroutine `handler(payload) -> result` into an A2A executor.
    The incoming message text is parsed as JSON; the result is returned as JSON
    text in an agent message."""

    def __init__(self, role: str, handler: Handler):
        self.role = role
        self.handler = handler

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        raw = context.get_user_input()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"_raw": raw}
        trace_id = payload.get("trace_id", "-")
        n = len(payload.get("points", []) or [])
        logger.info("[%s] role=%s received n=%s", trace_id, self.role, n)
        result = await self.handler(payload)
        result.setdefault("trace_id", trace_id)
        logger.info("[%s] role=%s done", trace_id, self.role)
        from a2a.utils import new_agent_text_message

        await event_queue.enqueue_event(new_agent_text_message(json.dumps(result)))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancellation not supported")


def serve(card: AgentCard, executor: AgentExecutor, *, host: str, port: int) -> None:
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=card, http_handler=handler).build()
    uvicorn.run(app, host=host, port=port, log_level="info")


# --------------------------- client side (delegation) ------------------------

async def discover_and_call(base_url: str, payload: dict, *, timeout: float = 120.0) -> dict:
    """Discover an agent via its card, then delegate a task over A2A and return
    the parsed JSON result. This is genuine A2A: we fetch the agent card and
    invoke the agent as a first-class participant, not a bare HTTP call."""
    async with httpx.AsyncClient(timeout=timeout) as hx:
        # Genuine A2A capability discovery: fetch the peer's agent card.
        resolver = A2ACardResolver(httpx_client=hx, base_url=base_url)
        card = await resolver.get_agent_card()
        logger.debug("discovered %s skills=%s", card.name, [s.id for s in card.skills])
        # Deliver the task to the endpoint we discovered it at (robust to the
        # card's self-advertised URL, which may differ on some platforms).
        client = A2AClient(httpx_client=hx, url=base_url, agent_card=card)
        msg = Message(
            role=Role.user,
            message_id=uuid.uuid4().hex,
            parts=[Part(root=TextPart(text=json.dumps(payload)))],
        )
        req = SendMessageRequest(
            id=uuid.uuid4().hex, params=MessageSendParams(message=msg)
        )
        resp = await client.send_message(req)
        return _extract_json(resp)


def _extract_json(resp) -> dict:
    """Pull the JSON text out of an A2A SendMessageResponse (message or task)."""
    root = resp.root
    result = getattr(root, "result", None)
    if result is None:
        raise RuntimeError(f"A2A error: {getattr(root, 'error', root)}")
    parts = getattr(result, "parts", None)
    if parts is None and getattr(result, "status", None) is not None:
        parts = result.status.message.parts
    for p in parts or []:
        text = getattr(p.root, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"_raw": text}
    return {}
