"""Single entrypoint for every deployable unit. ROLE selects which one.

    ROLE=mcp | zero_shot | fine_tuned | explainer | orchestrator

The orchestrator additionally exposes a bearer-protected REST endpoint so it can
be exercised with a single curl, exactly as the challenge describes:

    curl -X POST $URL/verify -H "Authorization: Bearer $KEY" -d @batch.json
"""

from __future__ import annotations

import json
import logging
import uuid

import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from .agents import _orchestrator_handler, build_role
from .config import get_service_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("truthfulness.run")

PUBLIC_PATHS = ("/.well-known", "/health")


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Require `Authorization: Bearer <token>` on everything except the agent
    card and health check. No-op if API_AUTH_TOKEN is unset (local dev)."""

    def __init__(self, app, token: str):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        if not self.token or any(request.url.path.startswith(p) for p in PUBLIC_PATHS):
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {self.token}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def _build_a2a_app(cfg):
    card, executor = build_role(cfg)
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    return A2AStarletteApplication(agent_card=card, http_handler=handler).build()


def main() -> None:
    cfg = get_service_config()
    logger.info("starting role=%s port=%s public_url=%s", cfg.role, cfg.port, cfg.resolved_public_url())

    if cfg.role == "mcp":
        from . import mcp_server
        mcp_server.main()
        return

    if cfg.role == "demo":
        # Public browser demo UI; proxies to the orchestrator with the token
        # held server-side. No bearer middleware on this service.
        from .demo import build_app
        uvicorn.run(build_app(), host=cfg.host, port=cfg.port, log_level="info")
        return

    app = _build_a2a_app(cfg)

    async def health(_request: Request):
        return JSONResponse({"status": "ok", "role": cfg.role})

    app.add_route("/health", health, methods=["GET"])

    if cfg.role == "orchestrator":
        orchestrate = _orchestrator_handler(cfg)

        async def verify(request: Request):
            body = await request.json()
            body.setdefault("trace_id", uuid.uuid4().hex)
            logger.info("[%s] /verify n=%s", body["trace_id"], len(body.get("points", [])))
            result = await orchestrate(body)
            return JSONResponse(result)

        app.add_route("/verify", verify, methods=["POST"])

    app.add_middleware(BearerAuthMiddleware, token=cfg.auth_token)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
