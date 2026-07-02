"""Service configuration from environment variables.

Every URL/port/token the agents and MCP server need is read from the
environment so the same image runs anywhere (locally via docker-compose, or on
Cloud Run where $PORT is injected).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class ServiceConfig:
    role: str = field(default_factory=lambda: _env("ROLE", "orchestrator"))
    # Cloud Run injects PORT; default 8080 locally.
    port: int = field(default_factory=lambda: int(_env("PORT", "8080")))
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))

    # URL this service advertises in its agent card / to peers. On Cloud Run set
    # to the public service URL; locally defaults to localhost:PORT.
    public_url: str = field(default_factory=lambda: _env("PUBLIC_URL", ""))

    # MCP tool server (consumed by every agent as an MCP client).
    mcp_url: str = field(default_factory=lambda: _env("MCP_URL", "http://localhost:8085/mcp"))

    # Peer agent base URLs (used by the orchestrator for A2A delegation).
    zero_shot_url: str = field(
        default_factory=lambda: _env("ZERO_SHOT_URL", "http://localhost:8081")
    )
    fine_tuned_url: str = field(
        default_factory=lambda: _env("FINE_TUNED_URL", "http://localhost:8082")
    )
    explainer_url: str = field(
        default_factory=lambda: _env("EXPLAINER_URL", "http://localhost:8083")
    )

    # Bearer token protecting the orchestrator's public endpoint.
    auth_token: str = field(default_factory=lambda: _env("API_AUTH_TOKEN", ""))

    # Reconciliation strategy when zero-shot and fine-tuned disagree:
    #   "fine_tuned" (default) -> defer to the trained model
    #   "confidence"           -> pick the higher-confidence verdict
    #   "zero_shot"            -> defer to zero-shot
    reconcile: str = field(default_factory=lambda: _env("RECONCILE", "fine_tuned"))

    def resolved_public_url(self) -> str:
        return self.public_url or f"http://localhost:{self.port}"


def get_service_config() -> ServiceConfig:
    return ServiceConfig()
