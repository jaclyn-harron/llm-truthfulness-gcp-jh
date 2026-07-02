"""Part 2: multi-agent, deployed truthfulness service.

A small network of A2A agents (orchestrator + zero-shot / fine-tuned predictors
+ explainer) that consume capabilities as MCP tools served by an MCP server. All
five units run from a single image; the ROLE env var selects which one.

    ROLE=mcp           python -m truthfulness_service.run   # MCP tool server
    ROLE=zero_shot     python -m truthfulness_service.run   # A2A predictor agent
    ROLE=fine_tuned    python -m truthfulness_service.run   # A2A predictor agent
    ROLE=explainer     python -m truthfulness_service.run   # A2A explainer agent
    ROLE=orchestrator  python -m truthfulness_service.run   # A2A entry point
"""

__all__ = ["config", "payloads"]
