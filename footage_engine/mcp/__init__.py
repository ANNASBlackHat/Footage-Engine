"""MCP server for Footage Retrieval Engine."""

from typing import Any
from footage_engine.mcp.server import create_mcp_server, get_default_server, main

__all__ = ["mcp", "create_mcp_server", "get_default_server", "main"]


def __getattr__(name: str) -> Any:
    if name == "mcp":
        return get_default_server()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
