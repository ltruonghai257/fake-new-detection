"""Shared A2A server base module.

Wraps the Google A2A protocol SDK (``a2a-sdk[http-server,fastapi]``) around the
existing factcheck agent functions. Every agent file defines a small handler
subclass exposing ``agent_fn(state) -> dict`` (the state diff the agent
mutates) plus an :class:`AgentCardConfig`; this module provides the shared
task lifecycle (deserialize state -> run agent -> attach diff -> complete),
agent-card construction, serialization helpers, and the uvicorn factory used
by all 10 services (ports 9001-9010).

Task contract (see .planning/phases/03 CONTEXT.md, decisions D-01..D-15):

- ``Task.input`` carries the full serialized ``FactCheckState`` dict (D-01).
- ``TaskResult.output`` is the state diff — the keys the agent mutated (D-02).
- On exception the task ends in ``failed`` state with ``{"error": ...}`` (D-04).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
from fastapi import FastAPI

from a2a.helpers import (
    get_data_parts,
    new_artifact,
    new_data_part,
    new_task_from_user_message,
    new_text_message,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)

from .state import FactCheckState

logger = logging.getLogger(__name__)


# ── Agent Card configuration (D-15) ────────────────────────────────────────


@dataclass
class AgentCardConfig:
    """Per-agent values for the A2A Agent Card (shared format, per-agent data)."""

    name: str
    description: str
    version: str
    skills: List[dict] = field(default_factory=list)
    port: int = 9001


def build_agent_card(cfg: AgentCardConfig) -> AgentCard:
    """Build the A2A ``AgentCard`` protobuf for one agent service."""
    url = f"http://localhost:{cfg.port}"
    return AgentCard(
        name=cfg.name,
        description=cfg.description,
        version=cfg.version,
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        skills=[
            AgentSkill(
                id=s.get("id", ""),
                name=s.get("name", ""),
                description=s.get("description", ""),
            )
            for s in cfg.skills
        ],
    )


# ── Serialization helpers (D-03) ───────────────────────────────────────────


def _json_safe(value: Any, _seen: Optional[set] = None) -> Any:
    """Recursively convert a value to a JSON-safe form for ``Task.input``.

    Handles datetime/date (ISO-8601 string), Path (string), sets/tuples
    (list), and falls back to ``repr`` for anything orjson cannot encode
    (e.g. the in-memory ``EvidenceGraph``), matching the graceful-degrade
    style of the rest of the codebase.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _seen) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, _seen) for v in value]
    # non-serializable object: best-effort repr so the payload stays valid JSON
    logger.warning(
        "Non-JSON-safe value in state: %s — using repr()", type(value).__name__
    )
    return repr(value)


def serialize_state(state: FactCheckState) -> dict:
    """Deep-copy + JSON-safe conversion of a state dict for ``Task.input`` (D-01)."""
    return orjson.loads(orjson.dumps(_json_safe(dict(state))))


def deserialize_state(data: dict) -> FactCheckState:
    """Reconstruct a state dict from a JSON-safe payload (reverse of serialize_state)."""
    return dict(orjson.loads(orjson.dumps(data)))


# ── Base task handler (D-14) ───────────────────────────────────────────────


class BaseTaskHandler(AgentExecutor):
    """Base class wrapping a factcheck agent as an A2A agent service.

    Subclasses implement :meth:`agent_fn` (the existing agent function body,
    taking a ``FactCheckState`` and returning the state diff) and set
    ``agent_card_config``. The shared :meth:`execute` flow handles the task
    lifecycle: deserialize state (D-01) -> run agent -> attach the diff as
    the ``output`` artifact (D-02) -> mark completed; on exception mark
    ``failed`` with ``{"error": ...}`` (D-04).
    """

    agent_card_config: AgentCardConfig

    async def agent_fn(self, state: FactCheckState) -> dict:
        """Run the agent logic and return the state diff. May be sync or async."""
        raise NotImplementedError

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )
        await updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message(f"{self.agent_card_config.name} processing..."),
        )

        try:
            state = self._extract_state(context)
            diff = await self._run_agent(state)
            await updater.add_artifact(parts=[new_data_part(diff)], name="output")
            await updater.complete(
                message=new_text_message("Task completed successfully")
            )
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the server
            logger.exception("[%s] agent_fn failed", self.agent_card_config.name)
            await updater.add_artifact(
                parts=[new_data_part({"error": str(exc)})], name="output"
            )
            await updater.failed(message=new_text_message(f"Task failed: {exc}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await updater.cancel(message=new_text_message("Task canceled"))

    # ── helpers ────────────────────────────────────────────────────────────

    def _extract_state(self, context: RequestContext) -> FactCheckState:
        """Pull the serialized FactCheckState from the incoming message parts."""
        parts = list((context.message.parts if context.message else None) or [])
        data_parts = get_data_parts(parts)
        if data_parts:
            data = data_parts[0]
            if isinstance(data, dict):
                return deserialize_state(data)
            if isinstance(data, str):
                return deserialize_state(orjson.loads(data))
        raw = context.get_user_input().strip()
        if raw:
            return deserialize_state(orjson.loads(raw))
        raise ValueError(
            "Task input must carry the serialized FactCheckState in a data part "
            "(contract D-01)"
        )

    async def _run_agent(self, state: FactCheckState) -> dict:
        result = self.agent_fn(state)
        if inspect.isawaitable(result):
            result = await result
        return result


# ── App / server factories ─────────────────────────────────────────────────


def create_app(handler: BaseTaskHandler, cfg: AgentCardConfig) -> FastAPI:
    """Build the FastAPI app with A2A JSON-RPC + REST + agent-card routes."""
    agent_card = build_agent_card(cfg)
    request_handler = DefaultRequestHandler(
        agent_executor=handler,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    app = FastAPI(title=cfg.name, version=cfg.version)
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card),
        jsonrpc_routes=create_jsonrpc_routes(request_handler, rpc_url="/"),
        rest_routes=create_rest_routes(request_handler),
    )

    # The SDK serves the card at /.well-known/agent-card.json; expose the
    # plan-contract path /.well-known/agent.json as an alias (contract D-12).
    # Must be inserted BEFORE the REST `/{tenant}` mount or it never matches.
    from a2a.server.request_handlers.response_helpers import agent_card_to_dict
    from fastapi.responses import JSONResponse
    from starlette.routing import Route

    async def _agent_card_alias(request):
        return JSONResponse(agent_card_to_dict(agent_card))

    app.router.routes.insert(
        0, Route("/.well-known/agent.json", _agent_card_alias, methods=["GET"])
    )

    return app


def run_server(handler: BaseTaskHandler, cfg: AgentCardConfig) -> None:
    """Run the agent service with uvicorn on ``cfg.port`` (blocks)."""
    import uvicorn

    uvicorn.run(
        create_app(handler, cfg), host="127.0.0.1", port=cfg.port, log_level="info"
    )
