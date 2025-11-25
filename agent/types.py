"""State models shared across the wtchtwr agent."""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


@dataclass
class ThinkingStep:
    """Lightweight trace item describing one reasoning phase."""

    phase: str
    title: str
    detail: str
    meta: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: Optional[int] = None

    def to_payload(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        payload = {
            "phase": self.phase,
            "title": self.title,
            "detail": self.detail,
            "meta": dict(self.meta or {}),
            "elapsed_ms": self.elapsed_ms,
        }
        return payload


@dataclass
class GraphState:
    """Conversation state shared across LangGraph nodes."""

    query: str
    tenant: Optional[str] = None
    intent: Optional[str] = None
    scope: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    sql: Dict[str, Any] = field(default_factory=dict)
    rag: Dict[str, Any] = field(default_factory=dict)
    result_bundle: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    model_used: Optional[str] = None
    timestamp: Optional[str] = None
    start_time: Optional[float] = None
    raw_input: Dict[str, Any] = field(default_factory=dict)
    guardrail_blocked: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)
    thinking: List[ThinkingStep] = field(default_factory=list)
    debug_thinking: bool = False


State = Dict[str, Any]


def state_to_graph(state: State) -> GraphState:
    """Convert a legacy mapping state into GraphState."""
    known = {
        "query",
        "tenant",
        "intent",
        "scope",
        "filters",
        "plan",
        "sql",
        "rag",
        "result_bundle",
        "telemetry",
        "memory",
        "history",
        "model_used",
        "timestamp",
        "_start_time",
        "_input",
        "guardrail_blocked",
        "content",
        "answer_text",
        "thinking",
        "debug_thinking",
    }
    extras = {k: v for k, v in state.items() if k not in known}

    content = (state.get("content") or "").strip()
    answer_text = (state.get("answer_text") or "").strip()
    if content:
        extras.setdefault("content", content)
    if answer_text:
        extras.setdefault("answer_text", answer_text)
    if content or answer_text:
        _LOGGER.warning(
            "[GRAPH_STATE_PATCH] Preserved content=%s answer_text=%s",
            bool(content),
            bool(answer_text),
        )

    sql_state_raw = state.get("sql", {}) or {}
    sql_state = {
        "text": sql_state_raw.get("text"),
        "df": sql_state_raw.get("df"),
        "explain": sql_state_raw.get("explain"),
        "table": sql_state_raw.get("table"),
        "rows": sql_state_raw.get("rows", []),
        "columns": sql_state_raw.get("columns", []),
        "markdown_table": sql_state_raw.get("markdown_table", ""),
    }

    raw_thinking = state.get("thinking") or []
    thinking_steps: List[ThinkingStep] = []
    if isinstance(raw_thinking, list):
        for item in raw_thinking:
            step = None
            if isinstance(item, ThinkingStep):
                step = item
            elif isinstance(item, dict):
                try:
                    step = ThinkingStep(
                        phase=str(item.get("phase") or "unknown"),
                        title=str(item.get("title") or "").strip() or "Untitled step",
                        detail=str(item.get("detail") or "").strip(),
                        meta=dict(item.get("meta") or {}),
                        elapsed_ms=item.get("elapsed_ms"),
                    )
                except Exception:
                    _LOGGER.debug("Unable to coerce thinking step payload: %s", item, exc_info=True)
            if step:
                thinking_steps.append(step)

    debug_flag = state.get("debug_thinking")
    if debug_flag is None and isinstance(state.get("_input"), dict):
        debug_flag = state["_input"].get("debug_thinking")

    return GraphState(
        query=state.get("query", ""),
        tenant=state.get("tenant"),
        intent=state.get("intent"),
        scope=state.get("scope"),
        filters=state.get("filters", {}) or {},
        plan=state.get("plan", {}) or {},
        sql=sql_state,
        rag=state.get("rag", {}) or {},
        result_bundle=state.get("result_bundle", {}) or {},
        telemetry=state.get("telemetry", {}) or {},
        memory=state.get("memory", {}) or {},
        history=state.get("history", []) or [],
        model_used=state.get("model_used"),
        timestamp=state.get("timestamp"),
        start_time=state.get("_start_time"),
        raw_input=state.get("_input", {}) or {},
        guardrail_blocked=bool(state.get("guardrail_blocked", False)),
        extras=extras,
        thinking=thinking_steps,
        debug_thinking=bool(debug_flag),
    )


def graph_to_state(graph_state: GraphState, base: Optional[State] = None) -> State:
    """Merge GraphState back into a legacy mapping state."""
    state: State = dict(base or {})
    rows = []
    if isinstance(graph_state.sql, dict):
        rows = graph_state.sql.get("rows", [])
        if isinstance(rows, list) and len(rows) > 5:
            rows = rows[:5]

    sql_payload = {
        "text": graph_state.sql.get("text") if isinstance(graph_state.sql, dict) else None,
        "df": graph_state.sql.get("df") if isinstance(graph_state.sql, dict) else None,
        "explain": graph_state.sql.get("explain") if isinstance(graph_state.sql, dict) else None,
        "table": graph_state.sql.get("table") if isinstance(graph_state.sql, dict) else None,
        "rows": rows,
        "columns": graph_state.sql.get("columns", []) if isinstance(graph_state.sql, dict) else [],
        "markdown_table": graph_state.sql.get("markdown_table", "") if isinstance(graph_state.sql, dict) else "",
    }
    _LOGGER.warning(
        "[GRAPH_STATE_PATCH] Preserving %d SQL rows during serialization.",
        len(sql_payload.get("rows", [])),
    )

    state.update(
        {
            "query": graph_state.query,
            "tenant": graph_state.tenant,
            "intent": graph_state.intent,
            "scope": graph_state.scope,
            "filters": graph_state.filters,
            "plan": graph_state.plan,
            "sql": sql_payload,
            "rag": graph_state.rag,
            "result_bundle": graph_state.result_bundle,
            "telemetry": graph_state.telemetry,
            "memory": graph_state.memory,
            "history": graph_state.history,
            "model_used": graph_state.model_used,
            "timestamp": graph_state.timestamp,
            "_start_time": graph_state.start_time,
            "_input": graph_state.raw_input,
            "guardrail_blocked": graph_state.guardrail_blocked,
        }
    )

    content = graph_state.extras.get("content")
    answer_text = graph_state.extras.get("answer_text")
    if content:
        state["content"] = content
    if answer_text:
        state["answer_text"] = answer_text
    if content or answer_text:
        _LOGGER.warning(
            "[GRAPH_STATE_PATCH] Restored content=%s answer_text=%s",
            bool(content),
            bool(answer_text),
        )

    if graph_state.thinking:
        state["thinking"] = [step.to_payload() for step in graph_state.thinking]
    if graph_state.debug_thinking:
        state["debug_thinking"] = True

    state.update(graph_state.extras)
    return state


def _extract_start_time(state_like: GraphState | State | None) -> Optional[float]:
    if state_like is None:
        return None
    if isinstance(state_like, GraphState):
        return state_like.start_time
    start = state_like.get("_start_time") if isinstance(state_like, dict) else None
    if start is not None:
        return float(start)
    start = state_like.get("start_time") if isinstance(state_like, dict) else None
    return float(start) if start is not None else None


def _coerce_step_payload(step: ThinkingStep | Dict[str, Any]) -> ThinkingStep:
    if isinstance(step, ThinkingStep):
        return step
    return ThinkingStep(
        phase=str(step.get("phase") or "unknown"),
        title=str(step.get("title") or "").strip() or "Untitled step",
        detail=str(step.get("detail") or "").strip(),
        meta=dict(step.get("meta") or {}),
        elapsed_ms=step.get("elapsed_ms"),
    )


def add_thinking_step(
    state_like: GraphState | State | None,
    phase: str,
    title: str,
    detail: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
    elapsed_ms: Optional[int] = None,
) -> None:
    """Append a thinking step when tracing is enabled."""
    if state_like is None:
        return
    if isinstance(state_like, GraphState):
        enabled = bool(state_like.debug_thinking)
    elif isinstance(state_like, dict):
        enabled = bool(state_like.get("debug_thinking"))
    else:
        enabled = False
    if not enabled:
        return

    start_time = _extract_start_time(state_like)
    computed_elapsed = elapsed_ms
    if computed_elapsed is None and isinstance(start_time, (int, float)):
        computed_elapsed = int(max((time.perf_counter() - start_time) * 1000, 0))

    step = ThinkingStep(
        phase=phase,
        title=title.strip() or phase.title(),
        detail=detail.strip(),
        meta=dict(meta or {}),
        elapsed_ms=computed_elapsed,
    )

    if isinstance(state_like, GraphState):
        state_like.thinking.append(step)
    elif isinstance(state_like, dict):
        payload_steps = state_like.setdefault("thinking", [])
        if isinstance(payload_steps, list):
            payload_steps.append(step.to_payload())
        else:
            state_like["thinking"] = [step.to_payload()]


__all__ = [
    "GraphState",
    "State",
    "state_to_graph",
    "graph_to_state",
    "ThinkingStep",
    "add_thinking_step",
]
