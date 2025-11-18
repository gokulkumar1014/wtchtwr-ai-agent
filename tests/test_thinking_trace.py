import time

from agent.guards import egress
from agent.types import ThinkingStep


def test_egress_includes_thinking_trace_when_enabled():
    state = {
        "answer_text": "ok",
        "result_bundle": {},
        "_start_time": time.perf_counter(),
        "telemetry": {},
        "thinking": [
            {
                "phase": "sql",
                "title": "Ran SQL",
                "detail": "Collected rows",
                "meta": {"rows": 5},
                "elapsed_ms": 120,
            }
        ],
        "debug_thinking": True,
    }
    response = egress(state)
    trace = response.get("thinking_trace")
    assert trace and trace[0]["phase"] == "sql"


def test_egress_hides_thinking_trace_without_debug():
    state = {
        "answer_text": "ok",
        "result_bundle": {},
        "_start_time": time.perf_counter(),
        "telemetry": {},
        "thinking": [
            {
                "phase": "sql",
                "title": "Ran SQL",
                "detail": "Collected rows",
                "meta": {"rows": 5},
                "elapsed_ms": 120,
            }
        ],
        "debug_thinking": False,
    }
    response = egress(state)
    assert "thinking_trace" not in response


def test_egress_accepts_thinking_objects():
    state = {
        "answer_text": "ok",
        "result_bundle": {},
        "_start_time": time.perf_counter(),
        "telemetry": {},
        "thinking": [
            ThinkingStep(
                phase="portfolio_triage",
                title="Scan",
                detail="looked at boroughs",
                meta={"rows": 2},
                elapsed_ms=42,
            )
        ],
        "debug_thinking": True,
    }
    response = egress(state)
    trace = response.get("thinking_trace")
    assert trace and trace[0]["meta"]["rows"] == 2
