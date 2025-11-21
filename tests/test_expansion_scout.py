import copy

import pytest

from agent import expansion_scout
from agent.compose import build_composer_input, fallback_text
from agent.guards import fallback_expansion_text


def test_expansion_scout_empty_results(monkeypatch):
    state = {"query": "Where should Highbury expand?"}
    monkeypatch.setattr(expansion_scout, "tavily_search", lambda *args, **kwargs: [])
    result = expansion_scout.exec_expansion_scout(copy.deepcopy(state))

    assert result["expansion_report"] == fallback_expansion_text()
    assert result["telemetry"]["expansion_source_count"] == 0
    assert result.get("result_bundle", {}).get("policy") == "EXPANSION_SCOUT"


def test_expansion_scout_basic_pipeline(monkeypatch):
    calls: dict = {}

    def fake_search(query: str, max_results: int = 3):
        calls.setdefault("queries", []).append(query)
        return [{"url": f"https://example.com/{len(calls['queries'])}", "title": "Example", "score": 0.9}]

    monkeypatch.setattr(expansion_scout, "tavily_search", fake_search)
    monkeypatch.setattr(
        expansion_scout,
        "load_article",
        lambda url: "Major growth project and transit development with infrastructure expansion.",
    )

    def fake_synthesize(normalized, ctx):
        calls["normalized"] = normalized
        calls["ctx"] = ctx
        return "SYNTHESIZED_EXPANSION_REPORT"

    monkeypatch.setattr(expansion_scout, "synthesize_expansion_report", fake_synthesize)
    monkeypatch.setattr(
        expansion_scout,
        "generate_dynamic_queries",
        lambda user_q: {
            "tourism_macro": "tourism query 1",
            "infrastructure": "infra query 2",
            "development": "dev query 3",
            "regulation": "reg query 4",
            "tourism_trending": "tourism query 5",
        },
    )

    result = expansion_scout.exec_expansion_scout({"query": "best neighborhood to invest"})

    assert result["expansion_report"] == "SYNTHESIZED_EXPANSION_REPORT"
    assert result["telemetry"]["expansion_source_count"] == 5
    normalized = calls["normalized"]
    assert all(
        bucket in normalized
        for bucket in ("tourism_signals", "infrastructure_signals", "regulation_signals", "development_signals")
    )
    assert any(item.get("key_points") for bucket in normalized.values() for item in bucket)


def test_tavily_failure_recovery(monkeypatch):
    def raise_search(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(expansion_scout, "tavily_search", raise_search)
    result = expansion_scout.exec_expansion_scout({"query": "where should we go next"})

    assert result["expansion_report"] == fallback_expansion_text()
    assert result["telemetry"]["expansion_source_count"] == 0


def test_compose_expansion_report_renders_correctly(monkeypatch):
    bundle = {
        "applied_filters": {},
        "expansion_report": "Draft expansion report",
        "expansion_sources": [{"url": "https://example.com/source"}],
        "policy": "EXPANSION_SCOUT",
    }
    messages = build_composer_input(
        history=[],
        user_question="Where should Highbury expand?",
        policy="EXPANSION_SCOUT",
        sql=None,
        rows=[],
        aggregates={},
        rag_snippets=[],
        applied_filters={},
        intent="EXPANSION_SCOUT",
        expansion_report=bundle["expansion_report"],
        expansion_sources=bundle["expansion_sources"],
    )
    context_block = messages[-2]["content"]
    assert "expansion report" in context_block.lower()
    assert "Web Sources Used" in context_block

    answer_text = fallback_text(bundle)
    assert "Draft expansion report" in answer_text
    assert "Web Sources Used" in answer_text
