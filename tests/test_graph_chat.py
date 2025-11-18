from agent.config import load_config
from agent.graph import run as run_agent, _compiled_graph  # type: ignore[attr-defined]


def test_graph_run_returns_bundle(monkeypatch):
    monkeypatch.setenv("HOPE_AGENT_STREAM_COMPOSER", "false")
    monkeypatch.setenv("HOPE_AGENT_DEFAULT_YEARS", "")
    load_config(refresh=True)
    _compiled_graph.cache_clear()  # type: ignore[attr-defined]

    filters = {
        "borough": ["Manhattan"],
        "neighbourhood": [],
        "month": [],
        "year": [],
        "listing_id": None,
        "is_highbury": None,
    }

    response = run_agent(
        "List our Highbury listings in Manhattan with prices",
        tenant="highbury",
        user_filters=filters,
        history=[{"role": "user", "content": "Hello"}],
        composer_enabled=False,
    )

    assert "answer_text" in response
    bundle = response.get("result_bundle", {})
    assert bundle.get("rows") is not None
    assert bundle.get("policy") == "SQL_HIGHBURY"
    applied = bundle.get("applied_filters", {})
    assert applied.get("borough") == "Manhattan"
    assert "highbury_listings" in (response.get("sql_text") or "")
