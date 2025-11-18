import sys
import types

if "duckdb" not in sys.modules:
    fake_duckdb = types.ModuleType("duckdb")

    def _fake_connect(*_args, **_kwargs):
        raise RuntimeError("duckdb not available in tests")

    fake_duckdb.connect = _fake_connect  # type: ignore[attr-defined]
    sys.modules["duckdb"] = fake_duckdb

if "pandas" not in sys.modules:
    fake_pd = types.ModuleType("pandas")

    class _FakeDataFrame:
        def __init__(self, *args, **kwargs):
            self.empty = True
            self.shape = (0, 0)

        def head(self, *_args, **_kwargs):
            return self

        def to_dict(self, *args, **kwargs):
            return []

        def __len__(self):
            return 0

    fake_pd.DataFrame = _FakeDataFrame  # type: ignore[attr-defined]
    sys.modules["pandas"] = fake_pd

if "openai" not in sys.modules:
    fake_openai = types.ModuleType("openai")

    class _FakeOpenAI:
        def __getattr__(self, _name):
            raise RuntimeError("openai not available in tests")

    fake_openai.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    sys.modules["openai"] = fake_openai

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["dotenv"] = fake_dotenv

if "numpy" not in sys.modules:
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.float32 = float  # type: ignore[attr-defined]
    fake_numpy.integer = int  # type: ignore[attr-defined]
    fake_numpy.ndarray = list  # type: ignore[attr-defined]
    sys.modules["numpy"] = fake_numpy

if "qdrant_client" not in sys.modules:
    fake_qdrant = types.ModuleType("qdrant_client")

    class _FakeQdrantClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("qdrant client not available in tests")

    fake_qdrant.QdrantClient = _FakeQdrantClient  # type: ignore[attr-defined]
    sys.modules["qdrant_client"] = fake_qdrant

    fake_http = types.ModuleType("qdrant_client.http")
    sys.modules["qdrant_client.http"] = fake_http

    fake_models = types.ModuleType("qdrant_client.http.models")

    class _FakeModel:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def dict(self):
            return self.kwargs

    fake_models.FieldCondition = _FakeModel  # type: ignore[attr-defined]
    fake_models.MatchValue = _FakeModel  # type: ignore[attr-defined]
    fake_models.MatchAny = _FakeModel  # type: ignore[attr-defined]
    fake_models.Range = _FakeModel  # type: ignore[attr-defined]
    fake_models.Filter = _FakeModel  # type: ignore[attr-defined]
    fake_models.SearchParams = _FakeModel  # type: ignore[attr-defined]
    fake_models.ScoredPoint = _FakeModel  # type: ignore[attr-defined]
    sys.modules["qdrant_client.http.models"] = fake_models

    fake_exceptions = types.ModuleType("qdrant_client.http.exceptions")

    class _FakeUnexpectedResponse(Exception):
        pass

    fake_exceptions.UnexpectedResponse = _FakeUnexpectedResponse  # type: ignore[attr-defined]
    sys.modules["qdrant_client.http.exceptions"] = fake_exceptions

if "pydantic" not in sys.modules:
    fake_pydantic = types.ModuleType("pydantic")

    class _FakeValidationError(Exception):
        pass

    fake_pydantic.ValidationError = _FakeValidationError  # type: ignore[attr-defined]
    sys.modules["pydantic"] = fake_pydantic

if "sentence_transformers" not in sys.modules:
    fake_st = types.ModuleType("sentence_transformers")

    class _FakeEmbedding:
        def __init__(self):
            self.dtype = float

        def astype(self, *_args, **_kwargs):
            return self

        def __getitem__(self, _index):
            return [0.0]

    class _FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *_args, **_kwargs):
            return _FakeEmbedding()

    fake_st.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = fake_st

from agent.portfolio_triage import run_portfolio_triage
from agent.types import GraphState


def _build_state(plan_kpi: str | None = None) -> GraphState:
    plan = {"mode": "portfolio_triage", "sql_table": "highbury_listings"}
    if plan_kpi:
        plan["kpi"] = plan_kpi
    return GraphState(
        query="portfolio triage",
        tenant="highbury",
        intent="PORTFOLIO_TRIAGE_ADVANCED",
        scope="Highbury",
        filters={},
        plan=plan,
        result_bundle={},
        telemetry={},
        raw_input={},
        extras={},
        thinking=[],
        debug_thinking=False,
    )


def test_portfolio_triage_builds_scope_and_backlog(monkeypatch):
    distribution_row = {
        "listing_count": 10,
        "avg_kpi": 68.5,
        "median_kpi": 70.0,
        "min_kpi": 20.0,
        "max_kpi": 95.0,
        "stddev_kpi": 5.0,
    }
    top_rows = [
        {
            "listings_id": 201,
            "listing_name": "H201",
            "neighbourhood_group": "Manhattan",
            "neighbourhood": "Midtown",
            "price_in_usd": 240,
            "estimated_revenue_30": 6200,
            "estimated_revenue_60": 11200,
            "estimated_revenue_90": 15000,
            "estimated_revenue_365": 64000,
            "review_scores_rating": 4.9,
            "occupancy_rate_30": 82,
            "occupancy_rate_60": 78,
            "occupancy_rate_90": 88,
            "occupancy_rate_365": 74,
            "selected_kpi": 88,
        },
        {
            "listings_id": 202,
            "listing_name": "H202",
            "neighbourhood_group": "Brooklyn",
            "neighbourhood": "Williamsburg",
            "price_in_usd": 260,
            "estimated_revenue_30": 7000,
            "estimated_revenue_60": 12400,
            "estimated_revenue_90": 16000,
            "estimated_revenue_365": 72000,
            "review_scores_rating": 4.7,
            "occupancy_rate_30": 85,
            "occupancy_rate_60": 79,
            "occupancy_rate_90": 86,
            "occupancy_rate_365": 70,
            "selected_kpi": 86,
        },
    ]
    bottom_rows = [
        {
            "listings_id": 301,
            "listing_name": "H301",
            "neighbourhood_group": "Manhattan",
            "neighbourhood": "Midtown",
            "price_in_usd": 310,
            "estimated_revenue_30": 4200,
            "estimated_revenue_60": 6800,
            "estimated_revenue_90": 9100,
            "estimated_revenue_365": 42000,
            "review_scores_rating": 4.1,
            "occupancy_rate_30": 40,
            "occupancy_rate_60": 38,
            "occupancy_rate_90": 35,
            "occupancy_rate_365": 30,
            "selected_kpi": 35,
        },
        {
            "listings_id": 302,
            "listing_name": "H302",
            "neighbourhood_group": "Brooklyn",
            "neighbourhood": "Williamsburg",
            "price_in_usd": 330,
            "estimated_revenue_30": 3800,
            "estimated_revenue_60": 6000,
            "estimated_revenue_90": 8000,
            "estimated_revenue_365": 36000,
            "review_scores_rating": 4.0,
            "occupancy_rate_30": 38,
            "occupancy_rate_60": 37,
            "occupancy_rate_90": 34,
            "occupancy_rate_365": 28,
            "selected_kpi": 34,
        },
    ]
    market_rows = [
        {
            "neighbourhood_group": "Manhattan",
            "neighbourhood": "Midtown",
            "market_median_kpi": 62,
            "market_median_price_usd": 280,
            "market_median_revenue_30": 5100,
            "market_avg_review_score": 4.6,
        },
        {
            "neighbourhood_group": "Brooklyn",
            "neighbourhood": "Williamsburg",
            "market_median_kpi": 58,
            "market_median_price_usd": 320,
            "market_median_revenue_30": 5400,
            "market_avg_review_score": 4.5,
        },
    ]

    executed_sql: list[str] = []

    def fake_execute(sql: str):
        executed_sql.append(sql)
        if "COUNT(*)" in sql:
            return {"rows": [distribution_row]}
        if "ORDER BY" in sql and "DESC" in sql:
            return {"rows": top_rows}
        if "ORDER BY" in sql and "ASC" in sql:
            return {"rows": bottom_rows}
        if "listings_cleaned" in sql:
            return {"rows": market_rows}
        return {"rows": []}

    def fake_sentiment(_state, listing_id, sentiment_label, **_kwargs):
        compound = 0.6 if sentiment_label == "positive" else -0.4
        return [
            {
                "listing_id": listing_id,
                "snippet": f"{sentiment_label} review for {listing_id}",
                "sentiment_label": sentiment_label,
                "compound": compound,
                "positive": 0.7,
                "neutral": 0.2,
                "negative": 0.1,
            }
        ]

    monkeypatch.setattr("agent.portfolio_triage.execute_duckdb", fake_execute)
    monkeypatch.setattr("agent.portfolio_triage._fetch_listing_sentiment", fake_sentiment)

    state = _build_state()
    state.filters = {"neighbourhood": ["Midtown"]}

    updated = run_portfolio_triage(state)
    triage = updated.result_bundle["portfolio_triage"]

    assert triage["scope"] == "Highbury / Midtown"
    glance = triage["portfolio_at_glance"]
    assert glance["top5_overview"]
    assert glance["bottom5_overview"]
    assert glance["sentiment_summary"]["total_positive"] == 2
    assert glance["market_benchmarks"]["entries"]
    backlog = triage["action_backlog"]
    assert backlog and backlog[0]["sample_reviews"]
    assert updated.result_bundle["summary"].startswith("Ranked the Highbury")
    assert updated.result_bundle["rag_snippets"]


def test_portfolio_triage_respects_custom_kpi(monkeypatch):
    distribution_row = {
        "listing_count": 2,
        "avg_kpi": 4800,
        "median_kpi": 4800,
        "min_kpi": 4500,
        "max_kpi": 5100,
        "stddev_kpi": 100,
    }
    revenue_rows = [
        {
            "listings_id": 501,
            "listing_name": "RevWin",
            "neighbourhood_group": "Manhattan",
            "neighbourhood": "Chelsea",
            "price_in_usd": 300,
            "estimated_revenue_30": 5200,
            "estimated_revenue_60": 9200,
            "estimated_revenue_90": 13800,
            "estimated_revenue_365": 61000,
            "review_scores_rating": 4.8,
            "occupancy_rate_30": 78,
            "occupancy_rate_60": 72,
            "occupancy_rate_90": 70,
            "occupancy_rate_365": 66,
            "selected_kpi": 5200,
        },
        {
            "listings_id": 601,
            "listing_name": "RevDrag",
            "neighbourhood_group": "Brooklyn",
            "neighbourhood": "Bushwick",
            "price_in_usd": 190,
            "estimated_revenue_30": 3100,
            "estimated_revenue_60": 5600,
            "estimated_revenue_90": 7800,
            "estimated_revenue_365": 29000,
            "review_scores_rating": 4.2,
            "occupancy_rate_30": 60,
            "occupancy_rate_60": 58,
            "occupancy_rate_90": 57,
            "occupancy_rate_365": 55,
            "selected_kpi": 3100,
        },
    ]
    market_rows = [
        {
            "neighbourhood_group": "Manhattan",
            "neighbourhood": "Chelsea",
            "market_median_kpi": 4700,
            "market_median_price_usd": 320,
            "market_median_revenue_30": 4700,
            "market_avg_review_score": 4.6,
        }
    ]
    executed_sql: list[str] = []

    def fake_execute(sql: str):
        executed_sql.append(sql)
        if "COUNT(*)" in sql:
            return {"rows": [distribution_row]}
        if "ORDER BY" in sql and "DESC" in sql:
            return {"rows": [revenue_rows[0]]}
        if "ORDER BY" in sql and "ASC" in sql:
            return {"rows": [revenue_rows[1]]}
        if "listings_cleaned" in sql:
            return {"rows": market_rows}
        return {"rows": []}

    monkeypatch.setattr("agent.portfolio_triage.execute_duckdb", fake_execute)
    monkeypatch.setattr("agent.portfolio_triage._fetch_listing_sentiment", lambda *args, **kwargs: [])

    state = _build_state(plan_kpi="estimated_revenue_30")

    updated = run_portfolio_triage(state)
    triage = updated.result_bundle["portfolio_triage"]

    assert triage["kpi_used"] == "estimated_revenue_30"
    combined_sql = "\n".join(executed_sql)
    assert "estimated_revenue_30" in combined_sql
    backlog_map = {entry["listing_id"]: entry["kpi_value"] for entry in triage["action_backlog"]}
    assert backlog_map.get("501") == 5200
