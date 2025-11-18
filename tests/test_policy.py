import os

from agent.config import load_config
from agent.guards import ingress
from agent.intents import classify_intent
from agent.policy import plan_steps, resolve_entities
from agent.nl2sql import plan_to_sql


os.environ.setdefault("HOPE_AGENT_DEFAULT_YEARS", "")


def run_pipeline(query, tenant=None, user_filters=None):
    load_config(refresh=True)
    state = ingress(query, tenant, user_filters)
    state = classify_intent(state)
    state = resolve_entities(state)
    state = plan_steps(state)
    return state


def test_highbury_policy_highbury_scope():
    state = run_pipeline("List our Highbury listings in Manhattan")
    assert state["policy"] == "SQL_HIGHBURY"
    assert state["plan"]["sql_table"] == "highbury_listings"
    assert state["scope"] == "Highbury"
    assert state["filters"]["year"] == []


def test_compare_policy_for_compare_query():
    state = run_pipeline("Compare our prices to the market in Manhattan", tenant="highbury")
    assert state["policy"] == "SQL_COMPARE"
    assert state["plan"]["sql_table"] == "both"
    assert state["scope"] == "Hybrid"


def test_market_policy_default():
    state = run_pipeline("List Manhattan listings")
    assert state["policy"] == "SQL_MARKET"
    assert state["plan"]["sql_table"] == "listings"
    assert state["scope"] == "Market"


def test_rag_policy_for_reviews():
    state = run_pipeline(
        "What complaints do guests mention in Brooklyn?",
        user_filters={"borough": ["Brooklyn"], "neighbourhood": [], "month": [], "year": [], "listing_id": None, "is_highbury": None},
    )
    assert state["policy"] == "RAG_REVIEWS"
    assert state["plan"]["mode"] == "rag"


def test_rag_policy_for_cleanliness_question():
    state = run_pipeline(
        "Are guests complaining about how clean the place is?",
        user_filters={"borough": [], "neighbourhood": [], "month": [], "year": [], "listing_id": None, "is_highbury": None},
    )
    assert state["policy"] == "RAG_REVIEWS"


def test_rag_policy_when_reviews_toggle_enabled():
    filters = {"borough": [], "neighbourhood": [], "month": [], "year": [], "listing_id": None, "is_highbury": True}
    state = run_pipeline("Show occupancy trend", user_filters=filters)
    assert state["policy"] == "RAG_REVIEWS"
    assert state["plan"]["mode"] == "rag"


def test_month_extraction():
    state = run_pipeline(
        "Show sentiment for Queens in March",
        user_filters={"borough": ["Queens"], "neighbourhood": [], "month": [], "year": [], "listing_id": None, "is_highbury": None},
    )
    assert "MAR" in state["filters"]["month"]


def test_listing_id_detection():
    state = run_pipeline(
        "Explain reviews for listing 54321",
        user_filters={"borough": [], "neighbourhood": [], "month": [], "year": [], "listing_id": None, "is_highbury": None},
    )
    assert state["filters"]["listing_id"] == 54321


def test_top_k_override():
    state = run_pipeline(
        "Guests feedback",
        user_filters={"borough": [], "neighbourhood": [], "month": [], "year": [], "listing_id": None, "is_highbury": None, "top_k": 12},
    )
    assert state["plan"]["top_k"] == 12


def test_time_filters_ignored_when_absent():
    filters = {"borough": ["Manhattan"], "neighbourhood": [], "month": [], "year": [], "listing_id": None, "is_highbury": None}
    state = run_pipeline("List Manhattan listings", tenant="market", user_filters=filters)
    state["policy"] = "SQL_MARKET"
    plan_to_sql(state)
    sql_text = state["sql"]["text"].lower()
    assert "year" not in sql_text
    assert "month" not in sql_text
