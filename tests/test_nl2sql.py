import re

from agent.nl2sql import plan_to_sql


BASE_FILTERS = {
    "borough": [],
    "neighbourhood": [],
    "month": [],
    "year": [],
    "listing_id": None,
    "is_highbury": None,
}


def build_state(query, scope="Market", filters=None, policy="SQL_MARKET"):
    filters_copy = dict(BASE_FILTERS)
    if filters:
        filters_copy.update(filters)
    return {
        "query": query,
        "scope": scope,
        "filters": filters_copy,
        "policy": policy,
        "plan": {"mode": "sql", "sql_table": "listings", "policy": policy},
        "sql": {},
    }


def sql_state_for(query, scope="Market", filters=None, policy="SQL_MARKET"):
    state = build_state(query, scope=scope, filters=filters, policy=policy)
    plan_to_sql(state)
    return state


def test_highbury_policy_uses_highbury_table_without_year():
    state = sql_state_for(
        "List our Highbury listings in Manhattan",
        scope="Highbury",
        filters={"borough": ["Manhattan"]},
        policy="SQL_HIGHBURY",
    )
    sql_text = state["sql"]["text"].lower()
    assert "from highbury_listings" in sql_text
    assert "union" not in sql_text
    assert "year" not in sql_text
    assert state["sql"]["params"] == ["Manhattan"]
    assert "year" not in state["applied_filters"]


def test_market_policy_uses_market_table():
    state = sql_state_for(
        "List Manhattan listings",
        scope="Market",
        filters={"borough": ["Manhattan"]},
        policy="SQL_MARKET",
    )
    sql_text = state["sql"]["text"].lower()
    assert "from listings" in sql_text
    assert state["sql"]["params"] == ["Manhattan"]


def test_multiple_borough_filters_use_any_clause():
    state = sql_state_for(
        "List listings across boroughs",
        scope="Market",
        filters={"borough": ["Manhattan", "Queens"]},
        policy="SQL_MARKET",
    )
    sql_text = state["sql"]["text"].upper()
    assert "NEIGHBOURHOOD_GROUP = ?" in sql_text
    assert sql_text.count("NEIGHBOURHOOD_GROUP = ?") == 2
    assert state["sql"]["params"] == ["Manhattan", "Queens"]


def test_compare_policy_uses_union():
    state = sql_state_for(
        "Compare our rating vs market",
        scope="Hybrid",
        filters={"borough": ["Manhattan"]},
        policy="SQL_COMPARE",
    )
    sql_text = state["sql"]["text"].lower()
    assert "with combined" in sql_text
    assert "union all" in sql_text
    assert state["sql"]["params"] == ["Manhattan", "Manhattan"]


def test_total_revenue_highbury_sum():
    state = sql_state_for(
        "Total estimated_revenue_90 for Highbury listings",
        scope="Highbury",
        policy="SQL_HIGHBURY",
    )
    sql_text = state["sql"]["text"].lower()
    assert "sum(estimated_revenue_90)" in sql_text
    assert "from highbury_listings" in sql_text


def test_output_never_contains_banned_keywords():
    state = sql_state_for("drop table listings")
    lowered = state["sql"]["text"].lower()
    for banned in ["drop", "delete", "update", "insert", "alter"]:
        assert banned not in lowered


def test_limit_respects_config_cap():
    state = sql_state_for("Show neighbourhood occupancy trends")
    match = re.search(r"LIMIT\s+(\d+)", state["sql"]["text"])
    assert match is not None
    assert int(match.group(1)) <= 500
