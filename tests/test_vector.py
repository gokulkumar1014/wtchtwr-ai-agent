from agent.vector import _build_where, need_rag, _to_bool


def test_build_where_highbury_scope_adds_flag_and_in_lists():
    filters = {
        "year": [2025],
        "month": [],
        "borough": ["Manhattan"],
        "neighbourhood": [],
        "listing_id": None,
        "is_highbury": None,
    }
    where = _build_where(filters, "Highbury")
    assert where == {
        "$and": [
            {"neighbourhood_group": {"$in": ["Manhattan"]}},
            {"year": {"$in": [2025]}},
            {"is_highbury": True},
        ]
    }


def test_build_where_market_scope_defaults_non_highbury():
    filters = {
        "year": [],
        "month": [],
        "borough": [],
        "neighbourhood": [],
        "listing_id": None,
        "is_highbury": None,
    }
    where = _build_where(filters, "Market", market_only=True)
    assert where == {}


def test_build_where_casts_listing_id_to_int():
    filters = {
        "year": [],
        "month": [],
        "borough": [],
        "neighbourhood": [],
        "listing_id": "54321",
        "is_highbury": False,
    }
    where = _build_where(filters, "Market")
    assert where == {"$and": [{"listing_id": 54321}, {"is_highbury": False}]}


def test_build_where_single_clause_returns_plain_dict():
    filters = {
        "year": [],
        "month": [],
        "borough": ["Manhattan"],
        "neighbourhood": [],
        "listing_id": None,
        "is_highbury": None,
    }
    where = _build_where(filters, "Market")
    assert where == {"neighbourhood_group": {"$in": ["Manhattan"]}}


def test_build_where_ignores_none_values():
    filters = {
        "year": [None, 2025],
        "month": ["", "MAR", None],
        "borough": [None, "Manhattan"],
        "neighbourhood": [],
        "listing_id": None,
        "is_highbury": None,
    }
    where = _build_where(filters, "Highbury")
    assert where == {
        "$and": [
            {"neighbourhood_group": {"$in": ["Manhattan"]}},
            {"month": {"$in": ["MAR"]}},
            {"year": {"$in": [2025]}},
            {"is_highbury": True},
        ]
    }


def test_need_rag_true_for_hybrid():
    state = {"plan": {"mode": "hybrid"}, "sql": {"df": None}}
    updated = need_rag(state)
    assert updated["rag_needed"] is True


def test_need_rag_sql_fallback_when_empty_df():
    class DummyDF:
        empty = True

    state = {"plan": {"mode": "sql"}, "sql": {"df": DummyDF()}}
    updated = need_rag(state)
    assert updated["rag_needed"] is True


def test_to_bool_handles_string_values():
    assert _to_bool(True) is True
    assert _to_bool(False) is False
    assert _to_bool("True") is True
    assert _to_bool("false") is False
    assert _to_bool("1") is True
    assert _to_bool("0") is False
    assert _to_bool("False") is False
