from agent.compose import build_composer_input, format_filters, render_result_markdown


def test_render_result_markdown_includes_table_and_truncated_snippet():
    rows = [
        {"listings_id": 101, "price_in_usd": 245.678, "room_type": "Entire home/apt"},
    ]
    aggregates = {"avg_price": 245.678, "avg_occupancy": 0.85}
    snippets = [
        {
            "listing_id": 101,
            "month": "JAN",
            "neighbourhood_group": "Manhattan",
            "comment_id": "r1",
            "snippet": "Quiet street and close to subway." * 10,
        }
    ]

    markdown = render_result_markdown(rows, aggregates, snippets, max_rows=5)

    assert "| listings_id |" in markdown
    assert "avg_price" in markdown
    assert "- [Manhattan | JAN" in markdown
    assert "]  #101" in markdown
    assert "#101 · #r1" in markdown
    assert "..." in markdown  # snippet should be truncated


def test_build_composer_input_uses_only_applied_filters():
    messages = build_composer_input(
        history=[],
        user_question="What do we know?",
        policy="SQL_HIGHBURY",
        sql="SELECT 1",
        rows=[],
        aggregates={},
        rag_snippets=[],
        applied_filters={},
    )
    context = messages[-2]["content"]
    assert "applied_filters: none" in context

    messages_with_year = build_composer_input(
        history=[],
        user_question="What about 2025?",
        policy="SQL_MARKET",
        sql="SELECT 1",
        rows=[],
        aggregates={},
        rag_snippets=[],
        applied_filters={"year": [2025]},
    )
    context_with_year = messages_with_year[-2]["content"]
    assert "2025" in context_with_year


def test_format_filters_only_lists_present():
    text = format_filters({"borough": ["Manhattan"], "year": []})
    assert text == "borough: Manhattan"
