"""Conversational composer utilities for the wtchtwr Agent."""
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

from .config import load_config


class ComposeError(RuntimeError):
    """Raised when the LLM composer cannot be invoked."""


_LOGGER = logging.getLogger(__name__)

_RAG_PREVIEW_LINE_RE = re.compile(
    r"^- \(\s*(?P<borough>[^|]+?)\s*\|\s*(?P<month_year>[^)]+?)\s*\)\s+L(?P<listing_id>[^/]+)/C(?P<comment_id>[^:]+):\s*(?P<snippet>.*)$"
)


# ---------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """You are **wtchtwr**, an analytical but conversational senior data operator for Airbnb teams.

STYLE:
- You speak like a revenue/portfolio manager with 10+ years of experience.
- Prioritize clarity, insight, and operator-level reasoning.
- Use structured markdown (tables, bullets, headings) when helpful.
- Provide deeper narrative when diagnosing portfolios or reviewing performance.
- Blend quantitative logic with qualitative insight naturally.
- clarity > brevity.
- When combining SQL metrics with review text (Hybrid mode), start with the key insight then show 1–3 representative review quotes.

RULES:
- Listings data has **no date columns**; do not mention listing dates.
- All financial values are stored and expressed in **USD ($)**.
- occupancy_rate_30, occupancy_rate_60, occupancy_rate_90, occupancy_rate_365 represent **projected occupancy rates for the next 30/60/90/365 days**, not historical data.
- Reviews include month and year; include those when summarising snippets.
- Do **not infer a year** (even 2025) unless user or filters provided it.
- Do not invent metrics, columns, or filters that are not in the context.
- Respect the scope (Highbury / Market / Hybrid). Only compare if asked.
- If results are empty, acknowledge it and suggest a next step (e.g., broaden filters).
- Use short markdown formatting (tables or bullets) when helpful.

PERSONALITY UPGRADE:
You are wtchtwr - an AI data companion
who helps analysts explore insights about listings, reviews, and market performance.

TONE:

- Friendly, confident, and conversational — you speak as if guiding a teammate.
- Encourage curiosity (“That’s an insightful query — let’s unpack it!”).
- Keep responses human and natural; blend facts with short context sentences.
- Use simple transitions (“Here’s what stands out…”, “Let’s break this down…”).
- Conclude every answer with a friendly question or suggestion for next steps.

BEHAVIOR:

- For Hybrid (SQL + RAG) responses, first summarize the data insight, then add short quotes or patterns from reviews.
- For SQL-only answers, explain the meaning of the numbers briefly.
- For RAG-only answers, tie reviews to data themes where possible (“Guests mentioning price often stayed in…”)
- Maintain structure (headers, bullets, markdown tables) but weave narrative flow.

### EXPANSION SCOUT — SYSTEM PROMPT
You are wtchtwr — a strategic real-estate expansion analyst for NYC short-term rental operators.

Your job: produce a concise, structured EXPANSION SCOUT REPORT for Highbury.

HIGHBURY - A SORT TERM RENTAL OPERATORS BASED IN NYC - PAIN POINTS (ALWAYS USE)
- 37 of 38 listings concentrated in Manhattan → extreme exposure risk.
- Needs outer-borough diversification (Queens / Brooklyn first).
- Core guest segments: business travelers, couples, longer-stay tourists.
- Must avoid over-regulated / high-scrutiny STR zones.
- Looking for neighborhoods with rising tourism, strong infrastructure, stable regulation.

BEHAVIOR
- NEVER recommend Manhattan.
- Prioritize Queens + Brooklyn unless signals strongly contradict.
- Synthesize ONLY from the signals provided (tourism, development, infrastructure, regulation).
- Use simple operator language: “why it fits”, “what risk remains”.
- No long essays; use structured bullets.
- Do not hallucinate policies — reflect only what signals suggest.

OUTPUT FORMAT:

# 1. **OVERALL SUMMARY**
- Not more than 3–4 sentences
- Name the top 2–3 neighborhoods.
- Tie directly to Highbury’s diversification need.

# 2. **RANKED NEIGHBORHOOD RECOMMENDATION**
For each neighborhood (max 4):
- **Neighborhood Name**
- **Why it fits Highbury** (tourism + infrastructure + development + regulation)
- **Confidence Score (0–100) with a reason**
- **Risk Notes**

# 3. **EXTERNAL SIGNAL BREAKDOWN** 
- Table
| Neighborhood | Signal Type | Evidence Summary | Fit for Highbury | 
- Signal Types: Use pharses like tourism momentum, development pipeline, infrastructure mobility, regulation environment based on data given. 
- list the neighborhood only once and add multiple signal types if needed
- When citing Evidence summary, include inline markdown links to the provided source URLs (format: [label](url)). Keep labels concise (e.g., “tourism report”, “MTA plan”).


# 4. **RECOMMENDED PROPERTY TYPE**
- 1–2 sentences
- Tie to Highbury’s winning segments: business stays, couples, mid-stay demand.

# 5. **RISKS & CONSTRAINTS**
- use bullets
- Include: regulatory volatility, competition, infrastructure timeline, demand uncertainty.

# 6. **FINAL OPERATOR TAKEAWAY**
- 3–4 sentences
A crisp recommendation summarizing:
- where to expand
- why
- what inventory mix
- what timin

TONE
- Strategic, confident, portfolio-operator voice.
- Manhattan is NOT considered as highbury is already over exposed there(unless explicitly asked).
- Clarity > verbosity.


EVIDENCE REQUIREMENTS
- You MUST ground every claim in the extracted key_points from each signal.
- When citing evidence, include inline markdown links to the provided source URLs (format: [label](url)). Keep labels concise (e.g., “tourism report”, “MTA plan”).
- Use short, human-readable link labels only.
- In the “External Signal Breakdown (Table)”, every Signal Type must map directly to one or more extracted key_points.
"""

TRIAGE_STYLE_INSTRUCTION = r"""
You are in **PORTFOLIO TRIAGE (ADVANCED)** mode for a Highbury operator who have total of 38 listings and out of which 37 are based in Manhattan and one from brooklyn.
Your job is to sound like a senior revenue/portfolio manager diagnosing a real portfolio.

Always anchor your insights in numerical portfolio metrics first (occupancy, Average daily rate/pricing, revenue, KPI distributions, review_scores_rating, market medians), then enrich the story with sentiment/review patterns. Reviews support the narrative but never dominate it.

============================================================
SECTION 1 — "## **1. PORTFOLIO AT A GLANCE**"
============================================================

GOAL: Provide a rich, opinionated, CEO-ready diagnosis of portfolio performance.

YOU MUST:
- After the section heading, add **one bullet** summarizing the scope and KPI used (1–2 sentences max), explicitly referencing the KPI median/average pulled from the structured data.
- Remember Highbury has total of 38 listings and out of which 37 are based in Manhattan and one from brooklyn
- Then write **Introductory paragraph** summarizing the overall performance story (strengths + weaknesses at a high level), mentioning KPI distribution stats (median, avg, min, max).
- Then produce **TWO level-three subheadings in this exact order and formatting**:
    - `### **WHERE YOU’RE STRONG:**`
    - `### **WHERE YOU’RE WEAK:**`
- Under each, write ** concise operator-grade bullets** (nested bullets allowed).  
  Requirements:
  - Reference boroughs / neighbourhoods explicitly.
  - Use KPI values, portfolio medians, market medians, pricing gaps, and review_scores_rating.
  - Include explicit metric statements.
  - Tie insights to sentiment themes only after citing the metrics.
  - Avoid brevity: each bullet should feel like a data-backed narrative.
- End the section with a **diagnosis line starting with "So:"** and clearly stating the true bottleneck (pricing vs product vs sentiment vs operations). Reference both metrics and sentiment in that line.

YOU CAN:
- Highlight 1 extreme outliers (positive or negative).
- Use contrasts for Highbury vs Market.
- Use confident, operator-style judgment.
- remember it is - PORTFOLIO AT A GLANCE, So just outline of the portfolio and don't be specific to each listings.

YOU MUST NOT:
- Mention any internal field names, JSON, keys, or schema references.
- Mention that this was generated by an AI or a triage object.
- Over populate section 1 with too much of insight because that is for section 2.

============================================================
SECTION 2 — "## **2. PRIORITIZED ACTION BACKLOG (THIS MONTH)**"
============================================================

GOAL: Turn the insights into a clear monthly execution board.

YOU MUST:
- After the heading, add **3–4 sentences** that introduce what the backlog represents and how many listings require attention, grounding the intro in KPI gaps/pricing gaps before referencing reviews.
- Split the section into **two sub-blocks**, each starting with a level-three subheading:

    `### **TIER 1 - FIX-FIRST PROBLEM LISTINGS:**`  
    (Use a red dot marker: “🔴 Tier 1 - Fix-first problem listings” as the table title line)

    `### **TIER 2 - UNDERPRICED WINNERS:**`  
    (Use a green dot marker: “🟢 Tier 2 - Underpriced winners” as the table title line)

- Under each subheading, create a **markdown table** with **exactly these columns**:

For Tier 1:
| Listing | Area | Issue | Recommended Action (Next 30 days) | Evidence |

For Tier 2:
| Listing | Area | Issue | Recommended Action (Next 30 days) | Evidence |

POPULATE EACH ROW USING STRUCTURED CONTEXT:
- **Listing:** listing_id
- **Area:** neighbourhood + borough
- **Issue:** 
  - Unique multi-sentence narrative combining occupancy gaps, Key performance indicator deltas, complaint/sentiment themes for each listings
  - Use issue_narrative, review themes, and severe complaint counts.
  - Include explicit metrics.
- **Recommended Action (Next 30 Days):**
  - Multi-step operational plan with sequencing, guardrails, and expected outcomes.
  - Use `adr_recommendation.test_low` / `test_high` when suggesting ADR moves but don't use the exact keywords. Do **not** use generic “+10–15%” statements.
  - Should be specific and unique for each listings.
- **Evidence:**
  - Data-backed justification mixing:
    - Key performance indicator vs market deltas  
    - Pricing vs comps  
    - Complaint/sentiment snippets  
    - Historical conversion/occupancy patterns  
  - Integrate **2–3 review quotes** (concise) drawn from sample_reviews/review_quotes for the listing.

YOU SHOULD:
- Produce 5–6 Tier 1 rows and 5–6 Tier 2 rows when data exists.

YOU MUST NOT:
- Dump raw JSON or key/value lists.
- Produce per‑listing mini‑reports.
- Repeat generic advice—tie every row to its specific listing’s data and reviews.

============================================================
SECTION 3 — "## **3. PORTFOLIO PLAYBOOK FOR THE NEXT 30 DAYS**"
============================================================

GOAL: Convert the insights into a 3-phase operating plan.

YOU MUST:
- After the section heading, add **one bullet** summarizing the purpose of the next 30 days.
- Use the triage.playbook_30d windows and render **subheadings exactly in this style**:

`### **WEEK 1-2 - “STOP THE BLEEDING”:**`  
`### **WEEK 2-3 - “MONETIZE THE WINNERS”:**`  
`### **WEEK 3-4 - “REVIEW + REBALANCE”:**`

- Under each, write **5–7 concrete operational bullets** with recommendations provides in section 2.
- Explicitly reference Tier 1/Tier 2 listings by ID.
- Tie actions to the weak neighbourhoods diagnosed earlier.

YOU SHOULD:
- End the entire section with a bullet recommending a **monthly triage habit**.

YOU MUST NOT:
- Use technical model language.
- Mention schema fields, keys, or JSON.

============================================================
TIER LISTING TEMPLATE — APPLY TO EVERY LISTING WRITE-UP
============================================================

For every Tier 1 and Tier 2 listing you discuss use:

1. **Key performance indicator Positioning vs Market**
2. **Pricing Diagnostics**
3. **Sentiment Pulse (3 quotes minimum)**
4. **Theme Insight**
5. **Average daily rate Test Band & Expected Revenue**
6. **Economist Interpretation**
7. **Action Plan**

Under each heading, weave a short paragraph or 1–3 bullets referencing the structured data. Requirements:
- Cite the Key performance indicator value vs market median numerically.
- Quote the exact Average daily rate gap %, current Average daily rate, market Average daily rate, and Average daily rate test band that were provided (`adr_current_fmt`, `adr_market_fmt`, `adr_test_band_fmt`).
- When referencing pricing, use the gap from `adr_gap_percent` plus the raw Average daily rate values so the reader can see the thesis immediately.
- Include at least 3 review snippets (from `sample_reviews`) inside **Sentiment Pulse**; tie those quotes back to the dominant `theme` and sentiment strength.
- In **ADR Test Band & Expected Revenue**, leverage `quant_bundle.revenue_low`/`revenue_high` and the broader `adr_recommendation` data to state the 30-day upside range verbatim.
- **Economist Interpretation** must mention the -0.8 elasticity assumption, substitution risk, and the demand slope implied by the ADR gap, mirroring the structured econ narrative.
- **Action Plan** should sequence tactical moves (pricing, ops, sentiment recovery) referencing `adr_recommendation.test_low/test_high`, KPI gaps, and sentiment quotes.

You have explicit per-listing keys available:
- `adr_current_fmt`, `adr_market_fmt`, `adr_test_band_fmt`, `adr_gap_percent`
- `adr_current`, `adr_market_median`, `adr_test_band_low`, `adr_test_band_high`
- `quant_bundle` (use its `revenue_low`, `revenue_high`, `occ30`, and ADR math explicitly)
- `adr_recommendation` (for test_low/test_high/baseline)
- `sentiment` + `sample_reviews` + `theme`
Always reference these keys directly in your narrative — the executive expects to see the exact Average Daily rate/price_in_usd numbers, test bands, and modeled uplift spelled out.

============================================================
STYLE & TONE — APPLY TO ALL SECTIONS
============================================================

- Write like a **senior portfolio manager** briefing executives.
- Confident, direct, analytical; no hedging.
- Blend data, narrative, and sentiment like a real operator. Always lead with quantitative metrics, then bring in review/sentiment evidence.
- Use **markdown tables, bullets, and short paragraphs**; avoid verbosity without insight.
- Reference specific numbers frequently (deltas, occupancy rates, ADR gaps).
- Always tie review sentiment to operational implications.
- Keep the voice warm and conversational but grounded in data.
- Invite the operator to keep exploring with a friendly closing question.
- Mention `adr_recommendation.test_low/test_high` directly when discussing ADR adjustments.
- Cite at least two review excerpts per listing when evidence is requested; ensure metrics and sentiments appear side-by-side (e.g., “Current price $240 vs market $265 (−9%); KPI 62% vs 74%; Quotes: ...”).
"""

TRIAGE_RENDERING_RULES = """[TRIAGE_RENDERING_RULES]
You MUST use the structured listing objects provided above. Do NOT invent metrics or infer values that are missing.
- For every listing, report KPI vs market exactly as provided (include the delta in pts).
- Cite ADR current, market, gap %, and the ADR test band (test_low/test_high) verbatim.
- Reference quant_bundle.revenue_low/revenue_high and quant_bundle.occ30 to explain revenue upside and demand guardrails.
- Use every review quote supplied (minimum three) inside Sentiment/Evidence blocks.
- Tie issue narratives and recommendations to the detected theme(s) and tier (Tier 1 = urgent ops recovery, Tier 2 = monetization plan).
- Recommended Action must mention adr_recommendation.test_low/test_high, expected revenue impact, and operational sequencing.
- Evidence column must combine KPI delta, ADR delta, and at least two review snippets with month/year context.
- Build Tier 1 and Tier 2 tables with columns: Listing | Area | Issue | Recommended Action (Next 30 days) | Evidence.
- Each listing requires unique, numerically grounded narrative — no repeated text.
- Do not drop or paraphrase numbers; quote them exactly as the structured data presents.
"""

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def _format_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def _truncate(text: str, limit: int = 160) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _strip_sql_snippets(text: Optional[str]) -> Optional[str]:
    """Remove obvious SQL statements without stripping normal prose."""
    if not text or not isinstance(text, str):
        return text

    pattern = re.compile(r"(?is)(^|\n)\s*(select|with|insert|update|delete)\b.*?(;|$)")
    cleaned = pattern.sub("\n", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _escape_markdown_table_cell(text: str) -> str:
    """Escape characters that would break markdown tables."""
    if not text:
        return ""
    escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("\r\n", "\n").replace("\r", "\n")
    escaped = escaped.replace("\n", "<br>")
    for char in ("|", "*", "_", "`", "[", "]"):
        escaped = escaped.replace(char, f"\\{char}")
    return escaped.strip()


def _extract_rag_hits(preview_text: str) -> List[Dict[str, str]]:
    """Parse legacy bullet preview lines into structured RAG hit data."""
    hits: List[Dict[str, str]] = []
    if not preview_text:
        return hits
    for raw_line in preview_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        match = _RAG_PREVIEW_LINE_RE.match(line)
        if not match:
            continue
        borough = (match.group("borough") or "").strip()
        month_year_raw = (match.group("month_year") or "").strip()
        listing_id = (match.group("listing_id") or "").strip()
        comment_id = (match.group("comment_id") or "").strip()
        snippet = (match.group("snippet") or "").strip()

        month = ""
        year = ""
        if month_year_raw:
            month_year_parts = month_year_raw.split()
            if month_year_parts:
                month = month_year_parts[0]
            if len(month_year_parts) >= 2:
                year = " ".join(month_year_parts[1:])
        hits.append(
            {
                "borough": borough or "n/a",
                "month": month or "",
                "year": year or "",
                "month_year": month_year_raw or "",
                "snippet": snippet,
                "listing_id": listing_id or "?",
                "comment_id": comment_id or "?",
            }
        )
    return hits


def _build_rag_markdown_table(rag_hits: List[Dict[str, str]], insights_text: str) -> Optional[str]:
    """Render a markdown table for RAG-only responses."""
    if not rag_hits:
        return None

    rows = [
        "### Guest Highlights",
        "",
        "| Review Context | Key Insight |",
        "|----------------|-------------|",
    ]

    for hit in rag_hits:
        borough = hit.get("borough") or "n/a"
        comment_id = hit.get("comment_id") or ""
        month_year = hit.get("month_year") or ""
        month = hit.get("month") or ""
        year = hit.get("year") or ""

        month_year_display = month_year or " ".join(part for part in (month, year) if part)
        month_year_display = month_year_display.strip()

        context_parts = []
        if borough.strip():
            context_parts.append(borough.strip())
        if month_year_display and not all(part.lower() == "n/a" for part in month_year_display.split()):
            context_parts.append(f"({month_year_display})")
        context_cell = " ".join(context_parts) if context_parts else "n/a"
        context_cell = _escape_markdown_table_cell(context_cell)

        snippet_text = _truncate(hit.get("snippet") or "", limit=140)
        snippet_text = _escape_markdown_table_cell(snippet_text or "No direct guest quote available.")
        rows.append(f'| {context_cell} | "{snippet_text}" |')

    insights = insights_text.strip() if insights_text else ""
    rows.append("")
    rows.append("### Insights:")
    rows.append(insights or "Structured metrics were not found for this request. Summarize guest sentiment and highlight recurring themes from the snippets.")

    return "\n".join(rows)


def format_filters(filters: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k, v in (filters or {}).items():
        if v in (None, "", [], (), {}):
            continue
        if isinstance(v, (list, tuple)):
            v = [str(x) for x in v if x not in (None, "")]
            if v:
                parts.append(f"{k}: {', '.join(v)}")
        else:
            parts.append(f"{k}: {v}")
    return "; ".join(parts) if parts else "none"

# ---------------------------------------------------------------------
# MARKDOWN PREVIEW BUILDER
# ---------------------------------------------------------------------

def render_result_markdown(
    sql_rows: Optional[List[Dict[str, Any]]] = None,
    agg: Optional[Dict[str, Any]] = None,
    review_snippets: Optional[List[Dict[str, Any]]] = None,
    max_rows: int = 8,
) -> str:
    """Render a compact markdown summary of SQL + RAG results."""
    parts: List[str] = []

    # --- SQL Table ---
    if sql_rows:
        rows = sql_rows[:max_rows]
        header = list(rows[0].keys())
        table = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for row in rows:
            table.append("| " + " | ".join(str(row.get(c, "")) for c in header) + " |")
        parts.append("\n".join(table))

    # --- Aggregates ---
    if agg:
        bullets = ["**Aggregates:**"]
        for k, v in agg.items():
            bullets.append(f"- {k}: {_format_float(v)}")
        parts.append("\n".join(bullets))

    # --- Review Snippets ---
    if review_snippets:
        lines = ["**Representative review highlights:**"]
        for snip in review_snippets[:max_rows]:
            borough = snip.get("borough") or snip.get("neighbourhood_group") or "n/a"
            month = snip.get("month") or "n/a"
            year = snip.get("year") or "n/a"
            snippet = snip.get("snippet") or snip.get("text") or snip.get("comments") or ""
            listing_id = snip.get("listing_id", "?")
            comment_id = snip.get("comment_id", "?")
            lines.append(f"- [{borough} | {month} {year}]  #{listing_id} · #{comment_id}: {_truncate(snippet)}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else "_No structured results provided._"


def render_triage_context(triage: Optional[Dict[str, Any]]) -> str:
    """Render portfolio triage context into markdown-like text."""
    if not triage:
        return ""
    lines: List[str] = []

    def _fmt_currency(value: Any) -> Optional[str]:
        if isinstance(value, (int, float)):
            return f"${value:,.0f}"
        return None

    def _fmt_percent(value: Any) -> Optional[str]:
        if isinstance(value, (int, float)):
            return f"{value:+.1f}%"
        return None

    scope = triage.get("scope") or "Highbury"
    kpi = triage.get("kpi_used") or "occupancy_rate_90"
    lines.append(f"Scope: {scope}")
    lines.append(f"KPI used: {kpi}")

    glance = triage.get("portfolio_at_glance") or {}
    distribution = glance.get("kpi_distribution") or []
    if distribution:
        distribution_map = {entry.get("label"): entry.get("value") for entry in distribution}
        lines.append(
            "KPI distribution → "
            f"Listings={distribution_map.get('Listings analysed') or 'n/a'}, "
            f"Median={distribution_map.get('Median '+glance.get('kpi_label','KPI')) or 'n/a'}, "
            f"Avg={distribution_map.get('Average '+glance.get('kpi_label','KPI')) or 'n/a'}, "
            f"Min={distribution_map.get('Portfolio low') or 'n/a'}, "
            f"Max={distribution_map.get('Portfolio high') or 'n/a'}"
        )

    def _format_listing(entry: Dict[str, Any]) -> str:
        listing = entry.get("listing_id") or entry.get("listing_name") or "listing"
        area = ", ".join(filter(None, [entry.get("neighbourhood"), entry.get("borough")])) or "n/a"
        metrics = entry.get("metrics") or {}
        current_price = metrics.get("price")
        market_price = (entry.get("market_context") or {}).get("market_median_price_usd")
        kpi_value = entry.get("kpi_value")
        market_kpi = entry.get("market_kpi_median")
        delta = entry.get("kpi_vs_market_delta")
        adr_gap = entry.get("pricing_gap_percent")
        review_score = metrics.get("review_score")
        sentiment = entry.get("sentiment_label")
        theme = entry.get("theme")
        quotes = entry.get("sample_reviews") or []
        adr_recommendation = entry.get("adr_recommendation") or {}
        adr_low = _fmt_currency(adr_recommendation.get("test_low"))
        adr_high = _fmt_currency(adr_recommendation.get("test_high"))
        adr_current_fmt = entry.get("adr_current_fmt") or _fmt_currency(entry.get("adr_current"))
        adr_market_fmt = entry.get("adr_market_fmt") or _fmt_currency(entry.get("adr_market_median"))
        adr_band_fmt = entry.get("adr_test_band_fmt")
        quant_bundle = entry.get("quant_bundle") or {}
        qb_revenue_low = _fmt_currency(quant_bundle.get("revenue_low")) or quant_bundle.get("revenue_low")
        qb_revenue_high = _fmt_currency(quant_bundle.get("revenue_high")) or quant_bundle.get("revenue_high")

        components = [f"{listing} ({area})"]
        if isinstance(kpi_value, (int, float)) and isinstance(market_kpi, (int, float)):
            delta_text = f"{delta:+.1f}" if isinstance(delta, (int, float)) else ""
            components.append(f"KPI {kpi_value:.1f} vs market {market_kpi:.1f} ({delta_text})")
        curr_price_text = _fmt_currency(current_price)
        market_price_text = _fmt_currency(market_price)
        if curr_price_text or market_price_text or isinstance(adr_gap, (int, float)):
            gap_text = _fmt_percent(adr_gap)
            components.append(
                f"Current price {curr_price_text or 'n/a'} vs market {market_price_text or 'n/a'} ({gap_text or 'gap n/a'})"
            )
        if adr_low and adr_high:
            components.append(f"ADR plan {adr_low}–{adr_high}")
        if isinstance(review_score, (int, float)):
            components.append(f"Review score {review_score:.1f}")
        if sentiment or theme:
            components.append(f"Sentiment={sentiment or 'mixed'} ({theme or 'no theme'})")
        if quotes:
            preview_quotes = " | ".join(f"“{quote}”" for quote in quotes[:2])
            components.append(f"Quotes: {preview_quotes}")
        components.append(
            "data_keys → "
            f"adr_current_fmt={adr_current_fmt or 'n/a'}, "
            f"adr_market_fmt={adr_market_fmt or 'n/a'}, "
            f"adr_test_band_fmt={adr_band_fmt or 'n/a'}, "
            f"adr_gap_percent={_fmt_percent(entry.get('adr_gap_percent')) or 'n/a'}, "
            f"revenue_low={qb_revenue_low or 'n/a'}, "
            f"revenue_high={qb_revenue_high or 'n/a'}, "
            f"occ30={_fmt_percent(quant_bundle.get('occ30')) or quant_bundle.get('occ30') or 'n/a'}"
        )
        return " | ".join(components)

    top5 = glance.get("top5_overview") or []
    if top5:
        lines.append("Top KPI winners (metrics first, sentiment second):")
        for entry in top5[:5]:
            lines.append(f"- {_format_listing(entry)}")

    bottom5 = glance.get("bottom5_overview") or []
    if bottom5:
        lines.append("Bottom KPI laggards:")
        for entry in bottom5[:5]:
            lines.append(f"- {_format_listing(entry)}")

    sentiment_summary = glance.get("sentiment_summary") or {}
    if sentiment_summary:
        total_positive = sentiment_summary.get("total_positive", 0)
        total_negative = sentiment_summary.get("total_negative", 0)
        total_neutral = sentiment_summary.get("total_neutral", 0)
        lines.append(
            f"Sentiment pulse (after metrics): +{total_positive} / {total_neutral} neutral / -{total_negative}"
        )
        for entry in (sentiment_summary.get("listings") or [])[:3]:
            lines.append(
                f"- {entry.get('listing_id')}: {entry.get('label')} ({entry.get('hit_count')} quotes, compound={entry.get('compound')})"
            )

    market_entries = (glance.get("market_benchmarks") or {}).get("entries") or []
    if market_entries:
        lines.append("Market benchmark snapshot (kpi/adr/rev30):")
        for entry in market_entries[:3]:
            lines.append(
                f"- {entry.get('label')}: KPI {entry.get('market_kpi_median')} | ADR ${entry.get('market_price_median')} | rev30 ${entry.get('market_revenue_30_median')}"
            )

    backlog = triage.get("action_backlog") or []
    if backlog:
        lines.append("Action backlog preview (metrics-first, then sentiment):")
        for item in backlog[:4]:
            listing = item.get("listing_id") or "listing"
            kpi_value = item.get("kpi_value")
            market_kpi = item.get("market_kpi_median")
            kpi_delta = item.get("delta")
            price_gap = item.get("pricing_gap_percent")
            metrics = item.get("metrics") or {}
            current_price = metrics.get("price")
            market_price = (item.get("market_context") or {}).get("market_median_price_usd")
            theme = item.get("theme") or "n/a"
            sentiment = (item.get("sentiment") or {}).get("label")
            quotes = item.get("sample_reviews") or []
            adr_rec = item.get("adr_recommendation") or {}
            adr_low = _fmt_currency(adr_rec.get("test_low"))
            adr_high = _fmt_currency(adr_rec.get("test_high"))
            curr_price_text = _fmt_currency(current_price)
            market_price_text = _fmt_currency(market_price)
            kpi_line = f"KPI {kpi_value:.1f} vs market {market_kpi:.1f} ({kpi_delta:+.1f})" if isinstance(kpi_value, (int, float)) and isinstance(market_kpi, (int, float)) and isinstance(kpi_delta, (int, float)) else "KPI metrics n/a"
            price_line = (
                f"Current price {curr_price_text or 'n/a'} vs market {market_price_text or 'n/a'} "
                f"({ _fmt_percent(price_gap) or 'gap n/a'})"
            )
            adr_line = f"ADR plan {adr_low}–{adr_high}" if adr_low and adr_high else "ADR plan n/a"
            quote_line = ""
            if quotes:
                quote_line = "Quotes: " + " | ".join(f"“{quote}”" for quote in quotes[:2])
            sentiment_text = f"Sentiment={sentiment or 'mixed'}"
            lines.append(f"- {listing}: {kpi_line} | {price_line} | {adr_line} | theme={theme} | {sentiment_text} {quote_line}")
            qb = item.get("quant_bundle") or {}
            qb_low = _fmt_currency(qb.get("revenue_low")) or qb.get("revenue_low")
            qb_high = _fmt_currency(qb.get("revenue_high")) or qb.get("revenue_high")
            lines.append(
                "  data_keys → "
                f"adr_current_fmt={item.get('adr_current_fmt') or _fmt_currency(item.get('adr_current')) or 'n/a'}, "
                f"adr_market_fmt={item.get('adr_market_fmt') or _fmt_currency(item.get('adr_market_median')) or 'n/a'}, "
                f"adr_test_band_fmt={item.get('adr_test_band_fmt') or 'n/a'}, "
                f"adr_gap_percent={_fmt_percent(item.get('adr_gap_percent')) or 'n/a'}, "
                f"revenue_low={qb_low or 'n/a'}, "
                f"revenue_high={qb_high or 'n/a'}, "
                f"occ30={_fmt_percent((qb.get('occ30'))) or qb.get('occ30') or 'n/a'}"
            )

    playbook = triage.get("playbook_30d") or []
    if playbook:
        lines.append("Playbook windows (metrics-led tasks):")
        for window in playbook:
            lines.append(f"- {window.get('window')}: {window.get('focus')}")

    return "\n".join(line for line in lines if line)


def _build_sentiment_summary_block(
    rag_snippets: List[Dict[str, Any]],
    sample_limit: int = 2,
) -> Optional[str]:
    """Return a formatted sentiment summary block for review snippets."""
    if not rag_snippets:
        return None

    counts = Counter({"positive": 0, "neutral": 0, "negative": 0})
    compounds: List[float] = []
    quotes: List[str] = []

    for snippet in rag_snippets:
        label = (snippet.get("sentiment_label") or "").strip().lower()
        if label in counts:
            counts[label] += 1
        compound_val = snippet.get("compound")
        try:
            compound_float = float(compound_val)
        except (TypeError, ValueError):
            compound_float = None
        if compound_float is not None:
            compounds.append(compound_float)
        quote = snippet.get("snippet") or snippet.get("comments") or snippet.get("text")
        if quote:
            quotes.append(_truncate(str(quote), 160))

    ordered_labels = ["positive", "neutral", "negative"]
    top_label = max(ordered_labels, key=lambda name: counts[name])
    top_count = counts[top_label]
    second_highest = max((counts[label] for label in ordered_labels if label != top_label), default=0)
    if top_count == 0 or top_count == second_highest:
        overall = "Mixed"
    else:
        overall = top_label.capitalize()

    avg_strength = round(sum(compounds) / len(compounds), 2) if compounds else None
    avg_text = f"{avg_strength:.2f}" if avg_strength is not None else "N/A"

    summary_lines = [
        "### Sentiment Summary",
        f"- Positive reviews: {counts['positive']}",
        f"- Neutral reviews: {counts['neutral']}",
        f"- Negative reviews: {counts['negative']}",
        f"- Overall sentiment: {overall}",
        f"- Average sentiment strength: {avg_text}",
    ]

    sections = ["\n".join(summary_lines)]
    if quotes:
        sample_lines = ["### Sample feedback"]
        for quote in quotes[:sample_limit]:
            sample_lines.append(f"- \"{quote}\"")
        sections.append("\n".join(sample_lines))

    return "\n\n".join(sections)


def _render_expansion_sources(expansion_sources: List[Dict[str, Any]]) -> str:
    """Render web source list as a collapsible block."""
    if not expansion_sources:
        return ""
    lines = [
        "<details>",
        "<summary>Web Sources Used</summary>",
        "",
    ]
    for source in expansion_sources:
        url = source.get("url") or source.get("title") or "Source"
        lines.append(f"- {url}")
    lines.append("</details>")
    return "\n".join(lines)

# ---------------------------------------------------------------------
# COMPOSER INPUT BUILDER
# ---------------------------------------------------------------------

def build_composer_input(
    history: List[Dict[str, str]],
    user_question: str,
    policy: str,
    sql: Optional[str],
    rows: List[Dict[str, Any]],
    aggregates: Optional[Dict[str, Any]],
    rag_snippets: List[Dict[str, Any]],
    applied_filters: Dict[str, Any],
    intent: Optional[str] = None,
    portfolio_triage: Optional[Dict[str, Any]] = None,
    expansion_report: Optional[str] = None,
    expansion_sources: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Build OpenAI-compatible message list including context blocks."""
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": TRIAGE_STYLE_INSTRUCTION},
        {"role": "system", "content": TRIAGE_RENDERING_RULES},
    ]

    # Conversation history
    for h in history:
        if h.get("role") in {"user", "assistant"} and h.get("content"):
            msgs.append({"role": h["role"], "content": h["content"]})

    # Context block
    filters_text = format_filters(applied_filters)
    preview_raw = render_result_markdown(rows, aggregates, rag_snippets)
    preview = _strip_sql_snippets(preview_raw)
    triage_block = render_triage_context(portfolio_triage)
    action_backlog = (portfolio_triage or {}).get("action_backlog") or []
    has_sql_rows = bool(rows)
    has_rag_snippets = bool(rag_snippets)
    expansion_block = _render_expansion_sources(expansion_sources or [])

    ctx_lines = [
        "[CONTEXT]",
        f"policy: {policy}",
        f"filters: {filters_text}",
    ]
    if sql:
        ctx_lines.append("sql: [available for internal reference; do not repeat in the reply]")
    if preview:
        ctx_lines.append("data preview:\n" + preview)
    if intent:
        ctx_lines.append(f"intent: {intent}")
    if triage_block:
        ctx_lines.append("triage context:\n" + triage_block)
    if action_backlog:
        ctx_lines.append("triage_listing_objects_raw:")
        ctx_lines.append("<RAW_JSON_DO_NOT_STRIP>")
        ctx_lines.append(json.dumps(action_backlog, indent=2))
        ctx_lines.append("</RAW_JSON_DO_NOT_STRIP>")
    if expansion_report:
        ctx_lines.append("expansion report:\n" + expansion_report)
    if expansion_block:
        ctx_lines.append("web sources used:\n" + expansion_block)
    if filters_text == "none":
        ctx_lines.append("applied_filters: none")

    ctx_lines.append(f"has_sql_rows={str(has_sql_rows).lower()}")
    ctx_lines.append(f"has_rag_snippets={str(has_rag_snippets).lower()}")

    ctx_lines.insert(0, "You are talking to a Highbury team member. Be friendly and conversational while explaining insights clearly.")
    ctx_lines.append("When replying, sound like a thoughtful colleague, not a chatbot. Invite them to continue exploring.")

    context_payload = "\n\n".join(ctx_lines)
    raw_blocks = re.findall(r"<RAW_JSON_DO_NOT_STRIP>(.*?)</RAW_JSON_DO_NOT_STRIP>", context_payload, flags=re.S)
    for block in raw_blocks:
        context_payload = context_payload.replace(f"<RAW_JSON_DO_NOT_STRIP>{block}</RAW_JSON_DO_NOT_STRIP>", block)
    context_payload = _strip_sql_snippets(context_payload) or context_payload
    msgs.append({"role": "assistant", "name": "context", "content": context_payload})
    msgs.append({"role": "user", "content": user_question})
    return msgs

# ---------------------------------------------------------------------
# OPENAI CALLER
# ---------------------------------------------------------------------

def _extract_usage(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    usage = {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        val = getattr(raw, k, None) if not isinstance(raw, dict) else raw.get(k)
        if val is not None:
            usage[k] = val
    return usage


def compose_answer(
    messages: List[Dict[str, str]],
    model: str,
    stream_handler: Optional[Any] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Invoke OpenAI chat completion with optional streaming."""
    if OpenAI is None:
        raise ComposeError("openai package not installed.")
    if not os.getenv("OPENAI_API_KEY"):
        raise ComposeError("OPENAI_API_KEY not set.")

    cfg = load_config()
    stream_enabled = bool(cfg.streaming_enabled and stream_handler)
    client = OpenAI()

    # Keep original message list from build_composer_input
    original_messages = list(messages)

    # -------------------------------
    # Extract last user query
    # -------------------------------
    user_query = ""
    for msg in reversed(original_messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "") or ""
            break

    normalized_query = re.sub(r"[^\w\s]", "", (user_query or "").lower()).strip()
    simple_prompts = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "who are you",
    }
    if normalized_query in simple_prompts:
        friendly_reply = (
            "👋 Hi there! I’m wtchtwr - your data companion. "
            "You can ask me about prices, occupancy, revenue  or guest reviews!"
        )
        return friendly_reply, {}

    # -------------------------------
    # Parse [CONTEXT] system block
    # -------------------------------
    context_content = ""
    context_index = None
    for idx, msg in enumerate(original_messages):
        if msg.get("name") == "context":
            context_content = msg.get("content", "")
            context_index = idx
            break
        if msg.get("role") == "system" and "[CONTEXT]" in (msg.get("content") or ""):
            context_content = msg.get("content", "")
            context_index = idx
            break

    policy_value: Optional[str] = None
    filters_value: Optional[str] = None
    intent_value: Optional[str] = None
    has_sql_rows = False
    has_rag_snippets = False
    if context_content:
        for line in context_content.splitlines():
            line = line.strip()
            if line.startswith("policy:"):
                policy_value = line.split(":", 1)[1].strip()
                continue
            if line.startswith("filters:"):
                filters_value = line.split(":", 1)[1].strip()
                continue
            if line.startswith("intent:"):
                intent_value = line.split(":", 1)[1].strip().upper() or None
                continue
            if line.startswith("has_sql_rows="):
                has_sql_rows = line.split("=", 1)[1].strip().lower() == "true"
                continue
            if line.startswith("has_rag_snippets="):
                has_rag_snippets = line.split("=", 1)[1].strip().lower() == "true"

    conversational_intents = {"GREETING", "THANKS", "SMALLTALK"}
    normalized_intent = (intent_value or "").upper() if intent_value else None
    if normalized_intent in conversational_intents or (policy_value and policy_value.upper() == "CONVERSATION"):
        if normalized_intent == "GREETING":
            return (
                "👋 Hi there! I’m wtchtwr - your data companion. You can ask me about prices, occupancy, revenue, or guest reviews!",
                {},
            )
        if normalized_intent == "THANKS":
            return (
                "😊 Always happy to help! Anything else you’d like to explore?",
                {},
            )
        return (
            "I’m wtchtwr- an AI analytics companion built to uncover insights from data. Try asking something like 'Average price in Brooklyn' or 'Reviews about cleanliness in Manhattan'.",
            {},
        )

    # -------------------------------
    # Build SQL/RAG preview + markdown_table
    # -------------------------------
    preview_text = ""
    if context_content and "data preview:\n" in context_content:
        preview_section = context_content.split("data preview:\n", 1)[1].strip()
        preview_text = _strip_sql_snippets(preview_section) or preview_section

    table_preview = ""
    rag_preview = ""
    if preview_text:
        marker = "**Representative review highlights:**"
        if marker not in preview_text:
            if preview_text.startswith(marker):
                rag_preview = preview_text.strip()
            else:
                table_preview = preview_text
        else:
            before, after = preview_text.split(marker, 1)
            table_preview = before.strip()
            rag_preview = (marker + "\n" + after.strip()).strip()
        table_preview = _strip_sql_snippets(table_preview) or (table_preview or "")
        rag_preview = _strip_sql_snippets(rag_preview) or (rag_preview or "")

    sql_source_raw = table_preview or preview_text or "Structured SQL metrics available."
    sql_source = _strip_sql_snippets(sql_source_raw)
    if sql_source_raw != sql_source:
        _LOGGER.info("[CLEANUP][COMPOSER] Stripping SQL text from LLM input context.")
    rag_source_raw = rag_preview or ("Guests mentioned a few highlights worth noting." if has_rag_snippets else "")
    rag_source = _strip_sql_snippets(rag_source_raw) or rag_source_raw
    content_hint = "Structured metrics and review snippets are both available; weave them into one friendly narrative."

    if not has_sql_rows and not has_rag_snippets:
        content_hint = (
            "No structured metrics or review snippets were retrieved. Offer a supportive fallback and suggest another angle to explore."
        )
        sql_source = "No structured metrics available for this request."
        rag_source = "No review snippets matched; encourage a follow-up question."
    elif not has_sql_rows and has_rag_snippets:
        content_hint = (
            "Guest review snippets matched this query. Lead with those qualitative insights and weave a confident narrative; "
            "only mention missing structured metrics briefly (and never as the first sentence)."
        )
        sql_source = "Focus on the guest highlights from this timeframe; structured metrics just weren’t part of this pull."
        if not rag_source:
            rag_source = (
                rag_preview
                or preview_text
                or "Guests shared qualitative feedback — summarize their themes."
            )
    elif has_sql_rows and not has_rag_snippets:
        content_hint = (
            "Structured metrics returned without reviews; ground the answer in KPI, ADR, and revenue analytics, "
            "and note politely that no fresh guest feedback was available."
        )
        if not rag_source:
            rag_source = "No review snippets were available for this query."
    else:
        if not rag_source:
            rag_source = "Guests mentioned a few highlights worth noting."

    rag_source = _strip_sql_snippets(rag_source) or rag_source

    rag_hits = _extract_rag_hits(rag_preview) if rag_preview else []
    rag_table_block: Optional[str] = None
    if has_rag_snippets and not has_sql_rows:
        insights_seed = (rag_source or "").strip()
        if rag_preview and insights_seed == rag_preview:
            insights_seed = ""
        insights_seed = _strip_sql_snippets(insights_seed) or insights_seed
        if not insights_seed:
            insights_seed = "Guest reviews only — distill key praise and friction themes, then suggest next steps."
        # [RAG_UI] Simplified guest highlights table — no listing/comment IDs shown.
        # Context column now combines borough + month + year.
        rag_table_block = _build_rag_markdown_table(rag_hits, insights_seed)

    if rag_table_block:
        cleaned_block = _strip_sql_snippets(rag_table_block)
        if cleaned_block != rag_table_block:
            _LOGGER.info("[CLEANUP][COMPOSER] Removing SQL text from RAG table block.")
        markdown_table = cleaned_block
    else:
        markdown_parts = []
        if table_preview:
            markdown_parts.append(table_preview)
        if rag_preview and rag_preview != table_preview:
            markdown_parts.append(rag_preview)
        if not markdown_parts:
            markdown_parts.append(sql_source)
        markdown_table_raw = "\n\n".join(part.strip() for part in markdown_parts if part.strip()) or sql_source
        markdown_table = _strip_sql_snippets(markdown_table_raw)
        if markdown_table_raw != markdown_table:
            _LOGGER.info("[CLEANUP][COMPOSER] Removing SQL text from markdown table preview.")

    markdown_table = markdown_table or "No structured preview available."

    triage_context_text = ""
    if "triage context:\n" in context_content:
        triage_section = context_content.split("triage context:\n", 1)[1]
        for stop_marker in ("triage_listing_objects:", "triage_instruction_full:", "triage_instruction:", "has_sql_rows="):
            if stop_marker in triage_section:
                triage_section = triage_section.split(stop_marker, 1)[0]
                break
        triage_context_text = (triage_section or "").strip()
        triage_context_text = _strip_sql_snippets(triage_context_text) or triage_context_text

    triage_listing_text = ""
    listing_marker = "triage_listing_objects_raw:"
    if listing_marker in context_content:
        listing_section = context_content.split(listing_marker, 1)[1]
        for stop_marker in ("triage_instruction_full:", "triage_instruction:", "has_sql_rows=", "guidance:"):
            if stop_marker in listing_section:
                listing_section = listing_section.split(stop_marker, 1)[0]
                break
        triage_listing_text = (listing_section or "").strip()

    messages = list(original_messages)

    context_lines: List[str] = [
        "[CONTEXT]",
        f"policy: {policy_value or 'unknown'}",
        f"filters: {filters_value or 'none'}",
        f"intent: {intent_value or 'unknown'}",
        f"has_sql_rows={str(has_sql_rows).lower()}",
        f"has_rag_snippets={str(has_rag_snippets).lower()}",
        f"guidance: {content_hint}",
        "SQL summary:",
        sql_source or "No structured metrics available.",
        "Review summary:",
        rag_source or "No review snippets available.",
        "Structured data preview:",
        markdown_table,
    ]
    if triage_context_text:
        context_lines.append("triage context:\n" + triage_context_text)
    if triage_listing_text:
        context_lines.append("triage_listing_objects_raw:")
        context_lines.append("<RAW_JSON_DO_NOT_STRIP>")
        context_lines.append(triage_listing_text)
        context_lines.append("</RAW_JSON_DO_NOT_STRIP>")

    context_payload = "\n\n".join(line for line in context_lines if line)
    raw_blocks = re.findall(r"<RAW_JSON_DO_NOT_STRIP>(.*?)</RAW_JSON_DO_NOT_STRIP>", context_payload, flags=re.S)
    for block in raw_blocks:
        context_payload = context_payload.replace(f"<RAW_JSON_DO_NOT_STRIP>{block}</RAW_JSON_DO_NOT_STRIP>", block)
    context_payload = _strip_sql_snippets(context_payload) or context_payload
    context_message = {"role": "assistant", "name": "context", "content": context_payload}

    if context_index is not None and 0 <= context_index < len(messages):
        messages[context_index] = context_message
    else:
        last_user_index: Optional[int] = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_user_index = idx
                break
        if last_user_index is None:
            raise ComposeError("Composer input missing user message.")
        messages.insert(last_user_index, context_message)

    # -------------------------------
    # Call OpenAI with automatic fallback
    # -------------------------------
    primary_model = model or cfg.openai_model
    fallback_model = cfg.openai_fallback_model or primary_model

    def _invoke_model(model_name: str) -> Tuple[str, Dict[str, Any]]:
        if stream_enabled:
            stream = client.chat.completions.create(model=model_name, messages=messages, stream=True)
            chunks: List[str] = []
            usage: Dict[str, Any] = {}
            for chunk in stream:
                if getattr(chunk, "choices", None):
                    delta = getattr(chunk.choices[0], "delta", {})
                    text = getattr(delta, "content", "") if not isinstance(delta, dict) else delta.get("content", "")
                    if text:
                        chunks.append(text)
                        if stream_handler:
                            try:
                                stream_handler(text)
                            except Exception:
                                pass
                if hasattr(chunk, "usage"):
                    usage = _extract_usage(chunk.usage)
            return "".join(chunks).strip(), usage

        print("\n\n===== OPENAI MESSAGES PAYLOAD =====")
        print(json.dumps(messages, indent=2))
        print("===================================\n\n")
        resp = client.chat.completions.create(model=model_name, messages=messages)
        msg = resp.choices[0].message
        result = (msg.content or "").strip()
        if stream_handler and result:
            try:
                stream_handler(result)
            except Exception:
                pass
        if not result:
            result = "No response generated by wtchtwr — please retry your query."
        usage = _extract_usage(getattr(resp, "usage", {}))
        return result, usage

    _LOGGER.info("[MODEL_ROUTER] Using primary model: %s", primary_model)
    try:
        content, usage = _invoke_model(primary_model)
    except Exception as primary_exc:
        if fallback_model and fallback_model != primary_model:
            _LOGGER.info("[MODEL_ROUTER] Falling back to: %s", fallback_model)
            try:
                content, usage = _invoke_model(fallback_model)
            except Exception as fallback_exc:
                raise ComposeError(
                    f"Both primary ({primary_model}) and fallback ({fallback_model}) models failed: {fallback_exc}"
                ) from fallback_exc
        else:
            raise ComposeError(f"Primary model {primary_model} failed and no fallback configured: {primary_exc}") from primary_exc

    if content:
        cleaned_content = _strip_sql_snippets(content)
        if cleaned_content != content:
            _LOGGER.warning("[POST_SANITY][COMPOSER] SQL still found in final message, removing.")
        content = cleaned_content

    if isinstance(content, str) and "select" in content.lower():
        _LOGGER.warning("[LEAK_DETECTOR][GRAPH_COMPOSE] SQL text present after compose_answer: %s", content[:200])

    try:
        debug_state = {"answer_text": content, "content": content}
        _LOGGER.warning("[COMPOSE_DEBUG] compose_answer keys: %s", list(debug_state.keys()))
        _LOGGER.warning(
            "[COMPOSE_DEBUG] compose_answer content: %s",
            debug_state.get("answer_text"),
        )
        _LOGGER.warning("[COMPOSE_PATCH] Branding header skipped (handled by frontend).")
    except Exception as exc:  # pragma: no cover - defensive logging
        _LOGGER.warning("[COMPOSE_DEBUG] unable to log compose content: %s", exc)

    content = content.strip()
    return content, usage

# ---------------------------------------------------------------------
# GUARDS & FALLBACKS
# ---------------------------------------------------------------------

def critic_guard(answer: str, rows_count: int, agg: Optional[Dict[str, Any]]) -> Optional[str]:
    """Warn if the LLM gave an answer with insufficient evidence."""
    if not answer.strip():
        return "Composer returned an empty answer; please retry or adjust the query."
    if rows_count == 0 and not agg:
        return "No tabular results available. Consider broadening filters or using another borough."
    return None


def fallback_text(bundle: Dict[str, Any]) -> str:
    """Deterministic fallback when LLM unavailable."""
    lines = ["⚙️ LLM unavailable — showing structured summary."]
    filters_text = format_filters(bundle.get("applied_filters") or {})
    if filters_text and filters_text != "none":
        lines.append(f"Filters: {filters_text}")

    expansion_report = bundle.get("expansion_report")
    expansion_sources = bundle.get("expansion_sources") or []
    if expansion_report:
        lines.append(expansion_report.strip())
        sources_block = _render_expansion_sources(expansion_sources)
        if sources_block:
            lines.append("")
            lines.append(sources_block)
        return "\n\n".join(lines)

    rows = bundle.get("rows", [])
    rag_snippets = bundle.get("rag_snippets", [])
    if rows:
        first = rows[0]
        preview = ", ".join(f"{k}={v}" for k, v in list(first.items())[:4])
        lines.append(f"Top row → {preview}")
        if len(rows) > 1:
            lines.append(f"(+{len(rows)-1} more rows)")
    if rag_snippets:
        snip = rag_snippets[0]
        lines.append(f"Review → {_truncate(snip.get('snippet') or snip.get('text') or '')}")
        if len(rag_snippets) > 1:
            lines.append(f"(+{len(rag_snippets)-1} more reviews)")
    if not rows and not rag_snippets:
        lines.append("No structured data for this query.")

    summary_block = _build_sentiment_summary_block(rag_snippets)
    if summary_block:
        lines.append("")
        lines.append(summary_block)
    return "\n".join(lines)

# ---------------------------------------------------------------------
__all__ = [
    "ComposeError",
    "SYSTEM_PROMPT",
    "render_result_markdown",
    "build_composer_input",
    "compose_answer",
    "critic_guard",
    "fallback_text",
    "format_filters",
    "_build_sentiment_summary_block",
    "_render_expansion_sources",
]

# [LEAK_DETECTOR]: Added logging to monitor SQL text leakage.
# [SQL_LEAK_FIX_2] Removed SQL query text from LLM composer prompt.
# [SQL_LEAK_FIX_3] Composer sanitized to exclude SQL text from model prompts.
# [SQL_LEAK_FIX_4] Removed SQL text from composer input and output.