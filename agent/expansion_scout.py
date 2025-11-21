"""Expansion Scout module for scouting new neighbourhood opportunities."""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from langchain_community.document_loaders import WebBaseLoader
except Exception:  # pragma: no cover - optional dependency
    WebBaseLoader = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from .config import load_config
from .guards import fallback_expansion_text
from .types import State, add_thinking_step

_LOGGER = logging.getLogger(__name__)

DEFAULT_EXPANSION_TEMPLATES: Dict[str, str] = {
    "tourism_macro": "NYC neighborhood tourism trends 2024 2025 growing visitor demand",
    "tourism_trending": "Top trending NYC neighborhoods for travelers 2025",
    "infrastructure": "NYC subway expansion 2025 neighborhood impact development pipeline",
    "development": "NYC major residential commercial developments by neighborhood 2024 2025",
    "regulation": "NYC short term rental law updates 2025 neighborhood risk friendly areas",
}

_KEYPOINT_TERMS = ("increase", "development", "rezoning", "project", "growth", "expansion", "pipeline")
_MAX_ARTICLE_CHARS = 12000  # ~3000 tokens


def tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Run a Tavily web search for the query and return simplified results."""
    cfg = load_config()
    api_key = cfg.tavily_api_key or os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set.")

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
    }
    resp = requests.post("https://api.tavily.com/search", json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json() or {}
    results = []
    for item in data.get("results", [])[:max_results]:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "url": item.get("url"),
                "title": item.get("title") or item.get("url") or query,
                "score": item.get("score"),
            }
        )
    return results


def load_article(url: str) -> str:
    """Fetch and clean article text, truncated to ~3000 tokens."""
    if not url:
        return ""
    if WebBaseLoader is None:
        _LOGGER.warning("[EXPANSION] WebBaseLoader unavailable; skipping article fetch.")
        return ""
    try:
        docs = WebBaseLoader(url).load()
    except Exception as exc:  # pragma: no cover - network/loader failures
        _LOGGER.warning("[EXPANSION] Failed to load article %s: %s", url, exc)
        return ""

    text_parts = [getattr(doc, "page_content", "") or "" for doc in docs]
    text = " ".join(text_parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_ARTICLE_CHARS]


def generate_dynamic_queries(user_question: str) -> Dict[str, str]:
    """Generate tailored Tavily queries from the user's question; fallback to defaults."""
    if not OpenAI or not (user_question or "").strip():
        return DEFAULT_EXPANSION_TEMPLATES

    client = OpenAI()
    cfg = load_config()
    primary_model = getattr(cfg, "openai_model", "gpt-5.1") or "gpt-5.1"
    fallback_model = getattr(cfg, "openai_fallback_model", "gpt-4o") or "gpt-4o"
    prompt = (
        "Generate 5 concise web search queries (one per line) for scouting NYC neighborhoods for Highbury.\n"
        "Focus on tourism momentum, infrastructure, development pipeline, and regulation.\n"
        "Exclude Manhattan; prioritize Queens/Brooklyn unless user explicitly requests otherwise.\n"
        f"User question: {user_question.strip()}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=primary_model,
            messages=[
                {"role": "system", "content": "Create targeted Tavily search queries for NYC neighborhood scouting."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        content = resp.choices[0].message.content or ""
        lines = [line.strip("- ").strip() for line in content.splitlines() if line.strip()]
        queries = [line for line in lines if len(line) > 8][:5]
        # Map onto template keys to keep downstream categories
        keys = list(DEFAULT_EXPANSION_TEMPLATES.keys())
        generated = {keys[i]: queries[i] for i in range(min(len(keys), len(queries)))}
        # Fill any missing with defaults
        for key in keys:
            generated.setdefault(key, DEFAULT_EXPANSION_TEMPLATES[key])
        return generated
    except Exception as exc:  # pragma: no cover - fallback behavior
        if fallback_model != primary_model:
            try:
                _LOGGER.warning("[EXPANSION] Primary query model failed (%s), retrying with fallback %s", primary_model, fallback_model)
                resp = client.chat.completions.create(
                    model=fallback_model,
                    messages=[
                        {"role": "system", "content": "Create targeted Tavily search queries for NYC neighborhood scouting."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                )
                content = resp.choices[0].message.content or ""
                lines = [line.strip("- ").strip() for line in content.splitlines() if line.strip()]
                queries = [line for line in lines if len(line) > 8][:5]
                keys = list(DEFAULT_EXPANSION_TEMPLATES.keys())
                generated = {keys[i]: queries[i] for i in range(min(len(keys), len(queries)))}
                for key in keys:
                    generated.setdefault(key, DEFAULT_EXPANSION_TEMPLATES[key])
                return generated
            except Exception:
                _LOGGER.warning("[EXPANSION] Fallback query model also failed: %s", exc)
        _LOGGER.warning("[EXPANSION] Dynamic query generation failed: %s", exc)
        return DEFAULT_EXPANSION_TEMPLATES


def _extract_key_points(text: str) -> List[str]:
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    highlighted = [
        sentence.strip()
        for sentence in sentences
        if sentence and any(term in sentence.lower() for term in _KEYPOINT_TERMS)
    ]
    candidates = highlighted or sentences
    key_points = [s.strip()[:280] for s in candidates if s.strip()]
    return key_points[:3]


def _map_category(key: str) -> str:
    lowered = (key or "").lower()
    if "regulation" in lowered:
        return "regulation"
    if "infra" in lowered or "subway" in lowered:
        return "infrastructure"
    if "develop" in lowered:
        return "development"
    if "tourism" in lowered or "travel" in lowered:
        return "tourism"
    return lowered or "tourism"


def normalize_external_signals(raw_sources: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Normalize raw web sources into structured signals."""
    normalized = {
        "tourism_signals": [],
        "infrastructure_signals": [],
        "regulation_signals": [],
        "development_signals": [],
    }
    for entry in raw_sources or []:
        category = (entry.get("category") or "").lower()
        bucket = f"{category}_signals" if category else None
        if bucket not in normalized:
            continue
        text = (entry.get("text") or "").strip()
        normalized[bucket].append(
            {
                "url": entry.get("url"),
                "title": entry.get("title") or entry.get("url") or "Source",
                "text": text,
                "key_points": _extract_key_points(text),
            }
        )
    return normalized


def _render_signals(signals: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for sig in signals or []:
        title = sig.get("title") or sig.get("url") or "Source"
        points = sig.get("key_points") or []
        preview = "; ".join(points) if points else (sig.get("text") or "")[:180]
        lines.append(f"- {title}: {preview}")
    return "\n".join(lines)


def synthesize_expansion_report(normalized_signals: Dict[str, Any], highbury_context: Dict[str, Any]) -> str:
    """Use GPT-5.1 to synthesize the expansion recommendation."""
    if OpenAI is None:
        raise RuntimeError("openai package not available for expansion report.")

    client = OpenAI()
    cfg = load_config()
    primary_model = getattr(cfg, "openai_model", "gpt-5.1") or "gpt-5.1"
    fallback_model = getattr(cfg, "openai_fallback_model", "gpt-4o") or "gpt-4o"

    tourism_block = _render_signals(normalized_signals.get("tourism_signals", []))
    infra_block = _render_signals(normalized_signals.get("infrastructure_signals", []))
    regulation_block = _render_signals(normalized_signals.get("regulation_signals", []))
    development_block = _render_signals(normalized_signals.get("development_signals", []))
    context_lines = [
        "You are generating a structured EXPANSION SCOUT REPORT for Highbury.",
        "Use ONLY the signals provided below.",
        "",
        "HIGHBURY CONTEXT:",
        "- 37 of 38 listings in Manhattan (overexposure).",
        "- Needs diversification into Queens/Brooklyn.",
        "- Core segments: business travelers, couples, longer-stay tourists.",
        "- Wants neighbourhoods with rising tourism + strong development + stable regulation.",
        "",
    "WEB SIGNALS (raw summaries):",
    "",
    "TOURISM SIGNALS:",
    tourism_block or "- none",
    "",
    "INFRASTRUCTURE SIGNALS:",
    infra_block or "- none",
    "",
    "DEVELOPMENT SIGNALS:",
    development_block or "- none",
    "",
    "REGULATION SIGNALS:",
    regulation_block or "- none",
    "",
        "STRUCTURE YOUR OUTPUT EXACTLY AS FOLLOWS:",
        "",
        "# 1. Summary Recommendation",
        "# 2. Ranked Neighborhood Recommendations",
        "# 3. External Signal Breakdown (Table)",
        "# 4. Recommended Property Types",
        "# 5. Risks & Constraints",
        "# 6. Final Operator Takeaway",
        "# 7. Sources (collapsible)",
        "",
        "Do NOT recommend Manhattan.",
        "Keep it concise and operator-ready.",
    ]
    prompt = "\n".join(context_lines)

    def _call_model(model_name: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strategic real estate advisor focused on NYC expansion.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
        )
        return (response.choices[0].message.content or "").strip()

    try:
        content = _call_model(primary_model)
    except Exception as primary_exc:  # pragma: no cover - defensive
        _LOGGER.warning("[EXPANSION] Primary model %s failed: %s", primary_model, primary_exc)
        if fallback_model and fallback_model != primary_model:
            try:
                content = _call_model(fallback_model)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Expansion synthesis failed for both primary ({primary_model}) and fallback ({fallback_model}) models: {fallback_exc}"
                ) from fallback_exc
        else:
            raise RuntimeError(f"Expansion synthesis failed for primary model {primary_model}: {primary_exc}") from primary_exc
    return content.strip()


def exec_expansion_scout(state: State) -> State:
    """Main LangGraph node to orchestrate Expansion Scout."""
    start = time.perf_counter()
    telemetry = state.setdefault("telemetry", {})
    cfg = load_config()
    raw_sources: List[Dict[str, Any]] = []
    max_results = min(getattr(cfg, "tavily_max_results", 3) or 3, 3)
    per_category_cap = 2
    global_cap = 10
    user_question = state.get("query") or ""
    queries_map = generate_dynamic_queries(user_question)

    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_meta = {}
            for category, query in queries_map.items():
                future = executor.submit(tavily_search, query, max_results)
                future_to_meta[future] = {"category": _map_category(category), "query": query}

            search_results: List[tuple[str, Dict[str, Any]]] = []
            category_counts: Dict[str, int] = {}
            for future in as_completed(future_to_meta):
                meta = future_to_meta[future]
                try:
                    hits = future.result()
                except Exception as search_exc:
                    _LOGGER.warning("[EXPANSION] Tavily search failed for %s: %s", meta["query"], search_exc)
                    continue
                for hit in hits:
                    if len(search_results) >= global_cap:
                        break
                    count = category_counts.get(meta["category"], 0)
                    if count >= per_category_cap:
                        continue
                    category_counts[meta["category"]] = count + 1
                    search_results.append((meta["category"], hit))

        # Fetch article content (can be parallelized; keep small pool)
        with ThreadPoolExecutor(max_workers=5) as executor:
            article_futures = {}
            for category, hit in search_results:
                url = hit.get("url")
                article_futures[executor.submit(load_article, url)] = (category, hit)
            for future in as_completed(article_futures):
                category, hit = article_futures[future]
                signal_category = _map_category(category)
                text = ""
                try:
                    text = future.result()
                except Exception as load_exc:  # pragma: no cover - loader failures
                    _LOGGER.debug("[EXPANSION] Article load failed for %s: %s", hit.get("url"), load_exc)
                raw_sources.append(
                    {
                        "category": signal_category,
                        "url": hit.get("url"),
                        "title": hit.get("title"),
                        "score": hit.get("score"),
                        "text": text,
                    }
                )
    except Exception as exc:  # pragma: no cover - defensive catch
        _LOGGER.warning("[EXPANSION] Tavily pipeline failed: %s", exc)

    normalized = normalize_external_signals(raw_sources)
    highbury_context = {
        "overexposure": "37 of 38 listings in Manhattan",
        "segments": ["business travelers", "couples", "tourists"],
    }

    try:
        report = synthesize_expansion_report(normalized, highbury_context) if raw_sources else fallback_expansion_text()
    except Exception as exc:
        _LOGGER.warning("[EXPANSION] Synthesis failed, using fallback: %s", exc)
        report = fallback_expansion_text()

    state["expansion_report"] = report
    state["expansion_sources"] = raw_sources
    bundle = state.setdefault("result_bundle", {})
    bundle.update(
        {
            "summary": report,
            "policy": "EXPANSION_SCOUT",
            "expansion_report": report,
            "expansion_sources": raw_sources,
        }
    )
    telemetry["expansion_latency_s"] = round(time.perf_counter() - start, 2)
    telemetry["expansion_source_count"] = len(raw_sources)
    state["policy"] = "EXPANSION_SCOUT"

    add_thinking_step(
        state,
        phase="expansion_scout",
        title="Expansion scout completed",
        detail=f"Collected {len(raw_sources)} sources.",
        meta={"source_count": len(raw_sources)},
    )
    return state


__all__ = [
    "DEFAULT_EXPANSION_TEMPLATES",
    "exec_expansion_scout",
    "load_article",
    "normalize_external_signals",
    "synthesize_expansion_report",
    "tavily_search",
    "generate_dynamic_queries",
]
