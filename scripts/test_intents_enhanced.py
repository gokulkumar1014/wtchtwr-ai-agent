# scripts/test_intents_enhanced.py

from agent.intents import classify_intent
from agent.types import State

tests = [
    "Which amenities are most common among Highbury listings?",
    "What do guests say about parking?",
    "Compare occupancy between Highbury and market.",
    "How do reviews reflect Highbury’s occupancy performance?",
    "Compare guest feedback between Highbury and market.",
]

for q in tests:
    state = {"query": q, "tenant": "highbury"}
    result = classify_intent(state)
    print(f"\n🧠 Query: {q}")
    print(f"Intent: {result['intent']} | Scope: {result['scope']}")
    print(f"Filters: {result['filters']}")
