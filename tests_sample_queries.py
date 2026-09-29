"""Runs benchmark queries against the running API.

Start the server first:  uvicorn app:app
Then:                    python tests_sample_queries.py
"""
import sys
import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/chat"

QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "What is the role of planning in AI agents?",
    "Who won the 2022 FIFA World Cup?",  # out-of-scope: must refuse
]

for i, q in enumerate(QUERIES, 1):
    r = requests.post(URL, json={"query": q}, timeout=60)
    r.raise_for_status()
    data = r.json()
    print(f"\n{'=' * 70}\nQ{i}: {q}")
    print(f"Answer    : {data['answer']}")
    print(f"Confidence: {data['confidence_score']}")
    print(f"Chunks    : {len(data['retrieved_chunks'])} retrieved")
    if data["retrieved_chunks"]:
        print(f"Top chunk : {data['retrieved_chunks'][0][:200].strip()}...")
