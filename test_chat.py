"""Headless smoke test of the full chatbot stack (no UI).

Requires OPENROUTER_API_KEY in Chatbot/.env (or the environment).
Usage: python test_chat.py
Exit code = number of failed cases.
"""
from __future__ import annotations

import bootstrap  # noqa: F401
from engine import ChatEngine

CASES = [
    ("What is the Job Zone of Registered Nurses?", []),
    ("Which tasks of Registered Nurses involve administering medication?", []),
    # forced-seed path: the question itself names no entity
    ("What are the most important transferable skills for this occupation?",
     ["29-1141.00"]),
]


def main() -> int:
    print("Loading engine (graph + linker)...")
    eng = ChatEngine()
    print(f"Graph: {eng.G.number_of_nodes()} nodes / {eng.G.number_of_edges()} edges")
    print(f"Backend: {eng.backend_description}\n")
    if not eng.is_configured:
        print("ERROR: set OPENROUTER_API_KEY in Chatbot/.env first.")
        return 1

    failures = 0
    for question, forced in CASES:
        print(f"Q: {question}" + (f"  [forced: {forced}]" if forced else ""))
        ans = eng.ask(question, forced_seeds=forced)
        ok = (ans.parsed is not None
              and {"answer", "citations", "reasoning"} <= set(ans.parsed)
              and bool(ans.seeds)
              and bool(ans.rec["context_text"]))
        status = "OK" if ok else "FAIL"
        print(f"[{status}] seeds={ans.seeds}")
        if ans.parsed:
            print(f"  answer   = {ans.parsed['answer']}")
            print(f"  citations= {ans.parsed['citations']}")
            print(f"  reasoning= {ans.parsed['reasoning']}")
        else:
            print(f"  RAW: {ans.raw[:300]}")
        u = ans.usage
        print(f"  context {ans.context_tokens} tok | in {u.get('input_tokens')} "
              f"/ out {u.get('output_tokens')} tok | {u.get('latency_s', 0):.1f}s\n")
        failures += (not ok)
    print(f"{len(CASES) - failures}/{len(CASES)} cases passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
