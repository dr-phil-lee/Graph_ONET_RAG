"""Live smoke test of the full chatbot stack against OpenRouter (no UI).

Requires OPENROUTER_API_KEY in Chatbot/.env (or the environment).
Usage: python test_chat.py [model-id]
Exit code = number of failed cases.

For the offline test suite (no key, fake LLM): python -m pytest -q
"""
from __future__ import annotations

import sys

import bootstrap  # noqa: F401
from engine import ChatEngine

RN = "29-1141.00"

# (question, pinned occupations, expected substring in evidence of a correct answer)
CASES = [
    ("What is the Job Zone of Registered Nurses?", [], "4"),
    ("Which tasks of Registered Nurses involve administering medication?", [], "medication"),
    # forced-seed path: the question itself names no entity
    ("What are the most important transferable skills for this occupation?", [RN], None),
    # two-job bridge path
    ("What work activities do Registered Nurses and Pharmacists share?", [], None),
    # comparison incl. an occupation without O*NET descriptor ratings (graph_ext)
    ("Compare Software Developers and Data Scientists.", [], "Data Scientists"),
    # career-change intent: Related Occupations must be used, not "Insufficient evidence"
    ("I am a tele marketer and want to change to another job. What could I do?",
     [], "Customer Service Representatives"),
    # no occupation title in the wording: LLM-assisted linking
    ("I sell things over the phone and want a different job. What could I do?", [], None),
]


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else None
    print("Loading engine (graph + linker)...")
    eng = ChatEngine()
    print(f"Graph: {eng.G.number_of_nodes()} nodes / {eng.G.number_of_edges()} edges")
    if not eng.is_configured:
        print("ERROR: set OPENROUTER_API_KEY in Chatbot/.env first.")
        return 1
    print(f"Backend: OpenRouter: {model or eng.default_model}\n")

    failures = 0
    for question, pinned, expect in CASES:
        print(f"Q: {question}" + (f"  [pinned: {pinned}]" if pinned else ""))
        try:
            ans = eng.ask(question, forced_seeds=pinned, model=model)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {type(e).__name__}: {e}\n")
            failures += 1
            continue
        answer_text = str(ans.parsed.get("answer")) if ans.parsed else ""
        ok = (ans.status == "grounded"
              and bool(ans.seeds)
              and (expect is None or expect.lower() in answer_text.lower()))
        status = "OK" if ok else "FAIL"
        print(f"[{status}] status={ans.status} seeds={ans.seeds} model={ans.model}")
        if ans.parsed:
            print(f"  answer   = {ans.parsed.get('answer')}")
            print(f"  citations= {ans.citations}"
                  + (f"  INVALID={ans.invalid_citations}" if ans.invalid_citations else ""))
            print(f"  reasoning= {ans.parsed.get('reasoning')}")
        else:
            print(f"  RAW: {ans.raw[:300]}")
        u = ans.usage
        print(f"  context {ans.context_tokens} tok | in {u.get('input_tokens')} "
              f"/ out {u.get('output_tokens')} tok | {u.get('latency_s', 0):.1f}s\n")
        failures += (not ok)
    total = len(CASES)

    # structured answer + AI exposure + conversation memory (two turns)
    total += 1
    print("Multi-turn: telemarketer -> follow-up about AI exposure")
    try:
        a1 = eng.ask("I was a tele marketer, but i got fired due to a new AI system. "
                     "Could you please recommend other jobs that I can use my skills?",
                     model=model)
        a2 = eng.ask("Which of those jobs is the least exposed to AI?", model=model,
                     history=[a1.memory_turn()])
        print(f"  Q1 headline : {a1.headline}")
        print(f"  Q1 cards    : {[(c['title'], (c['ai_exposure'] or {}).get('composite')) for c in a1.occupations]}")
        print(f"  Q2 rewritten: {a2.standalone_question}")
        print(f"  Q2 seeds    : {a2.seeds}")
        print(f"  Q2 headline : {a2.headline}")
        ok = (a1.status == "grounded" and len(a1.occupations) >= 2
              and any(c.startswith("X-") for c in a1.citations)
              and a2.status == "grounded" and a2.history_turns == 1
              and any(c.startswith("X-") for c in a2.citations))
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {type(e).__name__}: {e}")
        ok = False
    print(f"[{'OK' if ok else 'FAIL'}] multi-turn memory + AI exposure\n")
    failures += (not ok)

    print(f"{total - failures}/{total} cases passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
