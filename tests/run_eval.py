"""
Step 6b: Eval harness.

Replays each C1..C10 trace's user turns through our OWN /chat logic (in-process,
calling Agent.handle_turn directly -- no server needed) and computes Recall@10
against the labeled expected shortlist, per the assignment's own formula:

    Recall@K = (# relevant items in top K) / (total relevant items for the query)

We build up `messages` turn by turn exactly like the real evaluator will: send
user message N, get the agent's reply, append BOTH to history, send user message
N+1, etc. We stop feeding turns once the agent produces a non-null shortlist,
matching "the simulated user ... ends the conversation when the agent provides
a shortlist" -- but we also run all remaining turns for traces where the human
transcript kept refining (matches "refine" behavior).

Requires a real GROQ_API_KEY in the environment (loads .env). Run with:
    python -m tests.run_eval
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import app.env  # noqa: F401 -- loads .env
from app.agent import Agent
from tests.parse_traces import load_traces


def recall_at_10(expected: list[dict], predicted: list[dict] | None) -> float:
    if not expected:
        return 1.0 if not predicted else 1.0  # nothing to find
    if not predicted:
        return 0.0
    expected_urls = {e["url"] for e in expected}
    predicted_urls = {p["url"] for p in predicted[:10]}
    hit = len(expected_urls & predicted_urls)
    return hit / len(expected_urls)


def run_trace(agent: Agent, trace: dict, verbose: bool = True) -> dict:
    messages = []
    final_recs = None
    for i, user_msg in enumerate(trace["user_messages"]):
        messages.append({"role": "user", "content": user_msg})
        result = agent.handle_turn(messages)
        messages.append({"role": "assistant", "content": result["reply"]})
        if result.get("recommendations"):
            final_recs = result["recommendations"]
        if verbose:
            action_hint = "RECOMMEND" if result.get("recommendations") else "ask/refuse"
            print(f"    turn {i+1} [{action_hint}]: {result['reply'][:100].splitlines()[0]}")
        if result.get("end_of_conversation"):
            break

    score = recall_at_10(trace["expected_shortlist"], final_recs)
    return {
        "id": trace["id"],
        "recall_at_10": score,
        "expected_count": len(trace["expected_shortlist"]),
        "predicted_count": len(final_recs) if final_recs else 0,
        "predicted": final_recs,
    }


def main():
    agent = Agent()
    traces = load_traces()
    results = []
    for t in traces:
        print(f"\n=== {t['id']} ===")
        r = run_trace(agent, t)
        results.append(r)
        print(f"  Recall@10: {r['recall_at_10']:.2f}  (expected {r['expected_count']}, got {r['predicted_count']})")

    mean_recall = sum(r["recall_at_10"] for r in results) / len(results)
    print(f"\n{'='*40}\nMean Recall@10 across {len(results)} traces: {mean_recall:.3f}")

    Path("tests/eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Full results written to tests/eval_results.json")


if __name__ == "__main__":
    main()