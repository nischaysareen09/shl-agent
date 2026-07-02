"""
Diagnostic: isolates retrieval quality from LLM selection quality, without
spending any Groq tokens. For each trace, retrieves top-K candidates using the
FULL combined user context (all turns concatenated) and checks whether each
expected item's URL appears anywhere in that candidate pool.

If an expected item is MISSING here, no LLM could ever have recommended it --
that's a retrieval bug (fix: better query composition, larger K, or better
embeddings). If it's PRESENT but the eval still scored low, that's the LLM
picking the wrong subset -- a model-capability or prompt-tuning issue, not
retrieval.

Run with: python -m tests.check_retrieval_coverage
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval import Retriever
from tests.parse_traces import load_traces


def main():
    retriever = Retriever()
    print(f"Retriever mode: {retriever.mode}\n")

    total_expected = 0
    total_covered = 0

    for trace in load_traces():
        combined_context = "\n".join(trace["user_messages"])
        candidates = retriever.search_diverse(combined_context, k=25)
        candidate_urls = {c["url"] for c in candidates}

        expected = trace["expected_shortlist"]
        missing = [e for e in expected if e["url"] not in candidate_urls]
        covered = len(expected) - len(missing)
        total_expected += len(expected)
        total_covered += covered

        status = "OK" if not missing else "GAP"
        print(f"[{status}] {trace['id']}: {covered}/{len(expected)} expected items in top-25 candidates")
        for m in missing:
            print(f"    MISSING: {m['name']}")

    print(f"\nOverall coverage: {total_covered}/{total_expected} "
          f"({100*total_covered/total_expected:.0f}%) of expected items reachable by retrieval")


if __name__ == "__main__":
    main()