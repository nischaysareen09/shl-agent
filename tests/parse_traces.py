"""
Step 6a: Parse the provided C1..C10 trace files into structured conversations
we can replay against our own /chat logic (in-process, no server needed) to
sanity-check Recall@10 and behavior before deploying.

Each trace file alternates **User** / **Agent** turns. We extract:
  - the sequence of user messages (these become our simulated conversation)
  - the FINAL agent turn's markdown table -> the "expected" shortlist (name+url)
This mirrors the assignment's own definition: "each trace is a persona with a
fact set and a labeled expected shortlist."
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Any

TRACES_DIR = Path(__file__).parent / "traces"

USER_BLOCK_RE = re.compile(r"\*\*User\*\*\s*\n\s*>\s*(.+?)(?=\n###|\n\*\*Agent\*\*|\Z)", re.DOTALL)
TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*(.+?)\s*\|.*\|\s*<?(https?://\S+?)>?\s*\|$", re.MULTILINE)


def _extract_user_messages(text: str) -> List[str]:
    msgs = []
    for m in USER_BLOCK_RE.finditer(text):
        raw = m.group(1).strip()
        # user quotes can span multiple '>' lines; strip leading '>' per line
        lines = [re.sub(r"^>\s?", "", ln).strip() for ln in raw.splitlines()]
        msgs.append(" ".join(l for l in lines if l).strip())
    return msgs


def _expected_shortlist(text: str) -> List[Dict[str, str]]:
    """Take the LAST markdown table in the file as the expected/final shortlist."""
    all_rows = TABLE_ROW_RE.findall(text)
    if not all_rows:
        return []
    # dedupe preserving order, keep only rows from the final table block:
    # find the last '| # | Name |' header and take rows after it
    header_positions = [m.start() for m in re.finditer(r"\|\s*#\s*\|\s*Name\s*\|", text)]
    if not header_positions:
        rows = all_rows
    else:
        last_header = header_positions[-1]
        rows = TABLE_ROW_RE.findall(text[last_header:])
    seen = set()
    out = []
    for name, url in rows:
        if url not in seen:
            seen.add(url)
            out.append({"name": name.strip(), "url": url.strip()})
    return out


def load_traces() -> List[Dict[str, Any]]:
    traces = []
    for path in sorted(TRACES_DIR.glob("C*.md"), key=lambda p: (len(p.stem), p.stem)):
        text = path.read_text(encoding="utf-8")
        user_msgs = _extract_user_messages(text)
        expected = _expected_shortlist(text)
        traces.append({
            "id": path.stem,
            "user_messages": user_msgs,
            "expected_shortlist": expected,
        })
    return traces


if __name__ == "__main__":
    for t in load_traces():
        print(f"{t['id']}: {len(t['user_messages'])} user turns, {len(t['expected_shortlist'])} expected items")
        for m in t["user_messages"]:
            print("   >", m[:90])
        for e in t["expected_shortlist"]:
            print("   expect:", e["name"])
        print()
