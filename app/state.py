"""
Step 3a: Conversation state reconstruction.

The API is stateless -- every /chat call gets the full message history and
nothing else. Our own previous replies (which we always render WITH a markdown
table when we recommend, same format as the provided traces) come back to us
verbatim as assistant messages in that history. So "state" -- the current
shortlist, whether we already asked a clarifying question, etc. -- is recovered
by parsing that history, not stored anywhere.
"""
from __future__ import annotations
import re
from typing import List, Dict, Any, Optional

TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*(.+?)\s*\|.*\|\s*(https?://\S+?)\s*\|$", re.MULTILINE)


def extract_shortlist_from_text(text: str) -> List[Dict[str, str]]:
    """Pull (name, url) pairs out of a markdown recommendation table, if present."""
    rows = TABLE_ROW_RE.findall(text or "")
    return [{"name": name.strip(), "url": url.strip()} for name, url in rows]


def parse_conversation(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    user_turns = [m["content"] for m in messages if m.get("role") == "user"]
    assistant_turns = [m["content"] for m in messages if m.get("role") == "assistant"]

    prior_shortlist: List[Dict[str, str]] = []
    for a in reversed(assistant_turns):
        found = extract_shortlist_from_text(a)
        if found:
            prior_shortlist = found
            break

    combined_user_context = "\n".join(user_turns)

    return {
        "turn_count": len(messages),
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "latest_user_message": user_turns[-1] if user_turns else "",
        "combined_user_context": combined_user_context,
        "prior_shortlist": prior_shortlist,
        "has_prior_recommendation": bool(prior_shortlist),
    }
