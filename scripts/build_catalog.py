"""
Step 1: Normalize the raw scraped SHL catalog into a clean structured file
that the retrieval layer and agent can consume directly.

- Fixes the raw JSON (contains a literal newline inside one string -> use strict=False)
- Maps full category names ("Personality & Behavior") to the short codes used
  throughout SHL's own UI and in the provided conversation traces (P, K, A, B, S, C, D, E)
- Drops noisy raw_* duplicate fields, keeps what retrieval/display need
- Assigns a stable integer id for retrieval indexing
"""
import json
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent.parent / "shl_product_catalog.json"
# fallback if run from a place where that relative path doesn't resolve
if not RAW_PATH.exists():
    RAW_PATH = Path("/mnt/user-data/uploads/shl_product_catalog.json")

OUT_PATH = Path(__file__).parent.parent / "data" / "catalog.json"

CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def load_raw():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        return json.load(f, strict=False)


def clean_entry(idx, raw):
    keys = raw.get("keys", []) or []
    test_type_codes = sorted({CATEGORY_TO_CODE.get(k, k[:1]) for k in keys})
    return {
        "id": idx,
        "entity_id": raw.get("entity_id"),
        "name": raw.get("name", "").strip(),
        "url": raw.get("link", "").strip(),
        "description": (raw.get("description") or "").strip(),
        "test_type": test_type_codes,          # e.g. ["K"], ["P"], ["K","S"]
        "categories": keys,                     # full names, for display/compare
        "job_levels": raw.get("job_levels", []) or [],
        "languages": raw.get("languages", []) or [],
        "duration": (raw.get("duration") or "").strip(),
        "remote": raw.get("remote", ""),
        "adaptive": raw.get("adaptive", ""),
    }


def main():
    raw_list = load_raw()
    cleaned = []
    seen_urls = set()
    for i, raw in enumerate(raw_list):
        entry = clean_entry(i, raw)
        if not entry["name"] or not entry["url"]:
            continue
        if entry["url"] in seen_urls:
            continue
        seen_urls.add(entry["url"])
        cleaned.append(entry)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Loaded {len(raw_list)} raw entries -> wrote {len(cleaned)} cleaned entries to {OUT_PATH}")

    # quick sanity report
    from collections import Counter
    type_counts = Counter(t for e in cleaned for t in e["test_type"])
    print("Test type distribution:", dict(type_counts))
    no_duration = sum(1 for e in cleaned if not e["duration"])
    print(f"Entries with no duration listed: {no_duration}/{len(cleaned)}")


if __name__ == "__main__":
    main()
