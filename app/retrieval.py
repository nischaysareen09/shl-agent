"""
Step 2: Retrieval layer.

Primary strategy: local sentence-transformer embeddings (all-MiniLM-L6-v2) + cosine
similarity. Runs entirely offline/local at inference time -> no API cost, no rate
limits, good enough semantic matching for a ~400-item catalog.

Fallback: TF-IDF + cosine similarity, used automatically if sentence-transformers
isn't installed (e.g. while iterating in a restricted sandbox). Swap happens
transparently -- the rest of the app only calls Retriever.search(query, k).

Each catalog item is embedded from a composed text: name + description + categories
+ job_levels, so retrieval can match on role language ("senior", "graduate"),
skill language ("Rust", "AWS"), and behavioural language ("safety", "leadership").
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

if not _HAS_ST:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity


def _compose_text(item: Dict[str, Any]) -> str:
    parts = [
        item["name"],
        item.get("description", ""),
        " ".join(item.get("categories", [])),
        " ".join(item.get("job_levels", [])),
    ]
    return " . ".join(p for p in parts if p)


class Retriever:
    def __init__(self, catalog_path: Path = CATALOG_PATH):
        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog: List[Dict[str, Any]] = json.load(f)
        self.texts = [_compose_text(item) for item in self.catalog]
        self.by_url = {item["url"]: item for item in self.catalog}
        self.by_name_lower = {item["name"].lower(): item for item in self.catalog}

        if _HAS_ST:
            self.mode = "embeddings"
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.doc_vecs = self.model.encode(
                self.texts, normalize_embeddings=True, show_progress_bar=False
            )
        else:
            self.mode = "tfidf"
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.doc_matrix = self.vectorizer.fit_transform(self.texts)

    def search(
        self,
        query: str,
        k: int = 10,
        test_type_filter: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k catalog items ranked by relevance to `query`.
        Optionally restrict to items whose test_type intersects test_type_filter.
        """
        if not query.strip():
            return []

        candidate_idx = list(range(len(self.catalog)))
        if test_type_filter:
            tf = set(test_type_filter)
            candidate_idx = [
                i for i in candidate_idx
                if tf.intersection(self.catalog[i]["test_type"])
            ]
        if not candidate_idx:
            return []

        if self.mode == "embeddings":
            q_vec = self.model.encode([query], normalize_embeddings=True)[0]
            sims = self.doc_vecs[candidate_idx] @ q_vec
        else:
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.doc_matrix[candidate_idx]).flatten()

        order = np.argsort(-sims)
        ranked = [(candidate_idx[i], float(sims[i])) for i in order]
        results = []
        for idx, score in ranked[:k]:
            item = dict(self.catalog[idx])
            item["_score"] = score
            results.append(item)
        return results

    # Fixed category-anchor queries. Job descriptions rarely say "personality" or
    # "cognitive ability" explicitly, but batteries almost always need a
    # representative from each of these dimensions (OPQ32r, Verify G+, etc. are
    # exactly this kind of "default add-on" item). Pure semantic similarity to the
    # user's query text under-ranks them because they share little vocabulary with
    # a job description. Multi-query retrieval fixes this by always sampling a few
    # top items from each anchor, independent of how the main query is worded.
    _CATEGORY_ANCHORS = [
        "personality and workplace behaviour assessment",
        "cognitive ability aptitude reasoning test",
        "situational judgment biodata scenario test",
    ]

    def search_diverse(
        self,
        query: str,
        k: int = 25,
        per_anchor: int = 4,
        test_type_filter: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Main-query search merged with a few fixed category-anchor searches,
        deduped by URL, main-query results ranked first. This is what the agent
        should use for building its candidate pool (see agent.py) -- plain
        `search()` remains available for anything that wants pure relevance only
        (e.g. compare-by-name lookups)."""
        primary = self.search(query, k=k, test_type_filter=test_type_filter)
        seen = {item["url"] for item in primary}
        merged = list(primary)

        for anchor in self._CATEGORY_ANCHORS:
            for item in self.search(anchor, k=per_anchor, test_type_filter=test_type_filter):
                if item["url"] not in seen:
                    seen.add(item["url"])
                    merged.append(item)

        return merged

    def get_by_name(self, name: str) -> Dict[str, Any] | None:
        return self.by_name_lower.get(name.lower())

    def fuzzy_find(self, mention: str) -> Dict[str, Any] | None:
        """Best-effort lookup when the user names an assessment loosely
        (e.g. 'OPQ' instead of 'Occupational Personality Questionnaire OPQ32r')."""
        mention_norm = re.sub(r"[^a-z0-9 ]", "", mention.lower()).strip()
        if not mention_norm:
            return None
        best, best_score = None, 0.0
        for item in self.catalog:
            name_norm = re.sub(r"[^a-z0-9 ]", "", item["name"].lower())
            if mention_norm in name_norm or name_norm in mention_norm:
                score = len(mention_norm) / max(len(name_norm), 1)
                if score > best_score:
                    best, best_score = item, score
            else:
                mention_tokens = set(mention_norm.split())
                name_tokens = set(name_norm.split())
                overlap = len(mention_tokens & name_tokens)
                if overlap and overlap / max(len(mention_tokens), 1) > best_score:
                    best, best_score = item, overlap / max(len(mention_tokens), 1)
        return best if best_score >= 0.4 else None


if __name__ == "__main__":
    r = Retriever()
    print(f"Retriever mode: {r.mode}, catalog size: {len(r.catalog)}")
    for q in ["senior rust engineer high performance networking", "safety plant operator dependability", "excel word admin assistant"]:
        print(f"\nQuery: {q}")
        for item in r.search(q, k=5):
            print(f"  {item['_score']:.3f}  {item['name']} [{','.join(item['test_type'])}]")