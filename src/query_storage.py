"""
Query Storage Module

Stores Stage 2 query results per persona for incremental/resume runs.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class QueryStorage:
    """Persist queries per persona (one JSON file per persona)."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _persona_path(self, persona_id: str) -> Path:
        return self.output_dir / f"{persona_id}.json"

    def _derive_query_id(self, query: Dict[str, Any]) -> str:
        metadata = query.get('metadata', {}) or {}
        for key in ['query_id', 'id', 'source_id', 'uid', 'uuid', 'qid']:
            val = metadata.get(key) or query.get(key)
            if val:
                source = metadata.get('dataset') or metadata.get('source') or query.get('source_dataset') or query.get('source')
                if source:
                    return f"{source}:{val}"
                return str(val)

        text = query.get('original_query') or query.get('adapted_query') or query.get('text') or ""
        source = metadata.get('dataset') or metadata.get('source') or query.get('source_dataset') or query.get('source') or "unknown"
        base = f"{source}:{text}"
        digest = hashlib.sha1(base.encode('utf-8')).hexdigest()[:16]
        return f"hash:{digest}"

    def _normalize_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        normalized = []
        for q in queries:
            if not isinstance(q, dict):
                continue
            qid = q.get('query_id') or self._derive_query_id(q)
            q['query_id'] = qid
            if qid in seen:
                continue
            seen.add(qid)
            normalized.append(q)
        return normalized

    def load_persona(self, persona_id: str) -> List[Dict[str, Any]]:
        path = self._persona_path(persona_id)
        if not path.exists():
            return []

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'queries' in data:
            queries = data.get('queries', [])
        elif isinstance(data, list):
            queries = data
        else:
            queries = []

        normalized = self._normalize_queries(queries)
        # Persist normalized queries if we added missing IDs or removed duplicates
        if len(normalized) != len(queries) or any('query_id' not in q for q in queries):
            self.save_persona(persona_id, normalized)
        return normalized

    def save_persona(self, persona_id: str, queries: List[Dict[str, Any]]):
        normalized = self._normalize_queries(queries)
        payload = {
            'persona_id': persona_id,
            'count': len(normalized),
            'last_updated': datetime.now().isoformat(),
            'queries': normalized
        }
        path = self._persona_path(persona_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def append_persona(self, persona_id: str, new_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        existing = self.load_persona(persona_id)
        combined = existing + (new_queries or [])
        self.save_persona(persona_id, combined)
        return self.load_persona(persona_id)

    def load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        results: Dict[str, List[Dict[str, Any]]] = {}
        for path in self.output_dir.glob("*.json"):
            persona_id = path.stem
            try:
                results[persona_id] = self.load_persona(persona_id)
            except Exception:
                continue
        return results

    def get_all_query_ids(self) -> set:
        ids = set()
        for queries in self.load_all().values():
            for q in queries:
                qid = q.get('query_id')
                if qid:
                    ids.add(qid)
        return ids
