"""SQLite cache for LLM calls. Key = sha256 of the full request + config hash,
so identical reruns are free and any config change invalidates cleanly."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path


class LLMCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=60)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS calls ("
            "key TEXT PRIMARY KEY, model TEXT, response TEXT, usage TEXT, created REAL)")
        self._conn.commit()

    @staticmethod
    def make_key(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT response, usage FROM calls WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return {"text": row[0], "usage": json.loads(row[1]), "cached": True}

    def put(self, key: str, model: str, text: str, usage: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?, ?)",
            (key, model, text, json.dumps(usage), time.time()))
        self._conn.commit()

    def stats(self) -> dict:
        n, = self._conn.execute("SELECT COUNT(*) FROM calls").fetchone()
        return {"cached_calls": n, "path": str(self.path)}
