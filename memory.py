"""Conversation memory for multi-turn chat.

A turn is what later questions need to understand references such as
"those jobs" or "the second one":

    {"question": str,            # what the user asked
     "summary": str,             # the answer's headline (or its first ~400 chars)
     "occupations": [str],       # O*NET titles in focus (seeds + recommended)
     "seeds": [str],             # their O*NET-SOC codes
     "cited": [str]}             # evidence ids the answer cited

Memory is context, never evidence: ChatEngine uses it to rewrite a
follow-up into a standalone question and to carry the occupations in focus
forward, and the generator sees it marked "NOT evidence, never cite it".

ConversationStore keeps turns per session id for the REST API (the Streamlit
UI keeps them in st.session_state). It is in-memory and thread-safe, bounded
by an LRU on sessions, a cap on turns per session and an idle TTL. Set
CHATBOT_SESSION_DIR to also persist sessions as JSON files so they survive a
server restart.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

MAX_TURNS = 20
_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def turn_from_answer(ans) -> dict:
    """Memory record for one engine Answer."""
    summary = ans.headline or ""
    if not summary and ans.parsed:
        summary = str(ans.parsed.get("answer") or "")[:400]
    occ, socs = [], []
    for e in ans.entities:
        if e.label == "Job" and e.node_id not in socs:
            socs.append(e.node_id)
            occ.append(e.name)
    for c in ans.occupations:
        if c["soc"] not in socs:
            socs.append(c["soc"])
            occ.append(c["title"])
    return {"question": ans.question, "summary": summary, "occupations": occ,
            "seeds": socs, "cited": list(ans.citations)}


def normalize_turns(turns) -> list[dict]:
    """Accept caller-supplied history (API) defensively."""
    out = []
    for t in turns or []:
        if not isinstance(t, dict) or not str(t.get("question") or "").strip():
            continue
        out.append({"question": str(t["question"])[:2000],
                    "summary": str(t.get("summary") or "")[:2000],
                    "occupations": [str(x) for x in (t.get("occupations") or [])][:12],
                    "seeds": [str(x) for x in (t.get("seeds") or [])][:12],
                    "cited": [str(x) for x in (t.get("cited") or [])][:60]})
    return out[-MAX_TURNS:]


class ConversationStore:
    def __init__(self, max_sessions: int = 1000, max_turns: int = MAX_TURNS,
                 ttl_s: float = 24 * 3600, persist_dir: str | os.PathLike | None = None):
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self.ttl_s = ttl_s
        self._d: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        pd = persist_dir if persist_dir is not None else os.getenv("CHATBOT_SESSION_DIR", "")
        self.persist_dir = Path(pd) if pd else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def valid_id(session_id: str | None) -> bool:
        return bool(session_id) and bool(_SAFE_ID.match(session_id))

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def _expire(self, now: float) -> None:
        for sid in [s for s, v in self._d.items() if now - v["updated"] > self.ttl_s]:
            self._d.pop(sid, None)

    def _load(self, sid: str) -> dict | None:
        if not self.persist_dir:
            return None
        p = self.persist_dir / f"{sid}.json"
        try:
            v = json.loads(p.read_text(encoding="utf-8"))
            return {"turns": normalize_turns(v.get("turns")),
                    "created": float(v.get("created", time.time())),
                    "updated": float(v.get("updated", time.time()))}
        except (OSError, ValueError, AttributeError):
            return None

    def _save(self, sid: str, v: dict) -> None:
        if self.persist_dir:
            (self.persist_dir / f"{sid}.json").write_text(
                json.dumps(v, ensure_ascii=False), encoding="utf-8")

    def get(self, session_id: str) -> list[dict]:
        if not self.valid_id(session_id):
            return []
        now = time.time()
        with self._lock:
            self._expire(now)
            v = self._d.get(session_id) or self._load(session_id)
            if v is None:
                return []
            self._d[session_id] = v
            self._d.move_to_end(session_id)
            return [dict(t) for t in v["turns"]]

    def exists(self, session_id: str) -> bool:
        if not self.valid_id(session_id):
            return False
        with self._lock:
            return session_id in self._d or bool(
                self.persist_dir and (self.persist_dir / f"{session_id}.json").exists())

    def append(self, session_id: str, turn: dict) -> int:
        if not self.valid_id(session_id):
            raise ValueError("invalid session id")
        now = time.time()
        with self._lock:
            self._expire(now)
            v = self._d.get(session_id) or self._load(session_id) or {
                "turns": [], "created": now, "updated": now}
            v["turns"] = (v["turns"] + normalize_turns([turn]))[-self.max_turns:]
            v["updated"] = now
            self._d[session_id] = v
            self._d.move_to_end(session_id)
            while len(self._d) > self.max_sessions:
                self._d.popitem(last=False)
            self._save(session_id, v)
            return len(v["turns"])

    def delete(self, session_id: str) -> bool:
        if not self.valid_id(session_id):
            return False
        with self._lock:
            found = self._d.pop(session_id, None) is not None
            if self.persist_dir:
                p = self.persist_dir / f"{session_id}.json"
                if p.exists():
                    p.unlink()
                    found = True
            return found
