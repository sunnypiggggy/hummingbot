from __future__ import annotations

import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


class BotStore:
    def __init__(self, path: Path, session_ttl_seconds: int = 900):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.session_ttl_seconds = session_ttl_seconds
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS processed_updates(
              update_id INTEGER PRIMARY KEY,processed_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions(
              session_id TEXT PRIMARY KEY,user_id INTEGER NOT NULL,chat_id INTEGER NOT NULL,
              flow TEXT NOT NULL,step TEXT NOT NULL,payload TEXT NOT NULL,message_id INTEGER,
              expires_at REAL NOT NULL,updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS action_results(
              idempotency_key TEXT PRIMARY KEY,status TEXT NOT NULL,result TEXT NOT NULL,
              created_at REAL NOT NULL,updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_events(
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,event_type TEXT NOT NULL,user_id INTEGER,
              chat_id INTEGER,update_id INTEGER,callback_id TEXT,details TEXT NOT NULL,
              created_at REAL NOT NULL
            );
            """
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def metadata(self, key: str, default: str = "") -> str:
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def set_metadata(self, key: str, value: Any) -> None:
        self.db.execute(
            "INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )
        self.db.commit()

    def claim_update(self, update_id: int) -> bool:
        try:
            self.db.execute(
                "INSERT INTO processed_updates(update_id,processed_at) VALUES(?,?)",
                (int(update_id), time.time()),
            )
            self.db.execute(
                "DELETE FROM processed_updates WHERE processed_at < ?",
                (time.time() - 7 * 86400,),
            )
            self.db.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def create_session(self, user_id: int, chat_id: int, flow: str, message_id: Optional[int] = None) -> dict:
        session_id = secrets.token_urlsafe(7)[:10]
        now = time.time()
        payload: dict[str, Any] = {}
        self.db.execute(
            "INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?,?)",
            (session_id, user_id, chat_id, flow, "start", json.dumps(payload), message_id,
             now + self.session_ttl_seconds, now),
        )
        self.db.commit()
        return self.get_session(session_id) or {}

    def get_session(self, session_id: str) -> Optional[dict]:
        row = self.db.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not row:
            return None
        if float(row["expires_at"]) < time.time():
            self.delete_session(session_id)
            return None
        value = dict(row)
        value["payload"] = json.loads(value["payload"])
        return value

    def active_session(self, user_id: int, chat_id: int) -> Optional[dict]:
        row = self.db.execute(
            "SELECT session_id FROM sessions WHERE user_id=? AND chat_id=? AND expires_at>=? "
            "ORDER BY updated_at DESC LIMIT 1",
            (user_id, chat_id, time.time()),
        ).fetchone()
        return self.get_session(str(row["session_id"])) if row else None

    def update_session(self, session_id: str, *, step: Optional[str] = None,
                       payload: Optional[dict] = None, message_id: Optional[int] = None) -> dict:
        current = self.get_session(session_id)
        if current is None:
            raise KeyError("session expired")
        merged = dict(current["payload"])
        if payload:
            merged.update(payload)
        now = time.time()
        self.db.execute(
            "UPDATE sessions SET step=?,payload=?,message_id=COALESCE(?,message_id),expires_at=?,updated_at=? "
            "WHERE session_id=?",
            (step or current["step"], json.dumps(merged, ensure_ascii=False), message_id,
             now + self.session_ttl_seconds, now, session_id),
        )
        self.db.commit()
        return self.get_session(session_id) or {}

    def delete_session(self, session_id: str) -> None:
        self.db.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        self.db.commit()

    def clear_sessions(self, user_id: int, chat_id: int) -> None:
        self.db.execute("DELETE FROM sessions WHERE user_id=? AND chat_id=?", (user_id, chat_id))
        self.db.commit()

    def claim_action(self, key: str) -> tuple[bool, Optional[dict]]:
        row = self.db.execute("SELECT status,result FROM action_results WHERE idempotency_key=?", (key,)).fetchone()
        if row:
            return False, {"status": row["status"], "result": json.loads(row["result"])}
        now = time.time()
        self.db.execute(
            "INSERT INTO action_results VALUES(?,?,?,?,?)",
            (key, "RUNNING", "{}", now, now),
        )
        self.db.commit()
        return True, None

    def finish_action(self, key: str, status: str, result: dict) -> None:
        self.db.execute(
            "UPDATE action_results SET status=?,result=?,updated_at=? WHERE idempotency_key=?",
            (status, json.dumps(result, ensure_ascii=False, default=str), time.time(), key),
        )
        self.db.commit()

    def audit(self, event_type: str, *, user_id: Optional[int] = None, chat_id: Optional[int] = None,
              update_id: Optional[int] = None, callback_id: str = "", details: Optional[dict] = None) -> None:
        self.db.execute(
            "INSERT INTO audit_events(event_type,user_id,chat_id,update_id,callback_id,details,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (event_type, user_id, chat_id, update_id, callback_id,
             json.dumps(details or {}, ensure_ascii=False, default=str), time.time()),
        )
        self.db.commit()
