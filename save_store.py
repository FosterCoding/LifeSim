"""
save_store.py - isolated SQLite persistence for LifeSim playthroughs.

Nothing outside this file touches SQL directly. app.py calls these functions
only. This keeps Player.py / Narrator.py / Dice.py completely untouched by
the persistence layer.
"""
import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lifesim_saves.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Creates the saves table if it doesn't exist. Safe to call on every app startup."""
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS saves (
                save_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                location TEXT,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_playthrough(save_id: str, player_data: Dict[str, Any]) -> None:
    """
    Insert or overwrite a playthrough. save_id is stable for the life of one
    character (created once at new-game time, reused on every auto-save), so
    repeated calls UPDATE the same row rather than growing the table.

    Guards against a stale write clobbering a newer one using turn_number
    (see Player.turn_number): if the row already on disk has a HIGHER
    turn_number than the data being saved now, this save is silently
    skipped rather than applied. This closes a real race between the main
    request's synchronous save and the background mechanical-extraction
    thread's own follow-up save (app.py's _apply_extraction_async) - if the
    player submits their next action quickly enough, that next turn's save
    could land first, and the still-in-flight background save from the
    PREVIOUS turn could then overwrite it with older data (an earlier
    date, missing state) purely due to thread/request scheduling, not
    because it was actually the more current state. Comparing turn_number
    instead of wall-clock time is deliberate - clock skew or scheduling
    jitter can't fool an integer that only every increments by exactly 1
    per real turn.
    """
    incoming_turn = player_data.get("turn_number", 0)
    now = datetime.now(timezone.utc).isoformat()
    blob = json.dumps(player_data)
    conn = _get_conn()
    try:
        existing = conn.execute(
            "SELECT created_at, data FROM saves WHERE save_id = ?", (save_id,)
        ).fetchone()
        if existing:
            try:
                existing_turn = json.loads(existing["data"]).get("turn_number", 0)
            except (json.JSONDecodeError, TypeError):
                existing_turn = 0
            if existing_turn > incoming_turn:
                # A newer save already landed; this write is stale, drop it.
                return
        created_at = existing["created_at"] if existing else now
        conn.execute("""
            INSERT INTO saves (save_id, name, age, location, data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(save_id) DO UPDATE SET
                name=excluded.name,
                age=excluded.age,
                location=excluded.location,
                data=excluded.data,
                updated_at=excluded.updated_at
        """, (
            save_id,
            player_data.get("name", "Unknown"),
            player_data.get("age"),
            player_data.get("location", ""),
            blob,
            created_at,
            now,
        ))
        conn.commit()
    finally:
        conn.close()


def load_playthrough(save_id: str) -> Optional[Dict[str, Any]]:
    """Returns the raw player_data dict for a save_id, or None if not found."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT data FROM saves WHERE save_id = ?", (save_id,)).fetchone()
        return json.loads(row["data"]) if row else None
    finally:
        conn.close()


def list_playthroughs() -> List[Dict[str, Any]]:
    """Returns lightweight summaries (no full data blob) of every saved playthrough,
    most recently updated first, for rendering a 'load game' list in the UI."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT save_id, name, age, location, created_at, updated_at
            FROM saves ORDER BY updated_at DESC
        """).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def delete_playthrough(save_id: str) -> bool:
    """Deletes a save. Returns True if a row was actually removed."""
    conn = _get_conn()
    try:
        cur = conn.execute("DELETE FROM saves WHERE save_id = ?", (save_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
