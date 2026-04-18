"""SQLite 持久化 - 会谈历史记录"""
import json
import sqlite3
import os
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "meetings.db")

# 写操作锁：防止 SQLite 并发写入竞态
_db_lock = threading.Lock()


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL 模式：允许并发读写，减少 "database is locked" 错误
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meeting_records (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'ended',
            host_config TEXT NOT NULL,       -- JSON
            guests_config TEXT NOT NULL,     -- JSON
            history TEXT NOT NULL DEFAULT '[]',  -- JSON 消息列表
            summary TEXT,                    -- 总结内容
            agenda TEXT,                     -- JSON 议程
            materials TEXT,                  -- JSON [{name, content}]
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    # 兼容旧表：materials 列不存在时自动添加
    try:
        conn.execute("ALTER TABLE meeting_records ADD COLUMN materials TEXT")
        conn.commit()
    except Exception:
        pass
    # 兼容旧表：host_style 列不存在时自动添加
    try:
        conn.execute("ALTER TABLE meeting_records ADD COLUMN host_style TEXT DEFAULT 'neutral'")
        conn.commit()
    except Exception:
        pass
    # 兼容旧表：embedding_config 列不存在时自动添加
    try:
        conn.execute("ALTER TABLE meeting_records ADD COLUMN embedding_config TEXT")
        conn.commit()
    except Exception:
        pass
    # 兼容旧表：tavily_key 列不存在时自动添加
    try:
        conn.execute("ALTER TABLE meeting_records ADD COLUMN tavily_key TEXT")
        conn.commit()
    except Exception:
        pass
    # 兼容旧表：discussion_title 列不存在时自动添加（话题拆分：标题）
    try:
        conn.execute("ALTER TABLE meeting_records ADD COLUMN discussion_title TEXT")
        conn.commit()
    except Exception:
        pass
    # 兼容旧表：topic_content 列不存在时自动添加（话题拆分：内容）
    try:
        conn.execute("ALTER TABLE meeting_records ADD COLUMN topic_content TEXT")
        conn.commit()
    except Exception:
        pass
    conn.commit()
    conn.close()


def upsert_record(meeting_id: str, data: dict):
    """插入或更新会谈记录"""
    with _db_lock:
        conn = get_conn()
        now = datetime.now().isoformat()
        existing = conn.execute("SELECT id FROM meeting_records WHERE id=?", (meeting_id,)).fetchone()
        materials_json = json.dumps(data.get("materials") or [], ensure_ascii=False)
        if existing:
            conn.execute("""
                UPDATE meeting_records
                SET topic=?, state=?, host_config=?, guests_config=?,
                    history=?, summary=?, agenda=?, materials=?, host_style=?,
                    embedding_config=?, tavily_key=?, discussion_title=?, topic_content=?, updated_at=?
                WHERE id=?
            """, (
                data["topic"], data["state"],
                json.dumps(data["host_config"], ensure_ascii=False),
                json.dumps(data["guests_config"], ensure_ascii=False),
                json.dumps(data["history"], ensure_ascii=False),
                data.get("summary"),
                json.dumps(data.get("agenda"), ensure_ascii=False) if data.get("agenda") else None,
                materials_json,
                data.get("host_style", "neutral"),
                json.dumps(data.get("embedding_config"), ensure_ascii=False) if data.get("embedding_config") else None,
                data.get("tavily_key"),
                data.get("discussion_title"),
                data.get("topic_content"),
                now, meeting_id
            ))
        else:
            conn.execute("""
                INSERT INTO meeting_records
                    (id, topic, state, host_config, guests_config, history, summary, agenda, materials, host_style, embedding_config, tavily_key, discussion_title, topic_content, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                meeting_id, data["topic"], data["state"],
                json.dumps(data["host_config"], ensure_ascii=False),
                json.dumps(data["guests_config"], ensure_ascii=False),
                json.dumps(data["history"], ensure_ascii=False),
                data.get("summary"),
                json.dumps(data.get("agenda"), ensure_ascii=False) if data.get("agenda") else None,
                materials_json,
                data.get("host_style", "neutral"),
                json.dumps(data.get("embedding_config"), ensure_ascii=False) if data.get("embedding_config") else None,
                data.get("tavily_key"),
                data.get("discussion_title"),
                data.get("topic_content"),
                data.get("created_at", now), now
            ))
        conn.commit()
        conn.close()


def get_record(meeting_id: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM meeting_records WHERE id=?", (meeting_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_dict(row)


def list_records(limit: int = 50) -> List[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM meeting_records ORDER BY updated_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def delete_record(meeting_id: str):
    with _db_lock:
        conn = get_conn()
        conn.execute("DELETE FROM meeting_records WHERE id=?", (meeting_id,))
        conn.commit()
        conn.close()


def _row_to_dict(row) -> dict:
    d = dict(row)
    for key in ("host_config", "guests_config", "history", "agenda", "materials", "embedding_config"):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except Exception:
                pass
        else:
            d[key] = [] if key in ("history", "guests_config", "materials") else None
    # host_style 默认值
    if not d.get("host_style"):
        d["host_style"] = "neutral"
    return d


# 初始化
init_db()
