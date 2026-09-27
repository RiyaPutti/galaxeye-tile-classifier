import sqlite3
import json
import datetime
from contextlib import contextmanager

from config import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tile_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    predicted_label TEXT,
    confidence REAL,
    class_scores TEXT,
    needs_review INTEGER,
    model_version TEXT NOT NULL,
    feature_extractor_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'success',
    error_message TEXT,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(_SCHEMA)
        conn.commit()


def insert_result(filename, predicted_label, confidence, class_scores,
                   needs_review, model_version, feature_extractor_version):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO tile_results
               (filename, predicted_label, confidence, class_scores,
                needs_review, model_version, feature_extractor_version,
                status, error_message, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'success', NULL, ?)""",
            (filename, predicted_label, confidence, json.dumps(class_scores),
             int(needs_review), model_version, feature_extractor_version,
             datetime.datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cur.lastrowid


def insert_failure(filename, model_version, feature_extractor_version, error_message):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO tile_results
               (filename, predicted_label, confidence, class_scores,
                needs_review, model_version, feature_extractor_version,
                status, error_message, created_at)
               VALUES (?, NULL, NULL, NULL, NULL, ?, ?, 'failed', ?, ?)""",
            (filename, model_version, feature_extractor_version, error_message,
             datetime.datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cur.lastrowid


def list_results(label=None, needs_review=None, limit=50):
    query = "SELECT * FROM tile_results WHERE 1=1"
    params = []
    if label is not None:
        query += " AND predicted_label = ?"
        params.append(label)
    if needs_review is not None:
        query += " AND needs_review = ?"
        params.append(int(needs_review))
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("class_scores"):
                d["class_scores"] = json.loads(d["class_scores"])
            results.append(d)
        return results