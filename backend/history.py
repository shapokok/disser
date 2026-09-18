"""
Analysis journal: every prediction is stored in a small SQLite database so the
app can show monitoring over time (what was analysed, when, how often each
disease appears). Images are not stored, only a small thumbnail.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    image_name TEXT,
    model TEXT,
    model_label TEXT,
    explanation TEXT,
    dataset_type TEXT,
    class_raw TEXT,
    plant TEXT,
    confidence REAL,
    healthy INTEGER,
    inference_ms REAL,
    thumbnail TEXT,
    result_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_ts ON analyses(ts);
CREATE INDEX IF NOT EXISTS idx_analyses_class ON analyses(class_raw);
"""

COLUMNS = "id, ts, image_name, model, model_label, explanation, dataset_type, class_raw, plant, confidence, healthy, inference_ms, thumbnail"


class History:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self):
        con = sqlite3.connect(self.path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    # ---------------------------------------------------------------- write
    def add(self, result: dict, thumbnail: str | None) -> int:
        pred = result.get("prediction", {})
        slim = {k: v for k, v in result.items() if k != "images"}
        with self._lock, self._connect() as con:
            cur = con.execute(
                "INSERT INTO analyses (ts, image_name, model, model_label, explanation, dataset_type, class_raw, plant, "
                "confidence, healthy, inference_ms, thumbnail, result_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    result.get("timestamp") or datetime.now().isoformat(timespec="seconds"),
                    result.get("image_name"),
                    result.get("model"),
                    result.get("model_label"),
                    result.get("explanation"),
                    result.get("dataset_type"),
                    pred.get("class_raw"),
                    pred.get("plant"),
                    float(pred.get("confidence") or 0),
                    1 if pred.get("healthy") else 0,
                    result.get("inference_time_ms"),
                    thumbnail,
                    json.dumps(slim, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def delete(self, item_id: int) -> bool:
        with self._lock, self._connect() as con:
            return con.execute("DELETE FROM analyses WHERE id = ?", (item_id,)).rowcount > 0

    def clear(self) -> int:
        with self._lock, self._connect() as con:
            return con.execute("DELETE FROM analyses").rowcount

    # ---------------------------------------------------------------- read
    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        model: str | None = None,
        plant: str | None = None,
        query: str | None = None,
        healthy: bool | None = None,
    ) -> tuple[list[dict], int]:
        where, params = [], []
        if model:
            where.append("model = ?")
            params.append(model)
        if plant:
            where.append("plant = ?")
            params.append(plant)
        if healthy is not None:
            where.append("healthy = ?")
            params.append(1 if healthy else 0)
        if query:
            where.append("(class_raw LIKE ? OR image_name LIKE ?)")
            params += [f"%{query}%", f"%{query}%"]
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self._connect() as con:
            total = con.execute(f"SELECT COUNT(*) FROM analyses{clause}", params).fetchone()[0]
            rows = con.execute(
                f"SELECT {COLUMNS} FROM analyses{clause} ORDER BY id DESC LIMIT ? OFFSET ?", [*params, limit, offset]
            ).fetchall()
        return [dict(r) for r in rows], int(total)

    def get(self, item_id: int) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM analyses WHERE id = ?", (item_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        data["result"] = json.loads(data.pop("result_json") or "{}")
        return data

    def stats(self) -> dict:
        with self._connect() as con:
            total = con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            healthy = con.execute("SELECT COUNT(*) FROM analyses WHERE healthy = 1").fetchone()[0]
            by_class = con.execute(
                "SELECT class_raw, COUNT(*) AS n, AVG(confidence) AS conf, MAX(healthy) AS healthy FROM analyses "
                "GROUP BY class_raw ORDER BY n DESC LIMIT 12"
            ).fetchall()
            by_plant = con.execute(
                "SELECT plant, COUNT(*) AS n, SUM(healthy) AS healthy FROM analyses GROUP BY plant ORDER BY n DESC"
            ).fetchall()
            by_model = con.execute(
                "SELECT model, COUNT(*) AS n FROM analyses GROUP BY model ORDER BY n DESC"
            ).fetchall()
            by_day = con.execute(
                "SELECT substr(ts, 1, 10) AS day, COUNT(*) AS n, SUM(CASE WHEN healthy = 0 THEN 1 ELSE 0 END) AS unhealthy "
                "FROM analyses GROUP BY day ORDER BY day DESC LIMIT 30"
            ).fetchall()
            last = con.execute("SELECT ts FROM analyses ORDER BY id DESC LIMIT 1").fetchone()
        return {
            "total": int(total),
            "healthy": int(healthy),
            "healthy_share": (healthy / total) if total else None,
            "by_class": [dict(r) for r in by_class],
            "by_plant": [dict(r) for r in by_plant],
            "by_model": [dict(r) for r in by_model],
            "by_day": list(reversed([dict(r) for r in by_day])),
            "last_at": last[0] if last else None,
        }
