"""Small, transactional metadata store. No source image bytes are stored here."""
import json
import sqlite3
import time
from contextlib import contextmanager
from .config import DATA


@contextmanager
def connection():
    db = sqlite3.connect(DATA / 'naturegen.sqlite3', timeout=30)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, body TEXT, PRIMARY KEY(kind,id))')
    try:
        yield db
        db.commit()
    finally:
        db.close()


def put(kind: str, item: dict) -> dict:
    with connection() as db:
        db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?)', (kind, item['id'], json.dumps(item)))
    return item


def get(kind: str, key: str) -> dict | None:
    with connection() as db:
        row = db.execute('SELECT body FROM records WHERE kind=? AND id=?', (kind, key)).fetchone()
    return json.loads(row[0]) if row else None


def all_records(kind: str) -> list[dict]:
    with connection() as db:
        rows = db.execute('SELECT body FROM records WHERE kind=? ORDER BY rowid DESC', (kind,)).fetchall()
    return [json.loads(row[0]) for row in rows]


def delete(kind: str, key: str):
    with connection() as db:
        db.execute('DELETE FROM records WHERE kind=? AND id=?', (kind, key))


def recover_jobs():
    for job in all_records('job'):
        if job['state'] in ('queued', 'running', 'stopping'):
            put('job', {**job, 'state': 'interrupted', 'error': 'Backend restarted. Resume from the last checkpoint.', 'updated_at': time.time()})
