"""Persistent shared documents with short transactions around read/modify/write.

With DATABASE_URL configured, application CSV/JSON documents live in PostgreSQL.
Without it, the existing local files are used. SQLite is supported for local tests
and deployments that explicitly provide a persistent disk, not Community Cloud.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import os
from pathlib import Path
import sqlite3
import tempfile
from threading import RLock


DATA_ROOT = Path(__file__).resolve().parent
DOCUMENT_NAMES = frozenset({
    "donors.csv", "processed_articles.csv", "funding_research.csv", "funding_research.json",
    "news_articles.csv", "news_article_history.csv", "seller_contacts.csv",
    "events.csv", "event_people.csv", "processed_event_articles.csv",
})
_active_connection = ContextVar("research_storage_connection", default=None)
_local_lock = RLock()
_LOCK_ID = 731421607
_SCHEMA = "CREATE TABLE IF NOT EXISTS research_documents (name TEXT PRIMARY KEY, content TEXT NOT NULL)"


def uses_database(path):
    path = Path(path).resolve()
    return bool(os.environ.get("DATABASE_URL", "").strip()) and path.parent == DATA_ROOT and path.name in DOCUMENT_NAMES


def _execute(connection, statement, params=()):
    if not isinstance(connection, sqlite3.Connection):
        statement = statement.replace("?", "%s").replace("research_documents", "jmac_private.research_documents")
    return connection.execute(statement, params)


@contextmanager
def storage_transaction(path):
    if not uses_database(path):
        with _local_lock:
            yield
        return
    if _active_connection.get() is not None:
        yield
        return
    url = os.environ["DATABASE_URL"].strip()
    if url.startswith("sqlite:///"):
        connection = sqlite3.connect(url[len("sqlite:///"):], timeout=30)
    elif url.startswith(("postgres://", "postgresql://")):
        import psycopg
        connection = psycopg.connect(url, connect_timeout=15, prepare_threshold=None)
    else:
        raise ValueError("DATABASE_URL must use postgresql:// or sqlite:///.")
    token = _active_connection.set(connection)
    try:
        if isinstance(connection, sqlite3.Connection):
            connection.execute("BEGIN IMMEDIATE")
        else:
            # Released by commit/rollback; works with transaction poolers too.
            connection.execute("SET LOCAL lock_timeout = '30s'")
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_ID,))
            # Keep research data out of Supabase's exposed public API schema.
            connection.execute("CREATE SCHEMA IF NOT EXISTS jmac_private")
        _execute(connection, _SCHEMA)
        yield
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        _active_connection.reset(token)
        connection.close()


def transactional(function):
    """Serialize the complete mutation, including the read before the write."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        with storage_transaction(bound.arguments["path"]):
            return function(*args, **kwargs)

    return wrapped


def read_text(path):
    path = Path(path)
    if not uses_database(path):
        return path.read_text(encoding="utf-8-sig") if path.exists() else None
    with storage_transaction(path):
        row = _execute(
            _active_connection.get(), "SELECT content FROM research_documents WHERE name = ?", (path.name,),
        ).fetchone()
        # Never silently fall back to a stale bundled CSV when shared storage
        # is empty, unavailable, or a row has been deleted.
        return row[0] if row else None


def write_text(path, content):
    path = Path(path)
    if uses_database(path):
        with storage_transaction(path):
            _execute(
                _active_connection.get(),
                "INSERT INTO research_documents (name, content) VALUES (?, ?) "
                "ON CONFLICT (name) DO UPDATE SET content = excluded.content",
                (path.name, content),
            )
        return
    # Unique temporary names also avoid collisions between independent writers.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def import_local_documents():
    """Import existing data once; never replace an existing shared document."""
    imported = []
    anchor = DATA_ROOT / "news_articles.csv"
    if not uses_database(anchor):
        raise ValueError("Set DATABASE_URL before importing existing data.")
    with storage_transaction(anchor):
        for name in sorted(DOCUMENT_NAMES):
            path = DATA_ROOT / name
            if path.exists() and read_text(path) is None:
                write_text(path, path.read_text(encoding="utf-8-sig"))
                imported.append(name)
    return imported
