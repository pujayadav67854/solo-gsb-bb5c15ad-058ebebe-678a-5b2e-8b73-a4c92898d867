"""Checkpoint-safe storage retention for old readings.

``POST /sessions/{session_id}/retention`` reports (``dry_run=true``) or
performs (``dry_run=false``) the deletion of readings whose **event time** is
strictly earlier than the caller-supplied ``cutoff``.  Only rows that are
already terminal participate -- ``published`` and ``quarantined``; buffered
readings are never touched (they may still be published and drive the
watermark / alert state machines).

Quarantined rows carry no stream semantics beyond their own listing, so every
quarantined candidate is deletable.  Published rows, however, may still be
(re)read by consumers resuming from a checkpoint: a checkpoint resume returns
every published reading whose deterministic-order keyset
``(observed_at, source_id, seq)`` is *strictly greater* than the keyset the
checkpoint captured.  It is therefore enough to keep everything that the
session's **earliest** checkpoint may still return: a published reading whose
keyset is greater than that checkpoint's keyset is retained, while the
checkpoint boundary itself (the last reading it already covered) and every
reading before it can never be returned by any resume and is deletable.

When the session has no checkpoint at all no resume can be served, so no
published reading is protected.  A checkpoint captured on an empty stream
(``stream_offset = 0``, null keyset) protects *every* published reading:
resuming from it starts at the beginning.

Dry runs derive all counts in one deferred read snapshot and write nothing.
Confirmation executes the candidate deletes in a single serialised
``BEGIN IMMEDIATE`` write transaction, so the report and the delete are one
atomic unit: any failure rolls the transaction back unchanged.  Retention
never writes to ``sessions``, ``session_sources``, the alert tables or
``checkpoints`` -- watermarks, alerts and checkpoint semantics are unchanged.
Repeating the same request is idempotent: once the candidates are gone the
second call simply reports zero deletable rows.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Optional

from .database import read_transaction, transaction
from .service import _load_session
from .time_utils import format_event_time, parse_event_time, store_event_time

__all__ = ["apply_retention"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _earliest_checkpoint(
    conn: sqlite3.Connection, session_id: str
) -> Optional[sqlite3.Row]:
    """The checkpoint whose resume window is furthest back in the stream.

    Ordering is by the captured stream position (offset, then the
    deterministic keyset); ``name`` only breaks the tie for distinct
    checkpoints taken at the exact same position.  Retaining everything this
    checkpoint can still read trivially covers every later one.
    """
    return conn.execute(
        "SELECT * FROM checkpoints WHERE session_id = ?"
        " ORDER BY stream_offset ASC,"
        " last_observed_at ASC,"
        " last_source_id ASC,"
        " last_seq ASC,"
        " name ASC"
        " LIMIT 1",
        (session_id,),
    ).fetchone()


def _checkpoint_protection(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "checkpoint_name": row["name"],
        "stream_offset": row["stream_offset"],
        "last_observed_at": (
            format_event_time(parse_event_time(row["last_observed_at"]))
            if row["last_observed_at"]
            else None
        ),
        "last_source_id": row["last_source_id"],
        "last_seq": row["last_seq"],
    }


def _count(conn: sqlite3.Connection, where: str, *params: Any) -> int:
    return conn.execute(
        f"SELECT COUNT(*) AS c FROM readings WHERE {where}", params
    ).fetchone()["c"]


# ---------------------------------------------------------------------------
# retention report / delete
# ---------------------------------------------------------------------------

def apply_retention(
    conn: sqlite3.Connection,
    session_id: str,
    cutoff: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Report or atomically delete pre-cutoff terminal readings.

    Returns the retention report (same shape for dry runs and executions;
    ``deleted`` is 0 for dry runs and the number of rows removed otherwise).
    Invalid ``cutoff`` raises :class:`~app.time_utils.TimestampError` (400);
    an unknown session raises 404.
    """
    cutoff_store = store_event_time(parse_event_time(cutoff))

    if dry_run:
        with read_transaction(conn):
            _load_session(conn, session_id)
            earliest = _earliest_checkpoint(conn, session_id)
            counts = _compute_counts(conn, session_id, cutoff_store, earliest)
            protection = _checkpoint_protection(earliest) if earliest else None
        return _result(session_id, cutoff_store, dry_run, protection, counts,
                       deleted=0)

    deleted_quarantined = 0
    deleted_published = 0
    with transaction(conn):
        _load_session(conn, session_id)
        earliest = _earliest_checkpoint(conn, session_id)
        counts = _compute_counts(conn, session_id, cutoff_store, earliest)

        # Every quarantined candidate is unreachable from the published stream
        # and can be removed outright.
        cur = conn.execute(
            "DELETE FROM readings WHERE session_id = ?"
            " AND status = 'quarantined' AND observed_at < ?",
            (session_id, cutoff_store),
        )
        deleted_quarantined = cur.rowcount

        # Published candidates: keep precisely the rows the earliest
        # checkpoint can still return on resume (keyset strictly greater than
        # the captured keyset).  The boundary keyset itself is already covered
        # by the checkpoint, so it is deleted like everything before it.
        if earliest is None:
            cur = conn.execute(
                "DELETE FROM readings WHERE session_id = ?"
                " AND status = 'published' AND observed_at < ?",
                (session_id, cutoff_store),
            )
            deleted_published = cur.rowcount
        elif earliest["last_observed_at"] is None:
            # Empty-stream checkpoint: its resume starts at the very first
            # reading, so every published row is protected.
            deleted_published = 0
        else:
            b_ts = earliest["last_observed_at"]
            b_src = earliest["last_source_id"]
            b_seq = earliest["last_seq"]
            cur = conn.execute(
                "DELETE FROM readings WHERE session_id = ?"
                " AND status = 'published' AND observed_at < ?"
                " AND NOT (observed_at > ?"
                " OR (observed_at = ? AND source_id > ?)"
                " OR (observed_at = ? AND source_id = ? AND seq > ?))",
                (session_id, cutoff_store,
                 b_ts, b_ts, b_src, b_ts, b_src, b_seq),
            )
            deleted_published = cur.rowcount

        protection = _checkpoint_protection(earliest) if earliest else None

    return _result(
        session_id,
        cutoff_store,
        dry_run,
        protection,
        counts,
        deleted=deleted_published + deleted_quarantined,
    )


def _compute_counts(
    conn: sqlite3.Connection,
    session_id: str,
    cutoff_store: str,
    earliest: Optional[sqlite3.Row],
) -> dict[str, int]:
    """Count candidate / retained / deletable rows in the current snapshot.

    ``retained`` counts only *protected* candidates (checkpoint-retained
    published readings); buffered readings are not candidates and therefore
    not counted as retained.
    """
    quarantined = _count(
        conn,
        "session_id = ? AND status = 'quarantined' AND observed_at < ?",
        session_id, cutoff_store,
    )

    if earliest is None:
        published_deletable = _count(
            conn,
            "session_id = ? AND status = 'published' AND observed_at < ?",
            session_id, cutoff_store,
        )
        published_retained = 0
    elif earliest["last_observed_at"] is None:
        # Empty-stream checkpoint: all published candidates stay.
        published_deletable = 0
        published_retained = _count(
            conn,
            "session_id = ? AND status = 'published' AND observed_at < ?",
            session_id, cutoff_store,
        )
    else:
        b_ts = earliest["last_observed_at"]
        b_src = earliest["last_source_id"]
        b_seq = earliest["last_seq"]
        # Keyset strictly greater than the checkpoint boundary == still
        # readable on resume == retained.
        published_retained = _count(
            conn,
            "session_id = ? AND status = 'published' AND observed_at < ?"
            " AND (observed_at > ?"
            " OR (observed_at = ? AND source_id > ?)"
            " OR (observed_at = ? AND source_id = ? AND seq > ?))",
            session_id, cutoff_store,
            b_ts, b_ts, b_src, b_ts, b_src, b_seq,
        )
        published_total = _count(
            conn,
            "session_id = ? AND status = 'published' AND observed_at < ?",
            session_id, cutoff_store,
        )
        published_deletable = published_total - published_retained

    candidates = quarantined + published_retained + published_deletable
    return {
        "candidates": candidates,
        "retained": published_retained,
        "deletable": quarantined + published_deletable,
        "published_candidates": published_retained + published_deletable,
        "published_retained": published_retained,
        "published_deletable": published_deletable,
        "quarantined_candidates": quarantined,
        "quarantined_deletable": quarantined,
    }


def _result(
    session_id: str,
    cutoff_store: str,
    dry_run: bool,
    protection: Optional[dict[str, Any]],
    counts: dict[str, int],
    *,
    deleted: int,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "cutoff": format_event_time(parse_event_time(cutoff_store)),
        "dry_run": dry_run,
        "deleted": deleted,
        "protected_by_checkpoint": protection,
        "candidates": counts["candidates"],
        "retained": counts["retained"],
        "deletable": counts["deletable"],
        "by_status": {
            "published": {
                "candidates": counts["published_candidates"],
                "retained": counts["published_retained"],
                "deletable": counts["published_deletable"],
            },
            "quarantined": {
                "candidates": counts["quarantined_candidates"],
                "retained": 0,
                "deletable": counts["quarantined_deletable"],
            },
        },
    }
