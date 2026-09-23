"""Read-only impact rehearsal for quarantined readings.

``POST /sessions/{session_id}/quarantine-audit`` answers, without changing
anything: "if each quarantined reading could be inserted into the published
timeline by event time, *where* would it land, between which two published
neighbours, how far away are they, and which is the earliest consumer
checkpoint whose resume window would already contain it?"

The whole computation runs inside one deferred SQLite read transaction
(``read_transaction``), so the quarantined rows, the published stream and the
checkpoint rows all come from a single consistent snapshot.  Nothing is ever
written: the published stream, watermarks, alert state machines and
checkpoints are untouched.

Per quarantined reading the hypothetical insertion follows the deterministic
merge order ``(observed_at, source_id, seq)``:

* ``position`` is the number of published readings whose keyset is strictly
  smaller (0 = before the whole stream; equal to the published total = after
  the last reading).  ``previous`` / ``next`` are the keysets of the two
  published neighbours (``null`` at either end) and
  ``seconds_after_previous`` / ``seconds_before_next`` are the event-time
  distances to them.  Ties on ``observed_at`` are broken by
  ``(source_id, seq)`` exactly like the published ordering, so a quarantined
  reading sharing a timestamp slots in deterministically between the ties.
* ``affected_checkpoint`` is the earliest checkpoint (smallest captured
  stream position, mirroring the retention protection boundary) whose
  captured keyset is *strictly smaller* than the reading's keyset: resuming
  from it returns every published key strictly greater than its own, so the
  inserted reading would land inside that checkpoint's resume window.  A
  checkpoint captured on an empty stream (null keyset, offset 0) is affected
  by every reading; when no checkpoint precedes the insertion point the field
  is ``null``.

Optional filters narrow the quarantined set: ``source_id`` must be declared on
the session, ``from`` / ``to`` bound the *event time* inclusively (normalised
to UTC) and ``from`` must not be later than ``to``.  The surviving rows are
returned in the same stable ``(observed_at, source_id, seq)`` order, capped at
``limit`` (default 100, 1..1000).
"""
from __future__ import annotations

import json
import sqlite3
from bisect import bisect_left
from typing import Any, Optional

from .database import read_transaction
from .errors import BadRequestError
from .retention import _earliest_checkpoint
from .service import _load_session
from .time_utils import format_event_time, parse_event_time, store_event_time

__all__ = ["audit_quarantine"]

DEFAULT_LIMIT = 100
MAX_LIMIT = 1000


def _timeline_key_out(key: tuple[str, str, int]) -> dict[str, Any]:
    observed_at, source_id, seq = key
    return {
        "observed_at": format_event_time(parse_event_time(observed_at)),
        "source_id": source_id,
        "seq": seq,
    }


def _affected_checkpoint(
    reading_key: tuple[str, str, int], earliest: Optional[sqlite3.Row]
) -> Optional[dict[str, Any]]:
    """The earliest checkpoint whose resume window contains ``reading_key``.

    Checkpoint resume returns every published key strictly greater than the
    checkpoint's captured key, so a checkpoint is affected exactly while that
    captured key is strictly smaller than the inserted reading's key.  The
    earliest checkpoint owns the smallest captured keyset, so whenever any
    checkpoint is affected it is this one (its resume window is the widest);
    a null keyset (empty-stream checkpoint) precedes every reading.
    """
    if earliest is None:
        return None
    cp_ts = earliest["last_observed_at"]
    if cp_ts is None:
        precedes = True
    else:
        cp_key = (cp_ts, earliest["last_source_id"], earliest["last_seq"])
        precedes = cp_key < reading_key
    if not precedes:
        return None
    return {
        "name": earliest["name"],
        "stream_offset": earliest["stream_offset"],
        "last_observed_at": format_event_time(parse_event_time(cp_ts)) if cp_ts else None,
        "last_source_id": earliest["last_source_id"],
        "last_seq": earliest["last_seq"],
    }


def audit_quarantine(
    conn: sqlite3.Connection,
    session_id: str,
    source_id: Optional[str] = None,
    from_iso: Optional[str] = None,
    to_iso: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Rehearse inserting each (filtered) quarantined reading into the stream.

    Read-only: every read happens on one SQLite snapshot and no row is
    inserted, updated or deleted.  Invalid ``from``/``to`` timestamps raise
    :class:`~app.time_utils.TimestampError` (400); an inverted interval, an
    out-of-range ``limit`` or an undeclared ``source_id`` raise
    :class:`~app.errors.BadRequestError` (400); an unknown session raises 404.
    """
    if limit is None:
        limit = DEFAULT_LIMIT
    # bool is a subclass of int; reject it explicitly for a clean 400.
    if not isinstance(limit, int) or isinstance(limit, bool) or not (
        1 <= limit <= MAX_LIMIT
    ):
        raise BadRequestError(
            f"limit must be an integer between 1 and {MAX_LIMIT}, got {limit!r}"
        )

    from_store: Optional[str] = None
    to_store: Optional[str] = None
    if from_iso is not None:
        from_store = store_event_time(parse_event_time(from_iso))
    if to_iso is not None:
        to_store = store_event_time(parse_event_time(to_iso))
    if from_store is not None and to_store is not None and from_store > to_store:
        raise BadRequestError(
            f"'from' ({format_event_time(parse_event_time(from_store))}) must not be"
            f" later than 'to' ({format_event_time(parse_event_time(to_store))})"
        )

    where = "session_id = ? AND status = 'quarantined'"
    params: list[Any] = [session_id]
    if source_id is not None:
        where += " AND source_id = ?"
        params.append(source_id)
    if from_store is not None:
        where += " AND observed_at >= ?"
        params.append(from_store)
    if to_store is not None:
        where += " AND observed_at <= ?"
        params.append(to_store)

    with read_transaction(conn):
        session = _load_session(conn, session_id)
        if source_id is not None:
            allowed_sources = set(json.loads(session["source_ids"]))
            if source_id not in allowed_sources:
                raise BadRequestError(
                    f"unknown source_id {source_id!r} for this session;"
                    f" declared sources: {sorted(allowed_sources)}"
                )

        # The complete published keyset in deterministic merge order; the
        # publish index covers (session_id, status, observed_at, source_id, seq).
        published_keys: list[tuple[str, str, int]] = [
            (r["observed_at"], r["source_id"], r["seq"])
            for r in conn.execute(
                "SELECT observed_at, source_id, seq FROM readings"
                " WHERE session_id = ? AND status = 'published'"
                " ORDER BY observed_at, source_id, seq",
                (session_id,),
            ).fetchall()
        ]

        total_matched = conn.execute(
            f"SELECT COUNT(*) AS c FROM readings WHERE {where}",
            params,
        ).fetchone()["c"]

        rows = conn.execute(
            f"SELECT * FROM readings WHERE {where}"
            " ORDER BY observed_at, source_id, seq LIMIT ?",
            (*params, limit),
        ).fetchall()

        # Earliest checkpoint (same boundary ordering retention uses); its
        # resume window is the widest, so only it can be the *earliest*
        # affected checkpoint.
        earliest = _earliest_checkpoint(conn, session_id)

        items: list[dict[str, Any]] = []
        for row in rows:
            reading_key = (row["observed_at"], row["source_id"], row["seq"])
            # bisect_left: equal published keys cannot exist (a reading key is
            # unique across statuses), so left/right insertion coincide.
            position = bisect_left(published_keys, reading_key)
            previous_key = published_keys[position - 1] if position > 0 else None
            next_key = (
                published_keys[position] if position < len(published_keys) else None
            )
            reading_dt = parse_event_time(row["observed_at"])
            seconds_after_previous: Optional[float] = None
            seconds_before_next: Optional[float] = None
            if previous_key is not None:
                seconds_after_previous = (
                    reading_dt - parse_event_time(previous_key[0])
                ).total_seconds()
            if next_key is not None:
                seconds_before_next = (
                    parse_event_time(next_key[0]) - reading_dt
                ).total_seconds()

            items.append(
                {
                    "source_id": row["source_id"],
                    "seq": row["seq"],
                    "observed_at": format_event_time(reading_dt),
                    "value": (
                        json.loads(row["payload"]) if row["payload"] is not None else None
                    ),
                    "quarantined_watermark": format_event_time(
                        parse_event_time(row["quarantined_watermark"])
                    ),
                    "insertion": {
                        "position": position,
                        "previous": (
                            _timeline_key_out(previous_key)
                            if previous_key is not None
                            else None
                        ),
                        "next": (
                            _timeline_key_out(next_key) if next_key is not None else None
                        ),
                        "seconds_after_previous": seconds_after_previous,
                        "seconds_before_next": seconds_before_next,
                    },
                    "affected_checkpoint": _affected_checkpoint(reading_key, earliest),
                }
            )

    return {
        "session_id": session_id,
        "source_id": source_id,
        "from": (
            format_event_time(parse_event_time(from_store))
            if from_store is not None
            else None
        ),
        "to": (
            format_event_time(parse_event_time(to_store))
            if to_store is not None
            else None
        ),
        "limit": limit,
        "matched": len(items),
        "total_matched": total_matched,
        "readings": items,
    }
