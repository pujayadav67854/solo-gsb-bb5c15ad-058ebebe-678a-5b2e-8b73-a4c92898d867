"""End-to-end tests for the read-only quarantine impact audit.

POST /sessions/{session_id}/quarantine-audit rehearses, inside one SQLite
read snapshot, where each quarantined reading would be inserted into the
published timeline (ordered by (observed_at, source_id, seq)), reports the
neighbouring published keysets and event-time gaps, and names the earliest
checkpoint whose resume window the insertion would fall into.
"""
from __future__ import annotations

import json
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("SENSOR_DB_PATH", db_path)
    from app import database, main

    main.app.state.db_path = db_path
    database.init_db(db_path)
    with TestClient(main.app) as c:
        c.db_path = db_path  # type: ignore[attr-defined]
        yield c


def create_session(client, sources=("a", "b"), lateness=0.0, session_id="s"):
    r = client.post(
        "/sessions",
        json={
            "session_id": session_id,
            "source_ids": list(sources),
            "allowed_lateness_seconds": lateness,
        },
    )
    assert r.status_code == 201, r.text
    return session_id


def report(client, sid, readings):
    return client.post(f"/sessions/{sid}/readings", json={"readings": readings})


def audit(client, sid, body=None):
    return client.post(f"/sessions/{sid}/quarantine-audit", json=body if body is not None else {})


def make_checkpoint(client, sid, name):
    return client.post(f"/sessions/{sid}/checkpoints", json={"name": name})


def setup_scene(client, sid="s", lateness=0.0):
    """Published ladder at 10s spacing plus four quarantined readings.

    Published (lateness 0, strict-less-than watermarks, min-source clock):
        a@10:00:00/1, b@10:00:00/1, a@10:00:10/2, b@10:00:10/2,
        a@10:00:20/3, b@10:00:20/3
    Quarantined (each reading key is unique across the session), in the
    audit's deterministic (observed_at, source_id, seq) order:
        a@10:00:05/9   -> between b@10:00:00 and a@10:00:10
        a@10:00:15/10  -> between b@10:00:10 and a@10:00:20
        b@10:00:15/9   -> same timestamp as the previous one; source tie-break
        b@10:00:25/10  -> after the last published reading (tail)
    """
    # wm :00: both :00 readings buffered (strict <, on-the-boundary waits).
    report(client, sid, [
        {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 1},
        {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 2},
    ])
    # Both clocks advance to :10 -> wm :10 publishes the two :00 readings.
    report(client, sid, [
        {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T10:00:10Z", "value": 3},
        {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T10:00:10Z", "value": 4},
    ])
    # A :05 arrival is now behind the published watermark (:10).
    r = report(client, sid, [
        {"source_id": "a", "seq": 9, "observed_at": "2026-01-01T10:00:05Z", "value": 91},
    ])
    assert r.json()["quarantined"] == 1
    # Both clocks advance to :20 then :30: the :10 and :20 pairs publish.
    report(client, sid, [
        {"source_id": "a", "seq": 3, "observed_at": "2026-01-01T10:00:20Z", "value": 5},
        {"source_id": "b", "seq": 3, "observed_at": "2026-01-01T10:00:20Z", "value": 6},
    ])
    report(client, sid, [
        {"source_id": "a", "seq": 4, "observed_at": "2026-01-01T10:00:30Z", "value": 7},
        {"source_id": "b", "seq": 4, "observed_at": "2026-01-01T10:00:30Z", "value": 8},
    ])
    # Against wm :30, readings at :15/:25 are all behind it (quarantined rows
    # never advance the source clocks).
    r = report(client, sid, [
        {"source_id": "a", "seq": 10, "observed_at": "2026-01-01T10:00:15Z", "value": 92},
        {"source_id": "b", "seq": 9, "observed_at": "2026-01-01T10:00:15Z", "value": 93},
        {"source_id": "b", "seq": 10, "observed_at": "2026-01-01T10:00:25Z", "value": 94},
    ])
    assert r.json()["quarantined"] == 3


def item_key(item):
    return (item["source_id"], item["seq"], item["observed_at"])


class TestQuarantineAuditBasics:
    def test_all_quarantined_readings_audited_in_stable_order(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        r = audit(client, sid)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["session_id"] == sid
        assert body["source_id"] is None
        assert body["from"] is None
        assert body["to"] is None
        assert body["limit"] == 100
        assert body["matched"] == 4
        assert body["total_matched"] == 4
        # Stable order: (observed_at, source_id, seq) -- note a@:15 before b@:15.
        assert [(x["source_id"], x["seq"]) for x in body["readings"]] == [
            ("a", 9), ("a", 10), ("b", 9), ("b", 10)]
        assert [x["observed_at"] for x in body["readings"]] == [
            "2026-01-01T10:00:05Z",
            "2026-01-01T10:00:15Z",
            "2026-01-01T10:00:15Z",
            "2026-01-01T10:00:25Z",
        ]
        # Reading payload and intercepted watermark are carried through.
        first = body["readings"][0]
        assert first["value"] == 91
        assert first["quarantined_watermark"] == "2026-01-01T10:00:10Z"
        last = body["readings"][-1]
        assert last["quarantined_watermark"] == "2026-01-01T10:00:30Z"

    def test_insertion_between_two_neighbours(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        items = {item_key(x): x for x in audit(client, sid).json()["readings"]}
        m = items[("a", 9, "2026-01-01T10:00:05Z")]["insertion"]
        # Published keys < a@:05: a@:00, b@:00 (both at :00, a < b) -> position 2.
        assert m["position"] == 2
        assert m["previous"] == {
            "observed_at": "2026-01-01T10:00:00Z", "source_id": "b", "seq": 1}
        assert m["next"] == {
            "observed_at": "2026-01-01T10:00:10Z", "source_id": "a", "seq": 2}
        assert m["seconds_after_previous"] == 5.0
        assert m["seconds_before_next"] == 5.0

    def test_tied_timestamps_ordered_by_source_then_seq(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        items = {item_key(x): x for x in audit(client, sid).json()["readings"]}
        # a@:15/10 and b@:15/9 are both quarantined; each is positioned against
        # the *published* stream only (quarantined rows are not neighbours).
        ma = items[("a", 10, "2026-01-01T10:00:15Z")]["insertion"]
        mb = items[("b", 9, "2026-01-01T10:00:15Z")]["insertion"]
        # Published at :10 (a,b) then :20 (a,b): both :15 readings land at
        # offset 4; source tie-break only changes their relative ordering in
        # the response, not the published neighbours.
        assert ma["position"] == 4
        assert mb["position"] == 4
        assert ma["previous"] == mb["previous"] == {
            "observed_at": "2026-01-01T10:00:10Z", "source_id": "b", "seq": 2}
        assert ma["next"] == mb["next"] == {
            "observed_at": "2026-01-01T10:00:20Z", "source_id": "a", "seq": 3}
        assert ma["seconds_after_previous"] == 5.0
        assert ma["seconds_before_next"] == 5.0

    def test_insertion_before_the_whole_stream(self, client):
        sid = create_session(client)
        # Both clocks at :10 -> wm :10, the :10 readings stay buffered
        # (strict-less-than).  Advancing both to :20 publishes them.
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T10:00:10Z", "value": 1},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T10:00:10Z", "value": 2},
        ])
        report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T10:00:20Z", "value": 3},
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T10:00:20Z", "value": 4},
        ])
        r = report(client, sid, [
            {"source_id": "a", "seq": 5, "observed_at": "2026-01-01T10:00:05Z", "value": 7}])
        assert r.json()["quarantined"] == 1
        body = audit(client, sid).json()
        assert body["matched"] == 1
        m = body["readings"][0]["insertion"]
        assert m["position"] == 0
        assert m["previous"] is None
        assert m["seconds_after_previous"] is None
        assert m["next"] == {
            "observed_at": "2026-01-01T10:00:10Z", "source_id": "a", "seq": 1}
        assert m["seconds_before_next"] == 5.0

    def test_insertion_after_the_last_published_reading(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        items = {item_key(x): x for x in audit(client, sid).json()["readings"]}
        m = items[("b", 10, "2026-01-01T10:00:25Z")]["insertion"]
        # The :30 readings sit on the watermark and stay buffered, so the last
        # published reading remains b@:20/3.
        assert m["position"] == 6
        assert m["next"] is None
        assert m["seconds_before_next"] is None
        assert m["previous"] == {
            "observed_at": "2026-01-01T10:00:20Z", "source_id": "b", "seq": 3}
        assert m["seconds_after_previous"] == 5.0

    def test_timestamp_tie_with_published_reading_uses_source_seq(self, client):
        sid = create_session(client)
        # Publish a@:00 and b@:00 (wm :10 once both clocks reach :10).
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 1},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 2},
        ])
        report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T10:00:10Z", "value": 3},
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T10:00:10Z", "value": 4},
        ])
        # Publish readings at :20 in a pair (both clocks must advance).  The
        # :30 pair below sits on its own watermark and stays buffered.
        report(client, sid, [
            {"source_id": "a", "seq": 3, "observed_at": "2026-01-01T10:00:20Z", "value": 5},
            {"source_id": "b", "seq": 3, "observed_at": "2026-01-01T10:00:20Z", "value": 6},
        ])
        report(client, sid, [
            {"source_id": "a", "seq": 4, "observed_at": "2026-01-01T10:00:30Z", "value": 7},
            {"source_id": "b", "seq": 4, "observed_at": "2026-01-01T10:00:30Z", "value": 8},
        ])
        # Quarantine readings at the *published* time :20: a@:20/5 sorts
        # between the published b@:20/3 (later in the (source, seq) order) --
        # zero time delta, tie broken deterministically on source then seq.
        r = report(client, sid, [
            {"source_id": "a", "seq": 5, "observed_at": "2026-01-01T10:00:20Z", "value": 9},
            {"source_id": "b", "seq": 5, "observed_at": "2026-01-01T10:00:20Z", "value": 10},
        ])
        assert r.json()["quarantined"] == 2
        body = audit(client, sid).json()
        at_a, at_b = body["readings"]
        # Published: a/1,b/1,a/2,b/2,a/3,b/3 (offset 0..5); the :30 pair is
        # still buffered.  At :20 the published keys are a/3 then b/3.
        # a/5 lands strictly between them, b/5 after b/3:
        assert (at_a["source_id"], at_a["seq"]) == ("a", 5)
        ma = at_a["insertion"]
        assert ma["position"] == 5
        assert ma["previous"] == {
            "observed_at": "2026-01-01T10:00:20Z", "source_id": "a", "seq": 3}
        assert ma["next"] == {
            "observed_at": "2026-01-01T10:00:20Z", "source_id": "b", "seq": 3}
        assert ma["seconds_after_previous"] == 0.0
        assert ma["seconds_before_next"] == 0.0
        mb = at_b["insertion"]
        assert mb["position"] == 6
        assert mb["previous"] == {
            "observed_at": "2026-01-01T10:00:20Z", "source_id": "b", "seq": 3}
        assert mb["next"] is None
        assert mb["seconds_after_previous"] == 0.0
        assert mb["seconds_before_next"] is None

    def test_no_published_readings_yet(self, client):
        sid = create_session(client)
        # Publish a watermark is impossible before all sources report, and no
        # quarantine exists before a publish: simulate via two sources at the
        # same instant with no advancement -> nothing published, nothing
        # quarantined.
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 1},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 2},
        ])
        body = audit(client, sid).json()
        assert body["matched"] == 0
        assert body["total_matched"] == 0
        assert body["readings"] == []

    def test_empty_quarantine_on_closed_session(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 1},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 2},
        ])
        assert client.post(
            f"/sessions/{sid}/sources/a/seal",
            json={"final_observed_at": "2026-01-01T10:00:00Z"},
        ).status_code == 200
        assert client.post(
            f"/sessions/{sid}/sources/b/seal",
            json={"final_observed_at": "2026-01-01T10:00:00Z"},
        ).status_code == 200
        # Closed sessions can still be audited read-only.
        r = audit(client, sid)
        assert r.status_code == 200
        assert r.json()["readings"] == []


class TestQuarantineAuditFilters:
    def test_filter_by_source_id(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        r = audit(client, sid, {"source_id": "b"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source_id"] == "b"
        assert body["matched"] == 2
        assert body["total_matched"] == 2
        assert [(x["source_id"], x["seq"]) for x in body["readings"]] == [
            ("b", 9), ("b", 10)]

    def test_filter_by_inclusive_time_window(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        # Inclusive both ends; the two :15 readings share the boundary.
        body = audit(client, sid, {
            "from": "2026-01-01T10:00:15Z", "to": "2026-01-01T10:00:15Z"}).json()
        assert body["from"] == "2026-01-01T10:00:15Z"
        assert body["to"] == "2026-01-01T10:00:15Z"
        assert body["total_matched"] == 2
        assert [(x["source_id"], x["seq"]) for x in body["readings"]] == [
            ("a", 10), ("b", 9)]

    def test_filter_from_only_and_to_only(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        body = audit(client, sid, {"from": "2026-01-01T10:00:15Z"}).json()
        assert body["total_matched"] == 3
        body = audit(client, sid, {"to": "2026-01-01T10:00:05Z"}).json()
        assert body["total_matched"] == 1

    def test_time_bounds_normalise_offsets_to_utc(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        # 10:00:15+02:00 == 08:00:15Z -> earlier than every reading.
        body = audit(client, sid, {
            "to": "2026-01-01T12:00:15+02:00"}).json()
        assert body["to"] == "2026-01-01T10:00:15Z"
        assert body["total_matched"] == 3

    def test_combined_source_and_time_filters(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        body = audit(client, sid, {
            "source_id": "a", "from": "2026-01-01T10:00:10Z"}).json()
        assert [(x["source_id"], x["observed_at"]) for x in body["readings"]] == [
            ("a", "2026-01-01T10:00:15Z")]

    def test_limit_caps_rows_but_reports_total_matched(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        body = audit(client, sid, {"limit": 2}).json()
        assert body["limit"] == 2
        assert body["matched"] == 2
        assert body["total_matched"] == 4
        # The earliest two in the stable ordering.
        assert [(x["source_id"], x["seq"]) for x in body["readings"]] == [
            ("a", 9), ("a", 10)]

    def test_limit_equal_to_total(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        body = audit(client, sid, {"limit": 4}).json()
        assert body["matched"] == 4 == body["total_matched"]

    def test_limit_with_filters(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        body = audit(client, sid, {"source_id": "a", "limit": 1}).json()
        assert body["matched"] == 1
        assert body["total_matched"] == 2


class TestQuarantineAuditCheckpoints:
    def test_earliest_checkpoint_before_insertion_is_named(self, client):
        sid = create_session(client)
        # Publish a@:00 and b@:00 (both clocks reach :10 -> wm :10).
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 1},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 2},
        ])
        report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T10:00:10Z", "value": 3},
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T10:00:10Z", "value": 4},
        ])
        cp = make_checkpoint(client, sid, "cp-early")
        assert cp.status_code == 201
        assert cp.json()["stream_offset"] == 2
        # Advance both clocks again, then quarantine a reading at :05.
        report(client, sid, [
            {"source_id": "a", "seq": 3, "observed_at": "2026-01-01T10:00:20Z", "value": 5},
            {"source_id": "b", "seq": 3, "observed_at": "2026-01-01T10:00:20Z", "value": 6},
        ])
        r = report(client, sid, [
            {"source_id": "a", "seq": 9, "observed_at": "2026-01-01T10:00:05Z", "value": 91}])
        assert r.json()["quarantined"] == 1
        item = audit(client, sid).json()["readings"][0]
        # The reading would insert at offset 2: strictly after cp-early's key
        # (b@:00), so resuming from cp-early would return it.
        assert item["insertion"]["position"] == 2
        affected = item["affected_checkpoint"]
        assert affected is not None
        assert affected["name"] == "cp-early"
        assert affected["stream_offset"] == 2
        assert affected["last_observed_at"] == "2026-01-01T10:00:00Z"
        assert affected["last_source_id"] == "b"
        assert affected["last_seq"] == 1

    def test_later_checkpoint_is_earliest_affected_when_it_precedes(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 1},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 2},
        ])
        report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T10:00:10Z", "value": 3},
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T10:00:10Z", "value": 4},
        ])
        make_checkpoint(client, sid, "cp-early")   # offset 2, key b@:00
        report(client, sid, [
            {"source_id": "a", "seq": 3, "observed_at": "2026-01-01T10:00:20Z", "value": 5},
            {"source_id": "b", "seq": 3, "observed_at": "2026-01-01T10:00:20Z", "value": 6},
        ])
        make_checkpoint(client, sid, "cp-late")    # offset 4, key b@:10
        # Quarantine a reading at :05 (inserts at offset 2) and one at :15
        # (inserts at offset 4).
        r = report(client, sid, [
            {"source_id": "a", "seq": 9, "observed_at": "2026-01-01T10:00:05Z", "value": 91},
            {"source_id": "a", "seq": 10, "observed_at": "2026-01-01T10:00:15Z", "value": 92},
        ])
        assert r.json()["quarantined"] == 2
        items = audit(client, sid).json()["readings"]
        by_time = {x["observed_at"]: x for x in items}
        # Insertion at offset 2 is after cp-early's key but NOT after
        # cp-late's key: cp-early is the earliest affected.
        assert by_time["2026-01-01T10:00:05Z"]["affected_checkpoint"]["name"] == "cp-early"
        # Insertion at offset 4 follows both captured keys: still the earliest
        # checkpoint (widest resume window) is reported.
        assert by_time["2026-01-01T10:00:15Z"]["affected_checkpoint"]["name"] == "cp-early"

    def test_checkpoint_at_or_after_insertion_is_not_affected(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 1},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 2},
        ])
        report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T10:00:10Z", "value": 3},
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T10:00:10Z", "value": 4},
        ])
        report(client, sid, [
            {"source_id": "a", "seq": 3, "observed_at": "2026-01-01T10:00:20Z", "value": 5},
            {"source_id": "b", "seq": 3, "observed_at": "2026-01-01T10:00:20Z", "value": 6},
        ])
        # Only checkpoint: key b@:10, offset 4.
        make_checkpoint(client, sid, "cp-late")
        # Quarantined reading at :05 would insert at offset 2, before the
        # checkpoint's captured key (b@:10) -> not in its resume window.
        r = report(client, sid, [
            {"source_id": "a", "seq": 9, "observed_at": "2026-01-01T10:00:05Z", "value": 91}])
        assert r.json()["quarantined"] == 1
        item = audit(client, sid).json()["readings"][0]
        assert item["affected_checkpoint"] is None

    def test_empty_stream_checkpoint_is_affected_by_every_reading(self, client):
        sid = create_session(client)
        # Checkpoint before anything is published: null keyset, offset 0.
        cp = make_checkpoint(client, sid, "cp-empty")
        assert cp.json()["stream_offset"] == 0
        assert cp.json()["last_observed_at"] is None
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T10:00:10Z", "value": 1},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T10:00:10Z", "value": 2},
        ])
        report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T10:00:20Z", "value": 3},
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T10:00:20Z", "value": 4},
        ])
        r = report(client, sid, [
            {"source_id": "a", "seq": 9, "observed_at": "2026-01-01T10:00:05Z", "value": 91}])
        assert r.json()["quarantined"] == 1
        item = audit(client, sid).json()["readings"][0]
        affected = item["affected_checkpoint"]
        assert affected is not None
        assert affected["name"] == "cp-empty"
        assert affected["stream_offset"] == 0
        assert affected["last_observed_at"] is None

    def test_no_checkpoints_means_null_affected(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        for item in audit(client, sid).json()["readings"]:
            assert item["affected_checkpoint"] is None


class TestQuarantineAuditErrors:
    def test_unknown_session_returns_404(self, client):
        r = audit(client, "nope")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"]

    def test_unknown_source_id_returns_400(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        r = audit(client, sid, {"source_id": "zzz"})
        assert r.status_code == 400
        assert "unknown source_id" in r.json()["detail"]

    @pytest.mark.parametrize("bad_from", ["not-a-time", "2026-13-40T99:00:00Z", "   "])
    def test_invalid_from_returns_400(self, client, bad_from):
        sid = create_session(client)
        assert audit(client, sid, {"from": bad_from}).status_code == 400

    @pytest.mark.parametrize("bad_to", ["yesterday", "", "   "])
    def test_invalid_to_returns_400(self, client, bad_to):
        sid = create_session(client)
        assert audit(client, sid, {"to": bad_to}).status_code == 400

    def test_from_later_than_to_returns_400(self, client):
        sid = create_session(client)
        r = audit(client, sid, {
            "from": "2026-01-01T10:00:20Z", "to": "2026-01-01T10:00:10Z"})
        assert r.status_code == 400
        assert "must not be later" in r.json()["detail"]

    @pytest.mark.parametrize("limit", [0, -1, 1001, True])
    def test_invalid_limit_returns_400(self, client, limit):
        sid = create_session(client)
        r = audit(client, sid, {"limit": limit})
        assert r.status_code == 400, r.text

    @pytest.mark.parametrize("limit", [1.5, "abc"])
    def test_non_integer_limit_returns_422(self, client, limit):
        sid = create_session(client)
        assert audit(client, sid, {"limit": limit}).status_code == 422

    def test_blank_source_id_returns_422(self, client):
        sid = create_session(client)
        assert audit(client, sid, {"source_id": "   "}).status_code == 422

    def test_malformed_json_body_returns_400(self, client):
        sid = create_session(client)
        r = client.post(
            f"/sessions/{sid}/quarantine-audit",
            content="{not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400
        assert "valid JSON" in r.json()["detail"]

    def test_json_array_body_returns_400(self, client):
        sid = create_session(client)
        r = client.post(
            f"/sessions/{sid}/quarantine-audit",
            content="[1, 2]",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400

    def test_empty_body_is_accepted(self, client):
        sid = create_session(client)
        r = client.post(f"/sessions/{sid}/quarantine-audit")
        assert r.status_code == 200
        assert r.json()["limit"] == 100


class TestQuarantineAuditReadOnly:
    def test_audit_changes_nothing(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        make_checkpoint(client, sid, "cp-1")

        def snapshot():
            conn = sqlite3.connect(client.db_path)
            conn.row_factory = sqlite3.Row
            try:
                tables = {}
                for table in ("sessions", "session_sources", "readings",
                              "checkpoints", "alert_rules", "alert_events",
                              "correlated_rules", "correlated_conditions",
                              "correlated_events"):
                    tables[table] = [
                        dict(r) for r in conn.execute(
                            f"SELECT * FROM {table} ORDER BY 1,2,3")
                    ]
                return tables
            finally:
                conn.close()

        before = snapshot()
        # Repeated audits with varied filters must be pure reads.
        for body in ({}, {"source_id": "a"},
                     {"from": "2026-01-01T10:00:10Z", "to": "2026-01-01T10:00:20Z"},
                     {"limit": 1}, {"limit": 1000}):
            r = audit(client, sid, body)
            assert r.status_code == 200, r.text
        assert snapshot() == before

    def test_audit_concurrent_with_reports_never_500s(self, client):
        sid = create_session(client)
        setup_scene(client, sid)
        outcomes = {"audits": [], "reports": []}
        barrier = threading.Barrier(2)

        def do_audits():
            barrier.wait()
            for _ in range(30):
                outcomes["audits"].append(
                    audit(client, sid, {"limit": 2}).status_code)

        def do_reports():
            barrier.wait()
            for i in range(30):
                outcomes["reports"].append(report(client, sid, [
                    {"source_id": "a" if i % 2 else "b", "seq": 100 + i,
                     "observed_at": f"2026-01-01T10:03:{i:02d}Z", "value": i},
                ]).status_code)

        t1 = threading.Thread(target=do_audits)
        t2 = threading.Thread(target=do_reports)
        t1.start(); t2.start(); t1.join(); t2.join()

        assert outcomes["audits"] == [200] * 30
        # Reports at :03 with a watermark already at :20 quarantine cleanly.
        assert set(outcomes["reports"]) <= {200}

    def test_openapi_documents_route_and_alias_field(self, client):
        schema = client.get("/openapi.json").json()
        path = schema["paths"]["/sessions/{session_id}/quarantine-audit"]["post"]
        assert path["summary"] == "Quarantine Impact Audit"
        # The response schema serialises the reserved-word bound as "from".
        resp_schema = schema["components"]["schemas"]["QuarantineAuditResponse"]
        assert "from" in resp_schema["properties"]
        assert "from_" not in resp_schema["properties"]
        assert "total_matched" in resp_schema["properties"]
