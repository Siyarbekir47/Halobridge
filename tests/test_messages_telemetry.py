"""Observed Messages timings never invent tokens or replace engine timings."""
import json
import re
import unittest
from unittest.mock import patch

from test_dashboard import database, insert, trace


def event(item, at, payload):
    with patch("halogen_dashboard.time.monotonic", return_value=at):
        item.observe(b"data: " + json.dumps(payload).encode() + b"\n\n", "text/event-stream")


def new_trace():
    with patch("halogen_dashboard.time.monotonic", return_value=100):
        return trace("/v1/messages", True)


def complete(item, at=112, output=30):
    event(item, at, {"type": "message_delta", "usage": {"input_tokens": 5,
        "cache_read_input_tokens": 20, "output_tokens": output}})
    event(item, at, {"type": "message_stop"})
    item.finalize_parsing()


class ObservedMessagesTests(unittest.TestCase):
    def output(self, item, first=110, last=111):
        event(item, first, {"type": "content_block_delta", "delta": {
            "type": "thinking_delta", "thinking": "private thinking"}})
        event(item, last, {"type": "content_block_delta", "delta": {
            "type": "text_delta", "text": "private answer"}})

    def test_measures_first_output_and_stream_separately_from_initial_byte_and_prefill(self):
        item = new_trace()
        event(item, 100.1, {"type": "message_start", "message": {"usage": {"output_tokens": 0}}})
        event(item, 105, {"type": "ping"})
        self.output(item)
        complete(item)
        self.assertAlmostEqual(item.first_byte_mono, 100.1)
        self.assertEqual(item.first_output_ms, 10000)
        self.assertEqual(item.output_stream_ms, 2000)
        self.assertEqual(item.observed_tps, 15)
        self.assertEqual((item.input_tokens, item.output_tokens, item.cached_tokens), (25, 30, 20))
        self.assertIsNone(item.tps)
        self.assertIsNone(item.reasoning_tokens)
        self.assertEqual(item.thinking, "on")

    def test_reported_engine_speed_is_preserved_alongside_observation(self):
        item = new_trace()
        self.output(item)
        event(item, 112, {"timings": {"predicted_per_second": 42.5}})
        complete(item)
        self.assertEqual((item.tps, item.observed_tps), (42.5, 15))

    def test_content_start_and_tool_arguments_are_output_but_signature_and_empty_events_are_not(self):
        item = new_trace()
        event(item, 101, {"type": "content_block_start", "content_block": {"type": "thinking", "thinking": ""}})
        event(item, 102, {"type": "content_block_delta", "delta": {"type": "signature_delta", "signature": "private"}})
        event(item, 103, {"type": "content_block_delta", "delta": {"type": "text_delta", "text": ""}})
        event(item, 110, {"type": "content_block_start", "content_block": {"type": "tool_use", "name": "Read", "input": {}}})
        event(item, 111, {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{}"}})
        complete(item)
        self.assertEqual((item.first_output_ms, item.observed_tps), (10000, 15))
        text = new_trace()
        event(text, 110, {"type": "content_block_start", "content_block": {"type": "text", "text": "Hi"}})
        event(text, 111, {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "!"}})
        complete(text)
        self.assertEqual(text.observed_tps, 15)

    def test_one_buffered_chunk_and_sub_ten_millisecond_spans_have_no_rate(self):
        item = new_trace()
        events = [
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "first"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "last"}},
            {"type": "message_delta", "usage": {"output_tokens": 30}},
            {"type": "message_stop"},
        ]
        with patch("halogen_dashboard.time.monotonic", return_value=110):
            item.observe(b"".join(b"data: " + json.dumps(value).encode() + b"\n\n" for value in events), "text/event-stream")
        item.finalize_parsing()
        self.assertIsNone(item.observed_tps)
        item = new_trace()
        self.output(item, last=110.001)
        complete(item)
        self.assertIsNone(item.observed_tps)

    def test_no_rate_without_final_usage_normal_completion_or_positive_output(self):
        for ending in ("missing_usage", "aborted", "stream_error", "zero_output"):
            with self.subTest(ending=ending):
                item = new_trace()
                event(item, 101, {"type": "message_start", "message": {"usage": {"output_tokens": 0}}})
                self.output(item)
                if ending == "missing_usage":
                    event(item, 112, {"type": "message_stop"})
                    item.finalize_parsing()
                elif ending == "aborted":
                    item.finalize_parsing()
                else:
                    if ending == "stream_error":
                        event(item, 111.5, {"type": "error", "error": {"message": "failed"}})
                    complete(item, output=0 if ending == "zero_output" else 30)
                self.assertIsNone(item.observed_tps)

    def test_interleaved_streams_keep_independent_times_and_counters(self):
        first, second = new_trace(), new_trace()
        self.output(first, first=102, last=103)
        self.output(second, first=104, last=107)
        complete(second, at=108, output=80)
        complete(first, at=110, output=40)
        self.assertEqual((first.observed_tps, second.observed_tps), (5, 20))
        self.assertEqual((first.output_stream_ms, second.output_stream_ms), (8000, 4000))

    def test_malformed_content_events_do_not_interrupt_forwarding_or_later_usage(self):
        item = new_trace()
        for value in (None, [], {}, 1):
            event(item, 101, {"type": "content_block_delta", "delta": {"type": value}})
            event(item, 102, {"type": "content_block_start", "content_block": {"type": value}})
        self.output(item)
        complete(item)
        self.assertEqual(item.observed_tps, 15)

    def test_chunk_split_event_is_observed_when_its_content_becomes_parseable(self):
        item = new_trace()
        raw = b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}\n\n'
        with patch("halogen_dashboard.time.monotonic", return_value=110):
            item.observe(raw[:30], "text/event-stream")
        with patch("halogen_dashboard.time.monotonic", return_value=111):
            item.observe(raw[30:], "text/event-stream")
        event(item, 112, {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "!"}})
        complete(item, at=114)
        # The complete event is recognizable when its remaining bytes arrive.
        self.assertEqual(item.first_output_ms, 11000)
        self.assertEqual(item.observed_tps, 10)


class ObservedPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_observations_survive_storage_without_text_and_failed_requests_have_no_rate(self):
        dashboard = database()
        self.addCleanup(dashboard.db.close)
        for status in (200, 499):
            item = new_trace()
            ObservedMessagesTests().output(item)
            complete(item)
            await dashboard.finish_request(item, status)
        rows = dashboard.db.execute("SELECT first_output_ms,output_stream_ms,observed_tps,tps FROM requests ORDER BY id").fetchall()
        self.assertEqual(tuple(rows[0]), (10000, 2000, 15, None))
        self.assertIsNone(rows[1]["observed_tps"])
        history = dashboard._history(0, 10**12)
        self.assertEqual(history["items"][1]["observed_tps"], 15)
        dump = "\n".join(dashboard.db.iterdump())
        self.assertNotIn("private thinking", dump)
        self.assertNotIn("private answer", dump)

    async def test_existing_database_is_migrated_without_inventing_historical_timings(self):
        dashboard = database()
        self.addCleanup(dashboard.db.close)
        # Recreate the old requests schema while preserving the other tables.
        schema = dashboard.db.execute("SELECT sql FROM sqlite_master WHERE name='requests'").fetchone()[0]
        dashboard.db.execute("DROP TABLE requests")
        for name in ("first_output_ms", "output_stream_ms", "observed_tps"):
            schema = re.sub(r",\s*" + name + r" REAL\b", "", schema)
        dashboard.db.execute(schema)
        insert(dashboard)
        dashboard._initialize_schema()
        dashboard._initialize_schema()
        row = dashboard._history(0, 20000)["items"][0]
        self.assertEqual(row["output_tokens"], 100)
        self.assertIsNone(row["observed_tps"])
        self.assertIsNone(row["first_output_ms"])
