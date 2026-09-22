"""Regression tests: python -m unittest discover -s tests -v"""

import asyncio
import json
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from halogen_dashboard import Dashboard, TELEMETRY_VERSION, Trace, _client_name, _int
import halogen_router


def trace(path="/v1/chat/completions", stream=False):
    request = make_mocked_request("POST", path, headers={"X-Halogen-Client": "Test client"})
    return Trace(request, {"stream": stream}, "qwen3.8-flash")


def database():
    dashboard = Dashboard(None, None, "http://unused", halogen_router.MODELS)
    dashboard.db = sqlite3.connect(":memory:")
    dashboard._initialize_schema()
    return dashboard


def insert(dashboard, **overrides):
    row = dict(
        request_id=f"test-{time.time_ns()}", started_at=1000, completed_at=1010,
        client="Test client", client_ip="local", model="qwen3.8-flash",
        endpoint="/v1/chat/completions", status=200, duration_ms=10000,
        ttft_ms=100, stream=1, thinking="unknown", reasoning_effort="default",
        input_tokens=1000, output_tokens=100, cached_tokens=800, reasoning_tokens=20,
        response_bytes=10, telemetry_version=TELEMETRY_VERSION,
    )
    row.update(overrides)
    dashboard.db.execute(
        f"INSERT INTO requests ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
        list(row.values()),
    )
    dashboard.db.commit()
    return row


class TraceTests(unittest.TestCase):
    def test_zero_is_a_measurement_and_timings_do_not_override_usage(self):
        t = trace()
        t.observe(json.dumps({
            "usage": {"prompt_tokens": 1000, "completion_tokens": 0,
                      "prompt_tokens_details": {"cached_tokens": 800},
                      "completion_tokens_details": {"reasoning_tokens": 0}},
            "timings": {"prompt_n": 200, "predicted_n": 1, "cache_n": 799, "predicted_per_second": 0},
        }).encode(), "application/json")
        t.finalize_parsing()
        self.assertEqual((t.input_tokens, t.output_tokens, t.cached_tokens, t.reasoning_tokens, t.tps), (1000, 0, 800, 0, 0))

    def test_responses_stream_nested_usage_split_at_every_byte(self):
        t = trace("/v1/responses", True)
        event = {"type": "response.completed", "response": {"usage": {
            "input_tokens": 45, "output_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 8},
        }}}
        raw = b"event: response.completed\r\ndata: " + json.dumps(event).encode() + b"\r\n\r\ndata: [DONE]\r\n\r\n"
        for byte in raw:
            t.observe(bytes([byte]), "text/event-stream; charset=utf-8")
        t.finalize_parsing()
        self.assertEqual((t.input_tokens, t.output_tokens, t.cached_tokens, t.reasoning_tokens), (45, 12, 0, 8))
        self.assertEqual(t.thinking, "on")
        self.assertFalse(t._line_buffer)
        self.assertFalse(t._event_data)

    def test_chat_stream_usage_is_cumulative_not_added(self):
        t = trace(stream=True)
        for count in (5, 10, 10, 0):
            event = {"usage": {"prompt_tokens": 12, "completion_tokens": count}}
            t.observe(b"data: " + json.dumps(event).encode() + b"\n\n", "text/event-stream")
        t.finalize_parsing()
        self.assertEqual((t.input_tokens, t.output_tokens), (12, 0))

    def test_multiline_sse_and_unterminated_final_event(self):
        t = trace(stream=True)
        t.observe(b': keepalive\n\ndata: {"usage":\ndata: {"prompt_tokens": 4, "completion_tokens": 2}}', "text/event-stream")
        t.finalize_parsing()
        self.assertEqual((t.input_tokens, t.output_tokens), (4, 2))

    def test_malformed_event_does_not_hide_later_usage(self):
        t = trace(stream=True)
        t.observe(b'data: not json\n\ndata: {"usage":{"input_tokens":5,"output_tokens":7}}\n\n', "text/event-stream")
        t.finalize_parsing()
        self.assertEqual((t.input_tokens, t.output_tokens), (5, 7))

    def test_missing_usage_is_not_zero_or_a_token_estimate(self):
        t = trace()
        t.observe(b'{"choices":[{"message":{"content":"private text"}}],"timings":{"prompt_n":20,"predicted_n":9}}', "application/json")
        t.finalize_parsing()
        self.assertIsNone(t.input_tokens)
        self.assertIsNone(t.output_tokens)
        self.assertIsNone(t.cached_tokens)
        self.assertEqual(t.thinking, "unknown")

    def test_bad_counters_are_rejected(self):
        for value in (True, False, -1, 1.5, float("nan"), float("inf"), "10", 2**63, 10**400):
            with self.subTest(value=str(value)[:30]):
                self.assertIsNone(_int(value))
        self.assertEqual(_int(0), 0)
        self.assertEqual(_int(12.0), 12)

    def test_impossible_subtotals_are_unknown(self):
        t = trace()
        t._extract({"usage": {"input_tokens": 10, "output_tokens": 1,
                             "input_tokens_details": {"cached_tokens": 11},
                             "output_tokens_details": {"reasoning_tokens": 2}}})
        t.finalize_parsing()
        self.assertIsNone(t.cached_tokens)
        self.assertIsNone(t.reasoning_tokens)

    def test_oversized_json_is_discarded_and_cleared(self):
        t = trace()
        t.observe(b"x" * (4 * 1024 * 1024 + 1), "application/json")
        t.observe(b'{"usage":{"input_tokens":100}}', "application/json")
        t.finalize_parsing()
        self.assertIsNone(t.input_tokens)
        self.assertFalse(t._json_buffer)

    def test_generic_sdk_is_not_misidentified_as_openclaw(self):
        request = make_mocked_request("POST", "/", headers={"User-Agent": "OpenAI/JS 4.0"})
        self.assertEqual(_client_name(request), "OpenAI/JS")


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.dashboard = database()

    def tearDown(self):
        self.dashboard.db.close()

    def test_missing_and_legacy_values_are_not_counted_as_zero(self):
        insert(self.dashboard)
        insert(self.dashboard, input_tokens=500, output_tokens=50, cached_tokens=None, reasoning_tokens=None)
        insert(self.dashboard, status=499, input_tokens=None, output_tokens=None, cached_tokens=None, reasoning_tokens=None)
        insert(self.dashboard, telemetry_version=0, input_tokens=9000, output_tokens=9000)
        insert(self.dashboard, endpoint="/health", input_tokens=8000)
        s = self.dashboard._summary(0, 20000)
        self.assertEqual((s["requests"], s["successes"], s["errors"]), (4, 3, 1))
        self.assertEqual((s["input_tokens"], s["output_tokens"], s["total_tokens"]), (1500, 150, 1650))
        self.assertEqual((s["usage_reported"], s["cache_reported"], s["legacy_requests"]), (2, 1, 1))
        self.assertEqual(s["cache_ratio"], .8)
        self.assertEqual(s["new_input_tokens"], 200)
        self.assertEqual(s["cached_tokens"], 800)
        self.assertEqual(s["reasoning_tokens"], 20)

    def test_zero_counters_and_no_observations_remain_distinct(self):
        empty = self.dashboard._summary(0, 20000)
        self.assertEqual(empty["requests"], 0)
        self.assertIsNone(empty["input_tokens"])
        self.assertIsNone(empty["cache_ratio"])
        insert(self.dashboard, input_tokens=0, output_tokens=0, cached_tokens=0, reasoning_tokens=0)
        s = self.dashboard._summary(0, 20000)
        self.assertEqual((s["input_tokens"], s["output_tokens"], s["total_tokens"]), (0, 0, 0))
        self.assertEqual(s["usage_reported"], 1)
        self.assertIsNone(s["cache_ratio"])

    def test_chart_breakdowns_and_summary_have_identical_population(self):
        # Modern base timestamps: 1970-era values diverge between the OS
        # historical timezone database and the current fixed UTC offset.
        base = 1_700_000_000
        insert(self.dashboard, completed_at=base + 100)
        insert(self.dashboard, completed_at=base + 9810, input_tokens=None, output_tokens=None)
        insert(self.dashboard, completed_at=base + 9811, input_tokens=200, output_tokens=20, cached_tokens=0)
        summary = self.dashboard._summary(base, base + 14000)
        bucket, unit, timeline = self.dashboard._timeline(base, base + 14000, "custom")
        self.assertEqual(bucket, 3600)
        self.assertEqual(unit, "hour")
        self.assertEqual(len(timeline), 5)
        # The bucket holding the unreported pair must be marked partial;
        # index-independent because buckets align to the local clock.
        containing = [item for item in timeline
                      if item["bucket_start"] <= base + 9810 < item["bucket_end"]]
        self.assertEqual(len(containing), 1)
        self.assertTrue(containing[0]["partial"])
        self.assertTrue(any(item["requests"] == 0 and item["input_tokens"] == 0 for item in timeline))
        for column in ("input_tokens", "output_tokens", "requests"):
            self.assertEqual(sum(r[column] or 0 for r in timeline), summary[column])
            for group in ("client", "model"):
                self.assertEqual(sum(r[column] or 0 for r in self.dashboard._breakdown(base, base + 14000, group)), summary[column])

    def test_engine_does_not_replace_router_usage_or_match_by_time(self):
        insert(self.dashboard)
        self.dashboard.db.execute("""INSERT INTO engine_requests
            (cursor, completed_at, model, mode, output_tokens, generation_s, prompt_tokens)
            VALUES ('engine',1010,'qwen3.8-flash','mtp',9000,10,90000)""")
        self.assertEqual(self.dashboard._summary(0, 20000)["input_tokens"], 1000)
        history = self.dashboard._history(0, 20000)["items"][0]
        self.assertEqual(history["output_tokens"], 100)
        self.assertNotIn("engine_found", history)

    def test_engine_rates_are_weighted_by_time_and_rounds(self):
        self.dashboard.db.executemany("""INSERT INTO engine_requests
            (cursor,completed_at,model,mode,output_tokens,generation_s,rounds,commit_per_round)
            VALUES (?,1010,'model','mtp',?,?,?,?)""", [("a", 100, 10, 10, 2), ("b", 90, 1, 90, 4)])
        s = self.dashboard._engine_summary(0, 20000)
        self.assertAlmostEqual(s["weighted_tps"], 190 / 11)
        self.assertAlmostEqual(s["avg_commit_per_round"], 3.8)

    def test_history_pagination_is_deterministic_for_equal_timestamps(self):
        for i in range(30):
            insert(self.dashboard, request_id=str(i))
        first = self.dashboard._history(0, 20000, 1)
        second = self.dashboard._history(0, 20000, 2)
        third = self.dashboard._history(0, 20000, 3)
        ids = [r["request_id"] for r in first["items"] + second["items"] + third["items"]]
        self.assertEqual(ids, [str(i) for i in reversed(range(30))])
        self.assertEqual(first["pages"], 3)
        self.assertEqual(self.dashboard._history(0, 20000, 999)["page"], 3)

    def test_custom_ranges_reject_nonfinite_and_reversed_values(self):
        for start, end in (("nan", "100"), ("0", "inf"), ("1", "0")):
            request = make_mocked_request("GET", f"/dashboard/api/analytics?period=custom&from={start}&to={end}")
            with self.assertRaises(web.HTTPBadRequest):
                self.dashboard.analytics(request)

    def test_schema_upgrade_is_idempotent_and_preserves_legacy_rows(self):
        self.dashboard.db.execute("ALTER TABLE requests DROP COLUMN telemetry_version")
        self.dashboard._initialize_schema()
        insert(self.dashboard, telemetry_version=0)
        self.dashboard._initialize_schema()
        self.assertEqual(self.dashboard.db.execute("SELECT telemetry_version FROM requests").fetchone()[0], 0)
        self.assertEqual(self.dashboard._summary(0, 20000)["legacy_requests"], 1)

    def test_engine_parse_missing_prompt_remains_unknown(self):
        parsed = self.dashboard._parse_engine_line("serve_api: mtp 50 tok in 2.0s = 25.0 t/s")
        self.assertIsNone(parsed.get("new_tokens"))
        self.assertIsNone(parsed.get("prefill_tps"))

    def test_journal_cursor_is_deduplicated(self):
        item = {"MESSAGE": "serve_api: mtp 50 tok in 2.0s = 25.0 t/s | prompt 100 (80 cached, 80.0%), prefill 0.2s (20 new)",
                "_SYSTEMD_USER_UNIT": "halogen-official.service", "__CURSOR": "one",
                "__REALTIME_TIMESTAMP": str(int(time.time() * 1_000_000))}
        self.dashboard._store_engine_line(item)
        self.dashboard._store_engine_line(item)
        row = self.dashboard.db.execute("SELECT COUNT(*), new_tokens, prefill_tps FROM engine_requests").fetchone()
        self.assertEqual(tuple(row), (1, 20, 100))


class FinishTests(unittest.IsolatedAsyncioTestCase):
    async def test_buffered_response_does_not_create_fake_tps(self):
        dashboard = database()
        t = trace()
        t.observe(b'{"usage":{"prompt_tokens":100,"completion_tokens":1000}}', "application/json")
        await dashboard.finish_request(t, 200)
        row = dashboard.db.execute("SELECT tps, ttft_ms, telemetry_version FROM requests").fetchone()
        self.assertIsNone(row["tps"])
        self.assertIsNotNone(row["ttft_ms"])
        self.assertEqual(row["telemetry_version"], 2)
        await dashboard.close()

    async def test_response_content_is_not_persisted(self):
        dashboard = database()
        t = trace()
        t.observe(b'{"choices":[{"message":{"content":"PRIVATE-CONTENT"}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}', "application/json")
        await dashboard.finish_request(t, 200)
        dump = "\n".join(dashboard.db.iterdump())
        self.assertNotIn("PRIVATE-CONTENT", dump)
        self.assertFalse(t._json_buffer)
        await dashboard.close()


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.received = []

        async def backend(request):
            payload = await request.json() if request.method == "POST" else {}
            self.received.append((payload, dict(request.headers)))
            usage = {"prompt_tokens": 17, "completion_tokens": 3}
            if payload.get("stream"):
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                await response.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
                if payload.get("stream_options", {}).get("include_usage"):
                    await response.write(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n")
                await response.write(b"data: [DONE]\n\n")
                return response
            return web.json_response({"usage": usage})

        backend_app = web.Application()
        backend_app.router.add_route("*", "/{tail:.*}", backend)
        self.backend = TestServer(backend_app)
        await self.backend.start_server()
        self.url_patch = patch.object(halogen_router, "BACKEND_URL", str(self.backend.make_url("")).rstrip("/"))
        self.url_patch.start()
        self.session = ClientSession(auto_decompress=False)
        self.manager = SimpleNamespace(reserve_request=AsyncMock(return_value="qwen3.8-flash"), release_request=AsyncMock())
        self.dashboard = database()
        app = web.Application()
        app["session"], app["manager"], app["dashboard"] = self.session, self.manager, self.dashboard
        self.dashboard.register_routes(app)
        app.router.add_route("*", "/{tail:.*}", halogen_router.proxy_request)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.session.close()
        await self.backend.close()
        await self.dashboard.close()
        self.url_patch.stop()

    async def test_stream_requests_usage_without_changing_other_options(self):
        response = await self.client.post("/v1/chat/completions", json={
            "stream": True, "model": "qwen3.8-flash", "messages": [], "stream_options": {"custom": 1},
        })
        text = await response.text()
        self.assertEqual(response.status, 200)
        self.assertIn("[DONE]", text)
        self.assertEqual(self.received[0][0]["stream_options"], {"custom": 1, "include_usage": True})
        self.assertEqual(self.received[0][1]["Accept-Encoding"], "identity")
        row = self.dashboard.db.execute("SELECT input_tokens, output_tokens FROM requests").fetchone()
        self.assertEqual(tuple(row), (17, 3))
        self.manager.release_request.assert_awaited_once()

    async def test_explicit_usage_opt_out_is_respected_and_unknown(self):
        response = await self.client.post("/v1/chat/completions", json={"stream": True, "stream_options": {"include_usage": False}})
        await response.read()
        row = self.dashboard.db.execute("SELECT input_tokens, output_tokens FROM requests").fetchone()
        self.assertEqual(tuple(row), (None, None))

    async def test_health_requests_are_not_counted_as_inference(self):
        response = await self.client.get("/health")
        await response.read()
        self.assertEqual(self.dashboard.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0], 0)

    async def test_telemetry_failure_still_releases_model_reservation(self):
        self.dashboard.finish_request = AsyncMock(side_effect=RuntimeError("test telemetry error"))
        response = await self.client.post("/v1/chat/completions", json={})
        await response.read()
        self.manager.release_request.assert_awaited_once()

    async def test_dashboard_page_and_analytics_routes(self):
        response = await self.client.get("/dashboard")
        html = await response.text()
        self.assertIn('id="tokenChart"', html)
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        response = await self.client.get("/dashboard/api/analytics?period=24h")
        data = await response.json()
        self.assertEqual(data["summary"]["requests"], 0)
        self.assertIsNone(data["summary"]["input_tokens"])
        self.assertGreater(len(data["timeline"]), 0)


if __name__ == "__main__":
    unittest.main()
