"""Claude Code model names use the active backend without changing the wire."""
import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from halogen_router import ModelManager, proxy_request
from test_dashboard import database


OFFICIAL = "qwen3.8-flash"
SWIFT = "halogen-swift15"
REPLY = b'{"type":"message","model":"qwen3.8-flash","content":[{"type":"text","text":"OK"}],"usage":{"input_tokens":58,"output_tokens":29}}'
EVENTS = (b'event: message_start\ndata: {"type":"message_start","message":{"model":"qwen3.8-flash","usage":{"input_tokens":58,"output_tokens":0}}}\n\n'
          b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":"signed-thinking"}}\n\n'
          b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":29}}\n\n'
          b'event: message_stop\ndata: {"type":"message_stop"}\n\n')


class MessagesRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.received = []
        self.entered, self.release = asyncio.Event(), asyncio.Event()

        async def backend(request):
            if request.path == "/v1/models":
                return web.json_response({"data": [{"id": self.manager.current_model},
                    {"id": "qwen3.5-2b", "task": "generate"}]})
            raw = await request.read()
            self.received.append((str(request.rel_url), raw, request.headers))
            payload = json.loads(raw)
            if payload.get("hold"):
                self.entered.set()
                await self.release.wait()
            if payload.get("fail"):
                return web.Response(status=400, body=b'{"type":"error","error":{"type":"invalid_request_error","message":"backend validation"}}',
                                    content_type="application/json", headers={"X-Backend-Error": "preserved"})
            if payload.get("timed"):
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                for chunk in self.timed_chunks:
                    await response.write(chunk)
                    await asyncio.sleep(0.03)
                await response.write_eof()
                return response
            if request.path.endswith("/count_tokens"):
                return web.json_response({"input_tokens": 58})
            return web.Response(body=EVENTS if payload.get("stream") else REPLY,
                                content_type="text/event-stream" if payload.get("stream") else "application/json")

        backend_app = web.Application()
        backend_app.router.add_route("*", "/{tail:.*}", backend)
        self.backend = TestServer(backend_app)
        await self.backend.start_server()
        self.addAsyncCleanup(self.backend.close)
        self.session = ClientSession()
        self.addAsyncCleanup(self.session.close)
        self.manager = ModelManager(self.session, {OFFICIAL: "official.service", SWIFT: "swift.service"})
        self.manager.current_model = OFFICIAL
        self.manager.backend_url = str(self.backend.make_url("")).rstrip("/")
        self.manager.npu_names = lambda _: {"qwen3.5-2b"}
        self.manager.perform_switch = AsyncMock()
        self.dashboard = database()
        self.addAsyncCleanup(self.dashboard.close)
        app = web.Application()
        app.update(manager=self.manager, session=self.session, dashboard=self.dashboard,
                   backend_url=self.manager.backend_url)
        app.router.add_route("*", "/{tail:.*}", proxy_request)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_claude_names_forward_exact_messages_and_count_tokens(self):
        for model in ("claude-sonnet-4-5", "claude-haiku-4-5-20251001", "claude-opus-4-1", "custom-client-name"):
            for path in ("/v1/messages", "/v1/messages/count_tokens"):
                for stream in (False, True):
                    with self.subTest(model=model, path=path, stream=stream):
                        payload = {"model": model, "max_tokens": 100, "stream": stream,
                            "system": [{"type": "text", "text": "Keep this system prompt."}],
                            "messages": [{"role": "user", "content": [
                                {"type": "text", "text": "Antworte nur mit OK"},
                                {"type": "tool_reference", "tool_name": "read_file"}]}],
                            "tools": [{"name": "read_file", "defer_loading": True,
                                       "input_schema": {"type": "object"}}],
                            "thinking": {"type": "adaptive"}, "metadata": {"user_id": "test-session"}}
                        raw = json.dumps(payload, indent=2).encode()
                        headers = {"Content-Type": "application/json", "x-api-key": "anything",
                            "anthropic-version": "2023-06-01", "anthropic-beta": "test-beta",
                            "x-claude-code-session-id": "test-session", "X-Halogen-Priority": "1"}
                        response = await self.client.post(path + "?beta=true", data=raw, headers=headers)
                        self.assertEqual(response.status, 200, await response.text())
                        if path.endswith("count_tokens"):
                            self.assertEqual(await response.json(), {"input_tokens": 58})
                        else:
                            self.assertEqual(await response.read(), EVENTS if stream else REPLY)
                        self.assertEqual(self.received[-1][:2], (path + "?beta=true", raw))
                        for key, value in headers.items():
                            self.assertEqual(self.received[-1][2][key], value)
                        self.assertEqual(self.manager.active_requests, 0)
        rows = self.dashboard.db.execute("SELECT model,input_tokens,output_tokens FROM requests").fetchall()
        self.assertEqual(len(rows), 8)  # Token-count calls do not count as generations.
        self.assertTrue(all(tuple(row) == (OFFICIAL, 58, 29) for row in rows))
        self.manager.perform_switch.assert_not_awaited()

    async def test_installed_gpu_id_still_selects_its_profile(self):
        response = await self.client.post("/v1/messages", json={"model": SWIFT})
        self.assertEqual(response.status, 200, await response.text())
        self.manager.perform_switch.assert_awaited_once_with(SWIFT)
        self.assertEqual(self.manager.current_model, SWIFT)

    async def test_declared_npu_model_keeps_its_existing_admission(self):
        response = await self.client.post("/v1/messages", json={"model": "qwen3.5-2b"})
        self.assertEqual(response.status, 200, await response.text())
        row = self.dashboard.db.execute("SELECT model FROM requests").fetchone()
        self.assertEqual(row[0], "qwen3.5-2b")
        self.manager.perform_switch.assert_not_awaited()

    async def test_unknown_models_on_other_endpoints_still_fail(self):
        for path in ("/v1/chat/completions", "/v1/responses", "/v1/embeddings", "/v1/messages/unknown"):
            response = await self.client.post(path, json={"model": "claude-sonnet-4-5"})
            self.assertEqual(response.status, 404, await response.text())
        self.assertEqual(self.received, [])
        self.manager.perform_switch.assert_not_awaited()

    async def test_alias_waits_for_switch_and_uses_new_active_model(self):
        self.manager.switching = True
        self.manager.switch_target = SWIFT
        pending = asyncio.create_task(self.client.post("/v1/messages", json={"model": "claude-sonnet-4-5"}))
        await asyncio.sleep(0.02)
        was_waiting = not pending.done() and not self.received
        async with self.manager.condition:
            self.manager.current_model = SWIFT
            self.manager.switching = False
            self.manager.switch_target = None
            self.manager.condition.notify_all()
        response = await asyncio.wait_for(pending, 2)
        self.assertTrue(was_waiting)
        self.assertEqual(response.status, 200, await response.text())
        self.manager.perform_switch.assert_not_awaited()
        self.assertEqual(self.dashboard.db.execute("SELECT model FROM requests").fetchone()[0], SWIFT)

    async def test_alias_does_not_bypass_maintenance_or_start_a_missing_backend(self):
        for state, value in (("maintenance", True), ("current_model", None)):
            with self.subTest(state=state):
                previous = getattr(self.manager, state)
                setattr(self.manager, state, value)
                response = await self.client.post("/v1/messages", json={"model": "claude-sonnet-4-5"})
                self.assertEqual(response.status, 503, await response.text())
                self.assertIn("Retry-After", response.headers)
                setattr(self.manager, state, previous)
        self.assertEqual(self.received, [])
        self.manager.perform_switch.assert_not_awaited()

    async def test_alias_is_included_in_request_draining(self):
        pending = asyncio.create_task(self.client.post("/v1/messages", json={"model": "claude-sonnet-4-5", "hold": True}))
        try:
            await asyncio.wait_for(self.entered.wait(), 2)
            self.assertEqual(self.manager.active_requests, 1)
        finally:
            self.release.set()
        response = await asyncio.wait_for(pending, 2)
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(self.manager.active_requests, 0)

    async def test_backend_errors_remain_unchanged(self):
        response = await self.client.post("/v1/messages", json={"model": "claude-sonnet-4-5", "fail": True})
        self.assertEqual(response.status, 400)
        self.assertEqual(response.headers["X-Backend-Error"], "preserved")
        self.assertEqual((await response.json())["error"]["message"], "backend validation")
        self.assertEqual(self.manager.active_requests, 0)

    async def test_real_stream_records_observed_speed_without_modifying_wire_or_engine_counters(self):
        self.timed_chunks = [
            b'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"Think"}}\n\n',
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"OK"}}\n\n',
            b'data: {"type":"message_delta","usage":{"input_tokens":58,"output_tokens":29}}\n\ndata: {"type":"message_stop"}\n\n',
        ]
        payload = {"model": "claude-sonnet-4-5", "stream": True, "timed": True}
        response = await self.client.post("/v1/messages", json=payload)
        self.assertEqual(await response.read(), b"".join(self.timed_chunks))
        self.assertEqual(self.received[-1][1], json.dumps(payload).encode())
        row = self.dashboard.db.execute("SELECT * FROM requests").fetchone()
        self.assertGreater(row["observed_tps"], 0)
        self.assertGreaterEqual(row["output_stream_ms"], 10)
        self.assertGreaterEqual(row["first_output_ms"], row["ttft_ms"])
        self.assertEqual((row["model"], row["input_tokens"], row["output_tokens"]), (OFFICIAL, 58, 29))
        self.assertIsNone(row["tps"])
        self.assertIsNone(row["reasoning_tokens"])
        self.assertEqual(self.manager.active_requests, 0)
