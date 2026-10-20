"""Model discovery compatibility, including terminal-bench-mini's doctor."""
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer

from halogen_router import ModelManager, list_models, resolve_models
from settings import Config


MODEL = "halogen-qwen3.8-flash-next"


class ModelMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.entry = {"id": MODEL, "object": "model", "owned_by": "halogen",
                      "meta": {"n_ctx": 262144, "n_vocab": 248320},
                      "capabilities": {"vision": True}}
        self.entries = [self.entry]
        self.status = 200
        self.probes = 0
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()

        async def backend(request):
            if request.path == "/health":
                return web.json_response({"status": "ok", "model": MODEL})
            self.probes += 1
            self.entered.set()
            await self.release.wait()
            return web.json_response({"data": self.entries}, status=self.status)

        app = web.Application()
        app.router.add_get("/v1/models", backend)
        app.router.add_get("/health", backend)
        self.backend = TestServer(app)
        await self.backend.start_server()
        self.addAsyncCleanup(self.backend.close)
        self.session = ClientSession()
        self.addAsyncCleanup(self.session.close)
        # The issue's explicit manual.service setup: no Quadlets or deployment.
        self.config = Config.from_dict({
            "router": {"backend_url": str(self.backend.make_url("")).rstrip("/")},
            "models": {"auto_discover": False,
                       "quadlet_dir": str(Path(self.tmp.name) / "absent"),
                       "explicit": {MODEL: "manual.service"}},
            "deploy": {"enabled": False}, "updates": {"enabled": False},
        })
        specs, skipped, models = resolve_models(self.config)
        self.assertEqual((specs, skipped), ({}, []))
        self.manager = ModelManager(self.session, models, backend_url=self.config.router.backend_url)
        self.manager.run_command = AsyncMock(side_effect=AssertionError("Discovery must not run systemctl"))
        await self.manager.initialize()
        self.manager.npu_names = lambda _: set()
        app = web.Application()
        app["manager"] = self.manager
        app["models"] = {"stale-model": "old.service"}
        app.router.add_get("/v1/models", list_models)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def models(self):
        response = await self.client.get("/v1/models")
        self.assertEqual(response.status, 200, await response.text())
        return await response.json()

    async def test_manual_service_preserves_backend_context_and_metadata(self):
        payload = await self.models()
        self.assertEqual(len(payload["data"]), 1)
        entry = payload["data"][0]
        self.assertEqual(entry["meta"]["n_ctx"], 262144)
        for key, value in self.entry.items():
            self.assertEqual(entry[key], value)
        self.assertEqual(entry["type"], "model")
        self.assertIn("created_at", entry)
        self.assertFalse(payload["has_more"])
        self.assertEqual((payload["first_id"], payload["last_id"]), (MODEL, MODEL))
        self.assertEqual(self.probes, 1)
        self.manager.run_command.assert_not_awaited()

    async def test_context_field_variants_and_changed_runtime_limit(self):
        self.entry.pop("meta")
        for field in ("n_ctx", "context_length", "max_model_len"):
            with self.subTest(field=field):
                self.entry[field] = 131072
                entry = (await self.models())["data"][0]
                self.assertEqual(entry[field], 131072)
                self.entry.pop(field)

    async def test_active_and_npu_metadata_share_one_probe(self):
        auxiliary = {"id": "decider-4b", "task": "decision", "context_length": 32768}
        self.entries += [auxiliary, {"id": "unexpected", "meta": {"n_ctx": 1}}]
        self.manager.npu_names = lambda _: {"decider-4b"}
        entries = (await self.models())["data"]
        self.assertEqual(entries[0]["meta"], self.entry["meta"])
        self.assertEqual(entries[1:], [auxiliary])
        self.assertEqual(self.probes, 1)

    async def test_inactive_profiles_never_inherit_the_active_context(self):
        self.manager.models = {**self.manager.models, "other-model": "other.service"}
        self.entries.append({"id": "other-model", "meta": {"n_ctx": 123}})
        entries = (await self.models())["data"]
        self.assertEqual([e["id"] for e in entries], [MODEL, "other-model"])
        self.assertEqual(entries[0]["meta"]["n_ctx"], 262144)
        self.assertNotIn("meta", entries[1])

    async def test_unavailable_backend_keeps_configured_models_without_fake_context(self):
        self.status = 503
        entry = (await self.models())["data"][0]
        self.assertEqual(entry["id"], MODEL)
        self.assertNotIn("meta", entry)
        await self.backend.close()
        entries = (await self.models())["data"]
        self.assertEqual([e["id"] for e in entries], [MODEL])
        self.assertNotIn("meta", entries[0])

    async def test_mismatched_backend_is_not_used(self):
        self.entries = [{"id": "unexpected", "meta": {"n_ctx": 4096}},
                        {"id": "decider-4b", "task": "decision"}]
        self.manager.npu_names = lambda _: {"decider-4b"}
        entries = (await self.models())["data"]
        self.assertEqual([e["id"] for e in entries], [MODEL])
        self.assertNotIn("meta", entries[0])

    async def test_malformed_backend_lists_keep_configured_models(self):
        for value in (None, 123, "invalid", {"id": MODEL}, [None, "invalid", {"id": 123}]):
            with self.subTest(data=value):
                self.entries = value
                entries = (await self.models())["data"]
                self.assertEqual([e["id"] for e in entries], [MODEL])
                self.assertNotIn("meta", entries[0])

    async def test_no_probe_or_runtime_metadata_during_switch_or_maintenance(self):
        for state in ("switching", "maintenance"):
            with self.subTest(state=state):
                setattr(self.manager, state, True)
                entries = (await self.models())["data"]
                self.assertEqual([e["id"] for e in entries], [MODEL])
                self.assertNotIn("meta", entries[0])
                setattr(self.manager, state, False)
        self.assertEqual(self.probes, 0)

    async def test_switch_during_probe_discards_old_metadata_and_npu_models(self):
        self.manager.models["other-model"] = "other.service"
        self.entries.append({"id": "decider-4b", "task": "decision"})
        self.manager.npu_names = lambda _: {"decider-4b"}
        for state in ("current_model", "switching", "maintenance"):
            with self.subTest(state=state):
                self.manager.current_model = MODEL
                self.manager.switching = self.manager.maintenance = False
                self.entered.clear()
                self.release.clear()
                pending = asyncio.create_task(self.models())
                try:
                    await asyncio.wait_for(self.entered.wait(), 2)
                    setattr(self.manager, state, "other-model" if state == "current_model" else True)
                finally:
                    self.release.set()
                entries = (await pending)["data"]
                self.assertEqual([e["id"] for e in entries], [MODEL, "other-model"])
                self.assertTrue(all("meta" not in e for e in entries))
