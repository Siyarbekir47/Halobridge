"""NPU preparation uses real files/hashes; hardware/systemd are simulated."""
import asyncio
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from npu import (NPU_MODELS, NpuAssets, NpuManifest, bundled_manifest, host_status,
                 parse_models, patch_quadlet, system_mount_allowed, xrt_mounts)
from profiles import official_template, parse_quadlet, render_quadlet, validate_profile
from shared_assets import AssetError, SharedAssets
from halogen_router import ModelManager, list_models, proxy_request
from halogen_deploy import DeployError, DeployManager, DeployRoutes
from test_deploy import make_config
from test_dashboard import database
import test_updates as updater_tests

READY = {"ready": True, "errors": [], "xrt_mounts": []}
IMAGE = "ghcr.io/peonist-ai/halogen-flash-server:0.16.2"


def fixture_manifest(program=b"new program"):
    lines, contents = [], {}
    for model, task in NPU_MODELS.items():
        owner = "qwen3-embedding-0.6b" if task in {"score", "classify"} else model
        lines.append(f"model {model} repo=peonist-ai/{model} revision={'a' * 40} task={task}"
                     + (f" devices={owner}" if owner != model else ""))
        paths = {f"{model}.hnpw": (model + " weights").encode(),
                 "tokenizer/tokenizer.json": (model + " tokenizer").encode()}
        if owner == model:
            paths.update({"devices/devices.hnpm": b"device record", "devices/u0.elf": program})
        if task == "image":
            paths["devices/taef2.safetensors"] = b"image decoder"
        for name, data in paths.items():
            digest = hashlib.sha256(data).hexdigest()
            lines.append(f"file {model} {name} {len(data)} {digest}")
            contents[digest] = data
    return NpuManifest("\n".join(lines) + "\n"), contents


class ManifestTests(unittest.TestCase):
    def test_real_image_manifest_lists_all_six_and_shares_dense_program(self):
        manifest = bundled_manifest()
        self.assertEqual(set(manifest.models), set(NPU_MODELS))
        assets = manifest.select(["qwen3-reranker-0.6b", "qwen3guard-gen-0.6b"])
        paths = {(a.model, a.name) for a in assets}
        self.assertNotIn(("qwen3-embedding-0.6b", "qwen3-embedding-0.6b.hnpw"), paths)
        self.assertIn(("qwen3-embedding-0.6b", "devices/devices.hnpm"), paths)
        self.assertEqual(len(paths), len(assets))
        flux = manifest.select(["flux2-klein-4b"])
        self.assertEqual({a.model for a in flux}, {"flux2-klein-4b"})
        self.assertIn("devices/taef2.safetensors", {a.name for a in flux})
        self.assertEqual(sum(a.size for a in flux), 8_072_427_547)
        self.assertEqual(set(bundled_manifest("0.16.2").models), set(NPU_MODELS) - {"flux2-klein-4b"})

    def test_custom_programs_exclude_image_and_generation_models(self):
        manifest = bundled_manifest()
        with self.assertRaises(AssetError):
            manifest.select(["/models/custom"], ["flux2-klein-4b"])
        without_decoder = "\n".join(line for line in manifest.text.splitlines() if "taef2.safetensors" not in line)
        with self.assertRaisesRegex(AssetError, "Incomplete NPU manifest"):
            NpuManifest(without_decoder)

    def test_model_list_accepts_custom_paths_and_rejects_unsafe_or_duplicate_ids(self):
        self.assertEqual(parse_models("qwen3.5-2b,/models/my-guard"), ["qwen3.5-2b", "/models/my-guard"])
        for value in ("", "unknown", "qwen3.5-2b,", "qwen3.5-2b, qwen3-embedding-0.6b",
                      "/models/../outside", "/models/npu/custom", "/models/qwen3.5-2b",
                      "/models/a/guard,/models/b/guard", "qwen3.5-2b;reboot", "/etc/guard"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_models(value)

    def test_manifest_rejects_traversal_bad_hashes_unpinned_revisions_and_missing_programs(self):
        manifest, _ = fixture_manifest()
        bad = [manifest.text.replace("devices/u0.elf", "../u0.elf", 1),
               manifest.text.replace("revision=" + "a" * 40, "revision=main", 1),
               "\n".join(line for line in manifest.text.splitlines() if "devices/u0.elf" not in line),
               manifest.text + next(line for line in manifest.text.splitlines() if line.startswith("file"))]
        for text in bad:
            with self.assertRaises(AssetError):
                NpuManifest(text)

    def test_profile_roundtrip_supports_npu_and_new_flags(self):
        profile = official_template("0.16.2", Path("/models"), Path("/cache"))
        profile.env.update(HALOGEN_NPU_MODELS="qwen3.5-2b,qwen3guard-gen-0.6b", HALOGEN_CACHE_EVICT="0",
                           HALOGEN_NPU_QUEUE="128", HALOGEN_NPU_EMB_BATCH="0", HALOGEN_MTP="0",
                           HALOGEN_ADMISSION_RESERVE="1", HALOGEN_SCHEMA_ESCAPE="off", HALOGEN_CACHE_DISK_DEEPEN="0",
                           HALOGEN_PLE_PAR="0", HALOGEN_MTP_DEPTH="3", HALOGEN_PREFILL_KEEP_TRUNK="0",
                           HALOGEN_ADMIT_TICKS="2", HALOGEN_REPETITION_PENALTY="1.1", HALOGEN_REASONING_EFFORT="max")
        self.assertEqual(validate_profile(profile), [])
        rendered = render_quadlet(profile)
        self.assertIn("AddDevice=/dev/accel/accel0", rendered)
        self.assertEqual(parse_quadlet(rendered).env, profile.env)
        profile.env["HALOGEN_NPU_MODELS"] = "unknown"
        self.assertTrue(validate_profile(profile))

    def test_npu_patch_preserves_checkpoint_settings_and_mount_order(self):
        original = render_quadlet(official_template("0.16.2", Path("/host/models"), Path("/host/cache")))
        original = original.replace("\n", "\r\n").encode()
        result = patch_quadlet(original, models="qwen3.5-2b", shared_root=Path("/host/shared"))
        self.assertIn(b"Environment=HALOGEN_CHECKPOINT=/models/qwen38-flash-next-v2.hgn\r\n", result)
        self.assertLess(result.index(b"/models:ro,Z"), result.index(b"/models/npu:ro,z"))
        self.assertEqual(result.count(b"AddDevice=/dev/accel/accel0"), 1)
        self.assertEqual(patch_quadlet(result, models="qwen3.5-2b", shared_root=Path("/host/shared")), result)
        disabled = patch_quadlet(result, models="")
        self.assertNotIn(b"HALOGEN_NPU_MODELS=", disabled)
        self.assertNotIn(b"AddDevice=/dev/accel", disabled)
        self.assertIn(b"HALOGEN_CHECKPOINT=", disabled)
        old_xrt = patch_quadlet(result, mounts=[("/opt/xilinx/xrt", "/opt/xilinx/xrt", "ro")])
        repaired = patch_quadlet(old_xrt, mounts=[("/usr/lib/libxrt_core.so.2.19",
                                                  "/opt/xilinx/xrt/lib/libxrt_core.so.2", "ro")])
        self.assertNotIn(b"Volume=/opt/xilinx/xrt:/opt/xilinx/xrt:ro", repaired)
        self.assertIn(b"/opt/xilinx/xrt/lib/libxrt_core.so.2:ro", repaired)
        self.assertNotIn(b"libxrt_core", patch_quadlet(repaired, models=""))


class AssetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.verifier = SharedAssets(self.root / "models", self.root / "quadlets", self.root / "state")
        self.assets = NpuAssets(self.verifier, self.root / "state")
        self.old, self.old_data = fixture_manifest(b"old program")
        self.new, self.new_data = fixture_manifest(b"new program")
        self.downloads = []

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def download(self, asset, stage):
        self.downloads.append((asset.model, asset.name))
        path = stage / asset.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.new_data.get(asset.digest, self.old_data.get(asset.digest)))

    async def test_new_version_reuses_weights_and_keeps_old_program_for_rollback(self):
        model = "qwen3-embedding-0.6b"
        old = self.assets.selection(self.old, [model], [])
        await old.prepare(None, self.download, lambda _: None)
        previous = {path: path.read_bytes() for path in old.root.rglob("*") if path.is_file()}
        self.downloads.clear()
        new = self.assets.selection(self.new, [model], [])
        await new.prepare(None, self.download, lambda _: None)
        self.assertEqual(self.downloads, [(model, "devices/u0.elf")])
        self.assertEqual({path: path.read_bytes() for path in previous}, previous)
        self.downloads.clear()
        await new.prepare(None, self.download, lambda _: None)
        self.assertEqual(self.downloads, [])

    async def test_second_task_downloads_shared_program_once_and_no_unused_weights(self):
        first = self.assets.selection(self.new, ["qwen3-reranker-0.6b"], [])
        await first.prepare(None, self.download, lambda _: None)
        self.downloads.clear()
        second = self.assets.selection(self.new, ["qwen3guard-gen-0.6b"], [])
        await second.prepare(None, self.download, lambda _: None)
        self.assertEqual({model for model, _ in self.downloads}, {"qwen3guard-gen-0.6b"})
        self.assertFalse((second.root / "qwen3-embedding-0.6b/qwen3-embedding-0.6b.hnpw").exists())

    async def test_checksum_failure_does_not_promote_a_file(self):
        selection = self.assets.selection(self.new, ["qwen3.5-2b"], [])
        async def corrupt(asset, stage):
            path = stage / asset.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"wrong")
        with self.assertRaisesRegex(AssetError, "Checksum mismatch"):
            await selection.prepare(None, corrupt, lambda _: None)
        self.assertFalse((selection.root / "qwen3.5-2b/qwen3.5-2b.hnpw").exists())

    async def test_cancelled_download_keeps_resume_files(self):
        selection = self.assets.selection(self.new, ["qwen3.5-2b"], [])
        saved = []
        async def cancel(asset, stage):
            path = stage / asset.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"partial")
            saved.append(path)
            raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await selection.prepare(None, cancel, lambda _: None)
        self.assertEqual(saved[0].read_bytes(), b"partial")

    async def test_missing_files_block_start_without_download_and_old_manifest_still_works(self):
        model = "qwen3-embedding-0.6b"
        selection = self.assets.selection(self.old, [model], [])
        await selection.prepare(None, self.download, lambda _: None)
        profile = official_template("0.16.0", self.root / "models", self.root / "cache")
        profile.env["HALOGEN_NPU_MODELS"] = model
        profile.volumes.append((str(selection.root), "/models/npu", "ro,z"))
        with patch("npu.host_status", return_value=READY):
            await self.assets.validate_profile_start(profile)
            (selection.root / model / "devices/u0.elf").unlink()
            with self.assertRaisesRegex(AssetError, "NPU file missing"):
                await self.assets.validate_profile_start(profile)

    async def test_insufficient_disk_is_detected_before_download(self):
        selection = self.assets.selection(self.new, ["qwen3.5-2b"], [])
        from collections import namedtuple
        usage = namedtuple("usage", "total used free")(100, 100, 0)
        with patch("shared_assets.shutil.disk_usage", return_value=usage), self.assertRaises(AssetError):
            await selection.prepare(None, self.download, lambda _: None)
        self.assertEqual(self.downloads, [])


class HostTests(unittest.TestCase):
    def test_distro_library_mounts_are_resolved_readonly_and_not_relabelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("libxrt_coreutil.so.2", "libxrt_core.so.2", "libxrt_driver_xdna.so.2"):
                path = root / "usr/lib64" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"library")
            mounts = xrt_mounts(root)
            self.assertEqual(len(mounts), 6)
            self.assertTrue(all(mode == "ro" for _, _, mode in mounts))
            with patch("npu.xrt_mounts", return_value=mounts):
                self.assertTrue(system_mount_allowed(*mounts[0]))
                self.assertFalse(system_mount_allowed(mounts[0][0], "/etc/hosts", "ro"))
                self.assertFalse(system_mount_allowed(mounts[0][0], mounts[0][1], "ro,Z"))

    def test_host_diagnostics_report_device_iommu_xrt_and_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "proc").mkdir()
            (root / "proc/cmdline").write_text("amd_iommu=off")
            state = host_status(root)
            self.assertFalse(state["ready"])
            self.assertEqual(len(state["errors"]), 4)


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.received = []
        self.hold, self.entered = asyncio.Event(), asyncio.Event()
        self.backend_entries = [{"id": "halogen-swift15"}] + [{"id": name, "task": task} for name, task in NPU_MODELS.items()] + [{"id": "unexpected"}]
        self.headers = []
        async def backend(request):
            if request.path == "/v1/models":
                return web.json_response({"data": self.backend_entries})
            payload = await request.json()
            self.received.append((request.path, payload))
            self.headers.append(dict(request.headers))
            if payload.get("hold"):
                self.entered.set()
                await self.hold.wait()
            if request.path == "/v1/messages/count_tokens":
                return web.json_response({"input_tokens": 9})
            if request.path == "/v1/messages" and payload.get("stream"):
                return web.Response(text='event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":9,"output_tokens":0,"cache_read_input_tokens":0}}}\n\nevent: message_delta\ndata: {"type":"message_delta","usage":{"input_tokens":2,"output_tokens":4,"cache_read_input_tokens":7}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n',
                                    content_type="text/event-stream")
            if payload.get("stream"):
                return web.Response(text='data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2}}\n\ndata: [DONE]\n\n',
                                    content_type="text/event-stream")
            return web.json_response({"usage": {"prompt_tokens": 9, "completion_tokens": 2}})
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", backend)
        self.backend = TestServer(app)
        await self.backend.start_server()
        self.session = ClientSession()
        self.manager = ModelManager(self.session, {"halogen-swift15": "halogen-swift15.service"},
                                    backend_url=str(self.backend.make_url("")).rstrip("/"))
        self.manager.current_model = "halogen-swift15"
        self.manager.npu_names = lambda _: set(NPU_MODELS)
        self.manager.perform_switch = AsyncMock()
        self.dashboard = database()
        app = web.Application()
        app["manager"], app["session"], app["dashboard"] = self.manager, self.session, self.dashboard
        app["backend_url"] = self.manager.backend_url
        app.router.add_get("/v1/models", list_models)
        app.router.add_route("*", "/{tail:.*}", proxy_request)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.backend.close()
        await self.session.close()
        await self.dashboard.close()

    async def test_all_npu_routes_keep_gpu_loaded_and_preserve_payload(self):
        from npu import TASK_ROUTES
        for model, task in NPU_MODELS.items():
            payload = {"model": model, "input": "hello", "strict": True,
                       "response_format": {"type": "json_schema", "json_schema": {"schema": {"enum": ["a", "b"]}}}}
            response = await self.client.post(TASK_ROUTES[task], json=payload)
            self.assertEqual(response.status, 200)
            await response.read()
            self.assertEqual(self.received[-1], (TASK_ROUTES[task], payload))
        self.manager.perform_switch.assert_not_awaited()
        self.assertEqual(self.manager.current_model, "halogen-swift15")
        self.assertEqual(self.manager.active_requests, 0)
        rows = self.dashboard.db.execute("SELECT model FROM requests").fetchall()
        self.assertEqual({row[0] for row in rows}, set(NPU_MODELS))
        history = self.dashboard._history(0, 99999999999)
        self.assertEqual(history["total"], len(NPU_MODELS))

    async def test_systemone_alias_and_missing_model_use_decider_without_gpu_switch(self):
        self.manager.models["other-gpu"] = "halogen-other.service"
        for model in (None, "typesafe-model", "other-gpu", "decider-0.8b"):
            payload = {"state": {"text": "Charged twice"}, "questions": {
                "refund": {"type": "noul", "instructions": "Does this require billing support?"}}}
            if model is not None:
                payload["model"] = model
            response = await self.client.post("/v1/systemone", json=payload)
            self.assertEqual(response.status, 200)
            await response.read()
            self.assertEqual(self.received[-1], ("/v1/systemone", payload))
        self.manager.perform_switch.assert_not_awaited()
        rows = self.dashboard.db.execute("SELECT model FROM requests").fetchall()
        self.assertEqual({row[0] for row in rows}, {"decider-0.8b"})

    async def test_task_discovery_uses_custom_decider_and_rejects_ambiguity(self):
        self.manager.npu_names = lambda _: {"my-decider", "other-decider"}
        self.backend_entries = [{"id": "halogen-swift15"}, {"id": "my-decider", "task": "decision"}]
        response = await self.client.post("/v1/systemone", json={"model": "anything", "state": "text"})
        self.assertEqual(response.status, 200)
        await response.read()
        row = self.dashboard.db.execute("SELECT model FROM requests").fetchone()
        self.assertEqual(row[0], "my-decider")
        self.backend_entries.append({"id": "other-decider", "task": "decision"})
        response = await self.client.post("/v1/systemone", json={"model": "anything"})
        self.assertEqual(response.status, 422)
        response = await self.client.post("/v1/systemone", json={"model": "other-decider"})
        self.assertEqual(response.status, 200)
        self.manager.perform_switch.assert_not_awaited()

    async def test_image_alias_uses_loaded_flux_and_priority_header_is_preserved(self):
        payload = {"model": "dall-e-3", "prompt": "A cloud icon", "seed": 7, "size": "256x256", "n": 2}
        response = await self.client.post("/v1/images/generations", json=payload,
                                          headers={"X-Halogen-Priority": "1"})
        self.assertEqual(response.status, 200)
        await response.read()
        self.assertEqual(self.received[-1], ("/v1/images/generations", payload))
        self.assertEqual(self.headers[-1]["X-Halogen-Priority"], "1")
        row = self.dashboard.db.execute("SELECT model,input_tokens,output_tokens FROM requests").fetchone()
        self.assertEqual(tuple(row), ("flux2-klein-4b", None, None))
        self.manager.perform_switch.assert_not_awaited()

    async def test_unavailable_task_route_is_retryable_and_does_not_use_gpu_model(self):
        self.backend_entries = [{"id": "halogen-swift15"}]
        for route in ("/v1/systemone", "/v1/images/generations"):
            response = await self.client.post(route, json={"model": "halogen-swift15"})
            self.assertEqual(response.status, 503)
            self.assertEqual(response.headers["Retry-After"], "5")
        self.assertEqual(self.received, [])
        self.manager.perform_switch.assert_not_awaited()

    async def test_messages_stream_and_count_tokens_preserve_anthropic_shapes(self):
        payload = {"model": "halogen-swift15", "messages": [{"role": "user", "content": "hello"}],
                   "max_tokens": 32, "stream": True}
        response = await self.client.post("/v1/messages", json=payload,
                                          headers={"anthropic-version": "2023-06-01"})
        self.assertIn("event: message_stop", await response.text())
        self.assertEqual(self.received[-1][1], payload)
        self.assertEqual(self.headers[-1]["anthropic-version"], "2023-06-01")
        row = self.dashboard.db.execute("SELECT model,input_tokens,output_tokens,cached_tokens FROM requests").fetchone()
        self.assertEqual(tuple(row), ("halogen-swift15", 9, 4, 7))
        response = await self.client.post("/v1/messages/count_tokens", json={"model": "halogen-swift15", "messages": payload["messages"]})
        self.assertEqual((await response.json())["input_tokens"], 9)
        self.assertEqual(self.dashboard.db.execute("SELECT count(*) FROM requests").fetchone()[0], 1)
        models = await (await self.client.get("/v1/models")).json()
        self.assertFalse(models["has_more"])
        self.assertEqual(models["data"][0]["type"], "model")
        self.assertIn("created_at", models["data"][0])

    async def test_models_only_advertises_loaded_declared_auxiliary_models(self):
        response = await self.client.get("/v1/models")
        names = {entry["id"] for entry in (await response.json())["data"]}
        self.assertEqual(names, set(NPU_MODELS) | {"halogen-swift15"})
        self.backend_entries = [{"id": "other-backend"}, {"id": "qwen3.5-2b"}]
        response = await self.client.get("/v1/models")
        self.assertEqual([entry["id"] for entry in (await response.json())["data"]], ["halogen-swift15"])

    async def test_streaming_npu_usage_is_recorded_and_model_is_correct(self):
        response = await self.client.post("/v1/chat/completions", json={"model": "qwen3.5-2b", "stream": True})
        self.assertIn("[DONE]", await response.text())
        row = self.dashboard.db.execute("SELECT model,input_tokens,output_tokens FROM requests").fetchone()
        self.assertEqual(tuple(row), ("qwen3.5-2b", 9, 2))

    async def test_unknown_npu_and_maintenance_do_not_switch_or_forward(self):
        response = await self.client.post("/v1/embeddings", json={"model": "unknown"})
        self.assertEqual(response.status, 404)
        self.manager.maintenance = True
        response = await self.client.post("/v1/chat/completions", json={"model": "qwen3.5-2b"})
        self.assertEqual(response.status, 503)
        self.assertEqual(self.received, [])
        self.manager.perform_switch.assert_not_awaited()

    async def test_moderation_openai_alias_passes_to_current_backend(self):
        response = await self.client.post("/v1/moderations", json={"model": "omni-moderation-latest", "input": "hello"})
        self.assertEqual(response.status, 200)
        self.assertEqual(self.received[-1][1]["model"], "omni-moderation-latest")
        self.manager.perform_switch.assert_not_awaited()

    async def test_unavailable_declared_model_is_retryable_and_never_switches_gpu(self):
        self.backend_entries = [{"id": "halogen-swift15"}]
        response = await self.client.post("/v1/chat/completions", json={"model": "qwen3.5-2b"})
        self.assertEqual(response.status, 503)
        self.assertEqual(response.headers["Retry-After"], "5")
        self.manager.perform_switch.assert_not_awaited()

    async def test_maintenance_drains_npu_requests_and_rejects_new_work(self):
        self.manager.backend_health = AsyncMock(return_value={"status": "ok", "in_flight": 0, "queued": 0})
        pending = asyncio.create_task(self.client.post("/v1/embeddings", json={"model": "qwen3-embedding-0.6b", "hold": True}))
        await asyncio.wait_for(self.entered.wait(), 2)
        self.assertEqual(self.manager.active_requests, 1)
        await self.manager.begin_maintenance()
        drain = asyncio.create_task(self.manager.drain_maintenance())
        response = await self.client.post("/v1/chat/completions", json={"model": "qwen3.5-2b"})
        self.assertEqual(response.status, 503)
        self.assertFalse(drain.done())
        self.hold.set()
        response = await asyncio.wait_for(pending, 2)
        self.assertEqual(response.status, 200)
        await response.read()
        await asyncio.wait_for(drain, 2)
        await self.manager.end_maintenance()
        self.assertEqual(self.manager.active_requests, 0)


class EngineUpdateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patch = patch.multiple(updater_tests, OLD="0.16.0", NEW="0.16.2")
        self.patch.start()
        self.fixture = updater_tests.TransactionTests()
        await self.fixture.asyncSetUp()
        self.old, self.old_data = fixture_manifest(b"old program")
        self.new, self.new_data = fixture_manifest(b"new program")
        self.updater = self.fixture.updater
        self.fixture.manager.begin_maintenance = AsyncMock(wraps=self.fixture.manager.begin_maintenance)
        self.assets = NpuAssets(SharedAssets(self.fixture.root / "models", self.fixture.quadlets, self.fixture.root / "state"),
                               self.fixture.root / "state")
        self.updater.npu_assets = self.assets
        original_run = self.updater._run
        async def run(*args, **kwargs):
            if args[:2] == ("podman", "run") and "/bin/sh" in args:
                return "usage: halogen-npu"
            return await original_run(*args, **kwargs)
        self.updater._run = run
        self.downloads = []
        async def download(asset, stage):
            self.assertFalse(self.fixture.manager.maintenance)
            self.downloads.append(asset.name)
            path = stage / asset.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.new_data.get(asset.digest, self.old_data.get(asset.digest)))
        self.updater._download_checkpoint_asset = download
        old = self.assets.selection(self.old, ["qwen3-embedding-0.6b"], [])
        await old.prepare(None, download, lambda _: None)
        self.old_root = old.root
        self.old_files = {p: p.read_bytes() for p in old.root.rglob("*") if p.is_file()}
        self.downloads.clear()
        self.originals = {}
        from profiles import uncensored_template
        for maker in (official_template, uncensored_template):
            profile = maker("0.16.0", self.fixture.root / "models", self.fixture.root / "cache")
            profile.env["HALOGEN_NPU_MODELS"] = "qwen3-embedding-0.6b"
            profile.volumes.append((str(old.root), "/models/npu", "ro,z"))
            path = self.fixture.quadlets / f"{profile.container_name}.container"
            path.write_text(render_quadlet(profile), encoding="utf-8")
            self.originals[path] = path.read_bytes()
        runner = self.updater._run
        async def command(*args, **kwargs):
            if args[:2] == ("podman", "run") and "/bin/cat" in args:
                return self.old.text if args[-2].endswith(":0.16.0") else self.new.text
            return await runner(*args, **kwargs)
        self.updater._run = command
        self.fixture.manager.backend_model_entries = AsyncMock(return_value=[{"id": "qwen3-embedding-0.6b"}])
        self.host = patch("npu.host_status", return_value=READY)
        self.host.start()

    async def asyncTearDown(self):
        self.host.stop()
        await self.fixture.asyncTearDown()
        self.patch.stop()

    async def test_engine_update_prepares_new_npu_program_before_downtime(self):
        await self.fixture.execute()
        self.assertEqual(self.updater.job["phase"], "succeeded", self.updater.job)
        self.assertEqual(self.downloads, ["devices/u0.elf"])
        for path in self.originals:
            profile = parse_quadlet(path.read_text())
            self.assertTrue(profile.image.endswith(":0.16.2"))
            self.assertIn(self.new.key, next(host for host, target, _ in profile.volumes if target == "/models/npu"))
        self.assertEqual({p: p.read_bytes() for p in self.old_files}, self.old_files)

    async def test_failed_engine_activation_restores_old_npu_bindings_and_bytes(self):
        self.fixture.fail_new_start = True
        await self.fixture.execute()
        self.assertEqual(self.updater.job["phase"], "rolled_back", self.updater.job)
        self.assertEqual(self.fixture.running_version, "0.16.0")
        self.assertEqual({p: p.read_bytes() for p in self.originals}, self.originals)
        self.assertEqual({p: p.read_bytes() for p in self.old_files}, self.old_files)

    async def test_xrt_failure_keeps_old_engine_serving(self):
        original = self.updater._run
        async def run(*args, **kwargs):
            if args[:2] == ("podman", "run") and "/bin/sh" in args:
                raise AssetError("XRT ABI mismatch")
            return await original(*args, **kwargs)
        self.updater._run = run
        await self.fixture.execute()
        self.assertEqual(self.updater.job["phase"], "failed")
        self.assertEqual(self.fixture.running_version, "0.16.0")
        self.assertEqual({p: p.read_bytes() for p in self.originals}, self.originals)
        self.fixture.manager.begin_maintenance.assert_not_awaited()

    async def test_restart_during_asset_preparation_needs_no_rollback(self):
        self.updater.job = {"phase": "preparing_npu", "changed": False}
        self.updater._save_job(phase="preparing_npu", changed=False)
        await self.updater.recover_on_startup()
        self.assertFalse(self.updater.recovery_required)
        self.assertEqual(self.updater.job["phase"], "failed")
        self.fixture.manager.begin_maintenance.assert_not_awaited()


class InstallTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = make_config(self.root)
        self.config.models.quadlet_dir.mkdir()
        self.profile = official_template("0.16.2", self.root / "models", self.root / "cache")
        self.path = self.config.models.quadlet_dir / "halogen-official.container"
        self.original = render_quadlet(self.profile).encode()
        self.path.write_bytes(self.original)
        self.manager = ModelManager(None, {self.profile.model_id: self.profile.service_name})
        self.manager.current_model = self.profile.model_id
        self.manager.start_service = AsyncMock()
        self.manager.wait_for_gtt_release = AsyncMock()
        self.manager.backend_health = AsyncMock(return_value={"status": "ok", "model": self.profile.model_id,
            "in_flight": 0, "queued": 0, "capability_probe": "ok", "version": {"api": "0.16.2", "engine": "0.16.2"}})
        async def entries():
            current = parse_quadlet(self.path.read_text())
            return [{"id": current.model_id}] + [{"id": m.rsplit("/", 1)[-1]}
                for m in current.env.get("HALOGEN_NPU_MODELS", "").split(",") if m]
        self.manager.backend_model_entries = AsyncMock(side_effect=entries)
        self.deploy = DeployManager(self.manager, self.config)
        self.deploy._daemon_reload = AsyncMock()
        self.deploy._verify_unit = AsyncMock()
        self.deploy._run = AsyncMock(return_value="usage: halogen-npu")
        self.deploy._run_stream = AsyncMock()
        self.manifest, self.contents = fixture_manifest()
        self.deploy.npu_assets.manifest = AsyncMock(return_value=self.manifest)
        self.downloads = []
        async def download(asset, stage):
            self.assertFalse(self.manager.maintenance)
            self.downloads.append(asset.name)
            path = stage / asset.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.contents[asset.digest])
        self.deploy._download_asset = download
        self.patches = [patch("npu_install.host_status", return_value=READY),
                        patch("halogen_deploy.sys.platform", "linux")]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        if self.deploy._job_task and not self.deploy._job_task.done():
            self.deploy._job_task.cancel()
            await asyncio.gather(self.deploy._job_task, return_exceptions=True)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    async def install(self, models="qwen3.5-2b"):
        await self.deploy.quick_npu({"models": models})
        await self.deploy._job_task

    async def test_setup_repeat_and_disable_preserve_gpu_and_reuse_files(self):
        await self.install()
        self.assertEqual(self.deploy.job["state"], "done", self.deploy.job)
        current = parse_quadlet(self.path.read_text())
        self.assertEqual(current.env["HALOGEN_CHECKPOINT"], self.profile.env["HALOGEN_CHECKPOINT"])
        self.assertEqual(current.env["HALOGEN_NPU_MODELS"], "qwen3.5-2b")
        self.assertFalse(self.manager.maintenance)
        self.assertTrue(self.downloads)
        self.downloads.clear()
        self.manager.start_service.reset_mock()
        await self.install()
        self.assertEqual(self.downloads, [])
        self.manager.start_service.assert_not_awaited()
        await self.install("")
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(self.deploy.job["state"], "done")

    async def test_flux_requires_new_engine_before_any_download_or_mutation(self):
        with self.assertRaisesRegex(AssetError, "requires Halogen 0.17.0"):
            await self.deploy.npu_plan("flux2-klein-4b")
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(self.downloads, [])
        self.deploy._run_stream.assert_not_awaited()

    async def test_flux_setup_uses_versioned_shared_decoder_and_reuses_existing_files(self):
        self.profile.image = IMAGE.replace("0.16.2", "0.17.2")
        self.original = render_quadlet(self.profile).encode()
        self.path.write_bytes(self.original)
        self.manager.backend_health.return_value["version"] = {"api": "0.17.2", "engine": "0.17.2"}
        await self.install("flux2-klein-4b")
        self.assertEqual(self.deploy.job["state"], "done", self.deploy.job)
        current = parse_quadlet(self.path.read_text())
        root = next(Path(h) for h, c, _ in current.volumes if c == "/models/npu")
        self.assertTrue((root / "flux2-klein-4b/devices/taef2.safetensors").is_file())
        self.assertEqual(current.env["HALOGEN_CHECKPOINT"], self.profile.env["HALOGEN_CHECKPOINT"])
        self.downloads.clear()
        self.manager.start_service.reset_mock()
        await self.install("flux2-klein-4b")
        self.assertEqual(self.downloads, [])
        self.manager.start_service.assert_not_awaited()

    async def test_failed_activation_restores_quadlet_and_previous_backend(self):
        self.manager.backend_model_entries.return_value = []
        self.manager.backend_model_entries.side_effect = None
        await self.install()
        self.assertEqual(self.deploy.job["state"], "error")
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(self.manager.start_service.await_count, 2)
        self.assertFalse(self.manager.maintenance)
        self.assertEqual(json.loads(self.deploy.npu_journal.read_text())["phase"], "rolled_back")
        self.assertTrue(list((self.root / "models/shared/halogen-npu").rglob("*.hnpw")))

    async def test_cancel_during_activation_recovers_and_retains_downloads(self):
        self.manager.start_service.side_effect = [asyncio.CancelledError(), None]
        with self.assertRaises(asyncio.CancelledError):
            await self.install()
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertFalse(self.manager.maintenance)
        self.assertEqual(self.deploy.job["state"], "cancelled")
        self.assertFalse(self.deploy.recovery_required)

    async def test_restart_during_activation_recovers_only_affected_quadlet(self):
        await self.install()
        record = json.loads(self.deploy.npu_journal.read_text())
        record["phase"] = "activating"
        self.deploy.npu_journal.write_text(json.dumps(record))
        other = self.path.with_name("halogen-orca.container")
        other.write_bytes(b"untouched unrelated profile")
        await self.deploy.recover_npu_on_startup()
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(other.read_bytes(), b"untouched unrelated profile")
        self.assertFalse(self.deploy.recovery_required)

    async def test_external_edit_blocks_recovery_without_overwriting_user_change(self):
        await self.install()
        record = json.loads(self.deploy.npu_journal.read_text())
        record["phase"] = "activating"
        self.deploy.npu_journal.write_text(json.dumps(record))
        changed = self.path.read_bytes() + b"# user change\n"
        self.path.write_bytes(changed)
        await self.deploy.recover_npu_on_startup()
        self.assertTrue(self.deploy.recovery_required)
        self.assertTrue(self.manager.maintenance)
        self.assertEqual(self.path.read_bytes(), changed)

    async def test_xrt_failure_happens_before_downtime_and_any_profile_write(self):
        self.deploy._run.return_value = "error: XRT library missing"
        await self.install()
        self.assertEqual(self.deploy.job["state"], "error")
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertFalse(self.manager.maintenance)
        self.manager.start_service.assert_not_awaited()

    async def test_missing_host_unknown_targets_and_id_conflicts_are_rejected(self):
        with patch("npu_install.host_status", return_value={"ready": False, "errors": ["driver missing"]}):
            with self.assertRaisesRegex(DeployError, "driver missing"):
                await self.deploy.quick_npu({"models": "qwen3.5-2b"})
        for profiles in ([], ["unknown"]):
            with self.assertRaises(AssetError):
                await self.deploy.npu_plan("qwen3.5-2b", profiles)
        self.manager.models["qwen3.5-2b"] = "halogen-other.service"
        with self.assertRaisesRegex(AssetError, "conflict"):
            await self.deploy.npu_plan("qwen3.5-2b")

    async def test_custom_model_probed_and_mounted_readonly(self):
        source = self.root / "models/official/my-guard"
        source.mkdir(parents=True)
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            (source / name).write_bytes(b"fine-tune")
        async def run(*args, **kwargs):
            return "task=classify base=qwen3guard-gen-0.6b" if "probe" in args else "usage: halogen-npu"
        self.deploy._run.side_effect = run
        await self.install("/models/my-guard")
        self.assertEqual(self.deploy.job["state"], "done", self.deploy.job)
        current = parse_quadlet(self.path.read_text())
        self.assertIn((str(source), "/models/my-guard", "ro,z"), current.volumes)
        self.assertFalse(any(name.endswith(".hnpw") for name in self.downloads))

    async def test_deploy_route_requires_local_mutation_headers_and_protects_running_job(self):
        routes = DeployRoutes(self.deploy)
        app = web.Application()
        routes.register_routes(app)
        async with TestClient(TestServer(app)) as client:
            path = "/dashboard/api/deploy/quick/npu"
            response = await client.post(path, json={"models": "qwen3.5-2b"})
            self.assertEqual(response.status, 400)
            response = await client.post(path, json={"models": "qwen3.5-2b"},
                                         headers={"Origin": "http://foreign", "X-Halogen-Action": "deploy"})
            self.assertEqual(response.status, 400)
            self.deploy.job = {"state": "running", "kind": "quick-npu"}
            response = await client.post(path, json={"models": "qwen3.5-2b"}, headers={"X-Halogen-Action": "deploy"})
            self.assertEqual(response.status, 400)


if __name__ == "__main__":
    unittest.main()
