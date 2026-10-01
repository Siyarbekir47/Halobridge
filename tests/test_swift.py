"""Exercise real small files and real switch/recovery logic, without a GPU.

Only the published asset sizes/digests and external HF/systemd commands are
substituted. This tests corrupt/partial data, sharing and durable rollback.
"""

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import shared_assets
import swift_catalog
from halogen_deploy import DeployManager, DeployRoutes, DeployError
from halogen_router import ModelManager, list_models
from profiles import official_template, uncensored_template, swift_quick_template, parse_quadlet, render_quadlet, validate_profile
from shared_assets import SharedAssets, AssetError, atomic_json
from swift_install import shared_quadlet_content
from test_deploy import make_config
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


class SwiftFilesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.tmp = Path(self.tmpdir.name)
        self.content = {}

        def tiny(asset):
            content = (asset.name + " verified bytes\n").encode()
            self.content[asset.name] = content
            digest = (hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                      if asset.algorithm == "git-sha1" else hashlib.sha256(content).hexdigest())
            return replace(asset, size=len(content), digest=digest)

        self.shared = tuple(tiny(a) for a in swift_catalog.SHARED_ASSETS)
        self.variants = {key: {**value, "checkpoint": tiny(value["checkpoint"])}
                         for key, value in swift_catalog.SWIFT_VARIANTS.items()}
        for target, value in (("shared_assets.SHARED_ASSETS", self.shared),
                              ("swift_catalog.SHARED_ASSETS", self.shared),
                              ("swift_catalog.SWIFT_VARIANTS", self.variants),
                              ("shared_assets.SWIFT_VARIANTS", self.variants),
                              ("halogen_deploy.sys.platform", "linux")):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.config = make_config(self.tmp)
        self.manager = ModelManager(None, {}, gtt_path=None)
        self.manager.run_command = AsyncMock()
        self.manager.wait_for_backend_idle = AsyncMock()
        self.manager.wait_for_gtt_release = AsyncMock()
        self.manager.service_is_active = AsyncMock(return_value=False)
        self.manager.wait_for_backend = AsyncMock()
        self.deploy = DeployManager(self.manager, self.config)
        self.manager.asset_validator = self.deploy.assets.validate_start
        self.deploy._daemon_reload = AsyncMock()
        self.deploy._verify_unit = AsyncMock()
        self.deploy._run_stream = AsyncMock()
        self.deploy._stream_service_journal = AsyncMock()
        self.deploy._service_is_active = AsyncMock(return_value=False)
        self.deploy._run = AsyncMock()
        self.downloads = []

        async def download(asset, stage):
            self.downloads.append(asset.name)
            path = stage / asset.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.content[asset.name])

        self.download = download
        self.deploy._download_asset = download

        async def health():
            model = self.manager.switch_target or self.manager.current_model
            return {"status": "ok", "model": model, "version": {"api": "0.15.1", "engine": "0.15.1"}}

        self.manager.backend_health = health

    def write_profile(self, profile, extra=""):
        target = self.config.models.quadlet_dir / f"halogen-{profile.profile_id}.container"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_quadlet(profile) + extra, encoding="utf-8")
        return target

    async def install(self, variant="swift15"):
        await self.deploy.quick_swift({"variant": variant})
        await self.deploy._job_task
        return self.deploy.job

    async def test_fresh_install_and_second_variant_share_all_assets(self):
        job = await self.install()
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(len(self.downloads), 9)
        self.assertEqual(self.manager.current_model, "halogen-swift15")
        self.assertIn("halogen-swift15", self.manager.models)
        self.downloads.clear()
        job = await self.install("swift15-abliterated")
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(self.downloads, [self.variants["swift15-abliterated"]["checkpoint"].name])
        await self.deploy.assets.validate_start("halogen-swift15-abliterated")

    async def test_reuses_actual_official_mount_and_connects_all_profiles(self):
        official = official_template("0.15.1", self.tmp / "custom-disk", self.tmp / "cache")
        orca = uncensored_template("0.15.1", self.tmp / "custom-disk", self.tmp / "cache")
        official_path = self.write_profile(official, "# custom service settings\nEnvironment=KEPT=1\n")
        orca_path = self.write_profile(orca)
        original_checkpoint = orca.env["HALOGEN_CHECKPOINT"]
        for asset in self.shared:
            path = self.tmp / "custom-disk" / "official" / asset.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.content[asset.name])
        plan = self.deploy.swift_plan("swift15")
        self.assertEqual(plan["download_bytes"], self.variants["swift15"]["checkpoint"].size)
        self.assertTrue(all(f["status"] == "unverified" for f in plan["files"][1:]))
        self.manager.current_model = "qwen3.8-flash"
        job = await self.install()
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(len(self.downloads), 1)
        for path in (official_path, orca_path):
            profile = parse_quadlet(path.read_text(), path)
            self.assertIn("HALOGEN_NGRAM_TABLE", profile.env)
            self.assertTrue(all(v[2] == "ro,z" for v in profile.volumes if v[1].startswith("/shared/")))
        self.assertEqual(parse_quadlet(orca_path.read_text()).env["HALOGEN_CHECKPOINT"], original_checkpoint)
        self.assertIn("# custom service settings\nEnvironment=KEPT=1", official_path.read_text())
        asset = self.shared[0]
        source = self.tmp / "custom-disk" / "official" / asset.name
        self.assertTrue(source.exists())
        self.assertEqual(source.stat().st_ino, self.deploy.assets.shared_path(asset).stat().st_ino)

    async def test_repeat_install_keeps_custom_runtime_and_does_not_read_again(self):
        await self.install()
        target = self.config.models.quadlet_dir / "halogen-swift15.container"
        profile = parse_quadlet(target.read_text(), target)
        profile.env["HALOGEN_TEMPERATURE"] = "0.4"
        profile.env["HALOGEN_KV_SLOTS"] = "3"
        profile.env["HALOGEN_WEIGHTS_LOCK"] = "0"
        target.write_text(render_quadlet(profile))
        self.downloads.clear()
        # Cache survives a manager restart and avoids opening any model files.
        self.deploy.assets = SharedAssets(self.config.deploy.models_root, self.config.models.quadlet_dir,
                                         self.config.dashboard.state_dir)
        self.manager.asset_validator = self.deploy.assets.validate_start
        with patch("shared_assets.hashlib.sha256", side_effect=AssertionError("Unexpected reread")):
            job = await self.install()
        self.assertEqual(job["state"], "done", job)
        current = parse_quadlet(target.read_text())
        self.assertEqual(current.env["HALOGEN_TEMPERATURE"], "0.4")
        self.assertEqual(current.env["HALOGEN_KV_SLOTS"], "3")
        self.assertEqual(current.env["HALOGEN_WEIGHTS_LOCK"], "0")
        self.assertEqual(self.downloads, [])

    async def test_shared_import_copy_fallback_keeps_original_files(self):
        official = official_template("0.15.1", self.tmp / "legacy", self.tmp / "cache")
        self.write_profile(official)
        source = self.tmp / "legacy" / "official"
        for asset in self.shared:
            path = source / asset.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.content[asset.name])
        with patch("shared_assets.os.link", side_effect=OSError("Different filesystem")):
            job = await self.install()
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(len(self.downloads), 1)
        for asset in self.shared:
            self.assertEqual((source / asset.name).read_bytes(), self.content[asset.name])
            self.assertNotEqual((source / asset.name).stat().st_ino, self.deploy.assets.shared_path(asset).stat().st_ino)

    async def test_active_legacy_quick_setup_repairs_shared_bindings(self):
        for profile in (official_template("0.15.1", self.tmp / "legacy", self.tmp / "cache"),
                        uncensored_template("0.15.1", self.tmp / "legacy", self.tmp / "cache")):
            with self.subTest(model=profile.model_id):
                self.write_profile(profile)
                self.deploy._service_is_active = AsyncMock(return_value=True)
                self.manager.current_model = profile.model_id
                self.manager.backend_health = AsyncMock(return_value={"status": "ok", "model": profile.model_id})
                if profile.profile_id == "official":
                    self.deploy._run_quick_official = AsyncMock()
                    result = await self.deploy.quick_official()
                else:
                    output = self.config.deploy.models_root / "uncensored" / "qwen3.8-flash-uncensored.hgn"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"Existing converted Orca")
                    self.deploy._run_quick_uncensored = AsyncMock()
                    result = await self.deploy.quick_uncensored({})
                await self.deploy._job_task
                self.assertNotIn("already", result)
                self.deploy.job["state"] = "done"

    async def test_active_verified_canonical_profiles_need_no_job(self):
        await self.deploy.assets.prepare(None, self.download, lambda line: None)
        for profile in (official_template("0.15.1", self.tmp / "legacy", self.tmp / "cache"),
                        uncensored_template("0.15.1", self.tmp / "legacy", self.tmp / "cache")):
            self.write_profile(profile)
            path = self.config.models.quadlet_dir / f"halogen-{profile.profile_id}.container"
            path.write_text(shared_quadlet_content(path.read_text(), self.deploy.assets.locations()))
        self.deploy._service_is_active = AsyncMock(return_value=True)
        self.manager.backend_health = AsyncMock(return_value={"status": "ok", "model": "qwen3.8-flash"})
        self.assertTrue((await self.deploy.quick_official())["already"])
        self.assertTrue((await self.deploy.quick_uncensored({}))["already"])
        # Removing a canonical file makes Quick Setup a repair again.
        self.deploy.assets.shared_path(self.shared[0]).unlink()
        self.assertFalse(self.deploy.assets.profile_ready("qwen3.8-flash"))

    async def test_corrupt_download_is_not_activated_and_can_be_retried(self):
        async def corrupt(asset, stage):
            await self.download(asset, stage)
            (stage / asset.name).write_bytes(b"x" * asset.size)
        self.deploy._download_asset = corrupt
        job = await self.install()
        self.assertEqual(job["state"], "error")
        self.assertIn("Checksum mismatch", job["error"])
        self.assertFalse(self.config.models.quadlet_dir.exists())
        self.manager.run_command.assert_not_awaited()
        self.deploy._download_asset = self.download
        self.assertEqual((await self.install())["state"], "done")

    async def test_missing_one_tokenizer_file_is_repaired_once(self):
        await self.deploy.assets.prepare("swift15", self.download, lambda _: None)
        missing = next(a for a in self.shared if a.name == "tokenizer/chat_template.jinja")
        self.deploy.assets.shared_path(missing).unlink()
        self.downloads.clear()
        await self.deploy.assets.prepare("swift15-abliterated", self.download, lambda _: None)
        self.assertEqual(set(self.downloads), {missing.name, self.variants["swift15-abliterated"]["checkpoint"].name})

    async def test_missing_shared_file_refuses_switch_before_stopping_old_backend(self):
        await self.install()
        self.deploy.assets.shared_path(self.shared[0]).unlink()
        self.manager.current_model = "qwen3.8-flash"
        self.manager.models["qwen3.8-flash"] = "halogen-official.service"
        self.manager.run_command.reset_mock()
        with self.assertRaises(AssetError):
            await self.manager.perform_switch("halogen-swift15")
        self.manager.run_command.assert_not_awaited()
        self.assertEqual(self.manager.current_model, "qwen3.8-flash")

    async def test_file_mutation_invalidates_checksum_cache(self):
        await self.deploy.assets.prepare("swift15", self.download, lambda _: None)
        asset = self.shared[0]
        path = self.deploy.assets.shared_path(asset)
        self.assertTrue(self.deploy.assets.cached(path, asset))
        path.write_bytes(b"z" * asset.size)
        self.assertFalse(self.deploy.assets.cached(path, asset))
        self.assertFalse(await self.deploy.assets.verify(path, asset))

    async def test_insufficient_disk_is_reported_before_network_or_service_work(self):
        with patch("shared_assets.shutil.disk_usage", return_value=SimpleNamespace(free=1)):
            plan = self.deploy.swift_plan("swift15")
            self.assertFalse(plan["filesystems"][0]["sufficient"])
            job = await self.install()
        self.assertEqual(job["state"], "error")
        self.assertIn("disk space", job["error"])
        self.assertEqual(self.downloads, [])
        self.manager.run_command.assert_not_awaited()

    async def test_interrupted_download_retains_staging_and_old_backend(self):
        entered = asyncio.Event()
        async def interrupted(asset, stage):
            await self.download(asset, stage)
            entered.set()
            await asyncio.Event().wait()
        self.deploy._download_asset = interrupted
        self.manager.current_model = "qwen3.8-flash"
        await self.deploy.quick_swift({"variant": "swift15"})
        await entered.wait()
        self.deploy.cancel_job()
        with self.assertRaises(asyncio.CancelledError):
            await self.deploy._job_task
        self.assertEqual(self.deploy.job["state"], "cancelled")
        self.assertEqual(self.manager.current_model, "qwen3.8-flash")
        self.assertTrue(list(self.config.deploy.models_root.glob("swift15/.downloads/*.hgn")))
        self.manager.run_command.assert_not_awaited()

    async def test_failed_activation_restores_exact_originals_and_keeps_assets(self):
        official = self.write_profile(official_template("0.15.1", self.tmp / "models", self.tmp / "cache"))
        orca = self.write_profile(uncensored_template("0.15.1", self.tmp / "models", self.tmp / "cache"))
        originals = {p: p.read_bytes() for p in (official, orca)}
        self.deploy.reload_discovery()
        self.manager.current_model = "qwen3.8-flash"
        async def wait(model, **kwargs):
            if model == "halogen-swift15":
                raise RuntimeError("Broken Swift backend")
        self.manager.wait_for_backend = wait
        job = await self.install()
        self.assertEqual(job["state"], "error", job)
        self.assertFalse(self.deploy.recovery_required, job)
        self.assertFalse(self.manager.maintenance)
        self.assertEqual(self.manager.current_model, "qwen3.8-flash")
        for path, before in originals.items():
            self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.config.models.quadlet_dir / "halogen-swift15.container").exists())
        self.assertTrue(self.deploy.assets.shared_path(self.shared[0]).is_file())
        self.assertEqual(json.loads(self.deploy.swift_journal.read_text())["phase"], "rolled_back")

    async def test_restart_recovers_interrupted_activation_and_empty_discovery(self):
        await self.deploy.assets.prepare("swift15", self.download, lambda _: None)
        profile = swift_quick_template("swift15", self.config.deploy.models_root,
                                      self.config.deploy.cache_root, self.deploy.assets.locations())
        target = self.config.models.quadlet_dir / "halogen-swift15.container"
        record = self.deploy._installation_record("swift15", {target: render_quadlet(profile).encode()})
        self.deploy._write_installation(record)
        record["phase"] = "activating"
        atomic_json(self.deploy.swift_journal, record)
        self.deploy.reload_discovery()
        self.deploy._service_is_active = AsyncMock(return_value=True)
        await self.deploy.recover_swift_on_startup()
        self.assertFalse(target.exists())
        self.assertEqual(self.manager.models, {})
        self.assertFalse(self.manager.maintenance)
        self.deploy._run.assert_awaited_once()

    async def test_failure_to_verify_quadlet_restores_existing_swift_without_stopping_old(self):
        profile = swift_quick_template("swift15", self.config.deploy.models_root,
                                      self.config.deploy.cache_root, self.deploy.assets.locations())
        profile.env["HALOGEN_DOWNLOAD"] = "wrong/repo"
        target = self.write_profile(profile)
        before = target.read_bytes()
        self.deploy._verify_unit = AsyncMock(side_effect=DeployError("Bad generated unit"))
        reload_gates = []
        async def reload():
            reload_gates.append(self.manager.switching or self.manager.maintenance)
        self.deploy._daemon_reload = reload
        job = await self.install()
        self.assertEqual(job["state"], "error")
        self.assertFalse(self.deploy.recovery_required, job)
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(reload_gates, [True, True])
        self.assertFalse(self.manager.maintenance)
        self.manager.run_command.assert_not_awaited()

    async def test_external_edit_blocks_recovery_without_overwriting_it(self):
        profile = swift_quick_template("swift15", self.config.deploy.models_root,
                                      self.config.deploy.cache_root, self.deploy.assets.locations())
        target = self.config.models.quadlet_dir / "halogen-swift15.container"
        record = self.deploy._installation_record("swift15", {target: render_quadlet(profile).encode()})
        self.deploy._write_installation(record)
        target.write_text(target.read_text() + "# User edit\n")
        await self.deploy.recover_swift_on_startup()
        self.assertTrue(self.deploy.recovery_required)
        self.assertIn("# User edit", target.read_text())
        self.assertTrue(self.manager.maintenance)

    async def test_http_guard_and_live_model_discovery(self):
        app = web.Application()
        app["manager"] = self.manager
        app["models"] = {"stale-model": "old.service"}
        DeployRoutes(self.deploy).register_routes(app)
        app.router.add_get("/v1/models", list_models)
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/dashboard/api/deploy/quick/swift/plan?variant=bad")
            self.assertEqual(response.status, 400)
            response = await client.get("/dashboard/api/deploy/quick/swift/plan?variant=swift15")
            self.assertEqual(response.status, 200)
            response = await client.post("/dashboard/api/deploy/quick/swift", json={"variant": "swift15"})
            self.assertEqual(response.status, 400)
            response = await client.post("/dashboard/api/deploy/quick/swift", json={"variant": "swift15"},
                                         headers={"X-Halogen-Action": "deploy", "Origin": "https://evil.test"})
            self.assertEqual(response.status, 400)
            response = await client.post("/dashboard/api/deploy/quick/swift", json={"variant": "swift15"},
                                         headers={"X-Halogen-Action": "deploy"})
            self.assertEqual(response.status, 202)
            await self.deploy._job_task
            response = await client.get("/v1/models")
            self.assertEqual([entry["id"] for entry in (await response.json())["data"]], ["halogen-swift15"])

    async def test_resumed_download_plan_counts_only_remaining_allocated_bytes(self):
        asset = self.variants["swift15"]["checkpoint"]
        stage = self.deploy.assets.staging_root(asset, "swift15")
        partial = stage / ".cache" / "huggingface" / "download" / (asset.name + ".digest.incomplete")
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(self.content[asset.name][:12])
        plan = self.deploy.swift_plan("swift15")
        self.assertEqual(plan["files"][0]["download_bytes"], asset.size - 12)

    async def test_completed_staging_file_is_verified_and_reused_without_hf(self):
        asset = self.variants["swift15"]["checkpoint"]
        stage = self.deploy.assets.staging_root(asset, "swift15")
        await self.download(asset, stage)
        self.downloads.clear()
        await self.install()
        self.assertNotIn(asset.name, self.downloads)

    async def test_hash_cancel_closes_reader_without_saving_unverified_cache(self):
        asset = self.shared[0]
        path = self.deploy.assets.shared_path(asset)
        path.parent.mkdir(parents=True)
        path.write_bytes(self.content[asset.name])
        entered = asyncio.Event()
        original_thread = asyncio.to_thread
        async def paused_thread(fn, *args, **kwargs):
            entered.set()
            await asyncio.sleep(.02)
            return await original_thread(fn, *args, **kwargs)
        with patch("shared_assets.asyncio.to_thread", new=paused_thread):
            task = asyncio.create_task(self.deploy.assets.verify(path, asset))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertFalse(self.deploy.assets.cached(path, asset))
        self.assertTrue(await self.deploy.assets.verify(path, asset))

    async def test_cancel_during_activation_restores_previous_profile(self):
        original = self.write_profile(official_template("0.15.1", self.tmp / "models", self.tmp / "cache"))
        before = original.read_bytes()
        self.deploy.reload_discovery()
        self.manager.current_model = "qwen3.8-flash"
        entered = asyncio.Event()
        async def wait(model, **kwargs):
            if model == "halogen-swift15":
                entered.set()
                await asyncio.Event().wait()
        self.manager.wait_for_backend = wait
        await self.deploy.quick_swift({"variant": "swift15"})
        await entered.wait()
        self.deploy.cancel_job()
        with self.assertRaises(asyncio.CancelledError):
            await self.deploy._job_task
        self.assertEqual(self.deploy.job["state"], "cancelled")
        self.assertEqual(self.manager.current_model, "qwen3.8-flash")
        self.assertEqual(original.read_bytes(), before)
        self.assertFalse(self.manager.maintenance)
        self.assertFalse(self.manager.switching)

    async def test_cancel_while_draining_requests_always_releases_switch_gate(self):
        self.manager.active_requests = 1
        task = asyncio.create_task(self.manager.switch_with_hook("halogen-swift15"))
        await asyncio.sleep(0)
        self.assertTrue(self.manager.switching)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.manager.switching)
        self.manager.run_command.assert_not_awaited()

    async def test_repeated_cancel_does_not_interrupt_recovery(self):
        original = self.write_profile(official_template("0.15.1", self.tmp / "models", self.tmp / "cache"))
        before = original.read_bytes()
        self.deploy.reload_discovery()
        self.manager.current_model = "qwen3.8-flash"
        started, restoring, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def wait(model, **kwargs):
            if model == "halogen-swift15":
                started.set()
                await asyncio.Event().wait()
        calls = 0
        async def reload():
            nonlocal calls
            calls += 1
            if calls == 2:
                restoring.set()
                await release.wait()
        self.manager.wait_for_backend = wait
        self.deploy._daemon_reload = reload
        await self.deploy.quick_swift({"variant": "swift15"})
        await started.wait()
        self.deploy.cancel_job()
        await restoring.wait()
        self.deploy.cancel_job()
        await asyncio.sleep(0)
        self.assertTrue(self.manager.maintenance)
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await self.deploy._job_task
        self.assertEqual(self.deploy.job["state"], "cancelled")
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(self.manager.current_model, "qwen3.8-flash")
        self.assertFalse(self.manager.maintenance)
        self.assertEqual(json.loads(self.deploy.swift_journal.read_text())["phase"], "rolled_back")

    async def test_hf_argv_downloads_only_exact_file_and_revision(self):
        self.deploy._hf_executable = lambda: "hf"
        self.deploy._start_pipeline_job("test", 1)
        asset = self.variants["swift15-abliterated"]["checkpoint"]
        await DeployManager._download_asset(self.deploy, asset, self.tmp / "stage")
        args = self.deploy._run_stream.await_args.args
        self.assertEqual(args[0], ["hf", "download", asset.repo, asset.name, "--revision", asset.revision,
                                   "--local-dir", str(self.tmp / "stage")])
        self.assertNotIn("qwen38-flash-next-v2.hgn", args[0])
        self.assertEqual(args[1]["HF_HUB_OFFLINE"], "0")
        self.assertNotIn("HF_TOKEN", args[1])

    async def test_post_health_version_failure_is_recovered(self):
        original = self.write_profile(official_template("0.15.1", self.tmp / "models", self.tmp / "cache"))
        before = original.read_bytes()
        self.deploy.reload_discovery()
        self.manager.current_model = "qwen3.8-flash"
        self.deploy._swift_ready = AsyncMock(side_effect=AssetError("Wrong engine version"))
        self.deploy._service_is_active = AsyncMock(side_effect=lambda service: service == "halogen-swift15.service")
        job = await self.install()
        self.assertEqual(job["state"], "error")
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(self.manager.current_model, "qwen3.8-flash")
        self.assertFalse(self.manager.maintenance)
        self.deploy._run.assert_awaited_once_with("systemctl", "--user", "stop", "halogen-swift15.service", timeout=120)


class SwiftCatalogTests(unittest.TestCase):
    def test_pinned_versions_and_valid_runtime(self):
        assets = SharedAssets(Path("/models-root"), Path("/quadlets"), Path("/state"))
        for variant in swift_catalog.SWIFT_VARIANTS:
            profile = swift_quick_template(variant, Path("/models-root"), Path("/cache-root"), assets.locations())
            self.assertEqual(validate_profile(profile), [])
            self.assertTrue(profile.image.endswith(":0.15.1"))
            self.assertEqual(profile.env["HALOGEN_KV_POOL_POSITIONS"], "262144")
            self.assertEqual(profile.env["HALOGEN_WEIGHTS_LOCK"], "1")
            self.assertNotIn("HALOGEN_DOWNLOAD", profile.env)
            self.assertNotIn("HALOGEN_MTP_HEAD", profile.env)
            self.assertEqual(len([a for a in swift_catalog.SHARED_ASSETS if a.name.startswith("tokenizer/")]), 6)

    def test_shared_binding_preserves_sections_and_user_settings(self):
        original = render_quadlet(official_template("0.15.1", Path("/custom"), Path("/cache"))) + "# Keep\nTimeoutStartSec=1000\n"
        assets = SharedAssets(Path("/shared-root"), Path("/quadlets"), Path("/state"))
        result = shared_quadlet_content(original, assets.locations())
        self.assertIn("# Keep\nTimeoutStartSec=1000", result)
        self.assertIn("HALOGEN_CHECKPOINT=/models/qwen38-flash-next-v2.hgn", result)
        self.assertEqual(result, shared_quadlet_content(result, assets.locations()))


if __name__ == "__main__":
    unittest.main()
