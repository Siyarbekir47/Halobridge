"""Update transactions use temporary Quadlets and a fake Podman/systemd runner."""

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from halogen_router import MODELS, ModelManager, RouterError
import halogen_updates as updates
from halogen_updates import ContainerUpdater, IMAGE_REPOSITORY, UpdateError, atomic_write, quadlet_image, version_tuple


OLD, NEW = "0.13.1", "0.13.2"
MODEL = "qwen3.8-flash"


def quadlet(version=OLD):
    return (
        "[Unit]\r\nDescription=Keep my settings\r\n[Container]\r\n"
        f"Image={IMAGE_REPOSITORY}:{version}\r\n"
        "Environment=HALOGEN_KV_SLOTS=2\r\nEnvironment=HALOGEN_KV_POOL_POSITIONS=524288\r\n"
        "Environment=HALOGEN_MAX_TOK=32768\r\nEnvironment=HALOGEN_CACHE_DISK_GIB=750\r\n"
        "Volume=/models:/models:ro\r\n[Service]\r\nRestart=on-failure\r\n"
    ).encode()


class VersionTests(unittest.TestCase):
    def test_stable_numeric_versions_only(self):
        self.assertEqual(version_tuple("v0.13.1"), (0, 13, 1))
        self.assertGreater(version_tuple("0.13.10"), version_tuple("0.13.9"))
        for version in (None, {}, "latest", "0.13.2-rc1", "v0.13.1;reboot", "0.01.2", "0.13.1\n", "1.2.3+build"):
            self.assertIsNone(version_tuple(version))

    def test_only_image_changes_and_crlf_is_preserved(self):
        old, new = quadlet_image(quadlet(), f"{IMAGE_REPOSITORY}:{NEW}")
        self.assertEqual(old, f"{IMAGE_REPOSITORY}:{OLD}")
        self.assertEqual(new, quadlet().replace(f":{OLD}".encode(), f":{NEW}".encode()))

    def test_non_halogen_duplicate_or_missing_image_is_rejected(self):
        for data in (b"[Container]\n", b"[Container]\nImage=other/image:1.0.0\n",
                     quadlet() + f"[Container]\nImage={IMAGE_REPOSITORY}:{NEW}\n".encode()):
            with self.assertRaises(UpdateError):
                quadlet_image(data)


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = ModelManager(None)
        self.manager.current_model = MODEL
        self.manager.backend_health = AsyncMock(return_value={"status": "ok", "in_flight": 0, "queued": 0})

    async def test_drain_allows_existing_request_to_finish_and_blocks_new_ones(self):
        await self.manager.reserve_request(MODEL)
        await self.manager.begin_maintenance()
        drain = asyncio.create_task(self.manager.drain_maintenance())
        await asyncio.sleep(0)
        self.assertFalse(drain.done())
        with self.assertRaises(web.HTTPServiceUnavailable) as error:
            await self.manager.reserve_request(MODEL)
        self.assertEqual(error.exception.headers["Retry-After"], "30")
        self.assertEqual((await self.manager.status())["status"], "maintenance")
        await self.manager.release_request()
        await drain
        await self.manager.end_maintenance()
        self.assertEqual(await self.manager.reserve_request(MODEL), MODEL)

    async def test_update_waits_for_in_progress_model_switch(self):
        self.manager.switching = True
        waiting = asyncio.create_task(self.manager.begin_maintenance())
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        async with self.manager.condition:
            self.manager.switching = False
            self.manager.current_model = "qwen3.8-flash-uncensored"
            self.manager.condition.notify_all()
        await waiting
        self.assertTrue(self.manager.maintenance)
        self.assertEqual(self.manager.current_model, "qwen3.8-flash-uncensored")

    async def test_missing_health_is_not_assumed_idle(self):
        self.manager.backend_health.return_value = None
        await self.manager.begin_maintenance()
        with self.assertRaises(RouterError):
            await self.manager.drain_maintenance()

    async def test_failed_startup_recovery_reports_maintenance_without_a_model(self):
        self.manager.current_model = None
        self.manager.maintenance = True
        with self.assertRaises(web.HTTPServiceUnavailable):
            await self.manager.reserve_request(None)


class TransactionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("HALOGEN_TEST_TMP"))
        self.root = Path(self.temp.name)
        self.quadlets = self.root / "quadlets"
        self.quadlets.mkdir()
        for service in MODELS.values():
            (self.quadlets / service.replace(".service", ".container")).write_bytes(quadlet())
        self.manager = ModelManager(None)
        self.manager.current_model = MODEL
        self.manager.wait_for_gtt_release = AsyncMock()
        self.manager.backend_health = AsyncMock(side_effect=self.health)
        self.running_version = OLD
        self.running_model = MODEL
        self.commands = []
        self.fail_pull = False
        self.fail_new_start = False
        self.fail_all_starts = False
        self.pull_event = None
        self.updater = self.make_updater()

    def make_updater(self):
        updater = ContainerUpdater(self.manager, None, MODELS,
                                   quadlet_dir=self.quadlets, backup_root=self.root / "backups", state_dir=self.root / "state")
        updater._support_error = lambda: None
        updater._run = AsyncMock(side_effect=self.command)
        updater.latest_version = NEW
        updater.current_version = OLD
        updater.checked_at = updater.attempted_at = time.time()
        return updater

    async def health(self):
        return {"status": "ok", "model": self.running_model, "in_flight": 0, "queued": 0,
                "version": {"api": self.running_version, "engine": self.running_version}, "capability_probe": "ok"}

    async def command(self, *args, **kwargs):
        self.commands.append(args)
        if args[:2] == ("podman", "pull"):
            self.assertFalse(self.manager.maintenance, "Pull must not interrupt inference")
            if self.pull_event:
                await self.pull_event.wait()
            if self.fail_pull:
                raise UpdateError("Pull fehlgeschlagen")
        elif args[:3] == ("podman", "image", "inspect"):
            return json.dumps([{"Id": "sha256:image-" + args[3].rsplit(":", 1)[1]}])
        elif args[:3] == ("podman", "container", "inspect"):
            return json.dumps([{"Image": "image-" + self.running_version}])
        elif args[:3] == ("systemctl", "--user", "show"):
            if "--property=ActiveState" in args:
                return "inactive\n"
            image, _ = quadlet_image((self.quadlets / args[3].replace(".service", ".container")).read_bytes())
            return "/usr/bin/podman run " + image + " all"
        elif args[:3] == ("systemctl", "--user", "start"):
            self.assertTrue(self.manager.maintenance, "Admission gate must stay closed until verification")
            self.running_model = next(model for model, service in MODELS.items() if service == args[3])
            image, _ = quadlet_image((self.quadlets / args[3].replace(".service", ".container")).read_bytes())
            version = image.rsplit(":", 1)[1]
            if self.fail_all_starts or self.fail_new_start and version == NEW:
                raise UpdateError("Start fehlgeschlagen")
            self.running_version = version
        return ""

    async def asyncTearDown(self):
        await self.updater.close()
        self.temp.cleanup()

    async def execute(self):
        await self.updater.start(NEW)
        await self.updater.task

    def assert_originals(self):
        for path in self.updater._paths().values():
            self.assertEqual(path.read_bytes(), quadlet())

    async def test_success_updates_both_quadlets_and_only_starts_active_model(self):
        await self.execute()
        self.assertEqual(self.updater.job["phase"], "succeeded")
        self.assertFalse(self.manager.maintenance)
        self.assertEqual(self.running_version, NEW)
        for path in self.updater._paths().values():
            self.assertEqual(path.read_bytes(), quadlet(NEW))
            self.assertEqual((Path(self.updater.job["backup_dir"]) / path.name).read_bytes(), quadlet())
        starts = [command for command in self.commands if command[:3] == ("systemctl", "--user", "start")]
        self.assertEqual(starts, [("systemctl", "--user", "start", MODELS[MODEL])])
        self.assertFalse(self.updater.status()["update_available"])

    async def test_backup_retention_prunes_old_backups(self):
        self.updater.backup_keep = 2
        self.updater.backup_root.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            path = self.updater.backup_root / f"dashboard-update-0.13.1-{i}"
            path.mkdir()
            os.utime(path, (i, i))

        self.updater._prune_backups()

        remaining = sorted(p.name for p in self.updater.backup_root.glob("dashboard-update-*"))
        self.assertEqual(remaining, ["dashboard-update-0.13.1-2", "dashboard-update-0.13.1-3"])

    async def test_failed_pull_never_enters_maintenance_or_changes_quadlets(self):
        self.fail_pull = True
        await self.execute()
        self.assertEqual(self.updater.job["phase"], "failed")
        self.assertFalse(self.manager.maintenance)
        self.assert_originals()
        self.assertFalse(any(c[0] == "systemctl" for c in self.commands))

    async def test_drain_timeout_reopens_admission_without_changes(self):
        self.manager.drain_maintenance = AsyncMock(side_effect=asyncio.TimeoutError)
        await self.execute()
        self.assert_originals()
        self.assertFalse(self.manager.maintenance)
        self.assertEqual(self.updater.job["phase"], "failed")

    async def test_second_file_failure_rolls_back_the_first_file(self):
        original_write = atomic_write
        failed = False

        def write(path, content, mode=0o600):
            nonlocal failed
            if path.parent == self.quadlets and f":{NEW}".encode() in content:
                journal = json.loads(self.updater.journal_path.read_text(encoding="utf-8"))
                self.assertTrue(journal["changed"], "Recovery journal must precede every modification")
                if "uncensored" in path.name and not failed:
                    failed = True
                    raise OSError("simulated file write failure")
            return original_write(path, content, mode)

        with patch.object(updates, "atomic_write", side_effect=write):
            await self.execute()
        self.assertTrue(failed)
        self.assert_originals()
        self.assertEqual(self.updater.job["phase"], "rolled_back")
        self.assertFalse(self.manager.maintenance)

    async def test_backend_start_failure_restores_configs_and_old_running_image(self):
        self.fail_new_start = True
        await self.execute()
        self.assert_originals()
        self.assertEqual(self.running_version, OLD)
        self.assertEqual(self.updater.job["phase"], "rolled_back")
        self.assertFalse(self.manager.maintenance)

    async def test_rollback_failure_keeps_gate_closed_and_can_be_retried(self):
        self.fail_all_starts = True
        await self.execute()
        self.assertTrue(self.manager.maintenance)
        self.assertTrue(self.updater.recovery_required)
        self.assertTrue(self.updater.status()["can_recover"])
        self.assertEqual(self.updater.job["phase"], "recovery_required")
        self.fail_all_starts = False
        await self.updater._retry_recovery()
        self.assert_originals()
        self.assertEqual(self.running_version, OLD)
        self.assertFalse(self.updater.recovery_required)
        self.assertFalse(self.manager.maintenance)

    async def test_interrupted_update_is_recovered_from_durable_journal(self):
        await self.execute()
        self.updater._save_job(phase="verifying", changed=True)
        self.manager.current_model = None
        recovered = self.make_updater()
        await recovered.recover_on_startup()
        self.assert_originals()
        self.assertEqual(recovered.job["phase"], "rolled_back")
        self.assertEqual(self.manager.current_model, MODEL)
        self.assertFalse(self.manager.maintenance)

    async def test_corrupt_journal_keeps_gate_closed_and_does_not_overwrite_evidence(self):
        self.updater.state_dir.mkdir()
        self.updater.journal_path.write_bytes(b"broken journal")
        await self.updater.recover_on_startup()
        self.assertTrue(self.manager.maintenance)
        self.assertTrue(self.updater.recovery_required)
        self.assertEqual(self.updater.journal_path.read_bytes(), b"broken journal")

    async def test_concurrent_click_is_rejected_while_pull_continues(self):
        self.pull_event = asyncio.Event()
        await self.updater.start(NEW)
        await asyncio.sleep(0)
        with self.assertRaises(web.HTTPConflict):
            await self.updater.start(NEW)
        self.assertEqual(await self.manager.reserve_request(MODEL), MODEL)
        await self.manager.release_request()
        self.pull_event.set()
        await self.updater.task
        self.assertEqual(self.updater.job["phase"], "succeeded")

    async def test_active_model_is_selected_after_pull_not_from_old_check(self):
        self.pull_event = asyncio.Event()
        await self.updater.start(NEW)
        await asyncio.sleep(0)
        self.manager.current_model = self.running_model = "qwen3.8-flash-uncensored"
        self.pull_event.set()
        await self.updater.task
        self.assertEqual(self.updater.job["model"], "qwen3.8-flash-uncensored")

    async def test_unknown_or_older_target_cannot_be_installed(self):
        for target in ("0.13.0", "0.99.0", "evil:1.0.0"):
            with self.assertRaises(web.HTTPConflict):
                await self.updater.start(target)
        self.assertFalse(self.commands)

    async def test_shutdown_during_verification_rolls_back(self):
        reached = asyncio.Event()
        ready = self.updater._ready

        async def wait_for_new(model, version, image_id):
            if version == NEW:
                reached.set()
                await asyncio.Event().wait()
            else:
                await ready(model, version, image_id)

        self.updater._ready = wait_for_new
        await self.updater.start(NEW)
        await reached.wait()
        await self.updater.close()
        self.assert_originals()
        self.assertEqual(self.updater.job["phase"], "rolled_back")
        self.assertFalse(self.manager.maintenance)

    async def test_failed_generated_unit_verification_restores_quadlets(self):
        self.updater._verify_units = AsyncMock(side_effect=UpdateError("Falsches generiertes Image"))
        await self.execute()
        self.assert_originals()
        self.assertEqual(self.updater.job["phase"], "rolled_back")

    async def test_existing_newer_config_never_offers_a_downgrade(self):
        path = next(iter(self.updater._paths().values()))
        path.write_bytes(quadlet("0.14.0"))
        self.assertFalse(self.updater.status()["can_install"])
        self.assertFalse(self.updater.status()["up_to_date"])
        self.assertIsNotNone(self.updater.status()["blocked_reason"])

    async def test_journal_without_changed_flag_is_not_treated_as_no_changes(self):
        self.updater.state_dir.mkdir()
        raw = b'{"phase":"configuring"}'
        self.updater.journal_path.write_bytes(raw)
        await self.updater.recover_on_startup()
        self.assertTrue(self.updater.recovery_required)
        self.assertTrue(self.manager.maintenance)
        self.assertEqual(self.updater.journal_path.read_bytes(), raw)

    async def test_ready_requires_api_engine_probe_model_and_image(self):
        good = await self.health()
        good["version"] = {"api": NEW, "engine": NEW}
        bad_cases = [None, {**good, "model": "wrong"}, {**good, "capability_probe": "failed"},
                     {**good, "version": {"api": NEW, "engine": OLD}}]
        self.running_version = NEW
        for bad in bad_cases:
            self.manager.backend_health = AsyncMock(side_effect=[bad, good])
            with patch.object(updates.asyncio, "sleep", new=AsyncMock()):
                await self.updater._ready(MODEL, NEW, "image-" + NEW)
            self.assertEqual(self.manager.backend_health.await_count, 2)
        self.manager.backend_health = AsyncMock(return_value=good)
        with self.assertRaises(UpdateError):
            await self.updater._ready(MODEL, NEW, "wrong-image")

    async def test_same_origin_json_actions_and_invalid_versions(self):
        app = web.Application()
        self.updater.register_routes(app)
        async with TestClient(TestServer(app)) as client:
            url = "/dashboard/api/updates/install"
            for headers in ({}, {"Origin": "https://foreign.example", "X-Halogen-Action": "update"}):
                response = await client.post(url, json={"version": NEW}, headers=headers)
                self.assertEqual(response.status, 403)
            headers = {"Origin": str(client.make_url("")).rstrip("/"), "X-Halogen-Action": "update"}
            response = await client.post(url, json={"version": "0.13.2;reboot"}, headers=headers)
            self.assertEqual(response.status, 400)
            response = await client.post(url, json={"version": NEW}, headers=headers)
            self.assertEqual(response.status, 202)
            await self.updater.task
            response = await client.get("/dashboard/api/updates")
            self.assertEqual((await response.json())["job"]["phase"], "succeeded")


class CheckpointUpgradeTests(unittest.IsolatedAsyncioTestCase):
    """Switching the official profile from the 0.14 w4b checkpoint to v2."""

    VERSION = "0.15.0"

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("HALOGEN_TEST_TMP"))
        self.root = Path(self.temp.name)
        self.quadlets = self.root / "quadlets"
        self.quadlets.mkdir()
        self.models = self.root / "models"
        self.models.mkdir()
        for model, service in MODELS.items():
            content = self.checkpoint_quadlet() if model == MODEL else self.uncensored_quadlet()
            (self.quadlets / service.replace(".service", ".container")).write_bytes(content)
        self.manager = ModelManager(None)
        self.manager.current_model = MODEL
        self.manager.wait_for_gtt_release = AsyncMock()
        self.manager.backend_health = AsyncMock(side_effect=self.health)
        self.running_model = MODEL
        self.commands = []
        self.fail_start = False
        self.free_gib = 200
        self.updater = self.make_updater()

    def checkpoint_quadlet(self):
        # A Unix-style host path: on Linux the models volume host never carries
        # a drive-letter colon, which would break the host:container:mode split.
        return (
            "[Container]\r\n"
            f"Image={IMAGE_REPOSITORY}:{self.VERSION}\r\n"
            "ContainerName=halogen-official\r\n"
            "Environment=HALOGEN_MODEL_ID=qwen3.8-flash\r\n"
            "Environment=HALOGEN_CHECKPOINT=/models/qwen38-flash-next-w4b.hgn\r\n"
            "Environment=HALOGEN_FLASH_PIN_TRUNK=0\r\n"
            "Volume=/home/tester/halogen/models:/models:ro,Z\r\n"
            "[Service]\r\nRestart=on-failure\r\n"
        ).encode()

    def quick_quadlet(self):
        # Quick-install profile: HALOGEN_DOWNLOAD but no explicit HALOGEN_CHECKPOINT.
        return (
            "[Container]\r\n"
            f"Image={IMAGE_REPOSITORY}:{self.VERSION}\r\n"
            "ContainerName=halogen-official\r\n"
            "Environment=HALOGEN_MODEL_ID=qwen3.8-flash\r\n"
            "Environment=HALOGEN_DOWNLOAD=peonist-ai/halogen-qwen3.8-flash-next\r\n"
            "Volume=/home/tester/halogen/models:/models:Z\r\n"
            "[Service]\r\nRestart=on-failure\r\n"
        ).encode()

    def uncensored_quadlet(self, checkpoint="/models/qwen3.8-flash-uncensored.hgn"):
        return (
            "[Container]\r\n"
            f"Image={IMAGE_REPOSITORY}:{self.VERSION}\r\n"
            "ContainerName=halogen-uncensored\r\n"
            "Environment=HALOGEN_MODEL_ID=qwen3.8-flash-uncensored\r\n"
            f"Environment=HALOGEN_CHECKPOINT={checkpoint}\r\n"
            "Environment=HALOGEN_TOKENIZER=/models/tokenizer\r\n"
            "Volume=/home/tester/halogen/models/uncensored:/models:ro,Z\r\n"
            "Volume=/home/tester/halogen/models/uncensored:/uncensored:ro,Z\r\n"
            "[Service]\r\nRestart=on-failure\r\n"
        ).encode()

    def make_updater(self):
        updater = ContainerUpdater(self.manager, None, MODELS,
                                  quadlet_dir=self.quadlets, backup_root=self.root / "backups", state_dir=self.root / "state")
        updater._support_error = lambda: None
        updater._run = AsyncMock(side_effect=self.command)
        updater.current_version = self.VERSION
        updater.checked_at = updater.attempted_at = time.time()
        return updater

    async def health(self):
        return {"status": "ok", "model": self.running_model, "in_flight": 0, "queued": 0,
                "version": {"api": self.VERSION, "engine": self.VERSION}, "capability_probe": "ok"}

    async def command(self, *args, **kwargs):
        self.commands.append(args)
        if args[:3] == ("podman", "image", "inspect"):
            return json.dumps([{"Id": "sha256:image-" + args[3].rsplit(":", 1)[1]}])
        elif args[:3] == ("podman", "container", "inspect"):
            return json.dumps([{"Image": "image-" + self.VERSION}])
        elif args[:3] == ("systemctl", "--user", "start"):
            if self.fail_start:
                self.fail_start = False  # fail only the forward start; the rollback start must succeed
                raise UpdateError("Start fehlgeschlagen")
            self.running_model = next(m for m, s in MODELS.items() if s == args[3])
        return ""

    def patch_disk(self, free_gib):
        return patch.object(ContainerUpdater, "_free_gib", return_value=float(free_gib))

    async def asyncTearDown(self):
        await self.updater.close()
        self.temp.cleanup()

    async def execute(self):
        await self.updater.start_checkpoint()
        await self.updater.task

    async def test_detection_needs_download_and_can_install(self):
        with self.patch_disk(self.free_gib):
            info = self.updater._checkpoint_info()
            self.assertTrue(info["available"])
            self.assertTrue(info["needs_download"])
            self.assertEqual(info["download_gib"], updates.CHECKPOINT_DOWNLOAD_GIB)
            self.assertIsNone(info["blocked_reason"])
            self.assertTrue(self.updater.status()["checkpoint"]["can_install"])

    async def test_low_disk_blocks_the_upgrade(self):
        with self.patch_disk(50):
            info = self.updater._checkpoint_info()
        self.assertTrue(info["available"])
        self.assertIsNotNone(info["blocked_reason"])
        self.assertFalse(self.updater.status()["checkpoint"]["can_install"])
        with self.assertRaises(web.HTTPConflict):
            await self.updater.start_checkpoint()

    async def test_files_already_present_skips_download(self):
        present = patch.object(ContainerUpdater, "_v2_files_present", return_value=True)
        present.start()
        self.addCleanup(present.stop)
        with self.patch_disk(self.free_gib):
            info = self.updater._checkpoint_info()
        self.assertTrue(info["available"])
        self.assertFalse(info["needs_download"])
        await self.execute()
        self.assertEqual(self.updater.job["phase"], "succeeded")
        content = (self.quadlets / "halogen-official.container").read_text(encoding="utf-8")
        self.assertNotIn("HALOGEN_DOWNLOAD", content)
        self.assertIn(":ro,Z", content)

    async def test_success_rewrites_quadlet_and_backups(self):
        with self.patch_disk(self.free_gib):
            await self.execute()
        self.assertEqual(self.updater.job["phase"], "succeeded")
        self.assertFalse(self.manager.maintenance)
        content = (self.quadlets / "halogen-official.container").read_text(encoding="utf-8")
        self.assertIn("HALOGEN_CHECKPOINT=/models/qwen38-flash-next-v2.hgn", content)
        self.assertIn("HALOGEN_DOWNLOAD=peonist-ai/halogen-qwen3.8-flash-next", content)
        self.assertNotIn("HALOGEN_FLASH_PIN_TRUNK", content)
        self.assertIn("/models:Z", content)
        self.assertNotIn(":ro", content)
        backup = Path(self.updater.job["backup_dir"])
        self.assertIn("checkpoint-update-", backup.name)
        self.assertIn("w4b", (backup / "halogen-official.container").read_text(encoding="utf-8"))

    async def test_official_upgrade_preserves_both_uncensored_layouts(self):
        uncensored = self.quadlets / "halogen-uncensored.container"
        for checkpoint in ("/models/qwen3.8-flash-uncensored.hgn", "/uncensored/qwen3.8-flash-uncensored.hgn"):
            with self.subTest(checkpoint=checkpoint):
                original = self.uncensored_quadlet(checkpoint)
                uncensored.write_bytes(original)
                (self.quadlets / "halogen-official.container").write_bytes(self.checkpoint_quadlet())
                with self.patch_disk(self.free_gib):
                    await self.execute()
                self.assertEqual(self.updater.job["phase"], "succeeded")
                self.assertEqual(uncensored.read_bytes(), original)

    async def test_uncensored_is_not_offered_the_official_upgrade(self):
        uncensored = self.quadlets / "halogen-uncensored.container"
        self.manager.current_model = self.running_model = "qwen3.8-flash-uncensored"
        for checkpoint in ("/models/qwen3.8-flash-uncensored.hgn", updates.LEGACY_W4B_PATH, None):
            with self.subTest(checkpoint=checkpoint):
                content = self.uncensored_quadlet(checkpoint or updates.LEGACY_W4B_PATH)
                if checkpoint is None:
                    content = content.replace(f"Environment=HALOGEN_CHECKPOINT={updates.LEGACY_W4B_PATH}\r\n".encode(), b"")
                uncensored.write_bytes(content)
                with self.patch_disk(self.free_gib), patch.object(ContainerUpdater, "_w4b_files_present", return_value=True):
                    self.assertFalse(self.updater._checkpoint_info()["available"])
                    with self.assertRaises(web.HTTPConflict):
                        await self.updater.start_checkpoint()
                self.assertEqual(uncensored.read_bytes(), content)

    async def test_model_switch_before_upgrade_task_leaves_both_profiles_unchanged(self):
        originals = {path: path.read_bytes() for path in self.updater._paths().values()}
        with self.patch_disk(self.free_gib):
            await self.updater.start_checkpoint()
            self.manager.current_model = self.running_model = "qwen3.8-flash-uncensored"
            await self.updater.task
        self.assertEqual(self.updater.job["phase"], "failed")
        self.assertFalse(self.manager.maintenance)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertFalse(any(args[:3] == ("systemctl", "--user", "start") for args in self.commands))

    async def test_model_switch_during_image_check_rejects_upgrade(self):
        originals = {path: path.read_bytes() for path in self.updater._paths().values()}

        async def inspect_then_switch(image):
            self.manager.current_model = self.running_model = "qwen3.8-flash-uncensored"
            return "image-" + self.VERSION

        self.updater._inspect_image = AsyncMock(side_effect=inspect_then_switch)
        with self.patch_disk(self.free_gib):
            with self.assertRaises(web.HTTPConflict):
                await self.updater.start_checkpoint()
        self.assertIsNone(self.updater.job)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)

    async def test_model_switch_while_entering_maintenance_leaves_profiles_unchanged(self):
        originals = {path: path.read_bytes() for path in self.updater._paths().values()}
        begin_maintenance = self.manager.begin_maintenance

        async def finish_switch_then_enter():
            self.manager.current_model = self.running_model = "qwen3.8-flash-uncensored"
            await begin_maintenance()

        self.manager.begin_maintenance = finish_switch_then_enter
        with self.patch_disk(self.free_gib):
            await self.execute()
        self.assertEqual(self.updater.job["phase"], "failed")
        self.assertFalse(self.manager.maintenance)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)

    async def test_checkpoint_recovery_preserves_uncensored_changes(self):
        with self.patch_disk(self.free_gib):
            await self.execute()
        self.assertEqual(self.updater.job["phase"], "succeeded")
        self.updater._save_job(phase="verifying", changed=True)
        uncensored = self.quadlets / "halogen-uncensored.container"
        customized = self.uncensored_quadlet() + b"RestartSec=30\r\n"
        uncensored.write_bytes(customized)
        recovered = self.make_updater()
        await recovered.recover_on_startup()
        self.assertEqual(recovered.job["phase"], "rolled_back")
        self.assertFalse(self.manager.maintenance)
        self.assertEqual(uncensored.read_bytes(), customized)
        self.assertEqual((self.quadlets / "halogen-official.container").read_bytes(), self.checkpoint_quadlet())

    async def test_legacy_checkpoint_journal_still_restores_both_profiles(self):
        with self.patch_disk(self.free_gib):
            await self.execute()
        backup = Path(self.updater.job["backup_dir"])
        uncensored = self.quadlets / "halogen-uncensored.container"
        original = self.uncensored_quadlet()
        (backup / uncensored.name).write_bytes(original)
        uncensored.write_bytes(updates.quadlet_checkpoint_upgrade(
            original, updates.CHECKPOINT_V2_PATH,
            download_repo=updates.OFFICIAL_WEIGHTS_REPO, writable_models=True,
        ))
        legacy_job = {**self.updater.job, "phase": "verifying", "changed": True,
                      "modes": {**self.updater.job["modes"], "qwen3.8-flash-uncensored": 0o600}}
        legacy_job.pop("changed_models")
        self.updater.journal_path.write_text(json.dumps(legacy_job), encoding="utf-8")
        self.manager.current_model = None
        recovered = self.make_updater()
        await recovered.recover_on_startup()
        self.assertEqual(recovered.job["phase"], "rolled_back")
        self.assertFalse(self.manager.maintenance)
        self.assertEqual(uncensored.read_bytes(), original)
        self.assertEqual((self.quadlets / "halogen-official.container").read_bytes(), self.checkpoint_quadlet())

    async def test_start_failure_rolls_back_to_w4b(self):
        self.fail_start = True
        with self.patch_disk(self.free_gib):
            await self.execute()
        self.assertEqual(self.updater.job["phase"], "rolled_back")
        self.assertFalse(self.manager.maintenance)
        content = (self.quadlets / "halogen-official.container").read_text(encoding="utf-8")
        self.assertIn("w4b", content)
        self.assertIn("HALOGEN_FLASH_PIN_TRUNK=0", content)
        self.assertIn(":ro,Z", content)

    async def test_not_offered_below_0_15(self):
        self.updater.current_version = "0.14.2"
        info = self.updater._checkpoint_info()
        self.assertFalse(info["available"])
        with self.assertRaises(web.HTTPConflict):
            await self.updater.start_checkpoint()

    async def test_already_v2_is_not_offered(self):
        for path in self.updater._paths().values():
            path.write_bytes(self.checkpoint_quadlet().replace(
                b"w4b", b"v2"))
        info = self.updater._checkpoint_info()
        self.assertFalse(info["available"])

    async def test_unset_checkpoint_with_w4b_files_is_offered(self):
        for path in self.updater._paths().values():
            path.write_bytes(self.quick_quadlet())
        present = patch.object(ContainerUpdater, "_w4b_files_present", return_value=True)
        absent = patch.object(ContainerUpdater, "_v2_files_present", return_value=False)
        present.start(); absent.start()
        self.addCleanup(present.stop); self.addCleanup(absent.stop)
        with self.patch_disk(self.free_gib):
            info = self.updater._checkpoint_info()
            self.assertTrue(info["available"])
            self.assertEqual(info["current"], updates.LEGACY_W4B_PATH)
            self.assertTrue(info["needs_download"])
            await self.execute()
        self.assertEqual(self.updater.job["phase"], "succeeded")
        content = (self.quadlets / "halogen-official.container").read_text(encoding="utf-8")
        # The checkpoint line was inserted after the Image line.
        self.assertIn("HALOGEN_CHECKPOINT=/models/qwen38-flash-next-v2.hgn", content)
        self.assertIn("HALOGEN_DOWNLOAD=peonist-ai/halogen-qwen3.8-flash-next", content)
        self.assertEqual(content.count("HALOGEN_DOWNLOAD"), 1)

    async def test_unset_checkpoint_with_v2_files_not_offered(self):
        for path in self.updater._paths().values():
            path.write_bytes(self.quick_quadlet())
        present = patch.object(ContainerUpdater, "_v2_files_present", return_value=True)
        present.start()
        self.addCleanup(present.stop)
        info = self.updater._checkpoint_info()
        self.assertFalse(info["available"])

    async def test_api_requires_same_origin_and_returns_202(self):
        app = web.Application()
        self.updater.register_routes(app)
        async with TestClient(TestServer(app)) as client:
            url = "/dashboard/api/updates/checkpoint"
            response = await client.post(url, json={}, headers={"Origin": "https://foreign.example", "X-Halogen-Action": "update"})
            self.assertEqual(response.status, 403)
            headers = {"Origin": str(client.make_url("")).rstrip("/"), "X-Halogen-Action": "update"}
            with self.patch_disk(50):
                response = await client.post(url, json={}, headers=headers)
                self.assertEqual(response.status, 409)
            with self.patch_disk(self.free_gib):
                response = await client.post(url, json={}, headers=headers)
                self.assertEqual(response.status, 202)
                await self.updater.task
            response = await client.get("/dashboard/api/updates")
            self.assertEqual((await response.json())["job"]["phase"], "succeeded")


class DiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_tag_pagination_semver_and_prerelease_filtering(self):
        pages = []

        async def tags(request):
            page = int(request.query["page"])
            pages.append(page)
            self.assertEqual(request.headers["Accept-Encoding"], "identity")
            if page == 1:
                return web.json_response([{"name": "v0.13.9"}] + [{"name": "v0.14.0-rc1"}] * 99)
            return web.json_response([{"name": "v0.13.10"}, {"name": "latest"}, {"name": "v0.12.0"}])

        app = web.Application()
        app.router.add_get("/tags", tags)
        async with TestServer(app) as server, ClientSession(auto_decompress=False) as session:
            updater = ContainerUpdater(None, session, MODELS)
            with patch.object(updates, "TAGS_URL", str(server.make_url("/tags"))):
                self.assertEqual(await updater._fetch_latest(), "0.13.10")
        self.assertEqual(pages, [1, 2])

    async def test_registry_fetch_parses_stable_tags(self):
        updater = ContainerUpdater(None, None, MODELS, tag_source="registry")

        async def fake_run(*args, **kwargs):
            self.assertEqual(args[:5], ("podman", "search", "--list-tags", "--limit", "1000"))
            return "0.12.0\n0.13.1\n0.13.2\n0.14.0-rc1\nlatest\n"

        updater._run = AsyncMock(side_effect=fake_run)
        self.assertEqual(await updater._fetch_latest(), "0.13.2")

    async def test_check_cache_and_network_failure_do_not_claim_up_to_date(self):
        manager = ModelManager(None)
        manager.backend_health = AsyncMock(return_value={"version": {"api": OLD}})
        updater = ContainerUpdater(manager, None, MODELS)
        updater._fetch_latest = AsyncMock(return_value=NEW)
        await updater.check()
        await updater.check()
        updater._fetch_latest.assert_awaited_once()
        updater.attempted_at = 0
        updater._fetch_latest.side_effect = UpdateError("GitHub nicht erreichbar")
        status = await updater.check(force=True)
        self.assertEqual(status["check_error"], "GitHub nicht erreichbar")
        self.assertFalse(status["can_install"])


if __name__ == "__main__":
    unittest.main()
