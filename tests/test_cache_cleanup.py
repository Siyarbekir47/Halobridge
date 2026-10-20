import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cache_cleanup import CacheCleanup, candidates
from settings import Config, ConfigError


class CleanupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "cache"
        self.root.mkdir()
        config = Config()
        config.deploy.cache_root = self.root
        config.deploy.models_root = Path(self.tmp.name) / "models"
        manager = SimpleNamespace(current_model="official", models={"official": "halogen-official.service"},
            active_requests=0, maintenance=False, drain_timeout=1, stop_timeout=1,
            begin_maintenance=AsyncMock(), drain_maintenance=AsyncMock(), end_maintenance=AsyncMock(),
            wait_for_gtt_release=AsyncMock(), start_service=AsyncMock())
        self.deploy = SimpleNamespace(config=config, manager=manager, updater=None, job=None,
            deploy_lock=asyncio.Lock(), recovery_required=False,
            npu_assets=SimpleNamespace(profiles=lambda: []), _run=AsyncMock(), _service_is_active=AsyncMock(return_value=True))
        self.cleaner = CacheCleanup(self.deploy, SimpleNamespace(running=False))

    def group(self, name, stamp=10):
        p = self.root / "official" / name
        p.mkdir(parents=True)
        file = p / "conversation.rec"
        file.write_bytes(b"cached data")
        os.utime(file, (stamp, stamp))
        return p

    async def test_low_space_removes_old_configuration_and_restarts_backend(self):
        old = self.group("a" * 16)
        new = self.group("b" * 16, 20)
        sentinel = self.root / "unrelated.txt"
        sentinel.write_text("keep")
        def usage(_):
            return SimpleNamespace(total=100, free=10 if old.exists() else 25)
        with patch("cache_cleanup.shutil.disk_usage", side_effect=usage):
            await self.cleaner.scan()
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())
        self.assertTrue(sentinel.exists())
        self.deploy._run.assert_awaited_once()
        self.deploy.manager.start_service.assert_awaited_once_with("official")
        self.deploy.manager.end_maintenance.assert_awaited_once()
        self.assertEqual(self.cleaner.last_result["freed_bytes"], 15)

    async def test_normal_space_and_busy_backend_do_not_delete(self):
        old = self.group("a" * 16)
        with patch("cache_cleanup.shutil.disk_usage", return_value=SimpleNamespace(total=100, free=16)):
            await self.cleaner.scan()
        self.deploy.manager.active_requests = 1
        with patch("cache_cleanup.shutil.disk_usage", return_value=SimpleNamespace(total=100, free=0)):
            await self.cleaner.scan()
        self.assertTrue(old.exists())
        self.deploy._run.assert_not_awaited()

    async def test_disabled_cleanup_never_stops_backend(self):
        old = self.group("a" * 16)
        self.cleaner.config.cache_cleanup_enabled = False
        with patch("cache_cleanup.shutil.disk_usage", return_value=SimpleNamespace(total=100, free=0)):
            await self.cleaner.scan()
        self.assertTrue(old.exists())
        self.deploy._run.assert_not_awaited()

    async def test_another_explicit_backend_defers_cleanup(self):
        old = self.group("a" * 16)
        self.deploy.manager.models["manual"] = "manually-configured.service"
        with patch("cache_cleanup.shutil.disk_usage", return_value=SimpleNamespace(total=100, free=0)):
            await self.cleaner.scan()
        self.assertTrue(old.exists())
        self.deploy._run.assert_not_awaited()
        self.deploy.manager.end_maintenance.assert_awaited_once()

    def test_protected_record_directory_never_selected(self):
        group = self.group("a" * 16)
        record = group / "request-records"
        record.mkdir()
        self.assertEqual(candidates(self.root, self.deploy.config.deploy.models_root, [record.resolve()]), [])

    async def test_custom_cache_mount_uses_oldest_configuration_on_same_disk(self):
        newer = self.group("a" * 16, 20)
        custom = Path(self.tmp.name) / "custom-cache"
        older = custom / ("b" * 16)
        older.mkdir(parents=True)
        rec = older / "old.rec"
        rec.write_bytes(b"old cache")
        os.utime(rec, (10, 10))
        profile = SimpleNamespace(env={}, volumes=[(str(custom), "/cache", "rw")],
                                  model_id="official", service_name="halogen-official.service")
        self.deploy.npu_assets.profiles = lambda: [profile]
        with patch("cache_cleanup.shutil.disk_usage", side_effect=lambda _: SimpleNamespace(total=100, free=10 if older.exists() else 25)):
            await self.cleaner.scan()
        self.assertFalse(older.exists())
        self.assertTrue(newer.exists())

    async def test_delete_failure_still_restores_backend(self):
        self.group("a" * 16)
        with patch("cache_cleanup.shutil.disk_usage", return_value=SimpleNamespace(total=100, free=15)), \
             patch("cache_cleanup.shutil.rmtree", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                await self.cleaner.scan()
        self.deploy.manager.start_service.assert_awaited_once()
        self.deploy.manager.end_maintenance.assert_awaited_once()

    async def test_stop_failure_does_not_delete(self):
        old = self.group("a" * 16)
        self.deploy._run.side_effect = RuntimeError("could not stop")
        with patch("cache_cleanup.shutil.disk_usage", return_value=SimpleNamespace(total=100, free=0)):
            with self.assertRaises(RuntimeError):
                await self.cleaner.scan()
        self.assertTrue(old.exists())
        self.deploy.manager.start_service.assert_not_awaited()

    def test_unknown_directories_and_model_weights_never_selected(self):
        self.group("unknown")
        group = self.group("a" * 16)
        (group / "weights.hgn").write_bytes(b"weights")
        self.assertEqual(candidates(self.root, self.deploy.config.deploy.models_root), [])
        self.assertEqual(candidates(self.root, self.root), [])

    def test_cleanup_threshold_validation(self):
        with self.assertRaises(ConfigError):
            Config.from_dict({"deploy": {"cache_cleanup_free_percent": 90, "cache_cleanup_target_percent": 20}})

    async def test_cancellation_waits_for_backend_restoration(self):
        self.group("a" * 16)
        entered, release = asyncio.Event(), asyncio.Event()
        async def stop(*args, **kwargs):
            entered.set()
            await release.wait()
        self.deploy._run.side_effect = stop
        with patch("cache_cleanup.shutil.disk_usage", return_value=SimpleNamespace(total=100, free=10)):
            task = asyncio.create_task(self.cleaner.scan())
            await entered.wait()
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.deploy.manager.start_service.assert_awaited_once()
        self.deploy.manager.end_maintenance.assert_awaited_once()

    def test_symlink_configuration_and_nested_symlinks_are_skipped(self):
        group = self.group("a" * 16)
        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()
        try:
            (group / "linked").symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("Symlinks require privileges on this host")
        self.assertEqual(candidates(self.root, self.deploy.config.deploy.models_root), [])


if __name__ == "__main__":
    unittest.main()
