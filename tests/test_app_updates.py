"""Self-update discovery and handoff never install packages in these tests."""

import asyncio
import base64
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    updates = importlib.import_module("halobridge_updates")
except ModuleNotFoundError:
    updates = None


class AppUpdateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.assertIsNotNone(updates, "The self-update backend is missing")
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("HALOGEN_TEST_TMP"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = SimpleNamespace(
            app_updates=SimpleNamespace(enabled=True, check_interval_h=6, branch="develop", service="halobridge.service"),
            dashboard=SimpleNamespace(state_dir=self.root / "state"),
            security=SimpleNamespace(allow_install=True), router=SimpleNamespace(drain_timeout_s=2),
        )
        self.manager = SimpleNamespace(
            maintenance=False, switching=False, current_model="model", active_requests=0,
            begin_maintenance=AsyncMock(), drain_maintenance=AsyncMock(), end_maintenance=AsyncMock(),
        )
        self.version_patch = patch.object(updates.settings, "app_version", return_value="0.1.9")
        self.version_patch.start()
        self.addCleanup(self.version_patch.stop)
        self.updater = updates.AppUpdater(self.manager, None, config=self.config)
        self.updater._support_error = None
        self.updater._verify_support = AsyncMock(return_value=None)
        self.updater.source_ref = "develop"
        self.updater._source_error = None
        self.updater.commit = "a" * 40
        self.addAsyncCleanup(self.updater.close)

    async def test_check_compares_installed_package_versions_numerically(self):
        self.updater._fetch_latest = AsyncMock(return_value=("0.1.10", "a" * 40))
        result = await self.updater.check(force=True)
        self.assertEqual(result["current_version"], "0.1.9")
        self.assertEqual(result["latest_version"], "0.1.10")
        self.assertTrue(result["update_available"])
        self.assertTrue(result["can_install"])
        self.assertFalse(result["up_to_date"])

    async def test_failed_check_keeps_previous_data_but_disables_install_and_current_claim(self):
        self.updater._fetch_latest = AsyncMock(return_value=("0.1.9", "a" * 40))
        self.assertTrue((await self.updater.check(force=True))["up_to_date"])
        self.updater.attempted_at = 0
        self.updater._fetch_latest.side_effect = updates.AppUpdateError("GitHub unavailable")
        result = await self.updater.check(force=True)
        self.assertEqual(result["latest_version"], "0.1.9")
        self.assertIsNotNone(result["checked_at"])
        self.assertFalse(result["up_to_date"])
        self.assertFalse(result["can_install"])
        self.assertEqual(result["check_error"], "GitHub unavailable")

    async def test_local_prerelease_newer_or_stale_versions_never_install(self):
        self.updater.latest_version = "0.1.8"
        self.updater.checked_at = time.time()
        self.assertFalse(self.updater.status()["update_available"])
        self.assertFalse(self.updater.status()["can_install"])
        self.updater.latest_version = "0.1.10"
        self.updater.checked_at -= 7 * 3600
        self.assertFalse(self.updater.status()["can_install"])
        with patch.object(updates.settings, "app_version", return_value="0+local"):
            self.assertFalse(self.updater.status()["up_to_date"])
            self.assertFalse(self.updater.status()["can_install"])
        self.config.security.allow_install = False
        self.assertFalse(self.updater.status()["can_install"])

    async def test_repository_origin_and_requested_branch_are_validated(self):
        good = {"url": "https://github.com/Siyarbekir47/Halobridge.git", "vcs_info": {"vcs": "git", "requested_revision": "feature/fix", "commit_id": "b" * 40}}
        with patch.object(updates.metadata, "distribution", return_value=SimpleNamespace(read_text=lambda _: json.dumps(good))):
            self.assertEqual(updates.installed_source()["branch"], "feature/fix")
        for url in ("https://github.com/other/Halobridge.git", "https://github.com/Siyarbekir47/Halobridge.git.evil", "http://github.com/Siyarbekir47/Halobridge.git", "https://user:token@github.com/Siyarbekir47/Halobridge.git"):
            good["url"] = url
            with patch.object(updates.metadata, "distribution", return_value=SimpleNamespace(read_text=lambda _: json.dumps(good))):
                with self.assertRaises(updates.AppUpdateError):
                    updates.installed_source()

    async def test_github_contents_uses_branch_query_identity_and_verified_commit(self):
        requests = []
        async def contents(request):
            requests.append((request.query.get("ref"), request.headers.get("Accept-Encoding")))
            content = b'[project]\nname="halobridge"\nversion="0.1.10"\n'
            return web.json_response({"encoding": "base64", "content": base64.b64encode(content).decode(), "sha": "c" * 40})
        async def commit(request):
            self.assertEqual(request.match_info["ref"], "feature/fix")
            return web.json_response({"sha": "a" * 40})
        app = web.Application()
        app.router.add_get("/contents/pyproject.toml", contents)
        app.router.add_get("/commits/{ref:.*}", commit)
        async with TestServer(app) as server, ClientSession(auto_decompress=False) as session:
            self.updater.session = session
            self.updater.source_ref = "feature/fix"
            with patch.object(updates, "API_URL", str(server.make_url("/")).rstrip("/")):
                self.assertEqual(await self.updater._fetch_latest(), ("0.1.10", "a" * 40))
        self.assertEqual(requests, [("a" * 40, "identity")])

    async def test_oversized_or_unstable_project_response_is_rejected(self):
        async def oversized(request):
            return web.Response(body=b"x" * (1024 * 1024 + 1))
        app = web.Application()
        app.router.add_get("/commits/{ref}", oversized)
        async with TestServer(app) as server, ClientSession() as session:
            self.updater.session = session
            with patch.object(updates, "API_URL", str(server.make_url("/")).rstrip("/")):
                with self.assertRaises(updates.AppUpdateError):
                    await self.updater._fetch_latest()

    async def test_http_actions_require_same_origin_json_and_verified_version(self):
        app = web.Application()
        self.updater.register_routes(app)
        self.updater._fetch_latest = AsyncMock(return_value=("0.1.10", "a" * 40))
        async with TestClient(TestServer(app)) as client:
            path = "/dashboard/api/app-updates/install"
            for headers in ({}, {"Origin": "https://foreign.invalid", "X-Halogen-Action": "update"}):
                self.assertEqual((await client.post(path, json={"version": "0.1.10"}, headers=headers)).status, 403)
            headers = {"Origin": str(client.make_url("")).rstrip("/"), "X-Halogen-Action": "update"}
            for version, code in (("0.1.10;reboot", 400), (123, 400), (["0.1.10"], 400), ("0.1.8", 409)):
                self.assertEqual((await client.post(path, json={"version": version}, headers=headers)).status, code)
            self.config.security.allow_install = False
            self.assertEqual((await client.post(path, json={"version": "0.1.10"}, headers=headers)).status, 403)

    async def test_handoff_drains_requests_and_failed_dispatch_releases_maintenance(self):
        self.updater._fetch_latest = AsyncMock(return_value=("0.1.10", "a" * 40))
        gate = asyncio.Event()
        async def drain():
            await gate.wait()
        self.manager.drain_maintenance = AsyncMock(side_effect=drain)
        self.updater._dispatch = AsyncMock(side_effect=updates.AppUpdateError("cannot dispatch"))
        await self.updater.start("0.1.10")
        await asyncio.sleep(0)
        self.assertTrue(self.updater.running)
        self.assertFalse(self.updater._dispatch.called)
        with self.assertRaises(web.HTTPConflict):
            await self.updater.start("0.1.10")
        gate.set()
        await self.updater.task
        self.assertEqual(self.updater.status()["job"]["phase"], "failed")
        self.assertEqual(self.manager.end_maintenance.await_count, 1)

    async def test_other_maintenance_or_deploy_blocks_handoff(self):
        self.updater._fetch_latest = AsyncMock(return_value=("0.1.10", "a" * 40))
        for dependency in ("container", "deploy", "manager"):
            self.updater.container_updater = SimpleNamespace(running=dependency == "container", recovery_required=False)
            self.updater.deploy_manager = SimpleNamespace(deploy_lock=SimpleNamespace(locked=lambda: dependency == "deploy"))
            self.manager.maintenance = dependency == "manager"
            with self.assertRaises(web.HTTPConflict):
                await self.updater.start("0.1.10")
        self.assertFalse(self.manager.begin_maintenance.called)

    async def test_support_requires_current_service_pid_and_the_current_pipx_entry_point(self):
        prefix = self.root / "venvs/halobridge"
        (prefix / "bin").mkdir(parents=True)
        (prefix / "bin/halobridge").write_text("entry point")
        (prefix / "pipx_metadata.json").write_text(json.dumps({"main_package": {"package": "halobridge"}}))
        tools = self.root / "tools"
        tools.mkdir()
        for name in ("pipx", "systemctl", "systemd-run"):
            (tools / name).touch()
        async def run(*args, **kwargs):
            if args[1:4] == ("environment", "--value", "PIPX_LOCAL_VENVS"):
                return self.pipx_venvs
            if args[1:4] == ("environment", "--value", "PIPX_BIN_DIR"):
                return str(prefix / "bin")
            if args[1:4] == ("environment", "--value", "PIPX_HOME"):
                return str(prefix.parent.parent)
            return self.service_properties
        self.updater._run = run
        real_verify = updates.AppUpdater._verify_support.__get__(self.updater)
        self.service_properties = f"ActiveState=active\nMainPID={os.getpid()}\nExecStart={{ path={prefix / 'bin/halobridge'} ; argv[]=/ignored ; }}\n"
        self.pipx_venvs = str(prefix.parent)
        with patch.object(updates.sys, "platform", "linux"), patch.object(updates.sys, "prefix", str(prefix)), patch.object(updates.shutil, "which", side_effect=lambda name: str(tools / name)):
            self.assertIsNone(await real_verify())
            self.pipx_venvs = str(self.root / "different-venvs")
            self.assertIsNotNone(await real_verify())
            self.pipx_venvs = str(prefix.parent)
            self.service_properties = self.service_properties.replace(f"MainPID={os.getpid()}", "MainPID=1")
            self.assertIsNotNone(await real_verify())
            self.service_properties = f"ActiveState=active\nMainPID={os.getpid()}\nExecStart={{ path={tools / 'other-app'} ; }}\n"
            self.assertIsNotNone(await real_verify())
            (prefix / "pipx_metadata.json").unlink()
            self.assertIsNotNone(await real_verify())

    async def test_dispatch_copies_worker_and_launches_an_independent_systemd_unit(self):
        prefix = self.root / "venvs/halobridge"
        self.updater.tools = {"pipx": str(self.root / "pipx"), "systemctl": str(self.root / "systemctl"), "systemd_run": str(self.root / "systemd-run"), "python": sys.executable}
        self.updater.job = {"id": "test-update", "version": "0.1.10", "target_version": "0.1.10", "previous_version": "0.1.9", "phase": "draining", "source_ref": "develop"}
        calls = []
        async def run(*args, **kwargs):
            calls.append(args)
            return ""
        self.updater._run = run
        with patch.object(updates.sys, "prefix", str(prefix)):
            await self.updater._dispatch()
        command = calls[0]
        self.assertEqual(command[:3], (str(self.root / "systemd-run"), "--user", "--unit=halobridge-update-test-update.service"))
        self.assertIn("--property=Type=exec", command)
        self.assertIn("--property=Restart=no", command)
        self.assertIn("-I", command)
        copied = self.root / "state/app-update-worker-test-update.py"
        self.assertTrue(copied.is_file())
        self.assertNotIn(prefix, copied.parents)
        plan = json.loads((self.root / "state/app-update-plan-test-update.json").read_text())
        self.assertEqual(plan["previous_pid"], os.getpid())
        self.assertEqual(plan["commit"], "a" * 40)
        self.assertTrue(plan["drained"])

    async def test_worker_failure_observed_from_durable_journal_releases_admission_gate(self):
        self.updater._maintenance_owned = True
        self.updater._save_job(id="test-update", phase="rolled_back", message="previous version restored")
        with patch.object(updates.asyncio, "sleep", new=AsyncMock()):
            await self.updater._monitor()
        self.assertFalse(self.updater.running)
        self.assertEqual(self.manager.end_maintenance.await_count, 1)

    async def test_monitor_retries_unavailable_service_probe_and_observes_worker_completion(self):
        for error in (asyncio.TimeoutError(), OSError("temporary service probe failure")):
            with self.subTest(error=type(error).__name__):
                self.manager.end_maintenance.reset_mock()
                self.updater._maintenance_owned = True
                self.updater.tools = {"systemctl": "systemctl"}
                self.updater._save_job(id="test-update", phase="installing", worker_unit="halobridge-update-test.service")
                async def probe(*args, **kwargs):
                    self.updater._save_job(phase="rolled_back", message="previous version restored")
                    raise error
                with patch.object(self.updater, "_run", side_effect=probe), \
                     patch.object(updates.asyncio, "sleep", new=AsyncMock()):
                    await self.updater._monitor()
                self.assertFalse(self.updater.running)
                self.assertEqual(self.manager.end_maintenance.await_count, 1)

    async def test_orphan_probe_timeout_preserves_uncertain_worker_state(self):
        self.updater._save_job(id="test-update", phase="installing", worker_unit="halobridge-update-test.service", updated_at=0)
        with patch.object(updates.shutil, "which", return_value="systemctl"), \
             patch.object(self.updater, "_run", side_effect=asyncio.TimeoutError()):
            await self.updater._recover_orphan()
        self.assertEqual(self.updater.status()["job"]["phase"], "installing")

    async def test_malformed_source_metadata_cannot_break_startup(self):
        for data in ([], {"url": []}, {"url": "https://github.com/Siyarbekir47/Halobridge.git", "vcs_info": []}, {"url": "https://github.com/Siyarbekir47/Halobridge.git", "dir_info": [], "vcs_info": {"vcs": "git"}}):
            with patch.object(updates.metadata, "distribution", return_value=SimpleNamespace(read_text=lambda _: json.dumps(data))):
                with self.assertRaises(updates.AppUpdateError):
                    updates.installed_source()

    async def test_orphaned_preinstall_job_is_cleared_on_startup(self):
        self.updater._save_job(id="orphan", phase="draining", message="old process died")
        await self.updater._recover_orphan()
        self.assertEqual(self.updater.status()["job"]["phase"], "failed")
        self.assertFalse(self.updater.running)

    async def test_orphan_after_install_requires_recovery(self):
        self.updater._save_job(id="orphan", phase="installing", backup_dir=str(self.root / "backup"))
        await self.updater._recover_orphan()
        self.assertEqual(self.updater.status()["job"]["phase"], "recovery_required")
        self.assertFalse(self.updater.status()["can_install"])

    async def test_startup_never_overwrites_a_job_held_by_another_process(self):
        self.updater._save_job(id="live", phase="draining")
        lock = updates.acquire_lock(self.updater.lock_path)
        try:
            await self.updater._recover_orphan()
        finally:
            lock.close()
        self.assertEqual(self.updater.status()["job"]["phase"], "draining")

    async def test_local_source_startup_does_not_query_github(self):
        self.config.app_updates.branch = ""
        self.updater._source_error = "local source installation"
        self.updater._fetch_latest = AsyncMock()
        self.updater.start_checks()
        await asyncio.sleep(0)
        self.assertFalse(self.updater._fetch_latest.called)


if __name__ == "__main__":
    unittest.main()
