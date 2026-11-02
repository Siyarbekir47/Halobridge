"""Backend restart jobs use real admission/file handling and mocked systemd."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from test_deploy import make_config
from halogen_deploy import DeployError, DeployManager, DeployRoutes
from halogen_router import ModelManager, RouterError, make_auth_middleware
from profiles import official_template, render_quadlet


class BackendRestartTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.platform = patch("halogen_deploy.sys.platform", "linux")
        self.platform.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.profile = official_template("0.18.18", self.root / "models", self.root / "cache")
        self.profile.env["HALOGEN_CACHE_BRANCHES"] = "3"
        self.manager = ModelManager(None, {self.profile.model_id: self.profile.service_name})
        self.manager.current_model = self.profile.model_id
        self.manager.asset_validator = AsyncMock()
        self.manager.backend_health = AsyncMock(return_value={"status": "ok", "model": self.profile.model_id,
                                                            "in_flight": 0, "queued": 0})
        self.commands = []

        async def command(*args, **kwargs):
            self.commands.append(args)
            return 0, "", ""

        self.manager.run_command = AsyncMock(side_effect=command)
        self.manager.wait_for_gtt_release = AsyncMock(side_effect=lambda: self.commands.append(("gpu-free",)))
        self.manager.wait_for_backend = AsyncMock(side_effect=lambda model, timeout=None: self.commands.append(("ready", model)))
        self.manager.service_is_active = AsyncMock(return_value=True)
        self.deploy = DeployManager(self.manager, make_config(self.root))
        self.deploy.quadlet_dir.mkdir()
        self.path = self.deploy.quadlet_dir / "halogen-official.container"
        self.original = render_quadlet(self.profile).encode()
        self.path.write_bytes(self.original)
        self.deploy._service_is_active = AsyncMock(return_value=True)
        self.deploy._daemon_reload = AsyncMock(side_effect=lambda: self.commands.append(("reload",)))
        self.deploy._verify_unit = AsyncMock(side_effect=lambda profile: self.commands.append(("verify", profile.service_name)))
        self.deploy._stream_service_journal = AsyncMock()

    async def asyncTearDown(self):
        task = self.deploy._job_task
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.tmp.cleanup()
        self.platform.stop()

    async def finish_restart(self):
        result = await self.deploy.restart("official")
        await self.deploy._job_task
        return result

    async def test_restarts_same_backend_with_saved_settings_and_readiness_check(self):
        result = await self.finish_restart()
        service, model = self.profile.service_name, self.profile.model_id
        self.assertEqual(result, {"ok": True, "kind": "restart-backend", "service": service})
        self.assertEqual(self.commands, [("reload",), ("verify", service),
                         ("systemctl", "--user", "stop", service), ("gpu-free",),
                         ("systemctl", "--user", "reset-failed", service),
                         ("systemctl", "--user", "start", service), ("ready", model)])
        self.assertEqual(self.deploy.job_status()["state"], "done")
        self.assertEqual(self.deploy.job_status()["profile_id"], "official")
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertFalse(self.deploy.backup_root.exists())
        self.assertEqual(self.manager.current_model, model)
        self.assertFalse(self.manager.switching)

    async def test_active_requests_finish_before_any_stop_or_reload(self):
        self.manager.active_requests = 1
        await self.deploy.restart("official")
        await asyncio.sleep(0)
        self.assertTrue(self.manager.switching)
        self.assertEqual(self.commands, [])
        async with self.manager.condition:
            self.manager.active_requests = 0
            self.manager.condition.notify_all()
        await self.deploy._job_task
        self.assertEqual(self.deploy.job["state"], "done")

    async def test_running_job_remains_running_until_backend_is_ready(self):
        entered, ready = asyncio.Event(), asyncio.Event()

        async def wait(model, timeout=None):
            entered.set()
            await ready.wait()

        self.manager.wait_for_backend = AsyncMock(side_effect=wait)
        await self.deploy.restart("official")
        await asyncio.wait_for(entered.wait(), 2)
        self.assertEqual(self.deploy.job["state"], "running")
        self.assertTrue(self.manager.switching)
        ready.set()
        await self.deploy._job_task
        self.assertEqual(self.deploy.job["state"], "done")

    async def test_cancel_during_drain_keeps_backend_running(self):
        self.manager.active_requests = 1
        await self.deploy.restart("official")
        await asyncio.sleep(0)
        self.deploy.cancel_job()
        await asyncio.gather(self.deploy._job_task, return_exceptions=True)
        self.assertEqual(self.deploy.job["state"], "cancelled")
        self.assertEqual(self.commands, [])
        self.assertFalse(self.manager.switching)
        self.assertEqual(self.path.read_bytes(), self.original)

    async def test_start_failure_uses_switch_recovery_and_reports_error(self):
        self.manager.wait_for_backend.side_effect = [RouterError("not ready"), None]
        await self.finish_restart()
        self.assertEqual(self.deploy.job["state"], "error")
        self.assertIn("not ready", self.deploy.job["error"])
        self.assertEqual(self.commands.count(("systemctl", "--user", "start", self.profile.service_name)), 2)
        self.assertFalse(self.manager.switching)
        self.assertEqual(self.path.read_bytes(), self.original)

    async def test_cancel_after_stop_recovers_the_backend_before_finishing(self):
        entered = asyncio.Event()
        calls = 0

        async def wait(model, timeout=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await asyncio.Event().wait()

        self.manager.wait_for_backend = AsyncMock(side_effect=wait)
        await self.deploy.restart("official")
        await asyncio.wait_for(entered.wait(), 2)
        self.deploy.cancel_job()
        await asyncio.gather(self.deploy._job_task, return_exceptions=True)
        self.assertEqual(self.deploy.job["state"], "cancelled")
        self.assertEqual(calls, 2)
        self.assertFalse(self.manager.switching)
        self.assertEqual(self.path.read_bytes(), self.original)

    async def test_profile_changed_while_waiting_is_rejected_before_stop(self):
        self.manager.active_requests = 1
        await self.deploy.restart("official")
        await asyncio.sleep(0)
        changed = self.original.replace(b"HALOGEN_CACHE_BRANCHES=3", b"HALOGEN_CACHE_BRANCHES=4")
        self.path.write_bytes(changed)
        async with self.manager.condition:
            self.manager.active_requests = 0
            self.manager.condition.notify_all()
        await self.deploy._job_task
        self.assertEqual(self.deploy.job["state"], "error")
        self.assertIn("Profile changed", self.deploy.job["error"])
        self.assertEqual(self.commands, [])
        self.assertEqual(self.path.read_bytes(), changed)

    async def test_saved_model_id_change_still_restarts_the_same_service(self):
        self.profile.env["HALOGEN_MODEL_ID"] = "renamed-model"
        self.path.write_bytes(render_quadlet(self.profile).encode())
        await self.finish_restart()
        self.assertEqual(self.deploy.job["state"], "done")
        self.assertIn(("systemctl", "--user", "stop", self.profile.service_name), self.commands)
        self.manager.wait_for_backend.assert_awaited_once_with("renamed-model", timeout=self.manager.start_timeout)
        self.assertEqual(self.manager.current_model, "renamed-model")

    async def test_stopped_or_other_active_backend_is_not_restarted(self):
        self.deploy._service_is_active.return_value = False
        with self.assertRaisesRegex(DeployError, "use Start"):
            await self.deploy.restart("official")
        self.deploy._service_is_active.return_value = True
        self.deploy._any_other_service_active = AsyncMock(return_value=True)
        with self.assertRaisesRegex(DeployError, "Another backend"):
            await self.deploy.restart("official")
        self.assertIsNone(self.deploy.job)
        self.assertEqual(self.commands, [])

    async def test_invalid_disabled_and_busy_requests_cannot_start_jobs(self):
        for value in ("../other", "official\n", "missing"):
            with self.assertRaises(DeployError):
                await self.deploy.restart(value)
        self.deploy.config.deploy.enabled = False
        with self.assertRaisesRegex(DeployError, "disabled"):
            await self.deploy.restart("official")
        self.deploy.config.deploy.enabled = True
        self.deploy.job = {"kind": "verify", "state": "running"}
        with self.assertRaisesRegex(DeployError, "already running"):
            await self.deploy.restart("official")
        self.assertIsNone(self.deploy._job_task)
        self.assertEqual(self.commands, [])

    async def test_restart_route_uses_auth_origin_and_action_guards(self):
        app = web.Application(middlewares=[make_auth_middleware("secret")])
        DeployRoutes(self.deploy).register_routes(app)
        path = "/dashboard/api/deploy/official/restart"
        async with TestClient(TestServer(app)) as client:
            response = await client.post(path, allow_redirects=False)
            self.assertEqual(response.status, 401)
            response = await client.post(path, headers={"Authorization": "Bearer secret"})
            self.assertEqual(response.status, 400)
            headers = {"Authorization": "Bearer secret", "X-Halogen-Action": "deploy"}
            response = await client.post(path, headers={**headers, "Origin": "http://other.example"})
            self.assertEqual(response.status, 400)
            self.assertIsNone(self.deploy.job)
            response = await client.post(path, headers=headers)
            self.assertEqual(response.status, 202)
            await self.deploy._job_task
            response = await client.get("/dashboard/api/deploy/job", headers=headers)
            self.assertEqual((await response.json())["state"], "done")

    async def test_mutations_are_blocked_during_restart_but_cancel_is_allowed(self):
        self.manager.active_requests = 1
        await self.deploy.restart("official")
        await asyncio.sleep(0)
        app = web.Application()
        DeployRoutes(self.deploy).register_routes(app)
        async with TestClient(TestServer(app)) as client:
            headers = {"X-Halogen-Action": "deploy"}
            response = await client.post("/dashboard/api/deploy/apply", json={}, headers=headers)
            self.assertEqual(response.status, 400)
            response = await client.post("/dashboard/api/deploy/job/cancel", headers=headers)
            self.assertEqual(response.status, 200)
            await asyncio.gather(self.deploy._job_task, return_exceptions=True)
        self.assertFalse(self.manager.switching)
        self.assertEqual(self.commands, [])
