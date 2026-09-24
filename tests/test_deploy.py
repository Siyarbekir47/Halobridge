"""DeployManager: dry-run/diff, backup-before-write, rollback and guards.

systemd interaction is mocked; file handling runs against temp directories.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from halogen_deploy import (
    HF_CLI_PACKAGE,
    QUICK_OFFICIAL_START_TIMEOUT,
    DeployError,
    DeployManager,
    _hf_access_denied,
)
from profiles import official_template, parse_quadlet, profile_to_dict, render_quadlet


def make_config(tmp: Path):
    return SimpleNamespace(
        models=SimpleNamespace(
            quadlet_dir=tmp / "quadlets",
            explicit={},
            auto_discover=True,
        ),
        updates=SimpleNamespace(
            backup_dir=tmp / "backups",
            image_repo="ghcr.io/peonist-ai/halogen-flash-server",
        ),
        dashboard=SimpleNamespace(state_dir=tmp / "state"),
        deploy=SimpleNamespace(
            enabled=True,
            allowed_roots=[tmp],
            models_root=tmp / "models",
            cache_root=tmp / "cache",
        ),
    )


def make_manager(tmp: Path, service_active=False) -> DeployManager:
    manager = SimpleNamespace(
        current_model=None, switching=False, maintenance=False, models={}
    )
    deploy = DeployManager(manager, make_config(tmp))
    deploy._daemon_reload = AsyncMock()
    deploy._verify_unit = AsyncMock()
    deploy._service_is_active = AsyncMock(return_value=bool(service_active))
    return deploy


def official_payload(tmp: Path) -> dict:
    profile = official_template(
        "0.13.2", tmp / "models", tmp / "cache"
    )
    return profile_to_dict(profile)


class PosixTestCase(unittest.TestCase):
    """Tests for the deployment path pretend to run on Linux."""

    def setUp(self):
        self._platform = patch("halogen_deploy.sys.platform", "linux")
        self._platform.start()
        self.addCleanup(self._platform.stop)


class DryRunTests(PosixTestCase):
    def setUp(self):
        super().setUp()
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)
        self.manager = make_manager(self.tmp)

    def tearDown(self):
        self._dir.cleanup()

    def test_dry_run_ok_with_diff(self):
        result = self.manager.dry_run(official_payload(self.tmp))
        self.assertTrue(result["ok"], result["errors"])
        self.assertIn("HALOGEN_MODEL_ID", result["quadlet"])
        self.assertIn("PublishPort=127.0.0.1:8831:8731", result["diff"])
        self.assertFalse(result["exists"])

    def test_dry_run_rejects_path_outside_allowed_roots(self):
        payload = official_payload(self.tmp)
        payload["volumes"][0][0] = str(self.tmp.parent / "outside-models")
        result = self.manager.dry_run(payload)
        self.assertFalse(result["ok"])
        self.assertTrue(any("allowed roots" in e for e in result["errors"]))

    def test_dry_run_rejects_unknown_env(self):
        payload = official_payload(self.tmp)
        payload["env"]["HALOGEN_SOMETHING_ELSE"] = "1"
        result = self.manager.dry_run(payload)
        self.assertFalse(result["ok"])

    def test_shared_port_is_warning_not_error(self):
        self.manager.quadlet_dir.mkdir(parents=True, exist_ok=True)
        other = official_template("0.13.2", self.tmp / "models", self.tmp / "cache")
        other.profile_id = "other"
        other.env["HALOGEN_MODEL_ID"] = "other-model"
        other.host_port = 8831
        (self.manager.quadlet_dir / "halogen-other.container").write_text(
            render_quadlet(other), encoding="utf-8"
        )
        result = self.manager.dry_run(official_payload(self.tmp))
        self.assertTrue(result["ok"])
        self.assertTrue(any("Port" in w for w in result["warnings"]))

    def test_dry_run_detects_model_id_conflict(self):
        self.manager.quadlet_dir.mkdir(parents=True, exist_ok=True)
        other = official_template("0.13.2", self.tmp / "models", self.tmp / "cache")
        other.profile_id = "other"
        (self.manager.quadlet_dir / "halogen-other.container").write_text(
            render_quadlet(other), encoding="utf-8"
        )
        result = self.manager.dry_run(official_payload(self.tmp))
        self.assertFalse(result["ok"])
        self.assertTrue(any("Model ID" in e for e in result["errors"]))


class ApplyTests(PosixTestCase):
    def setUp(self):
        super().setUp()
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)
        self.manager = make_manager(self.tmp)

    def tearDown(self):
        self._dir.cleanup()

    def test_apply_writes_quadlet(self):
        result = asyncio.run(self.manager.apply(official_payload(self.tmp)))
        target = self.tmp / "quadlets" / "halogen-official.container"
        self.assertTrue(result["ok"])
        self.assertTrue(target.exists())
        parsed = parse_quadlet(target.read_text(encoding="utf-8"))
        self.assertEqual(parsed.model_id, "qwen3.8-flash")
        self.manager._daemon_reload.assert_awaited()

    def test_apply_creates_backup_on_change(self):
        asyncio.run(self.manager.apply(official_payload(self.tmp)))
        payload = official_payload(self.tmp)
        payload["env"]["HALOGEN_KV_SLOTS"] = "4"
        result = asyncio.run(self.manager.apply(payload))
        self.assertIsNotNone(result["backup"])
        backup = Path(result["backup"])
        self.assertTrue(backup.exists())
        old = parse_quadlet(backup.read_text(encoding="utf-8"))
        self.assertEqual(old.env["HALOGEN_KV_SLOTS"], "2")

    def test_apply_rolls_back_when_unit_verification_fails(self):
        asyncio.run(self.manager.apply(official_payload(self.tmp)))
        self.manager._verify_unit = AsyncMock(side_effect=DeployError("Unit kaputt"))
        with self.assertRaises(DeployError):
            asyncio.run(
                self.manager.apply(
                    {**official_payload(self.tmp), "image": "ghcr.io/peonist-ai/halogen-flash-server:9.9.9"}
                )
            )
        target = self.tmp / "quadlets" / "halogen-official.container"
        self.assertIn("0.13.2", target.read_text(encoding="utf-8"))

    def test_apply_rejects_invalid_payload(self):
        payload = official_payload(self.tmp)
        payload["env"]["HALOGEN_KV_SLOTS"] = "9999"
        with self.assertRaises(DeployError):
            asyncio.run(self.manager.apply(payload))

    def test_disabled_deployment_refuses_everything(self):
        config = make_config(self.tmp)
        config.deploy.enabled = False
        manager = DeployManager(SimpleNamespace(current_model=None, switching=False, maintenance=False), config)
        with self.assertRaises(DeployError):
            asyncio.run(manager.apply(official_payload(self.tmp)))


class DeleteRollbackTests(PosixTestCase):
    def setUp(self):
        super().setUp()
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)
        self.manager = make_manager(self.tmp)
        asyncio.run(self.manager.apply(official_payload(self.tmp)))

    def tearDown(self):
        self._dir.cleanup()

    def test_delete_refused_while_service_active(self):
        active = make_manager(self.tmp, service_active=True)
        active._daemon_reload = AsyncMock()
        with self.assertRaises(DeployError):
            asyncio.run(active.delete("official"))
        self.assertTrue((self.tmp / "quadlets" / "halogen-official.container").exists())

    def test_delete_removes_file_keeps_backup(self):
        result = asyncio.run(self.manager.delete("official"))
        self.assertTrue(result["ok"])
        self.assertFalse((self.tmp / "quadlets" / "halogen-official.container").exists())
        self.assertTrue(Path(result["backup"]).exists())

    def test_backups_in_same_microsecond_do_not_overwrite_each_other(self):
        target = self.tmp / "quadlets" / "halogen-official.container"
        original = target.read_bytes()
        with patch("halogen_deploy.datetime") as clock, patch(
            "halogen_deploy.time.time_ns", return_value=1
        ):
            clock.now.return_value.strftime.return_value = "20260924-121235-123456"
            first = self.manager._backup_existing(target)
            target.write_bytes(b"new profile")
            second = self.manager._backup_existing(target)

        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(second.read_bytes(), b"new profile")

    def test_rollback_restores_previous_state(self):
        payload = official_payload(self.tmp)
        payload["env"]["HALOGEN_KV_SLOTS"] = "4"
        asyncio.run(self.manager.apply(payload))
        result = asyncio.run(self.manager.rollback("official"))
        target = self.tmp / "quadlets" / "halogen-official.container"
        parsed = parse_quadlet(target.read_text(encoding="utf-8"))
        self.assertEqual(
            parsed.env["HALOGEN_KV_SLOTS"],
            "2",
            [(path.name, path.read_bytes()) for path in self.manager._backups_for(target.name)],
        )
        self.assertTrue(Path(result["restored_from"]).exists())

    def test_rollback_without_backup_fails(self):
        with self.assertRaises(DeployError):
            asyncio.run(self.manager.rollback("nonexistent"))

    def test_delete_rejects_bad_ids(self):
        for bad in ("../evil", "Bad ID", ""):
            with self.assertRaises(DeployError):
                asyncio.run(self.manager.delete(bad))


class InstallDirsTests(PosixTestCase):
    def setUp(self):
        super().setUp()
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)
        self.manager = make_manager(self.tmp)

    def tearDown(self):
        self._dir.cleanup()

    def test_install_dirs_creates_volume_roots(self):
        result = self.manager.install_dirs(official_payload(self.tmp))
        self.assertTrue(result["ok"])
        for created in result["created"]:
            self.assertTrue(Path(created).is_dir())

    def test_install_dirs_refuses_outside_roots(self):
        payload = official_payload(self.tmp)
        payload["volumes"][0][0] = "/var/lib/secret"
        with self.assertRaises(DeployError):
            self.manager.install_dirs(payload)


class RouteGuardTests(PosixTestCase):
    def test_guard_requires_action_header(self):
        from halogen_deploy import DeployRoutes

        routes = DeployRoutes(make_manager(Path(".")))
        request = SimpleNamespace(
            headers={"Host": "localhost:8820", "Origin": "http://localhost:8820"}
        )
        with self.assertRaises(DeployError):
            routes._guard(request)
        request = SimpleNamespace(
            headers={
                "Host": "localhost:8820",
                "Origin": "http://evil.example",
                "X-Halogen-Action": "deploy",
            }
        )
        with self.assertRaises(DeployError):
            routes._guard(request)
        request = SimpleNamespace(
            headers={
                "Host": "localhost:8820",
                "Origin": "http://localhost:8820",
                "X-Halogen-Action": "deploy",
            }
        )
        routes._guard(request)  # must not raise


class RoutePayloadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._platform = patch("halogen_deploy.sys.platform", "linux")
        self._platform.start()
        self.addCleanup(self._platform.stop)
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)
        self.addCleanup(self._dir.cleanup)

    async def test_dry_run_route_reads_json_body(self):
        from halogen_deploy import DeployRoutes

        routes = DeployRoutes(make_manager(self.tmp))
        payload = official_payload(self.tmp)

        class Req:
            headers = {
                "Host": "localhost:8820",
                "Origin": "http://localhost:8820",
                "X-Halogen-Action": "deploy",
            }
            content_type = "application/json"

            async def text(self):
                return json.dumps(payload)

        response = await routes.api_dry_run(Req())
        self.assertEqual(response.status, 200)
        body = json.loads(response.body.decode("utf-8"))
        self.assertTrue(body["ok"], body["errors"])


class ConvertVerifyJobTests(PosixTestCase):
    def setUp(self):
        super().setUp()
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)
        self.manager = make_manager(self.tmp)
        self.gguf = self.tmp / "models" / "uncensored" / "input-IQ4_XS.gguf"
        self.gguf.parent.mkdir(parents=True, exist_ok=True)
        self.gguf.write_bytes(b"fake gguf")
        self.hgn = self.gguf.parent / "converted.hgn"
        self.hgn.write_bytes(b"fake hgn")
        self.head = self.tmp / "models" / "official" / "qwen38-flash-next-mtp.hgn"
        self.head.parent.mkdir(parents=True, exist_ok=True)
        self.head.write_bytes(b"head")

    def tearDown(self):
        self._dir.cleanup()

    def _capture_start(self):
        captured = {}
        def fake_start(kind, argv, env_extra=None, timeout=3600):
            captured["kind"] = kind
            captured["argv"] = argv
            captured["env"] = env_extra
            return {"ok": True, "kind": kind}
        return captured, fake_start

    def test_convert_builds_expected_argv(self):
        captured, fake = self._capture_start()
        with patch.object(self.manager, "_start_job", fake):
            self.manager.convert({
                "image": "ghcr.io/peonist-ai/halogen-flash-server:0.13.1",
                "gguf": str(self.gguf),
                "output": "qwen3.8-flash-uncensored.hgn",
                "download_head": True,
            })
        argv = captured["argv"]
        self.assertEqual(captured["kind"], "convert")
        self.assertEqual(argv[:2], ["podman", "run"])
        self.assertIn("convert", argv)
        self.assertIn(f"/models/{self.gguf.name}", argv)
        self.assertIn("/models/qwen3.8-flash-uncensored.hgn", argv)
        self.assertIn(
            "HALOGEN_DOWNLOAD=peonist-ai/halogen-qwen3.8-flash-next", " ".join(argv)
        )

    def test_convert_rejects_missing_file(self):
        with self.assertRaises(DeployError):
            self.manager.convert({
                "image": "ghcr.io/peonist-ai/halogen-flash-server:0.13.1",
                "gguf": str(self.tmp / "nope.gguf"),
            })

    def test_convert_rejects_path_traversal_output(self):
        with self.assertRaises(DeployError):
            self.manager.convert({
                "image": "ghcr.io/peonist-ai/halogen-flash-server:0.13.1",
                "gguf": str(self.gguf),
                "output": "../evil.hgn",
            })

    def test_convert_with_explicit_head_skips_download(self):
        captured, fake = self._capture_start()
        with patch.object(self.manager, "_start_job", fake):
            self.manager.convert({
                "image": "ghcr.io/peonist-ai/halogen-flash-server:0.13.1",
                "gguf": str(self.gguf),
                "mtp_head": str(self.head),
            })
        joined = " ".join(captured["argv"])
        self.assertIn("HALOGEN_MTP_HEAD=/heads/qwen38-flash-next-mtp.hgn", joined)
        self.assertNotIn("HALOGEN_DOWNLOAD", joined)

    def test_verify_builds_argv(self):
        captured, fake = self._capture_start()
        with patch.object(self.manager, "_start_job", fake):
            self.manager.verify({
                "image": "ghcr.io/peonist-ai/halogen-flash-server:0.13.1",
                "hgn": str(self.hgn),
            })
        self.assertEqual(captured["kind"], "verify")
        self.assertIn("verify", captured["argv"])
        self.assertIn("/models/converted.hgn", captured["argv"])

    def test_hf_download_requires_token(self):
        with patch.object(self.manager, "_hf_executable", return_value="hf"):
            with self.assertRaises(DeployError):
                self.manager.hf_download({"repo": "orcarouter/x", "dest": str(self.tmp / "d")})

    def test_hf_install_uses_package_with_built_in_cli(self):
        captured, fake = self._capture_start()
        with patch.object(self.manager, "_hf_executable", return_value=None):
            with patch.object(self.manager, "_start_job", fake):
                self.manager.hf_install()
        self.assertEqual(captured["argv"][-1], HF_CLI_PACKAGE)
        self.assertNotIn("[cli]", captured["argv"][-1])

    def test_hf_access_denied_recognizes_gated_repository_error(self):
        self.assertTrue(
            _hf_access_denied(
                "Error: Access denied. This repository requires approval."
            )
        )
        self.assertFalse(_hf_access_denied("network connection timed out"))

    def test_run_stream_forwards_carriage_return_progress(self):
        self.manager._start_pipeline_job("test-progress", 1)
        command = (
            "import sys; "
            "sys.stdout.write('Fetching 1%\\rFetching 2%\\r"
            "\\x1b[32mDone\\x1b[0m\\n'); "
            "sys.stdout.flush()"
        )

        asyncio.run(
            self.manager._run_stream([sys.executable, "-c", command], timeout=30)
        )

        self.assertEqual(
            self.manager.job["lines"],
            ["Fetching 1%", "Fetching 2%", "Done"],
        )

    def test_hf_download_rejects_bad_repo(self):
        with patch.object(self.manager, "_hf_executable", return_value="hf"):
            with self.assertRaises(DeployError):
                self.manager.hf_download(
                    {"repo": "bad repo", "dest": str(self.tmp / "d"), "token": "t"}
                )

    def test_hf_download_requires_hf_cli(self):
        with patch.object(self.manager, "_hf_executable", return_value=None):
            with self.assertRaises(DeployError):
                self.manager.hf_download(
                    {
                        "repo": "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF",
                        "dest": str(self.tmp / "models" / "uncensored"),
                        "token": "hf_secret123",
                    }
                )

    def test_hf_download_token_never_in_argv(self):
        captured, fake = self._capture_start()
        with patch.object(self.manager, "_hf_executable", return_value="hf"):
            with patch.object(self.manager, "_start_job", fake):
                self.manager.hf_download({
                    "repo": "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF",
                    "dest": str(self.tmp / "models" / "uncensored"),
                    "token": "hf_secret123",
                })
        self.assertEqual(captured["kind"], "hf-download")
        self.assertEqual(
            captured["argv"][:3],
            ["hf", "download", "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF"],
        )
        self.assertEqual(captured["env"]["HF_TOKEN"], "hf_secret123")
        self.assertNotIn("hf_secret123", " ".join(captured["argv"]))

    def test_hf_status_detects_venv_local_hf(self):
        hf = Path(sys.executable).parent / ("hf.exe" if os.name == "nt" else "hf")
        with patch("halogen_deploy.shutil.which", return_value=None):
            with patch.object(Path, "is_file", return_value=True):
                with patch("os.access", return_value=True):
                    self.assertTrue(self.manager.hf_status()["installed"])
        self.assertTrue(hf.name.startswith("hf"))

    def test_uncensored_template_via_manager(self):
        data = self.manager.template("uncensored")
        self.assertEqual(data["profile"]["model_id"], "qwen3.8-flash-uncensored")
        self.assertIn("/uncensored", data["quadlet"])


class QuickDeployTests(PosixTestCase):
    """Quick install runs as a cancelable job that takes over the active backend
    and rolls back on failure."""

    def setUp(self):
        super().setUp()
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)
        self.deploy = make_manager(self.tmp)

    def tearDown(self):
        self._dir.cleanup()

    async def _drive(self, coro):
        result = await coro
        if self.deploy._job_task is not None:
            await self.deploy._job_task
        return result

    def test_quick_official_starts_job_and_switches(self):
        self.deploy.manager.switch_with_hook = AsyncMock()
        with patch.object(DeployManager, "reload_discovery", return_value={"ok": True}):
            result = asyncio.run(self._drive(self.deploy.quick_official()))
        self.assertEqual(result["kind"], "quick-official")
        self.assertEqual(self.deploy.job["state"], "done")
        args, kwargs = self.deploy.manager.switch_with_hook.await_args
        self.assertEqual(args[0], "qwen3.8-flash")
        self.assertEqual(
            kwargs.get("start_timeout"), QUICK_OFFICIAL_START_TIMEOUT
        )

    def test_service_journal_is_forwarded_to_job_log(self):
        class Output:
            def __init__(self):
                self.lines = iter((b"Downloading shard 1/12\n", b""))

            async def readline(self):
                return next(self.lines)

        process = SimpleNamespace(
            stdout=Output(),
            returncode=0,
            wait=AsyncMock(return_value=0),
        )
        self.deploy._start_pipeline_job("quick-official", 2)
        with patch("halogen_deploy.shutil.which", return_value="/usr/bin/journalctl"), \
             patch(
                 "halogen_deploy.asyncio.create_subprocess_exec",
                 new=AsyncMock(return_value=process),
             ) as create:
            asyncio.run(
                self.deploy._stream_service_journal(
                    "halogen-official.service"
                )
            )

        self.assertIn("Downloading shard 1/12", self.deploy.job["lines"])
        self.assertIn("halogen-official.service", create.await_args.args)

    def test_quick_official_does_not_raise_when_other_backend_active(self):
        async def fake_active(service):
            return service == "halogen-uncensored.service"

        self.deploy._service_is_active = AsyncMock(side_effect=fake_active)
        self.deploy.manager.switch_with_hook = AsyncMock()
        with patch.object(DeployManager, "reload_discovery", return_value={"ok": True}):
            result = asyncio.run(self._drive(self.deploy.quick_official()))
        self.assertEqual(result["kind"], "quick-official")
        self.assertEqual(self.deploy.job["state"], "done")
        self.deploy.manager.switch_with_hook.assert_awaited_once()

    def test_quick_official_tracks_already_running_download(self):
        self.deploy._service_is_active = AsyncMock(return_value=True)
        self.deploy.manager.models = {
            "qwen3.8-flash": "halogen-official.service"
        }
        self.deploy.manager.backend_health = AsyncMock(return_value=None)
        self.deploy.manager.wait_for_backend = AsyncMock()
        self.deploy._stream_service_journal = AsyncMock()

        result = asyncio.run(self._drive(self.deploy.quick_official()))

        self.assertTrue(result["already_running"])
        self.assertEqual(self.deploy.job["state"], "done")
        self.deploy.manager.wait_for_backend.assert_awaited_once_with(
            "qwen3.8-flash", timeout=QUICK_OFFICIAL_START_TIMEOUT
        )

    def test_quick_uncensored_requires_token_when_download_needed(self):
        with self.assertRaises(DeployError):
            asyncio.run(self.deploy.quick_uncensored({}))

    def test_quick_uncensored_no_token_needed_when_hgn_exists(self):
        uncensored_dir = self.tmp / "models" / "uncensored"
        uncensored_dir.mkdir(parents=True)
        (uncensored_dir / "qwen3.8-flash-uncensored.hgn").write_text("x")
        (uncensored_dir / "tokenizer").mkdir()
        (uncensored_dir / "tokenizer" / "tok.json").write_text("{}")
        self.deploy._hf_executable = lambda: "hf"
        self.deploy._run_quick_uncensored = AsyncMock()
        # No token, but weights + tokenizer already present -> no raise.
        result = asyncio.run(self._drive(self.deploy.quick_uncensored({})))
        self.assertEqual(result["kind"], "quick-uncensored")

    def test_quick_uncensored_starts_job_when_other_backend_active(self):
        async def fake_active(service):
            return service == "halogen-official.service"

        self.deploy._service_is_active = AsyncMock(side_effect=fake_active)
        self.deploy._run_quick_uncensored = AsyncMock()
        result = asyncio.run(self._drive(self.deploy.quick_uncensored({"token": "hf_x"})))
        self.assertEqual(result["kind"], "quick-uncensored")
        self.assertEqual(self.deploy.job["state"], "running")

    def test_run_quick_uncensored_switches_with_convert_hook(self):
        uncensored_dir = self.tmp / "models" / "uncensored"
        uncensored_dir.mkdir(parents=True)
        gguf = uncensored_dir / "x.gguf"
        output = uncensored_dir / "qwen3.8-flash-uncensored.hgn"
        streams = []

        async def fake_stream(argv, *a, **k):
            streams.append(list(argv))

        self.deploy._run_stream = fake_stream
        self.deploy._hf_executable = lambda: "hf"
        self.deploy.install_dirs = lambda payload: {"ok": True}
        self.deploy.dry_run = lambda payload: {"ok": True, "errors": []}
        self.deploy.apply = AsyncMock()
        switch_calls = []

        async def fake_switch(model, hook=None, start_timeout=None):
            switch_calls.append(model)
            if hook is not None:
                await hook()

        self.deploy.manager.switch_with_hook = fake_switch
        self.deploy._start_pipeline_job("quick-uncensored", 6)
        with patch.object(DeployManager, "reload_discovery", return_value={"ok": True}):
            asyncio.run(
                self.deploy._run_quick_uncensored(
                    "tok", "0.13.2", uncensored_dir, gguf, output, True, True
                )
            )
        self.assertEqual(switch_calls, ["qwen3.8-flash-uncensored"])
        self.assertTrue(any("convert" in s for s in streams))
        self.assertEqual(self.deploy.job["state"], "done")

    def test_run_quick_uncensored_repairs_missing_tokenizer(self):
        uncensored_dir = self.tmp / "models" / "uncensored"
        uncensored_dir.mkdir(parents=True)
        output = uncensored_dir / "qwen3.8-flash-uncensored.hgn"
        output.write_text("x")  # weights present, tokenizer missing
        streams = []

        async def fake_stream(argv, *a, **k):
            streams.append(list(argv))

        self.deploy._run_stream = fake_stream
        self.deploy._hf_executable = lambda: "hf"
        self.deploy.install_dirs = lambda payload: {"ok": True}
        self.deploy.dry_run = lambda payload: {"ok": True, "errors": []}
        self.deploy.apply = AsyncMock()
        self.deploy.manager.switch_with_hook = AsyncMock()
        self.deploy._start_pipeline_job("quick-uncensored", 3)
        with patch.object(DeployManager, "reload_discovery", return_value={"ok": True}):
            asyncio.run(
                self.deploy._run_quick_uncensored(
                    "tok",
                    "0.13.2",
                    uncensored_dir,
                    uncensored_dir / "x.gguf",
                    output,
                    False,
                    True,
                )
            )
        joined = [" ".join(s) for s in streams]
        self.assertTrue(any("tokenizer/*" in j for j in joined))
        self.assertFalse(any("orcarouter" in j for j in joined))  # no gguf download
        self.assertEqual(self.deploy.job["state"], "done")

    def test_cancel_job_cancels_running_switch(self):
        gate = asyncio.Event()

        async def blocking_switch(model, hook=None, start_timeout=None):
            await gate.wait()

        self.deploy.manager.switch_with_hook = blocking_switch
        with patch.object(DeployManager, "reload_discovery", return_value={"ok": True}):
            async def run():
                await self.deploy.quick_official()
                await asyncio.sleep(0.05)
                res = self.deploy.cancel_job()
                try:
                    await self.deploy._job_task
                except asyncio.CancelledError:
                    pass
                return res

            res = asyncio.run(run())
        self.assertTrue(res["ok"])
        self.assertEqual(self.deploy.job["state"], "cancelled")


if __name__ == "__main__":
    unittest.main()
