"""DeployManager: dry-run/diff, backup-before-write, rollback and guards.

systemd interaction is mocked; file handling runs against temp directories.
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from halogen_deploy import DeployError, DeployManager
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
        self.assertTrue(result["warnings"])  # 118 GiB download warning

    def test_dry_run_rejects_path_outside_allowed_roots(self):
        payload = official_payload(self.tmp)
        payload["volumes"][0][0] = str(self.tmp.parent / "outside-models")
        result = self.manager.dry_run(payload)
        self.assertFalse(result["ok"])
        self.assertTrue(any("erlaubten Wurzeln" in e for e in result["errors"]))

    def test_dry_run_rejects_unknown_env(self):
        payload = official_payload(self.tmp)
        payload["env"]["HALOGEN_SOMETHING_ELSE"] = "1"
        result = self.manager.dry_run(payload)
        self.assertFalse(result["ok"])

    def test_dry_run_detects_port_conflict(self):
        self.manager.quadlet_dir.mkdir(parents=True, exist_ok=True)
        other = official_template("0.13.2", self.tmp / "models", self.tmp / "cache")
        other.profile_id = "other"
        other.host_port = 8831
        (self.manager.quadlet_dir / "halogen-other.container").write_text(
            render_quadlet(other), encoding="utf-8"
        )
        result = self.manager.dry_run(official_payload(self.tmp))
        self.assertFalse(result["ok"])
        self.assertTrue(any("Port" in e for e in result["errors"]))

    def test_dry_run_detects_model_id_conflict(self):
        self.manager.quadlet_dir.mkdir(parents=True, exist_ok=True)
        other = official_template("0.13.2", self.tmp / "models", self.tmp / "cache")
        other.profile_id = "other"
        (self.manager.quadlet_dir / "halogen-other.container").write_text(
            render_quadlet(other), encoding="utf-8"
        )
        result = self.manager.dry_run(official_payload(self.tmp))
        self.assertFalse(result["ok"])
        self.assertTrue(any("Modell-ID" in e for e in result["errors"]))


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

    def test_rollback_restores_previous_state(self):
        payload = official_payload(self.tmp)
        payload["env"]["HALOGEN_KV_SLOTS"] = "4"
        asyncio.run(self.manager.apply(payload))
        result = asyncio.run(self.manager.rollback("official"))
        target = self.tmp / "quadlets" / "halogen-official.container"
        parsed = parse_quadlet(target.read_text(encoding="utf-8"))
        self.assertEqual(parsed.env["HALOGEN_KV_SLOTS"], "2")
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


if __name__ == "__main__":
    unittest.main()
