"""The copied self-update worker uses only fake package and service commands."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    worker = importlib.import_module("halobridge_update_worker")
except ModuleNotFoundError:
    worker = None


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(worker, "The standalone update worker is missing")
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("HALOGEN_TEST_TMP"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = self.root / "venvs" / "halobridge"
        (self.prefix / "bin").mkdir(parents=True)
        (self.prefix / "bin" / "python").write_text("old interpreter")
        (self.prefix / "original").write_text("working app")
        (self.prefix / "pipx_metadata.json").write_text(json.dumps({"main_package": {"package": "halobridge", "package_or_url": "git+https://github.com/Siyarbekir47/Halobridge.git@develop"}}))
        self.state = self.root / "state"
        self.state.mkdir()
        self.plan = {
            "id": "test-update", "version": "0.1.10", "previous_version": "0.1.9", "commit": "a" * 40,
            "source_ref": "develop", "prefix": str(self.prefix), "pipx": str(self.root / "tools/pipx"), "systemctl": str(self.root / "tools/systemctl"),
            "service": "halobridge.service", "journal": str(self.state / "app-update.json"), "lock": str(self.state / "app-update.lock"),
            "previous_pid": 100, "drained": True,
        }
        (self.state / "app-update.json").write_text(json.dumps({"id": "test-update", "phase": "starting"}))
        self.commands = []
        self.fail_install = False
        self.installed_version = "0.1.10"
        self.installed_commit = "a" * 40
        self.service_pid = "200"

    def runner(self, args, **kwargs):
        self.commands.append(args)
        if args[:3] == [self.plan["pipx"], "install", "--force"]:
            (self.prefix / "original").unlink()
            (self.prefix / "replacement").write_text("new app")
            return subprocess.CompletedProcess(args, 1 if self.fail_install else 0, "", "installation failed")
        if args[0] == str(self.prefix / "bin" / "python"):
            info = {"version": "0.1.9" if (self.prefix / "original").exists() else self.installed_version, "commit": self.installed_commit, "url": "https://github.com/Siyarbekir47/Halobridge.git"}
            return subprocess.CompletedProcess(args, 0, json.dumps(info), "")
        if "show" in args:
            pid = self.service_pid if any("restart" in command for command in self.commands) else "100"
            return subprocess.CompletedProcess(args, 0, f"ActiveState=active\nMainPID={pid}\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    def job(self):
        return json.loads((self.state / "app-update.json").read_text())

    def test_success_verifies_version_before_service_restart_and_records_success(self):
        self.assertEqual(worker.execute(self.plan, runner=self.runner), 0)
        self.assertEqual(self.job()["phase"], "succeeded")
        self.assertTrue((self.prefix / "replacement").exists())
        install = next(command for command in self.commands if "install" in command)
        self.assertEqual(install, [self.plan["pipx"], "install", "--force", "git+https://github.com/Siyarbekir47/Halobridge.git@develop"])
        restart = self.commands.index([self.plan["systemctl"], "--user", "restart", "halobridge.service"])
        verification = next(i for i, command in enumerate(self.commands) if command[0] == str(self.prefix / "bin" / "python"))
        self.assertLess(verification, restart)

    def test_failed_install_restores_previous_environment_and_never_restarts(self):
        self.fail_install = True
        self.assertEqual(worker.execute(self.plan, runner=self.runner), 1)
        self.assertEqual(self.job()["phase"], "rolled_back")
        self.assertEqual((self.prefix / "original").read_text(), "working app")
        self.assertFalse((self.prefix / "replacement").exists())
        self.assertFalse(any("restart" in command for command in self.commands))

    def test_wrong_version_or_branch_movement_restores_backup_without_restart(self):
        for attribute, value in (("installed_version", "0.1.9"), ("installed_commit", "b" * 40)):
            setattr(self, attribute, value)
            self.assertEqual(worker.execute(self.plan, runner=self.runner), 1)
            self.assertEqual(self.job()["phase"], "rolled_back")
            self.assertTrue((self.prefix / "original").exists())
            self.assertFalse(any("restart" in command for command in self.commands))
            setattr(self, attribute, "0.1.10" if attribute == "installed_version" else "a" * 40)
            self.plan["id"] += "-next"
            (self.state / "app-update.json").write_text(json.dumps({"id": self.plan["id"], "phase": "starting"}))

    def test_restart_without_new_service_process_is_not_claimed_successful(self):
        self.service_pid = "100"
        self.assertEqual(worker.execute(self.plan, runner=self.runner), 1)
        self.assertEqual(self.job()["phase"], "recovery_required")
        self.assertIsNotNone(self.job().get("backup_dir"))

    def test_untrusted_source_or_missing_drain_authorization_never_installs(self):
        for field, value in (("source_ref", "develop;reboot"), ("service", "--all"), ("drained", False), ("commit", "HEAD")):
            original = self.plan[field]
            self.plan[field] = value
            with self.assertRaises(worker.WorkerError):
                worker.execute(self.plan, runner=self.runner)
            self.assertEqual(self.commands, [])
            self.plan[field] = original

    def test_failed_install_restores_exposed_pipx_executable_links(self):
        entry = self.root / "bin/halobridge"
        entry.parent.mkdir()
        entry.write_text("original pipx entry")
        self.plan["app_links"] = [str(entry)]
        original = self.runner
        def runner(args, **kwargs):
            result = original(args, **kwargs)
            if "install" in args:
                entry.unlink()
            return result
        self.fail_install = True
        self.assertEqual(worker.execute(self.plan, runner=runner), 1)
        self.assertEqual(entry.read_text(), "original pipx entry")
        self.assertEqual(self.job()["phase"], "rolled_back")

    def test_failed_restore_is_durable_and_does_not_restart(self):
        self.fail_install = True
        original = worker.shutil.copytree
        def copytree(source, destination, *args, **kwargs):
            if Path(destination) == self.prefix:
                raise OSError("disk unavailable")
            return original(source, destination, *args, **kwargs)
        with patch.object(worker.shutil, "copytree", side_effect=copytree):
            self.assertEqual(worker.execute(self.plan, runner=self.runner), 1)
        self.assertEqual(self.job()["phase"], "recovery_required")
        self.assertTrue(Path(self.job()["backup_dir"]).is_dir())
        self.assertFalse(any("restart" in command for command in self.commands))

    def test_new_service_must_respond_with_the_installed_target_version(self):
        self.plan["health_url"] = "http://127.0.0.1:8080/dashboard/api/app-updates"
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self, limit):
                return b'{"current_version":"0.1.9"}'
        with patch.object(worker, "urlopen", return_value=Response(), create=True), patch.object(worker.time, "sleep"):
            self.assertEqual(worker.execute(self.plan, runner=self.runner), 1)
        self.assertEqual(self.job()["phase"], "recovery_required")

    def test_revoked_handoff_during_install_restores_and_never_restarts(self):
        original = self.runner
        def runner(args, **kwargs):
            result = original(args, **kwargs)
            if "install" in args:
                job = self.job()
                job["cancelled"] = True
                (self.state / "app-update.json").write_text(json.dumps(job))
            return result
        self.assertEqual(worker.execute(self.plan, runner=runner), 1)
        self.assertEqual(self.job()["phase"], "rolled_back")
        self.assertTrue(self.job()["cancelled"])
        self.assertTrue((self.prefix / "original").exists())
        self.assertFalse(any("restart" in command for command in self.commands))

    def test_command_timeout_terminates_install_process_group(self):
        process = unittest.mock.Mock(pid=123, returncode=-9)
        process.communicate.side_effect = [subprocess.TimeoutExpired(["pipx"], 1), ("", "")]
        with patch.object(worker.subprocess, "Popen", return_value=process), patch.object(worker.os, "name", "posix"), patch.object(worker.signal, "SIGKILL", 9, create=True), patch.object(worker.os, "killpg", create=True) as killpg:
            with self.assertRaises(subprocess.TimeoutExpired):
                worker.run_command(["pipx"], timeout=1, capture_output=True, text=True, check=False)
        killpg.assert_called_once_with(123, 9)

    def test_cli_removes_the_plan_containing_the_dashboard_token(self):
        path = self.state / "plan.json"
        path.write_text(json.dumps({**self.plan, "health_token": "private-token"}))
        with patch.object(worker.sys, "argv", ["worker", "--plan", str(path)], create=True), patch.object(worker, "execute", return_value=0):
            self.assertEqual(worker.main(), 0)
        self.assertFalse(path.exists())

    def test_worker_must_not_restart_a_different_live_service_process(self):
        original = self.runner
        def runner(args, **kwargs):
            if "show" in args and not any("restart" in command for command in self.commands):
                self.commands.append(args)
                return subprocess.CompletedProcess(args, 0, "ActiveState=active\nMainPID=999\n", "")
            return original(args, **kwargs)
        self.assertEqual(worker.execute(self.plan, runner=runner), 1)
        self.assertEqual(self.job()["phase"], "rolled_back")
        self.assertFalse(any("restart" in command for command in self.commands))

    @unittest.skipUnless(sys.platform == "linux", "flock belongs to the supported Linux installation")
    def test_durable_worker_lock_blocks_another_update(self):
        import fcntl
        with (self.state / "app-update.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(worker.WorkerError):
                worker.execute(self.plan, runner=self.runner)
        self.assertEqual(self.commands, [])


if __name__ == "__main__":
    unittest.main()
