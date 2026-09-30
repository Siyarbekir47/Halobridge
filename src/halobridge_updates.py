"""Check Halobridge package versions and safely hand pipx updates to systemd."""

from __future__ import annotations

import asyncio
import base64
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import time
import tomllib
from typing import Any
from urllib.parse import quote
import uuid
import ipaddress

from aiohttp import ClientSession, ClientTimeout, web

import settings
from semver import normalize_version, version_tuple
from halobridge_update_worker import REPO_URL, TERMINAL_PHASES, WorkerError, acquire_lock, atomic_json, read_job, valid_ref

LOG = logging.getLogger("halobridge.updates")
API_URL = "https://api.github.com/repos/Siyarbekir47/Halobridge"
REPO_PAGE = "https://github.com/Siyarbekir47/Halobridge"
MAX_RESPONSE = 1024 * 1024
CHECK_RETRY = 300


class AppUpdateError(RuntimeError):
    pass


def installed_source() -> dict:
    """Trust only the official HTTPS Git origin recorded by pip."""
    try:
        raw = metadata.distribution("halobridge").read_text("direct_url.json")
        data = json.loads(raw or "{}")
    except (metadata.PackageNotFoundError, ValueError, OSError) as error:
        raise AppUpdateError("The installed Halobridge Git source cannot be verified.") from error
    if not isinstance(data, dict):
        raise AppUpdateError("The installed Halobridge Git source cannot be verified.")
    vcs = data.get("vcs_info", {})
    directory = data.get("dir_info", {})
    if (not isinstance(data.get("url"), str) or data["url"] not in {REPO_URL, REPO_URL.removesuffix(".git")}
            or not isinstance(vcs, dict) or vcs.get("vcs") != "git"
            or not isinstance(directory, dict) or directory.get("editable")):
        raise AppUpdateError("Automatic updates require the official Halobridge Git installation.")
    revision = vcs.get("requested_revision")
    # A pinned commit has no branch to track: resolve the repository default.
    branch = revision if valid_ref(revision) and not re.fullmatch(r"[0-9a-f]{40}", revision) else None
    return {"branch": branch, "commit": vcs.get("commit_id")}


class AppUpdater:
    def __init__(self, manager: Any, session: ClientSession, *, config: Any,
                 container_updater=None, deploy_manager=None):
        self.manager, self.session, self.config = manager, session, config
        self.container_updater, self.deploy_manager = container_updater, deploy_manager
        self.enabled = config.app_updates.enabled
        self.check_interval = config.app_updates.check_interval_h * 3600
        self.state_dir = Path(config.dashboard.state_dir)
        self.journal_path = self.state_dir / "app-update.json"
        self.lock_path = self.state_dir / "app-update.lock"
        self.start_lock, self.check_lock = asyncio.Lock(), asyncio.Lock()
        self.task: asyncio.Task | None = None
        self.check_task: asyncio.Task | None = None
        self.monitor_task: asyncio.Task | None = None
        self.latest_version: str | None = None
        self.checked_at: float | None = None
        self.attempted_at = 0.0
        self.check_error: str | None = None
        self.commit: str | None = None
        self.job = read_job(self.journal_path)
        self._support_error: str | None = "Installation support has not been verified."
        self.tools: dict[str, str] = {}
        self._source_error = None
        self._maintenance_owned = False
        self._handoff = False
        self.operation_requests = 0
        configured = config.app_updates.branch
        try:
            source = installed_source()
            if configured and not valid_ref(configured):
                raise AppUpdateError("The configured update branch is invalid.")
            self.source_ref = configured or source["branch"]
        except AppUpdateError as error:
            self.source_ref = configured if valid_ref(configured) else None
            self._source_error = str(error)

    def _load_job(self) -> None:
        durable = read_job(self.journal_path)
        if durable:
            self.job = durable

    @property
    def running(self) -> bool:
        self._load_job()
        return bool((self.task and not self.task.done()) or (self.job and self.job.get("phase") not in TERMINAL_PHASES))

    def _busy(self) -> bool:
        return bool(self.operation_requests or getattr(self.manager, "maintenance", False) or getattr(self.manager, "switching", False)
                    or (self.container_updater and (self.container_updater.running or self.container_updater.recovery_required))
                    or (self.deploy_manager and (self.deploy_manager.deploy_lock.locked()
                        or (getattr(self.deploy_manager, "_job_task", None) and not self.deploy_manager._job_task.done())
                        or (getattr(self.deploy_manager, "job", None) and self.deploy_manager.job.get("state") == "running"))))

    def status(self) -> dict:
        self._load_job()
        current = settings.app_version()
        installed, latest = version_tuple(current), version_tuple(self.latest_version)
        fresh = self.checked_at is not None and time.time() - self.checked_at < self.check_interval and not self.check_error
        available = bool(installed and latest and installed < latest)
        support_error = ("App updates are disabled." if not self.enabled else self._source_error or self._support_error)
        running = self.running
        recovery = bool(self.job and self.job.get("phase") == "recovery_required")
        manual = (f"pipx install --force {shlex.quote('git+' + REPO_URL + '@' + self.source_ref)} && "
                  f"systemctl --user restart {shlex.quote(self.config.app_updates.service)}") if self.source_ref else None
        return {
            "current_version": current, "latest_version": self.latest_version,
            "update_available": available, "up_to_date": bool(fresh and installed and latest and installed >= latest),
            "checked_at": self.checked_at, "check_error": self.check_error,
            "supported": support_error is None, "support_error": support_error,
            "can_install": bool(available and fresh and self.enabled and self.config.security.allow_install
                                and not support_error and not running and not recovery and not self._busy()),
            "running": running, "job": self.job, "recovery_required": recovery,
            "blocked_reason": "App installation is disabled." if not self.config.security.allow_install else "An update, deployment or maintenance is already active." if self._busy() else None,
            "source_ref": self.source_ref, "release_url": f"{REPO_PAGE}/blob/{quote(self.source_ref, safe='')}/CHANGELOG.md" if self.source_ref else REPO_PAGE,
            "manual_command": manual,
        }

    async def _run(self, *argv: str, timeout=20) -> str:
        process = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, _ = await asyncio.wait_for(process.communicate(), timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        if process.returncode:
            raise AppUpdateError(f"{Path(argv[0]).name} failed (exit {process.returncode}).")
        return out.decode("utf-8", errors="replace")

    async def _verify_support(self) -> str | None:
        if sys.platform != "linux":
            return "Dashboard installation requires Linux, pipx and an active systemd user service."
        if self._source_error:
            return self._source_error
        service = self.config.app_updates.service
        if not re.fullmatch(r"halobridge[a-zA-Z0-9_.@-]*\.service", service):
            return "The configured Halobridge user service name is invalid."
        prefix = Path(sys.prefix)
        try:
            data = json.loads((prefix / "pipx_metadata.json").read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("main_package"), dict):
                raise AppUpdateError("Halobridge must run in its own pipx environment.")
            package = data.get("main_package", {})
            if prefix.name != "halobridge" or prefix.is_symlink() or package.get("package") != "halobridge":
                raise AppUpdateError("Halobridge must run in its own pipx environment.")
            python = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
            if prefix == python or prefix in python.parents or not python.is_file():
                raise AppUpdateError("A Python interpreter outside the pipx environment is required.")
            pipx = shutil.which("pipx") or str(Path.home() / ".local/bin/pipx")
            paths = {"pipx": pipx, "systemctl": shutil.which("systemctl"), "systemd_run": shutil.which("systemd-run"), "python": str(python)}
            if any(not value or not Path(value).is_file() for value in paths.values()):
                raise AppUpdateError("pipx, systemctl and systemd-run must be available under the service user.")
            if prefix in Path(pipx).resolve().parents:
                raise AppUpdateError("pipx must run outside the Halobridge environment.")
            venvs = Path((await self._run(pipx, "environment", "--value", "PIPX_LOCAL_VENVS")).strip())
            pipx_home = Path((await self._run(pipx, "environment", "--value", "PIPX_HOME")).strip())
            pipx_bin = Path((await self._run(pipx, "environment", "--value", "PIPX_BIN_DIR")).strip())
            if not all(path.is_absolute() for path in (venvs, pipx_home, pipx_bin)) or venvs.resolve() != prefix.parent.resolve() or (pipx_home / "venvs").resolve() != venvs.resolve():
                raise AppUpdateError("pipx does not manage the environment running this Halobridge process.")
            properties = dict(line.split("=", 1) for line in (await self._run(
                paths["systemctl"], "--user", "show", service, "--property=ActiveState", "--property=MainPID", "--property=ExecStart",
            )).splitlines() if "=" in line)
            if properties.get("ActiveState") != "active" or properties.get("MainPID") != str(os.getpid()):
                raise AppUpdateError("The configured user service does not own this Halobridge process.")
            # systemctl's structured ExecStart has a path= field; do not execute
            # or interpret its argv. Resolve the existing pipx symlink only.
            match = re.search(r"(?:^|[ {;])path=([^ ;}]+)", properties.get("ExecStart", ""))
            if not match or Path(match[1]).resolve() != (prefix / "bin/halobridge").resolve():
                raise AppUpdateError("The user service does not start this pipx Halobridge entry point.")
            entry = pipx_bin / "halobridge"
            if not entry.is_file() or entry.resolve() != (prefix / "bin/halobridge").resolve():
                raise AppUpdateError("The exposed pipx Halobridge command does not match this environment.")
            if not all(callable(getattr(self.manager, method, None)) for method in ("begin_maintenance", "drain_maintenance", "end_maintenance")):
                raise AppUpdateError("The router cannot safely drain active requests.")
            self.tools = {**paths, "pipx_home": str(pipx_home), "pipx_bin": str(pipx_bin), "app_entry": str(entry)}
            return None
        except (OSError, ValueError, asyncio.TimeoutError, AppUpdateError) as error:
            return str(error) if isinstance(error, AppUpdateError) else "The pipx environment or user service could not be verified."

    async def _get_json(self, suffix: str, *, params=None) -> dict:
        headers = {"Accept": "application/vnd.github+json", "Accept-Encoding": "identity", "User-Agent": "halobridge-updater", "X-GitHub-Api-Version": "2022-11-28"}
        async with self.session.get(API_URL + suffix, params=params, headers=headers, timeout=ClientTimeout(total=20)) as response:
            if response.status in {403, 429}:
                raise AppUpdateError("GitHub rate limit reached. Try again later.")
            if response.status != 200:
                raise AppUpdateError(f"GitHub version check failed (HTTP {response.status}).")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise AppUpdateError("GitHub returned an unsupported encoded response.")
            if response.content_length is not None and response.content_length > MAX_RESPONSE:
                raise AppUpdateError("GitHub version response is too large.")
            content = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                content.extend(chunk)
                if len(content) > MAX_RESPONSE:
                    raise AppUpdateError("GitHub version response is too large.")
        try:
            data = json.loads(content)
        except ValueError as error:
            raise AppUpdateError("Invalid GitHub version response.") from error
        if not isinstance(data, dict):
            raise AppUpdateError("Invalid GitHub version response.")
        return data

    async def _fetch_latest(self) -> tuple[str, str]:
        if not self.source_ref:
            self.source_ref = (await self._get_json("")).get("default_branch")
        if not valid_ref(self.source_ref):
            raise AppUpdateError("A valid Halobridge source branch could not be determined.")
        commit = (await self._get_json("/commits/" + quote(self.source_ref, safe=""))).get("sha")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise AppUpdateError("GitHub returned an invalid source revision.")
        data = await self._get_json("/contents/pyproject.toml", params={"ref": commit})
        try:
            if data.get("encoding") != "base64" or not isinstance(data.get("content"), str):
                raise ValueError()
            content = base64.b64decode("".join(data["content"].split()), validate=True)
            project = tomllib.loads(content.decode("utf-8"))["project"]
            latest = normalize_version(project.get("version"))
            if project.get("name") != "halobridge" or not latest:
                raise ValueError()
        except (ValueError, KeyError, UnicodeError) as error:
            raise AppUpdateError("No stable Halobridge package version found on the source branch.") from error
        return latest, commit

    async def check(self, force=False) -> dict:
        async with self.check_lock:
            if not self.enabled or self.running:
                return self.status()
            delay = CHECK_RETRY if self.check_error else self.check_interval
            if time.time() - self.attempted_at < (30 if force else delay):
                return self.status()
            self.attempted_at = time.time()
            try:
                self.latest_version, self.commit = await self._fetch_latest()
                self.checked_at = time.time()
                self.check_error = None
                self._support_error = await self._verify_support()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.check_error = str(error) if isinstance(error, AppUpdateError) else "Version check failed. Check your network connection."
            return self.status()

    def _save_job(self, **changes):
        current = read_job(self.journal_path)
        if current and self.job and current.get("id") == self.job.get("id"):
            self.job = {**self.job, **current}
        self.job = {**(self.job or {}), **changes, "updated_at": time.time()}
        atomic_json(self.journal_path, self.job)

    async def start(self, version: str) -> None:
        async with self.start_lock:
            if not self.enabled or not self.config.security.allow_install:
                raise web.HTTPForbidden(text="App installation is disabled.")
            if self.running or self._busy() or (self.job and self.job.get("phase") == "recovery_required"):
                raise web.HTTPConflict(text="An update, deployment or maintenance is already active.")
            await self.check()
            self._support_error = await self._verify_support()
            status = self.status()
            if version != self.latest_version or not status["can_install"]:
                raise web.HTTPConflict(text=status["support_error"] or status["check_error"] or "Check versions again; this app version cannot be installed.")
            try:
                lock_handle = acquire_lock(self.lock_path)
            except (OSError, WorkerError) as error:
                raise web.HTTPConflict(text=str(error)) from error
            self._load_job()
            if self.job and self.job.get("phase") not in TERMINAL_PHASES:
                lock_handle.close()
                raise web.HTTPConflict(text="Another app update is already running.")
            self.job = {"id": str(uuid.uuid4()), "phase": "starting", "version": version, "target_version": version,
                        "previous_version": normalize_version(settings.app_version()), "source_ref": self.source_ref, "owner_pid": os.getpid(),
                        "message": "Preparing the Halobridge update.", "started_at": time.time()}
            try:
                self._save_job()
                self.task = asyncio.create_task(self._prepare(lock_handle))
            except Exception:
                lock_handle.close()
                raise

    async def _prepare(self, lock_handle) -> None:
        try:
            timeout = self.config.router.drain_timeout_s
            if self._busy():
                raise AppUpdateError("An update, deployment or maintenance is already active.")
            self._save_job(phase="draining", message="Waiting for active requests to finish.")
            await asyncio.wait_for(self.manager.begin_maintenance(), timeout)
            self._maintenance_owned = True
            if getattr(self.manager, "current_model", None):
                await asyncio.wait_for(self.manager.drain_maintenance(), timeout)
            elif getattr(self.manager, "active_requests", 0):
                raise AppUpdateError("Active requests could not be drained.")
            # The durable nonterminal job blocks other app instances during the
            # handoff gap; release flock before starting the worker so it can
            # acquire the same lock in its separate process.
            lock_handle.close()
            await self._dispatch()
            self._handoff = True
            self.monitor_task = asyncio.create_task(self._monitor())
        except (Exception, asyncio.CancelledError) as error:
            self._save_job(phase="failed", cancelled=True, message=str(error) if isinstance(error, AppUpdateError) else "The app update could not be started.")
            await self._release_maintenance()
        finally:
            lock_handle.close()

    async def _dispatch(self) -> None:
        assert self.job is not None
        # A copied worker and an external Python survive pipx replacing src and
        # the target venv. The transient unit has its own cgroup, so restarting
        # Halobridge cannot kill the process that finishes this transaction.
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if Path(sys.prefix).resolve() in self.state_dir.resolve().parents:
            raise AppUpdateError("The update state directory must be outside the pipx environment.")
        worker = self.state_dir / f"app-update-worker-{self.job['id']}.py"
        plan_path = self.state_dir / f"app-update-plan-{self.job['id']}.json"
        import halobridge_update_worker
        shutil.copyfile(halobridge_update_worker.__file__, worker)
        os.chmod(worker, 0o600)
        unit = f"halobridge-update-{self.job['id']}.service"
        plan = {**self.job, "commit": self.commit, "prefix": sys.prefix,
                "pipx": self.tools["pipx"], "systemctl": self.tools["systemctl"],
                "service": self.config.app_updates.service, "journal": str(self.journal_path.absolute()),
                "lock": str(self.lock_path.absolute()), "previous_pid": os.getpid(), "drained": True}
        bind = getattr(self.config.router, "bind", ["127.0.0.1"])[0]
        address = "127.0.0.1" if bind in {"0.0.0.0", "localhost"} else "::1" if bind == "::" else bind
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as error:
            raise AppUpdateError("Service verification requires an IP address in router.bind.") from error
        host = f"[{ip}]" if ip.version == 6 else str(ip)
        plan["health_url"] = f"http://{host}:{getattr(self.config.router, 'port', 8731)}/dashboard/api/app-updates"
        plan["health_token"] = getattr(self.config.security, "auth_token", "")
        if self.tools.get("app_entry"):
            plan["app_links"] = [self.tools["app_entry"]]
        for field in ("pipx_home", "pipx_bin"):
            if self.tools.get(field):
                plan[field] = self.tools[field]
        atomic_json(plan_path, plan)
        self._save_job(phase="starting", message="Starting the separate update worker.", worker_unit=unit)
        await self._run(self.tools["systemd_run"], "--user", f"--unit={unit}", "--collect",
                        "--property=Type=exec", "--property=Restart=no", self.tools["python"], "-I", str(worker.absolute()), "--plan", str(plan_path.absolute()))

    async def _release_maintenance(self):
        if self._maintenance_owned:
            self._maintenance_owned = False
            await self.manager.end_maintenance()

    async def _monitor(self):
        while True:
            await asyncio.sleep(2)
            self._load_job()
            if self.job and self.job.get("phase") in TERMINAL_PHASES:
                await self._release_maintenance()
                return
            # Detect a worker that was killed rather than leaving the dashboard
            # indefinitely busy. After installation starts, require recovery.
            try:
                output = await self._run(self.tools["systemctl"], "--user", "show", self.job["worker_unit"], "--property=ActiveState", "--value")
                active = output.strip() in {"active", "activating", "deactivating"}
            except AppUpdateError:
                active = False
            except (asyncio.TimeoutError, OSError):
                # An unavailable probe cannot establish that the worker died.
                # Keep watching its durable journal, including rollback results.
                LOG.warning("App update service probe unavailable; retrying.")
                continue
            if not active:
                self._load_job()
                if self.job.get("phase") not in TERMINAL_PHASES:
                    phase = "recovery_required" if self.job.get("backup_dir") else "failed"
                    self._save_job(phase=phase, message="The update worker stopped unexpectedly. Check its journal and saved environment.")
                await self._release_maintenance()
                return

    def start_checks(self):
        checks_allowed = self.enabled and (not self._source_error or self.config.app_updates.branch)
        if (checks_allowed or self.running) and (not self.check_task or self.check_task.done()):
            self.check_task = asyncio.create_task(self._check_loop())

    async def _recover_orphan(self):
        """A dead preparer/worker must not leave a permanently busy journal."""
        self._load_job()
        if not self.job or self.job.get("phase") in TERMINAL_PHASES:
            return
        try:
            handle = acquire_lock(self.lock_path)
        except (OSError, WorkerError):
            return
        try:
            self._load_job()
            if not self.job or self.job.get("phase") in TERMINAL_PHASES:
                return
            owner = self.job.get("owner_pid")
            if sys.platform == "linux" and isinstance(owner, int) and owner > 0 and owner != os.getpid():
                try:
                    os.kill(owner, 0)
                    return
                except ProcessLookupError:
                    pass
                except OSError:
                    return
            unit = self.job.get("worker_unit")
            if unit and re.fullmatch(r"halobridge-update-[a-zA-Z0-9-]+\.service", unit):
                systemctl = shutil.which("systemctl")
                if not systemctl:
                    return
                try:
                    active = (await self._run(systemctl, "--user", "show", unit, "--property=ActiveState", "--value")).strip()
                    if active in {"active", "activating", "deactivating"}:
                        return
                except (asyncio.TimeoutError, OSError):
                    return
                except AppUpdateError:
                    pass
                # Give a just-dispatched transient unit time to acquire flock.
                if time.time() - self.job.get("updated_at", 0) < 30:
                    return
            phase = "recovery_required" if self.job.get("backup_dir") or self.job.get("phase") in {"installing", "verifying", "restarting", "rolling_back"} else "failed"
            self._save_job(phase=phase, message="An interrupted app update was found. Check the worker journal and saved environment before retrying.")
        finally:
            handle.close()

    async def _check_loop(self):
        await self._recover_orphan()
        # A restarted app still sees the copied worker's durable job.
        if self.running and not self.monitor_task:
            self._support_error = await self._verify_support()
            if not self.tools.get("systemctl") and sys.platform == "linux":
                systemctl = shutil.which("systemctl")
                if systemctl:
                    self.tools["systemctl"] = systemctl
            if self.tools.get("systemctl") and self.job.get("worker_unit"):
                self.monitor_task = asyncio.create_task(self._monitor())
        if not self.enabled or (self._source_error and not self.config.app_updates.branch):
            return
        while True:
            await self.check()
            await asyncio.sleep(CHECK_RETRY if self.check_error else self.check_interval)

    async def close(self):
        tasks = [task for task in (self.check_task, self.monitor_task) if task]
        if self.task and not self.task.done() and not self._handoff:
            tasks.append(self.task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # After dispatch the service owns the transaction. Keep admission closed
        # until its restart, or until the monitoring task observes a failure.
        if not self._handoff:
            await self._release_maintenance()

    @staticmethod
    def _require_same_origin(request):
        if (request.headers.get("Origin") != f"{request.scheme}://{request.host}"
                or request.content_type != "application/json" or request.headers.get("X-Halogen-Action") != "update"):
            raise web.HTTPForbidden(text="Update actions require JSON from the local dashboard.")

    async def api_status(self, request):
        return web.json_response(self.status(), headers={"Cache-Control": "no-store"})

    async def api_check(self, request):
        self._require_same_origin(request)
        return web.json_response(await self.check(force=True), headers={"Cache-Control": "no-store"})

    async def api_install(self, request):
        self._require_same_origin(request)
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            raise web.HTTPBadRequest(text="Invalid JSON")
        if not isinstance(body, dict) or not isinstance(body.get("version"), str) or not body["version"] or normalize_version(body["version"]) != body["version"]:
            raise web.HTTPBadRequest(text="Specify a verified stable app version.")
        await self.start(body["version"])
        return web.json_response(self.status(), status=202, headers={"Cache-Control": "no-store"})

    def register_routes(self, app):
        app.router.add_get("/dashboard/api/app-updates", self.api_status)
        app.router.add_post("/dashboard/api/app-updates/check", self.api_check)
        app.router.add_post("/dashboard/api/app-updates/install", self.api_install)
