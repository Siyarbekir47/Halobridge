"""One-click updates of the two local Halogen Quadlets, coordinated with routing.

Only stable upstream tags and the fixed GHCR repository are accepted. A durable
journal is written before changing either file, allowing recovery on restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

# Re-exported so existing callers/tests keep working; the definitions live in
# semver.py, shared with discovery.
from semver import normalize_version, version_tuple


LOG = logging.getLogger("halogen-updates")
REPOSITORY = "peonist-ai/halogen-flash-server"
IMAGE_REPOSITORY = f"ghcr.io/{REPOSITORY}"
TAGS_URL = f"https://api.github.com/repos/{REPOSITORY}/tags"
CHECK_INTERVAL = 6 * 3600
CHECK_RETRY = 300
DRAIN_TIMEOUT = 7200
START_TIMEOUT = 600
TERMINAL_PHASES = {"succeeded", "failed", "rolled_back"}
JOB_PHASES = TERMINAL_PHASES | {"starting", "pulling", "draining", "configuring", "restarting", "verifying", "rolling_back", "recovery_required"}


class UpdateError(RuntimeError):
    pass


def quadlet_image(
    content: bytes,
    replacement: str | None = None,
    image_repo: str | None = None,
) -> tuple[str, bytes]:
    """Change exactly one Image in [Container], preserving every other byte."""
    lines = content.decode("utf-8").splitlines(keepends=True)
    section = ""
    found = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        match = re.fullmatch(r"([ \t]*Image[ \t]*=[ \t]*)(\S+)([ \t]*)(\r?\n)?", line)
        if section == "[Container]" and match:
            found.append(match[2])
            if replacement:
                lines[index] = match[1] + replacement + match[3] + (match[4] or "")
    if len(found) != 1:
        raise UpdateError("Jedes Quadlet muss genau eine Image-Zeile in [Container] enthalten.")
    prefix = (image_repo or IMAGE_REPOSITORY) + ":"
    if not found[0].startswith(prefix) or version_tuple(found[0][len(prefix):]) is None:
        raise UpdateError("Das konfigurierte Image ist kein versioniertes Halogen-Image.")
    return found[0], "".join(lines).encode("utf-8")


def atomic_write(path: Path, content: bytes, mode: int = 0o600) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class ContainerUpdater:
    def __init__(
        self,
        manager: Any,
        session: ClientSession,
        models: dict[str, str],
        *,
        quadlet_dir: Path | None = None,
        backup_root: Path | None = None,
        state_dir: Path | None = None,
        config: Any | None = None,
        image_repo: str | None = None,
        github_repo: str | None = None,
        tag_source: str | None = None,
        check_interval: float | None = None,
        drain_timeout: float | None = None,
        start_timeout: float | None = None,
        stop_timeout: float | None = None,
        backup_keep: int | None = None,
        enabled: bool | None = None,
        allow_install: bool | None = None,
    ) -> None:
        self.manager, self.session, self.models = manager, session, models

        if config is not None:
            image_repo = image_repo or config.updates.image_repo
            github_repo = github_repo or config.updates.github_repo
            tag_source = tag_source or config.updates.tag_source
            check_interval = (
                check_interval
                if check_interval is not None
                else config.updates.check_interval_h * 3600
            )
            quadlet_dir = quadlet_dir or config.models.quadlet_dir
            backup_root = backup_root or config.updates.backup_dir
            state_dir = state_dir or config.dashboard.state_dir
            drain_timeout = (
                drain_timeout
                if drain_timeout is not None
                else config.router.drain_timeout_s
            )
            start_timeout = (
                start_timeout
                if start_timeout is not None
                else config.router.start_timeout_s
            )
            stop_timeout = (
                stop_timeout
                if stop_timeout is not None
                else config.router.stop_timeout_s
            )
            backup_keep = (
                backup_keep if backup_keep is not None else config.updates.backup_keep
            )
            enabled = config.updates.enabled if enabled is None else enabled
            allow_install = (
                config.security.allow_install if allow_install is None else allow_install
            )

        self.image_repo = image_repo or IMAGE_REPOSITORY
        self.github_repo = github_repo or REPOSITORY
        self.tag_source = tag_source or "github"
        self.check_interval = check_interval if check_interval is not None else CHECK_INTERVAL
        self.check_retry = CHECK_RETRY
        self.drain_timeout = drain_timeout if drain_timeout is not None else DRAIN_TIMEOUT
        self.start_timeout = start_timeout if start_timeout is not None else START_TIMEOUT
        self.stop_timeout = stop_timeout if stop_timeout is not None else 180
        self.backup_keep = backup_keep if backup_keep is not None else 5
        self.enabled = True if enabled is None else enabled
        self.allow_install = True if allow_install is None else allow_install

        self.quadlet_dir = quadlet_dir or Path.home() / ".config/containers/systemd"
        self.backup_root = backup_root or Path.home() / "halogen/backups"
        self.state_dir = state_dir or Path.home() / ".local/state/halogen-dashboard"
        self.journal_path = self.state_dir / "container-update.json"
        self.tags_url: str | None = None
        self.check_lock = asyncio.Lock()
        self.start_lock = asyncio.Lock()
        self.latest_version: str | None = None
        self.current_version: str | None = None
        self.checked_at: float | None = None
        self.attempted_at = 0.0
        self.check_error: str | None = None
        self.job: dict[str, Any] | None = None
        self.task: asyncio.Task | None = None
        self.check_task: asyncio.Task | None = None
        self.recovery_required = False

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def _paths(self) -> dict[str, Path]:
        return {model: self.quadlet_dir / service.replace(".service", ".container") for model, service in self.models.items()}

    def _configuration(self) -> dict[str, dict[str, Any]]:
        result = {}
        for model, path in self._paths().items():
            if path.is_symlink() or not path.is_file():
                raise UpdateError(f"Lokales Quadlet fehlt oder ist ein Symlink: {path.name}")
            content = path.read_bytes()
            image, _ = quadlet_image(content, image_repo=self.image_repo)
            result[model] = {"image": image, "version": normalize_version(image.rsplit(":", 1)[1]), "content": content}
        return result

    def _support_error(self) -> str | None:
        if not self.enabled:
            return "Updates sind deaktiviert."
        if os.name != "posix" or not shutil.which("podman") or not shutil.which("systemctl"):
            return "Updates benötigen den Linux-Host mit Podman und systemd im Router-Benutzerkonto."
        try:
            self._configuration()
        except (OSError, UnicodeError, UpdateError) as error:
            return str(error)
        return None

    async def _run(self, *args: str, timeout: float = 120) -> str:
        """Fixed argv only; never invoke a shell or log container environments."""
        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            raise
        if process.returncode:
            raise UpdateError(f"{' '.join(args[:2])} fehlgeschlagen (Exit {process.returncode}).")
        return stdout.decode("utf-8", errors="replace")

    async def _inspect_image(self, image: str) -> str:
        data = json.loads(await self._run("podman", "image", "inspect", image))
        image_id = data[0].get("Id")
        if not isinstance(image_id, str) or not image_id:
            raise UpdateError("Image-ID konnte nicht geprüft werden.")
        return image_id.removeprefix("sha256:")

    async def _container_image(self, model: str) -> str:
        name = self.models[model].removesuffix(".service")
        data = json.loads(await self._run("podman", "container", "inspect", name))
        image_id = data[0].get("Image")
        if not isinstance(image_id, str) or not image_id:
            raise UpdateError("Laufendes Container-Image konnte nicht geprüft werden.")
        return image_id.removeprefix("sha256:")

    async def _fetch_latest(self) -> str:
        if self.tag_source == "registry":
            return await self._fetch_latest_registry()
        return await self._fetch_latest_github()

    async def _fetch_latest_github(self) -> str:
        versions = []
        headers = {"Accept": "application/vnd.github+json", "Accept-Encoding": "identity",
                   "User-Agent": f"{self.github_repo}-updater", "X-GitHub-Api-Version": "2022-11-28"}
        url = self.tags_url or TAGS_URL
        # Tags, not /releases/latest: upstream currently publishes no Releases.
        for page in range(1, 11):
            async with self.session.get(
                url, params={"per_page": "100", "page": str(page)},
                headers=headers, timeout=ClientTimeout(total=15),
            ) as response:
                if response.status in {403, 429}:
                    raise UpdateError("GitHub-Abfragelimit erreicht. Später erneut prüfen.")
                if response.status != 200:
                    raise UpdateError(f"GitHub-Versionen nicht erreichbar (HTTP {response.status}).")
                tags = await response.json()
            if not isinstance(tags, list):
                raise UpdateError("Ungültige Versionsantwort von GitHub.")
            versions.extend(parsed for tag in tags if isinstance(tag, dict)
                            if (parsed := version_tuple(tag.get("name"))) is not None)
            if len(tags) < 100:
                break
        else:
            raise UpdateError("Tagliste zu groß für eine vollständige Versionsprüfung.")
        if not versions:
            raise UpdateError("Keine stabile Halogen-Version gefunden.")
        return ".".join(map(str, max(versions)))

    async def _fetch_latest_registry(self) -> str:
        try:
            output = await self._run(
                "podman", "search", "--list-tags", "--limit", "1000",
                "--format", "{{.Tag}}", self.image_repo, timeout=120,
            )
        except UpdateError as error:
            raise UpdateError(f"Registry-Versionen nicht erreichbar: {error}") from error
        versions = []
        for line in output.splitlines():
            parsed = version_tuple(line.strip())
            if parsed is not None:
                versions.append(parsed)
        if not versions:
            raise UpdateError("Keine stabile Halogen-Version in der Registry gefunden.")
        return ".".join(map(str, max(versions)))

    async def check(self, force: bool = False) -> dict[str, Any]:
        async with self.check_lock:
            if not self.enabled:
                return self.status()
            delay = self.check_retry if self.check_error else self.check_interval
            if self.running or self.recovery_required or (not force and time.time() - self.attempted_at < delay):
                return self.status()
            # Even manual checks have a small cooldown across browser tabs.
            if force and time.time() - self.attempted_at < 30:
                return self.status()
            self.attempted_at = time.time()
            try:
                self.latest_version = await self._fetch_latest()
                health = await self.manager.backend_health()
                version = (health or {}).get("version") or {}
                self.current_version = normalize_version(version.get("api")) if isinstance(version, dict) else None
                self.checked_at = time.time()
                self.check_error = None
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.check_error = str(error) if isinstance(error, UpdateError) else "Versionsprüfung fehlgeschlagen. Netzwerkverbindung prüfen."
            return self.status()

    def status(self) -> dict[str, Any]:
        support_error = self._support_error()
        configured = {}
        try:
            configured = {model: value["version"] for model, value in self._configuration().items()}
        except (OSError, UnicodeError, UpdateError):
            pass
        latest = version_tuple(self.latest_version)
        versions = [version_tuple(value) for value in configured.values()]
        if self.current_version:
            versions.append(version_tuple(self.current_version))
        available = bool(latest and versions and all(value is not None and value <= latest for value in versions)
                         and any(value < latest for value in versions if value is not None))
        up_to_date = bool(latest and self.current_version and len(configured) == len(self.models)
                          and all(value is not None and value >= latest for value in versions))
        blocked_reason = None
        if latest and any(value < latest for value in versions if value is not None) and any(value > latest for value in versions if value is not None):
            blocked_reason = "Unterschiedliche Versionsstände: Eine Konfiguration ist neuer als der GitHub-Tag. Kein automatischer Downgrade."
        fresh = self.checked_at is not None and time.time() - self.checked_at < self.check_interval
        return {
            "latest_version": self.latest_version, "current_version": self.current_version,
            "configured_versions": configured, "checked_at": self.checked_at,
            "check_error": self.check_error, "supported": support_error is None,
            "support_error": support_error, "update_available": available,
            "up_to_date": up_to_date, "blocked_reason": blocked_reason,
            "can_install": (
                available
                and fresh
                and self.enabled
                and self.allow_install
                and not self.check_error
                and not support_error
                and not self.running
                and not self.recovery_required
            ),
            "running": self.running, "recovery_required": self.recovery_required,
            "can_recover": self.recovery_required and bool(self.job and self.job.get("backup_dir")) and not self.running,
            "job": self.job,
            "release_url": f"https://github.com/{self.github_repo}/blob/v{self.latest_version}/CHANGELOG.md" if latest else None,
        }

    def _save_job(self, **changes: Any) -> None:
        assert self.job is not None
        updated = {**self.job, **changes, "updated_at": time.time()}
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        atomic_write(self.journal_path, json.dumps(updated, ensure_ascii=False).encode("utf-8"))
        self.job = updated

    def _prune_backups(self) -> None:
        if self.backup_keep <= 0:
            return
        try:
            backups = [
                path
                for path in self.backup_root.glob("dashboard-update-*")
                if path.is_dir()
            ]
        except OSError:
            return
        backups.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for old in backups[self.backup_keep:]:
            try:
                shutil.rmtree(old)
            except OSError:
                pass

    async def start(self, version: str) -> None:
        async with self.start_lock:
            if not self.enabled:
                raise web.HTTPForbidden(text="Updates sind deaktiviert.")
            if not self.allow_install:
                raise web.HTTPForbidden(text="Installation ist deaktiviert.")
            if self.running or self.recovery_required:
                raise web.HTTPConflict(text="Ein Update oder eine Wiederherstellung läuft bereits.")
            await self.check()
            status = self.status()
            if version != self.latest_version or not status["can_install"]:
                raise web.HTTPConflict(text=status["support_error"] or status["check_error"] or "Bitte Versionsstand erneut prüfen; diese Version kann nicht installiert werden.")
            self.job = {"version": version, "phase": "starting", "message": "Update wird vorbereitet.",
                        "started_at": time.time(), "changed": False, "backup_dir": None}
            self._save_job()
            self.task = asyncio.create_task(self._update(version))

    async def _verify_units(self, image: str) -> None:
        for service in self.models.values():
            command = await self._run("systemctl", "--user", "show", service, "--property=ExecStart", "--value")
            if image not in command.split():
                raise UpdateError(f"Generierter Dienst {service} verwendet nicht das erwartete Image.")

    async def _ready(self, model: str, version: str, image_id: str) -> None:
        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            health = await self.manager.backend_health() or {}
            versions = health.get("version") or {}
            if isinstance(versions, dict) and health.get("status") == "ok" and health.get("model") == model:
                api = normalize_version(versions.get("api"))
                engine = normalize_version(versions.get("engine"))
                # Old releases may lack capability_probe; 0.13.1+ must report ok.
                probe_ok = health.get("capability_probe") == "ok" if version_tuple(version) >= (0, 13, 1) else True
                if api == version and engine == version and probe_ok:
                    if await self._container_image(model) != image_id:
                        raise UpdateError("Der gestartete Container verwendet eine unerwartete Image-ID.")
                    return
            await asyncio.sleep(2)
        raise UpdateError("Backend wurde nicht mit erwarteter API-/Engine-Version und Capability-Probe bereit.")

    async def _restart(self, model: str) -> None:
        # Stop + wait for GTT release avoids starting a second model over memory
        # still held by the driver. The router admission gate stays closed.
        await self._run("systemctl", "--user", "stop", self.models[model], timeout=self.stop_timeout)
        await self.manager.wait_for_gtt_release()
        await self._run("systemctl", "--user", "start", self.models[model], timeout=self.start_timeout)

    async def _update(self, version: str) -> None:
        maintenance = False
        try:
            image = f"{self.image_repo}:{version}"
            self._save_job(phase="pulling", message=f"Image {version} wird geladen. Anfragen laufen weiter.")
            await self._run("podman", "pull", image, timeout=1800)
            image_id = await self._inspect_image(image)
            self._save_job(phase="draining", message="Warte auf Modellwechsel und laufende Anfragen.")
            await asyncio.wait_for(self.manager.begin_maintenance(), self.drain_timeout)
            maintenance = True
            await asyncio.wait_for(self.manager.drain_maintenance(), self.drain_timeout)
            model = self.manager.current_model
            if model not in self.models:
                raise UpdateError("Aktives Modell ist nicht bekannt.")
            configurations = self._configuration()
            if any(version_tuple(value["version"]) > version_tuple(version) for value in configurations.values()):
                raise UpdateError("Konfiguration wurde inzwischen aktualisiert; Downgrade abgebrochen.")
            health = await self.manager.backend_health()
            if not health or health.get("status") != "ok" or health.get("model") != model:
                raise UpdateError("Aktives Backend konnte vor dem Update nicht eindeutig geprüft werden.")
            for other, service in self.models.items():
                if other != model:
                    active = await self._run("systemctl", "--user", "show", service, "--property=ActiveState", "--value")
                    if active.strip() not in {"inactive", "failed"}:
                        raise UpdateError("Ein zweiter Modelldienst ist aktiv; Update abgebrochen.")
            old_version = normalize_version(((health or {}).get("version") or {}).get("api"))
            if not old_version or version_tuple(old_version) > version_tuple(version):
                raise UpdateError("Laufende Version ist unbekannt oder neuer als das Update.")
            old_image_id = await self._container_image(model)
            if await self._inspect_image(configurations[model]["image"]) != old_image_id:
                raise UpdateError("Laufendes Image und aktives Quadlet stimmen nicht überein.")
            backup = self.backup_root / f"dashboard-update-{version}-{time.time_ns()}"
            backup.mkdir(parents=True, mode=0o700)
            modes = {}
            for name, path in self._paths().items():
                modes[name] = stat.S_IMODE(path.stat().st_mode)
                atomic_write(backup / path.name, configurations[name]["content"], modes[name])
            self._save_job(phase="configuring", message="Quadlets gesichert; beide Image-Versionen werden aktualisiert.",
                           backup_dir=str(backup), model=model, old_version=old_version,
                           old_image_id=old_image_id, modes=modes, changed=True)
            # changed is durable BEFORE the first write (including partial failure).
            for name, path in self._paths().items():
                if path.read_bytes() != configurations[name]["content"]:
                    raise UpdateError("Quadlet wurde während des Updates extern verändert.")
                _, content = quadlet_image(configurations[name]["content"], image, image_repo=self.image_repo)
                atomic_write(path, content, modes[name])
            await self._run("systemctl", "--user", "daemon-reload")
            await self._verify_units(image)
            self._save_job(phase="restarting", message=f"{model} wird mit {version} gestartet.")
            await self._restart(model)
            self._save_job(phase="verifying", message="Prüfe API, Engine, Modell, Capability-Probe und Container-Image.")
            await self._ready(model, version, image_id)
            self.current_version = version
            self._save_job(phase="succeeded", message=f"{version} aktiv. Beide Quadlets aktualisiert; das inaktive Modell folgt beim nächsten Wechsel.", changed=False)
            self._prune_backups()
        except (Exception, asyncio.CancelledError) as error:
            LOG.warning("Container-Update fehlgeschlagen: %s", type(error).__name__)
            reason = str(error) if isinstance(error, UpdateError) else "Update abgebrochen oder Zeitlimit überschritten."
            if self.job and self.job.get("changed"):
                try:
                    await self._rollback(reason)
                except (Exception, asyncio.CancelledError):
                    self.recovery_required = True
                    self._save_job(phase="recovery_required", message="Rollback nicht abgeschlossen. Wiederherstellung erneut starten; Router bleibt im Wartungsmodus.")
            elif self.job:
                self._save_job(phase="failed", message=reason, changed=False)
        finally:
            if maintenance and not self.recovery_required:
                await self.manager.end_maintenance()

    async def _rollback(self, reason: str) -> None:
        assert self.job is not None
        self._save_job(phase="rolling_back", message=f"{reason} Vorherige Konfiguration wird wiederhergestellt.")
        model = self.job["model"]
        backup = Path(self.job["backup_dir"])
        if model not in self.models or backup.resolve().parent != self.backup_root.resolve():
            raise UpdateError("Ungültiger Wiederherstellungsdatensatz.")
        # Read and validate BOTH backups before restoring either file.
        originals = {name: (backup / path.name).read_bytes() for name, path in self._paths().items()}
        for content in originals.values():
            quadlet_image(content, image_repo=self.image_repo)
        for name, path in self._paths().items():
            atomic_write(path, originals[name], self.job["modes"][name])
        await self._run("systemctl", "--user", "daemon-reload")
        # A process restart could have occurred midway through startup. Stop both
        # managed units before bringing back exactly the originally active model.
        await self._run("systemctl", "--user", "stop", *self.models.values(), timeout=self.stop_timeout)
        await self.manager.wait_for_gtt_release()
        await self._run("systemctl", "--user", "reset-failed", self.models[model])
        await self._run("systemctl", "--user", "start", self.models[model], timeout=self.start_timeout)
        await self._ready(model, self.job["old_version"], self.job["old_image_id"])
        self.manager.current_model = model
        self.current_version = self.job["old_version"]
        self.recovery_required = False
        self._save_job(phase="rolled_back", message=f"{reason} Vorherige Version {self.current_version} ist wieder aktiv.", changed=False)

    async def recover_on_startup(self) -> None:
        if not self.journal_path.exists():
            return
        try:
            self.job = json.loads(self.journal_path.read_text(encoding="utf-8"))
            if not isinstance(self.job, dict) or self.job.get("phase") not in JOB_PHASES or not isinstance(self.job.get("changed"), bool):
                raise ValueError
            if self.job["phase"] in TERMINAL_PHASES and not self.job["changed"]:
                return
            if not self.job.get("changed"):
                self._save_job(phase="failed", message="Update durch Router-Neustart unterbrochen; Quadlets wurden noch nicht geändert.")
                return
            await self.manager.begin_maintenance()
            await self._rollback("Unterbrochenes Update erkannt.")
            await self.manager.end_maintenance()
        except Exception:
            self.recovery_required = True
            self.manager.maintenance = True
            # Keep a damaged journal intact for diagnosis rather than overwrite it.
            if not isinstance(self.job, dict):
                self.job = None
            LOG.error("Automatische Update-Wiederherstellung fehlgeschlagen; Wartungsmodus aktiv.")

    async def _retry_recovery(self) -> None:
        try:
            await self._rollback("Wiederherstellung erneut angefordert.")
            await self.manager.end_maintenance()
        except (Exception, asyncio.CancelledError):
            self.recovery_required = True
            if self.job:
                self._save_job(phase="recovery_required", message="Wiederherstellung fehlgeschlagen. Backup und Routerjournal prüfen.")

    def start_checks(self) -> None:
        if self.enabled and self._support_error() is None:
            self.check_task = asyncio.create_task(self._check_loop())

    async def _check_loop(self) -> None:
        while True:
            await self.check()
            await asyncio.sleep(self.check_retry if self.check_error else self.check_interval)

    async def close(self) -> None:
        if self.check_task:
            self.check_task.cancel()
            await asyncio.gather(self.check_task, return_exceptions=True)
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    @staticmethod
    def _require_same_origin(request: web.Request) -> None:
        # No CORS: a foreign page must not be able to trigger maintenance. These
        # actions are intended for the existing trusted local/NetBird interface.
        expected = f"{request.scheme}://{request.host}"
        if request.headers.get("Origin") != expected or request.content_type != "application/json" or request.headers.get("X-Halogen-Action") != "update":
            raise web.HTTPForbidden(text="Updateaktionen nur aus dem lokalen Dashboard mit JSON zulässig.")

    async def api_status(self, _: web.Request) -> web.Response:
        return web.json_response(self.status(), headers={"Cache-Control": "no-store"})

    async def api_check(self, request: web.Request) -> web.Response:
        self._require_same_origin(request)
        return web.json_response(await self.check(force=True), headers={"Cache-Control": "no-store"})

    async def api_install(self, request: web.Request) -> web.Response:
        self._require_same_origin(request)
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            raise web.HTTPBadRequest(text="Ungültiges JSON")
        if not isinstance(body, dict) or normalize_version(body.get("version")) != body.get("version") or not body.get("version"):
            raise web.HTTPBadRequest(text="Eine geprüfte stabile Version muss angegeben werden.")
        await self.start(body["version"])
        return web.json_response(self.status(), status=202, headers={"Cache-Control": "no-store"})

    async def api_recover(self, request: web.Request) -> web.Response:
        self._require_same_origin(request)
        async with self.start_lock:
            if self.running or not self.recovery_required or not self.job or not self.job.get("backup_dir"):
                raise web.HTTPConflict(text="Keine automatisch wiederherstellbare Aktualisierung vorhanden.")
            self.task = asyncio.create_task(self._retry_recovery())
        return web.json_response(self.status(), status=202, headers={"Cache-Control": "no-store"})

    def register_routes(self, app: web.Application) -> None:
        app.router.add_get("/dashboard/api/updates", self.api_status)
        app.router.add_post("/dashboard/api/updates/check", self.api_check)
        app.router.add_post("/dashboard/api/updates/install", self.api_install)
        app.router.add_post("/dashboard/api/updates/recover", self.api_recover)
