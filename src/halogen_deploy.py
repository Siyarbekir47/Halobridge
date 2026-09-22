"""Deployment manager for Halogen backend profiles (quadlet files).

Owns the write path for ``~/.config/containers/systemd/halogen-*.container``:
dry-run with diff, backup-before-write, atomic apply, daemon-reload, rollback
and a safe start action. Same safety primitives as the container updater:
fixed-argv subprocess calls (never a shell), atomic writes, generated-unit
verification, one mutation at a time. Model and cache directories are created
but never deleted by this module.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from discovery import discover
from profiles import (
    ENV_FIELDS,
    Profile,
    ProfileError,
    custom_template,
    official_template,
    parse_quadlet,
    profile_to_dict,
    render_quadlet,
    validate_profile,
)

logger = logging.getLogger("halobridge.deploy")

PROFILE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
QUADLET_FILENAME_RE = re.compile(r"^halogen-[a-z0-9][a-z0-9-]{0,31}\.container$")


class DeployError(ValueError):
    """Raised for a rejected deployment; the message is safe to show the user."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class DeployManager:
    def __init__(
        self,
        manager: Any,
        config: Any,
        updater: Any = None,
        dashboard: Any = None,
    ) -> None:
        self.manager = manager
        self.config = config
        self.updater = updater
        self.dashboard = dashboard
        self.quadlet_dir = Path(config.models.quadlet_dir)
        self.backup_root = Path(config.updates.backup_dir)
        self.deploy_lock = asyncio.Lock()

    # ------------------------------------------------------------------ state

    async def status(self) -> dict:
        enabled = bool(getattr(self.config, "deploy", None) and self.config.deploy.enabled)
        return {
            "enabled": enabled,
            "posix": sys.platform != "win32",
            "quadlet_dir": str(self.quadlet_dir),
            "allowed_roots": [str(p) for p in self._allowed_roots()],
            "profiles": await self.list_profiles(),
            "env_keys": sorted(ENV_FIELDS),
            "backups": self._backup_names(),
            "active_model": self.manager.current_model,
            "deploying": self.deploy_lock.locked(),
        }

    async def list_profiles(self) -> list[dict]:
        result = []
        if not self.quadlet_dir.is_dir():
            return result
        for path in sorted(self.quadlet_dir.glob("halogen-*.container")):
            try:
                profile = parse_quadlet(path.read_text(encoding="utf-8"), path)
            except ProfileError as exc:
                result.append({"profile_id": path.stem, "error": str(exc)})
                continue
            entry = profile_to_dict(profile)
            entry["active"] = self.manager.current_model == profile.model_id
            entry["service_active"] = await self._service_is_active(profile.service_name)
            entry["has_backup"] = bool(self._backups_for(path.name))
            result.append(entry)
        return result

    # ------------------------------------------------------------- templates

    def template(self, kind: str) -> dict:
        tag = self._suggested_tag()
        models_root, cache_root = self._default_roots()
        if kind == "official":
            profile = official_template(tag, models_root, cache_root)
        elif kind == "custom":
            profile = custom_template(tag, models_root, cache_root)
        else:
            raise DeployError(f"Unbekannte Vorlage: {kind}")
        return {
            "template": kind,
            "profile": profile_to_dict(profile),
            "quadlet": render_quadlet(profile),
        }

    # ------------------------------------------------------------- dry run

    def dry_run(self, payload: dict) -> dict:
        profile = self._profile_from_payload(payload)
        errors = validate_profile(profile)
        errors += self._validate_host_paths(profile)
        errors += self._validate_conflicts(profile)
        rendered = render_quadlet(profile) if not errors else ""
        target = self.quadlet_dir / f"{profile.container_name}.container"
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        diff = "\n".join(
            difflib.unified_diff(
                current.splitlines(),
                rendered.splitlines(),
                fromfile="aktuell",
                tofile="neu",
                lineterm="",
            )
        )
        warnings = []
        if profile.downloads_weights:
            warnings.append(
                "Erster Start laedt ~118 GiB Modellgewichte aus dem Internet "
                "(HALOGEN_DOWNLOAD); das kann Stunden dauern."
            )
        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "quadlet": rendered,
            "diff": diff or "(keine Aenderung)",
            "exists": target.exists(),
        }

    # --------------------------------------------------------------- apply

    async def apply(self, payload: dict) -> dict:
        self._require_enabled()
        profile = self._profile_from_payload(payload)
        errors = validate_profile(profile)
        errors += self._validate_host_paths(profile)
        errors += self._validate_conflicts(profile)
        if errors:
            raise DeployError("; ".join(errors))
        target = self.quadlet_dir / f"{profile.container_name}.container"
        rendered = render_quadlet(profile)
        async with self.deploy_lock:
            self.quadlet_dir.mkdir(parents=True, exist_ok=True)
            backup = self._backup_existing(target)
            self._atomic_write(target, rendered.encode("utf-8"))
            try:
                await self._daemon_reload()
                await self._verify_unit(profile)
            except DeployError:
                if backup is not None:
                    self._atomic_write(target, backup.read_bytes())
                    await self._daemon_reload()
                raise
            self._audit("deploy_apply", profile, backup)
        return {
            "ok": True,
            "quadlet_path": str(target),
            "backup": str(backup) if backup else None,
            "restart_required": True,
        }

    def install_dirs(self, payload: dict) -> dict:
        """Create (but never delete) the host directories a profile needs."""
        self._require_enabled()
        profile = self._profile_from_payload(payload)
        created = []
        for host_path, _container, _mode in profile.volumes:
            root = self._validate_host_path(host_path)
            root.mkdir(parents=True, exist_ok=True)
            created.append(str(root))
        return {"ok": True, "created": created}

    # -------------------------------------------------------------- delete

    async def delete(self, profile_id: str) -> dict:
        self._require_enabled()
        if not PROFILE_SLUG_RE.match(profile_id):
            raise DeployError("Ungueltige Profil-ID")
        target = self.quadlet_dir / f"halogen-{profile_id}.container"
        if not QUADLET_FILENAME_RE.match(target.name) or not target.exists():
            raise DeployError(f"Kein Quadlet fuer Profil {profile_id}")
        profile = parse_quadlet(target.read_text(encoding="utf-8"), target)
        if await self._service_is_active(profile.service_name):
            raise DeployError(
                f"Dienst {profile.service_name} laeuft noch. Erst stoppen, dann loeschen."
            )
        async with self.deploy_lock:
            backup = self._backup_existing(target)
            target.unlink()
            await self._daemon_reload()
            self._audit("deploy_delete", profile, backup)
        return {"ok": True, "backup": str(backup) if backup else None}

    # ------------------------------------------------------------ rollback

    async def rollback(self, profile_id: str) -> dict:
        self._require_enabled()
        if not PROFILE_SLUG_RE.match(profile_id):
            raise DeployError("Ungueltige Profil-ID")
        quadlet_name = f"halogen-{profile_id}.container"
        backups = self._backups_for(quadlet_name)
        if not backups:
            raise DeployError(f"Kein Backup fuer Profil {profile_id}")
        latest = backups[-1]
        target = self.quadlet_dir / quadlet_name
        async with self.deploy_lock:
            current_backup = self._backup_existing(target)
            self._atomic_write(target, latest.read_bytes())
            await self._daemon_reload()
            self._audit("deploy_rollback", quadlet_name, current_backup)
        return {"ok": True, "restored_from": str(latest)}

    # ------------------------------------------------------- service action

    async def start(self, profile_id: str) -> dict:
        self._require_enabled()
        if not PROFILE_SLUG_RE.match(profile_id):
            raise DeployError("Ungueltige Profil-ID")
        if self.manager.switching or self.manager.maintenance:
            raise DeployError(
                "Modellwechsel oder Wartung laeuft gerade - bitte kurz warten."
            )
        if await self._any_other_service_active(profile_id):
            raise DeployError(
                "Ein anderes Backend ist bereits aktiv. Halobridge bedient immer "
                "nur ein Backend gleichzeitig; wechsle zuerst das Modell."
            )
        target = self.quadlet_dir / f"halogen-{profile_id}.container"
        if not target.exists():
            raise DeployError(f"Kein Quadlet fuer Profil {profile_id}")
        profile = parse_quadlet(target.read_text(encoding="utf-8"), target)
        await self._run(
            "systemctl", "--user", "start", profile.service_name, timeout=900
        )
        self._audit("deploy_start", profile, None)
        return {"ok": True, "service": profile.service_name}

    # -------------------------------------------------- live rediscovery

    def reload_discovery(self) -> dict:
        """Re-run discovery and swap the router maps so new profiles become
        usable without restarting halobridge."""
        self._require_enabled()
        try:
            specs, _skipped = discover(
                self.config.models.quadlet_dir, self.config.updates.image_repo
            )
        except Exception as exc:
            raise DeployError(f"Discovery fehlgeschlagen: {exc}")
        discovered = {spec.model_id: spec.service for spec in specs.values()}
        discovered.update(self.config.models.explicit)
        if not discovered:
            raise DeployError("Keine Halogen-Backends gefunden - nichts geladen.")
        self.manager.models = discovered
        if self.updater is not None:
            self.updater.models = discovered
        if self.dashboard is not None:
            self.dashboard.models = discovered
        return {"ok": True, "models": sorted(discovered)}

    # ------------------------------------------------------------ helpers

    def _require_enabled(self) -> None:
        deploy = getattr(self.config, "deploy", None)
        if deploy is None or not deploy.enabled:
            raise DeployError("Deployment ist in der Konfiguration deaktiviert.")
        if sys.platform == "win32":
            raise DeployError("Deployment ist nur unter Linux/POSIX verfuegbar.")

    def _profile_from_payload(self, payload: dict) -> Profile:
        try:
            volumes = [tuple(v) for v in payload.get("volumes", [])]
            if not all(len(v) == 3 for v in volumes):
                raise DeployError("volumes: jede Angabe braucht host, container, mode")
            profile = Profile(
                profile_id=str(payload.get("profile_id", "")),
                image=str(payload.get("image", "")),
                volumes=volumes,
                host_port=int(payload.get("host_port", 0)),
                env={
                    str(k): str(v)
                    for k, v in (payload.get("env") or {}).items()
                    if str(v) != ""
                },
            )
        except (TypeError, ValueError) as exc:
            raise DeployError(f"Ungueltiges Profil: {exc}")
        return profile

    def _allowed_roots(self) -> list[Path]:
        deploy = getattr(self.config, "deploy", None)
        raw = getattr(deploy, "allowed_roots", None) if deploy else None
        if not raw:
            return [Path.home()]
        return [Path(str(p)).expanduser().resolve() for p in raw]

    def _default_roots(self) -> tuple[Path, Path]:
        deploy = getattr(self.config, "deploy", None)
        models = getattr(deploy, "models_root", None) if deploy else None
        cache = getattr(deploy, "cache_root", None) if deploy else None
        home = Path.home()
        return (
            Path(str(models)).expanduser() if models else home / "halogen" / "models",
            Path(str(cache)).expanduser() if cache else home / "halogen" / "cache",
        )

    def _validate_host_path(self, host_path: str) -> Path:
        path = Path(str(host_path)).expanduser()
        if not path.is_absolute():
            raise DeployError(f"Pfad muss absolut sein: {host_path}")
        resolved = path.resolve()
        roots = self._allowed_roots()
        if not any(_is_within(resolved, root) for root in roots):
            raise DeployError(
                f"Pfad ausserhalb der erlaubten Wurzeln: {host_path} "
                f"(erlaubt: {', '.join(str(r) for r in roots)})"
            )
        if resolved == Path("/") or len(resolved.parts) < 2:
            raise DeployError(f"Zu unspezifischer Pfad: {host_path}")
        return resolved

    def _validate_host_paths(self, profile: Profile) -> list[str]:
        errors = []
        for host_path, _container, _mode in profile.volumes:
            try:
                self._validate_host_path(host_path)
            except DeployError as exc:
                errors.append(str(exc))
        return errors

    def _validate_conflicts(self, profile: Profile) -> list[str]:
        errors = []
        if not self.quadlet_dir.is_dir():
            return errors
        for path in self.quadlet_dir.glob("halogen-*.container"):
            if path.name == f"{profile.container_name}.container":
                continue
            try:
                other = parse_quadlet(path.read_text(encoding="utf-8"), path)
            except ProfileError:
                continue
            if other.model_id == profile.model_id:
                errors.append(
                    f"Modell-ID '{profile.model_id}' wird schon von Profil "
                    f"'{other.profile_id}' verwendet"
                )
            if other.host_port == profile.host_port:
                errors.append(
                    f"Port {profile.host_port} wird schon von Profil "
                    f"'{other.profile_id}' verwendet"
                )
        return errors

    def _suggested_tag(self) -> str:
        if self.updater is not None:
            latest = getattr(self.updater, "latest_version", None)
            if latest:
                return str(latest)
        return "0.13.2"

    async def _any_other_service_active(self, profile_id: str) -> bool:
        if not self.quadlet_dir.is_dir():
            return False
        for path in self.quadlet_dir.glob("halogen-*.container"):
            if path.stem == f"halogen-{profile_id}":
                continue
            try:
                other = parse_quadlet(path.read_text(encoding="utf-8"), path)
            except ProfileError:
                continue
            if await self._service_is_active(other.service_name):
                return True
        return False

    async def _service_is_active(self, service_name: str) -> bool:
        if not service_name.startswith("halogen-") or not service_name.endswith(".service"):
            raise DeployError(f"Ungueltiger Dienstname: {service_name}")
        process = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            "is-active",
            "--quiet",
            service_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            returncode = await asyncio.wait_for(process.wait(), 15)
        except asyncio.TimeoutError:
            process.kill()
            return False
        return returncode == 0

    async def _run(self, *args: str, timeout: float = 120) -> str:
        """Fixed argv only; never invoke a shell."""
        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout
            )
        except asyncio.TimeoutError:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise DeployError(f"Timeout bei: {' '.join(args[:2])}")
        if process.returncode:
            raise DeployError(
                f"{' '.join(args[:2])} fehlgeschlagen (Exit {process.returncode}): "
                f"{(stderr or stdout).decode('utf-8', errors='replace').strip()[:200]}"
            )
        return stdout.decode("utf-8", errors="replace")

    async def _daemon_reload(self) -> None:
        await self._run("systemctl", "--user", "daemon-reload", timeout=120)

    async def _verify_unit(self, profile: Profile) -> None:
        unit = await self._run(
            "systemctl", "--user", "cat", profile.service_name, timeout=60
        )
        if profile.image not in unit:
            raise DeployError(
                f"Generiertes Unit fuer {profile.service_name} enthaelt nicht das "
                "erwartete Image"
            )

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> None:
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.write_bytes(data)
        Path.replace(tmp, target)

    def _backup_existing(self, target: Path) -> Path | None:
        if not target.exists():
            return None
        self.backup_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup = self.backup_root / f"profile-{target.stem}-{stamp}.container"
        backup.write_bytes(target.read_bytes())
        return backup

    def _backups_for(self, quadlet_name: str) -> list[Path]:
        stem = quadlet_name[: -len(".container")]
        if not self.backup_root.is_dir():
            return []
        return sorted(self.backup_root.glob(f"profile-{stem}-*.container"))

    def _backup_names(self) -> list[str]:
        if not self.backup_root.is_dir():
            return []
        return sorted(p.name for p in self.backup_root.glob("profile-*.container"))

    def _audit(self, action: str, subject: Any, backup: Path | None) -> None:
        name = subject if isinstance(subject, str) else getattr(subject, "profile_id", "?")
        logger.info("%s profile=%s backup=%s", action, name, backup)


class DeployRoutes:
    """HTTP surface for profile deployment. Mutations require same-origin plus
    the ``X-Halogen-Action: deploy`` header, mirroring the update endpoints."""

    def __init__(self, manager: DeployManager) -> None:
        self.manager = manager

    def _guard(self, request: web.Request) -> None:
        expected = f"http://{request.headers.get('Host', '')}"
        origin = request.headers.get("Origin")
        if origin is not None and origin != expected:
            raise DeployError("Cross-Origin-Request abgelehnt")
        if request.headers.get("X-Halogen-Action") != "deploy":
            raise DeployError("Header X-Halogen-Action: deploy erforderlich")

    def _payload(self, request: web.Request) -> dict:
        if request.content_type != "application/json":
            raise DeployError("JSON-Body erwartet")
        try:
            data = json.loads(request.text or "{}")
        except json.JSONDecodeError as exc:
            raise DeployError(f"Ungueltiges JSON: {exc}")
        if not isinstance(data, dict):
            raise DeployError("JSON-Objekt erwartet")
        return data

    async def api_status(self, _: web.Request) -> web.Response:
        return web.json_response(await self.manager.status())

    async def api_template(self, request: web.Request) -> web.Response:
        try:
            return web.json_response(self.manager.template(request.match_info["kind"]))
        except DeployError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def api_dry_run(self, request: web.Request) -> web.Response:
        self._guard(request)
        try:
            return web.json_response(self.manager.dry_run(self._payload(request)))
        except DeployError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def api_apply(self, request: web.Request) -> web.Response:
        self._guard(request)
        try:
            return web.json_response(await self.manager.apply(self._payload(request)))
        except DeployError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def api_install_dirs(self, request: web.Request) -> web.Response:
        self._guard(request)
        try:
            return web.json_response(self.manager.install_dirs(self._payload(request)))
        except DeployError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def api_delete(self, request: web.Request) -> web.Response:
        self._guard(request)
        try:
            return web.json_response(
                await self.manager.delete(request.match_info["profile_id"])
            )
        except DeployError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def api_rollback(self, request: web.Request) -> web.Response:
        self._guard(request)
        try:
            return web.json_response(
                await self.manager.rollback(request.match_info["profile_id"])
            )
        except DeployError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def api_start(self, request: web.Request) -> web.Response:
        self._guard(request)
        try:
            return web.json_response(
                await self.manager.start(request.match_info["profile_id"])
            )
        except DeployError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def api_reload(self, _: web.Request) -> web.Response:
        try:
            return web.json_response(self.manager.reload_discovery())
        except DeployError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    def register_routes(self, app: web.Application) -> None:
        app.router.add_get("/dashboard/api/deploy", self.api_status)
        app.router.add_get("/dashboard/api/deploy/template/{kind}", self.api_template)
        app.router.add_post("/dashboard/api/deploy/dry-run", self.api_dry_run)
        app.router.add_post("/dashboard/api/deploy/apply", self.api_apply)
        app.router.add_post("/dashboard/api/deploy/install-dirs", self.api_install_dirs)
        app.router.add_post("/dashboard/api/deploy/{profile_id}/delete", self.api_delete)
        app.router.add_post(
            "/dashboard/api/deploy/{profile_id}/rollback", self.api_rollback
        )
        app.router.add_post("/dashboard/api/deploy/{profile_id}/start", self.api_start)
        app.router.add_post("/dashboard/api/deploy/reload", self.api_reload)
