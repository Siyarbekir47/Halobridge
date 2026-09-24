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
import codecs
import difflib
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from discovery import discover
from profiles import (
    ENV_FIELDS,
    OFFICIAL_WEIGHTS_REPO,
    REPO_ID_RE,
    UNCENSORED_DEFAULT_GGUF_NAME,
    UNCENSORED_DEFAULT_OUTPUT,
    UNCENSORED_HF_REPO,
    Profile,
    ProfileError,
    custom_template,
    official_quick_template,
    official_template,
    parse_quadlet,
    profile_to_dict,
    render_quadlet,
    uncensored_quick_template,
    uncensored_template,
    validate_profile,
)

logger = logging.getLogger("halobridge.deploy")

PROFILE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
QUADLET_FILENAME_RE = re.compile(r"^halogen-[a-z0-9][a-z0-9-]{0,31}\.container$")
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")

# How long a quick-install takeover waits for the new backend to become healthy
# before giving up and rolling back to the previous model. Kept above the normal
# 2-3 min model load so a healthy start finishes, but bounded so a broken one
# recovers instead of freezing the host for the full router start timeout.
QUICK_START_TIMEOUT = 300
# The official one-click profile downloads roughly 118 GiB inside the backend
# container on its first start. Keep that operation bounded, but give normal
# internet connections enough time to complete it.
QUICK_OFFICIAL_START_TIMEOUT = 6 * 60 * 60
HF_CLI_PACKAGE = "huggingface_hub"
HF_GATED_ACCESS_MESSAGE = (
    "Hugging Face denied access to the uncensored repository. Open "
    "https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF "
    "in a browser, accept the terms or request access, wait for approval, "
    "then use a READ token from the same account."
)


class DeployError(ValueError):
    """Raised for a rejected deployment; the message is safe to show the user."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _hf_access_denied(output: str) -> bool:
    text = output.lower()
    return any(
        marker in text
        for marker in (
            "access denied",
            "requires approval",
            "gated repo",
            "cannot access gated",
            "invalid user token",
        )
    )


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
        self.job: dict | None = None
        self._job_task: asyncio.Task | None = None

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
        elif kind == "uncensored":
            profile = uncensored_template(tag, models_root, cache_root)
        elif kind == "official-quick":
            profile = official_quick_template(tag, models_root, cache_root)
        elif kind == "uncensored-quick":
            profile = uncensored_quick_template(tag, models_root, cache_root)
        elif kind == "custom":
            profile = custom_template(tag, models_root, cache_root)
        else:
            raise DeployError(f"Unknown template: {kind}")
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
                fromfile="current",
                tofile="new",
                lineterm="",
            )
        )
        warnings = list(self._conflict_warnings(profile))
        if profile.downloads_weights:
            warnings.append(
                "First start downloads ~118 GiB of model weights from the internet "
                "(HALOGEN_DOWNLOAD); this can take hours."
            )
        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "quadlet": rendered,
            "diff": diff or "(no changes)",
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
            raise DeployError("Invalid profile ID")
        target = self.quadlet_dir / f"halogen-{profile_id}.container"
        if not QUADLET_FILENAME_RE.match(target.name) or not target.exists():
            raise DeployError(f"No Quadlet for profile {profile_id}")
        profile = parse_quadlet(target.read_text(encoding="utf-8"), target)
        if await self._service_is_active(profile.service_name):
            raise DeployError(
                f"Service {profile.service_name} is still running. Stop it before deleting."
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
            raise DeployError("Invalid profile ID")
        quadlet_name = f"halogen-{profile_id}.container"
        backups = self._backups_for(quadlet_name)
        if not backups:
            raise DeployError(f"No backup for profile {profile_id}")
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
            raise DeployError("Invalid profile ID")
        if self.manager.switching or self.manager.maintenance:
            raise DeployError(
                "A model switch or maintenance is in progress. Please wait."
            )
        if await self._any_other_service_active(profile_id):
            raise DeployError(
                "Another backend is already active. Halobridge serves only "
                "one backend at a time; switch the model first."
            )
        target = self.quadlet_dir / f"halogen-{profile_id}.container"
        if not target.exists():
            raise DeployError(f"No Quadlet for profile {profile_id}")
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
            raise DeployError(f"Discovery failed: {exc}")
        discovered = {spec.model_id: spec.service for spec in specs.values()}
        discovered.update(self.config.models.explicit)
        if not discovered:
            raise DeployError("No Halogen backends found. Nothing loaded.")
        self.manager.models = discovered
        if self.updater is not None:
            self.updater.models = discovered
        if self.dashboard is not None:
            self.dashboard.models = discovered
        return {"ok": True, "models": sorted(discovered)}

    # ------------------------------------------------- HF download & convert

    def _hf_executable(self) -> str | None:
        found = shutil.which("hf")
        if found:
            return found
        local = Path(sys.executable).parent / ("hf.exe" if os.name == "nt" else "hf")
        if local.is_file() and os.access(local, os.X_OK):
            return str(local)
        return None

    def hf_status(self) -> dict:
        return {
            "installed": self._hf_executable() is not None,
            "default_repo": UNCENSORED_HF_REPO,
            "default_file": "",
            "default_gguf_name": UNCENSORED_DEFAULT_GGUF_NAME,
            "default_output": UNCENSORED_DEFAULT_OUTPUT,
        }

    def job_status(self) -> dict:
        if self.job is None:
            return {"active": False}
        return {
            "active": True,
            "kind": self.job["kind"],
            "state": self.job["state"],
            "lines": self.job["lines"][-200:],
            "error": self.job.get("error"),
            "elapsed": round(time.time() - self.job["started"], 1),
            "step": self.job.get("step"),
            "total_steps": self.job.get("total_steps"),
        }

    def hf_install(self) -> dict:
        self._require_enabled()
        if self._hf_executable():
            raise DeployError("HF CLI is already installed.")
        return self._start_job(
            "hf-install",
            [sys.executable, "-m", "pip", "install", "--upgrade", HF_CLI_PACKAGE],
            timeout=900,
        )

    def hf_download(self, payload: dict) -> dict:
        """Download a (gated) HuggingFace repo into a host directory.

        The token is passed to the child process as HF_TOKEN only. It is
        never written to disk, never rendered into a quadlet and redacted
        from all captured output.
        """
        self._require_enabled()
        repo = str(payload.get("repo", "")).strip()
        if not REPO_ID_RE.match(repo):
            raise DeployError("Repository must be 'org/name'")
        filename = str(payload.get("file", "")).strip()
        if filename and ("/" in filename or "\\" in filename or ".." in filename):
            raise DeployError("Filename contains invalid characters")
        dest = self._validate_host_path(str(payload.get("dest", "")))
        token = str(payload.get("token", "")).strip()
        if not token:
            raise DeployError(
                "HF token is missing. A fine-grained token with read access is sufficient."
            )
        hf = self._hf_executable()
        if not hf:
            raise DeployError(
                "HF CLI is missing. Install it first or add `hf` to PATH."
            )
        cmd = [hf, "download", repo]
        if filename:
            cmd.append(filename)
        cmd += ["--local-dir", str(dest)]
        return self._start_job(
            "hf-download", cmd, {"HF_TOKEN": token}, timeout=21600
        )

    def convert(self, payload: dict) -> dict:
        """Convert a GGUF into a complete .hgn checkpoint via the image's
        convert mode (runs ~10 minutes, output beside the input).

        Lossless repack: no requantization. The lookup table and the MTP
        draft head are folded in so the engine loads the .hgn directly
        instead of repacking the GGUF into RAM on every start.
        """
        self._require_enabled()
        image = self._validated_image(str(payload.get("image", "")))
        gguf = self._validate_host_path(str(payload.get("gguf", "")))
        if not gguf.is_file():
            raise DeployError(f"GGUF not found: {gguf}")
        if not gguf.name.lower().endswith(".gguf"):
            raise DeployError("Input file must end in .gguf")
        out_name = str(payload.get("output", "")).strip() or (gguf.stem + ".hgn")
        if not re.match(r"^[A-Za-z0-9._-]+\.hgn$", out_name):
            raise DeployError("Output must be a simple filename ending in .hgn")
        cmd = [
            "podman", "run", "--rm",
            "--device", "/dev/kfd",
            "--device", "/dev/dri",
            "--group-add", "keep-groups",
            "--ipc=host",
            "--ulimit", "memlock=-1:-1",
            "-v", f"{gguf.parent}:/models:Z",
        ]
        head = str(payload.get("mtp_head", "")).strip()
        if head:
            head_path = self._validate_host_path(head)
            if not head_path.is_file():
                raise DeployError(f"MTP head not found: {head_path}")
            cmd += [
                "-e", f"HALOGEN_MTP_HEAD=/heads/{head_path.name}",
                "-v", f"{head_path.parent}:/heads:ro,Z",
            ]
        elif payload.get("download_head", True):
            cmd += ["-e", f"HALOGEN_DOWNLOAD={OFFICIAL_WEIGHTS_REPO}"]
        cmd += [image, "convert", f"/models/{gguf.name}", f"/models/{out_name}"]
        return self._start_job("convert", cmd, timeout=7200)

    def verify(self, payload: dict) -> dict:
        """Verify a .hgn checkpoint with the image's own verify mode."""
        self._require_enabled()
        image = self._validated_image(str(payload.get("image", "")))
        hgn = self._validate_host_path(str(payload.get("hgn", "")))
        if not hgn.is_file():
            raise DeployError(f"HGN file not found: {hgn}")
        if not hgn.name.lower().endswith(".hgn"):
            raise DeployError("File must end in .hgn")
        cmd = [
            "podman", "run", "--rm",
            "-v", f"{hgn.parent}:/models:ro,Z",
            image,
            "verify", f"/models/{hgn.name}",
        ]
        return self._start_job("verify", cmd, timeout=1800)

    def _validated_image(self, image: str) -> str:
        parts = image.rsplit(":", 1)
        if len(parts) != 2 or not parts[1] or ":" in parts[0]:
            raise DeployError("Image must be 'repository:tag'")
        from semver import normalize_version

        if normalize_version(parts[1]) is None:
            raise DeployError("Image tag must be a stable version X.Y.Z")
        return image

    def _start_job(
        self, kind: str, argv: list[str], env_extra: dict | None = None, timeout: float = 3600
    ) -> dict:
        if self.job is not None and self.job["state"] == "running":
            raise DeployError("A job is already running. Please wait.")
        self.job = {
            "kind": kind,
            "state": "running",
            "lines": [],
            "error": None,
            "started": time.time(),
        }
        self._job_task = asyncio.create_task(
            self._run_job(argv, env_extra, timeout)
        )
        return {"ok": True, "kind": kind}

    async def _run_job(
        self, argv: list[str], env_extra: dict | None, timeout: float
    ) -> None:
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        token = (env_extra or {}).get("HF_TOKEN")
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
        except FileNotFoundError:
            self.job["state"] = "error"
            self.job["error"] = f"Program not found: {argv[0]}"
            return

        try:
            recent_lines = await asyncio.wait_for(
                self._pump_process_output(process.stdout, token), timeout
            )
            returncode = await process.wait()
        except asyncio.TimeoutError:
            process.kill()
            self.job["state"] = "error"
            self.job["error"] = "Timed out"
            return
        if returncode == 0:
            self.job["state"] = "done"
        else:
            self.job["state"] = "error"
            output = "\n".join(recent_lines)
            self.job["error"] = (
                HF_GATED_ACCESS_MESSAGE
                if _hf_access_denied(output)
                else f"Exit {returncode}"
            )

    def _append_job_line(self, text: str) -> None:
        if self.job is None:
            return
        self.job["lines"].append(text)
        if len(self.job["lines"]) > 200:
            del self.job["lines"][:-200]

    async def _pump_process_output(
        self, stream: asyncio.StreamReader, token: str | None = None
    ) -> list[str]:
        """Forward newline and terminal-style carriage-return updates.

        Tools such as ``hf``/tqdm redraw progress bars with ``\r`` instead of
        emitting newline-terminated records. Reading with ``readline()`` hides
        those updates until the command finishes, so consume chunks and treat
        either delimiter as a dashboard log record.
        """
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        recent_lines: list[str] = []

        def emit(raw: str) -> None:
            text = ANSI_ESCAPE_RE.sub("", raw).replace("\b", "").rstrip()
            if token and token in text:
                text = text.replace(token, "[token]")
            if not text:
                return
            self._append_job_line(text)
            recent_lines.append(text)
            if len(recent_lines) > 20:
                del recent_lines[:-20]

        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            pending += decoder.decode(chunk)
            records = re.split(r"[\r\n]+", pending)
            pending = records.pop()
            for record in records:
                emit(record)

        pending += decoder.decode(b"", final=True)
        emit(pending)
        return recent_lines

    async def _run_stream(
        self,
        argv: list[str],
        env_extra: dict | None = None,
        timeout: float = 3600,
        token: str | None = None,
    ) -> None:
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
        except FileNotFoundError as exc:
            raise DeployError(f"Program not found: {argv[0]} ({exc})")

        try:
            recent_lines = await asyncio.wait_for(
                self._pump_process_output(process.stdout, token), timeout
            )
            returncode = await process.wait()
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise DeployError(f"Timed out running: {argv[0]}")
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        if returncode != 0:
            if _hf_access_denied("\n".join(recent_lines)):
                raise DeployError(HF_GATED_ACCESS_MESSAGE)
            raise DeployError(f"{argv[0]} failed (exit {returncode})")

    def _start_pipeline_job(self, kind: str, total_steps: int) -> dict:
        if self.job is not None and self.job["state"] == "running":
            raise DeployError("A job is already running. Please wait.")
        self.job = {
            "kind": kind,
            "state": "running",
            "lines": [],
            "error": None,
            "started": time.time(),
            "step": 0,
            "total_steps": total_steps,
        }
        return {"ok": True, "kind": kind}

    async def quick_official(self) -> dict:
        self._require_enabled()
        if await self._service_is_active("halogen-official.service"):
            health = await self.manager.backend_health()
            if health and health.get("model") == "qwen3.8-flash":
                return {
                    "ok": True,
                    "already": True,
                    "service": "halogen-official.service",
                }
            model_id = next(
                (
                    model
                    for model, service in self.manager.models.items()
                    if service == "halogen-official.service"
                ),
                "qwen3.8-flash",
            )
            self._start_pipeline_job("quick-official", 1)
            self._job_task = asyncio.create_task(
                self._track_active_official(model_id)
            )
            return {
                "ok": True,
                "kind": "quick-official",
                "already_running": True,
            }
        self._start_pipeline_job("quick-official", 2)
        self._job_task = asyncio.create_task(self._run_quick_official())
        return {"ok": True, "kind": "quick-official"}

    async def _track_active_official(self, model_id: str) -> None:
        self.job["step"] = 1
        self._append_job_line("--- Continue active official download/startup ---")
        stop = asyncio.Event()
        hb = asyncio.create_task(self._heartbeat(stop, "waiting for official"))
        service = self.manager.models.get(model_id, "halogen-official.service")
        journal = asyncio.create_task(self._stream_service_journal(service))
        try:
            await self.manager.wait_for_backend(
                model_id, timeout=QUICK_OFFICIAL_START_TIMEOUT
            )
            self.job["state"] = "done"
        except asyncio.CancelledError:
            self.job["state"] = "cancelled"
            self._append_job_line("--- Cancelled ---")
            raise
        except Exception as exc:
            self.job["state"] = "error"
            self.job["error"] = str(exc)
        finally:
            stop.set()
            hb.cancel()
            journal.cancel()
            await asyncio.gather(hb, journal, return_exceptions=True)

    async def _run_quick_official(self) -> None:
        try:
            self.job["step"] = 1
            self._append_job_line("--- Apply official profile ---")
            target = self.quadlet_dir / "halogen-official.container"
            if not target.exists():
                tag = self._suggested_tag()
                models_root, cache_root = self._default_roots()
                profile = official_quick_template(tag, models_root, cache_root)
                payload = profile_to_dict(profile)
                self.install_dirs(payload)
                preview = self.dry_run(payload)
                if not preview["ok"]:
                    raise DeployError("; ".join(preview["errors"]))
                await self.apply(payload)
            self.reload_discovery()
            model_id = parse_quadlet(target.read_text(encoding="utf-8"), target).model_id
            self.job["step"] = 2
            self._append_job_line("--- Stop active backend and start official ---")
            await self._switch_with_heartbeat(
                model_id,
                "waiting for official",
                start_timeout=QUICK_OFFICIAL_START_TIMEOUT,
            )
            self.job["state"] = "done"
        except asyncio.CancelledError:
            self.job["state"] = "cancelled"
            self._append_job_line("--- Cancelled ---")
            raise
        except Exception as exc:
            self.job["state"] = "error"
            self.job["error"] = str(exc)

    async def _switch_with_heartbeat(
        self,
        model_id: str,
        label: str,
        hook=None,
        start_timeout: float = QUICK_START_TIMEOUT,
    ) -> None:
        """Run a takeover while forwarding service logs and emitting a heartbeat."""
        stop = asyncio.Event()
        hb = asyncio.create_task(self._heartbeat(stop, label))
        service = self.manager.models.get(model_id)
        journal = (
            asyncio.create_task(self._stream_service_journal(service))
            if service
            else None
        )
        try:
            await self.manager.switch_with_hook(
                model_id, hook, start_timeout=start_timeout
            )
        finally:
            stop.set()
            hb.cancel()
            if journal is not None:
                journal.cancel()
            await asyncio.gather(
                *([hb, journal] if journal is not None else [hb]),
                return_exceptions=True,
            )

    async def _stream_service_journal(self, service: str) -> None:
        """Forward new systemd user-unit output into the deployment job log."""
        journalctl = shutil.which("journalctl")
        if not journalctl:
            return
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                journalctl,
                "--user",
                "-f",
                "-n",
                "0",
                "-o",
                "cat",
                "-u",
                service,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert process.stdout is not None
            while line := await process.stdout.readline():
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._append_job_line(text[:4000])
            await process.wait()
        except asyncio.CancelledError:
            raise
        except OSError as exc:
            self._append_job_line(f"Journal output unavailable: {exc}")
        finally:
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 3)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

    async def _heartbeat(self, stop: asyncio.Event, label: str) -> None:
        start = time.time()
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), 10)
            except asyncio.TimeoutError:
                self._append_job_line(f"... {label} ({int(time.time() - start)}s)")

    def cancel_job(self) -> dict:
        if self.job is None or self.job["state"] != "running":
            return {"ok": False, "reason": "no running job"}
        if self._job_task is not None and not self._job_task.done():
            self._job_task.cancel()
        return {"ok": True}

    async def quick_uncensored(self, payload: dict) -> dict:
        self._require_enabled()
        if await self._service_is_active("halogen-uncensored.service"):
            return {"ok": True, "already": True, "service": "halogen-uncensored.service"}
        token = str(payload.get("token", "")).strip()
        tag = self._suggested_tag()
        models_root, cache_root = self._default_roots()
        uncensored_dir = models_root / "uncensored"
        self._validate_host_path(str(uncensored_dir))
        uncensored_dir.mkdir(parents=True, exist_ok=True)
        gguf = uncensored_dir / UNCENSORED_DEFAULT_GGUF_NAME
        output = uncensored_dir / UNCENSORED_DEFAULT_OUTPUT
        tokenizer_dir = uncensored_dir / "tokenizer"
        need_gguf = not output.exists()
        # The tokenizer is needed at runtime even when the .hgn already exists,
        # so a partial earlier install (weights but no tokenizer) is repaired.
        need_tokenizer = not tokenizer_dir.is_dir() or not any(tokenizer_dir.iterdir())
        # A token is only needed when we still have to download the GGUF.
        if need_gguf and not token:
            raise DeployError(
                "HF token is missing. A fine-grained token with read access is sufficient."
            )
        hf = self._hf_executable()
        steps = 2  # apply profile + switch
        if (need_gguf or need_tokenizer) and not hf:
            steps += 1
        if need_gguf:
            steps += 2  # gguf + draft head
        if need_tokenizer:
            steps += 1
        self._start_pipeline_job("quick-uncensored", steps)
        self._job_task = asyncio.create_task(
            self._run_quick_uncensored(
                token, tag, uncensored_dir, gguf, output, need_gguf, need_tokenizer
            )
        )
        return {"ok": True, "kind": "quick-uncensored"}

    async def _run_quick_uncensored(
        self,
        token: str,
        tag: str,
        uncensored_dir: Path,
        gguf: Path,
        output: Path,
        need_gguf: bool,
        need_tokenizer: bool,
    ) -> None:
        try:
            step = 1
            if need_gguf or need_tokenizer:
                if not self._hf_executable():
                    self.job["step"] = step
                    self._append_job_line(f"--- {step}: Install HF CLI ---")
                    await self._run_stream(
                        [sys.executable, "-m", "pip", "install", "--upgrade", HF_CLI_PACKAGE],
                        timeout=900,
                    )
                    step += 1
                hf = self._hf_executable()
                if not hf:
                    raise DeployError("HF CLI is missing after installation.")

                if need_gguf:
                    self.job["step"] = step
                    self._append_job_line(f"--- {step}: Download uncensored GGUF ---")
                    await self._run_stream(
                        [hf, "download", UNCENSORED_HF_REPO, "--local-dir", str(uncensored_dir)],
                        {"HF_TOKEN": token},
                        timeout=21600,
                        token=token,
                    )
                    step += 1

                    self.job["step"] = step
                    self._append_job_line("--- Download draft head ---")
                    await self._run_stream(
                        [
                            hf,
                            "download",
                            OFFICIAL_WEIGHTS_REPO,
                            "qwen38-flash-next-mtp.hgn",
                            "--local-dir",
                            str(uncensored_dir),
                        ],
                        timeout=3600,
                    )
                    step += 1

                if need_tokenizer:
                    self.job["step"] = step
                    self._append_job_line("--- Download tokenizer ---")
                    await self._run_stream(
                        [
                            hf,
                            "download",
                            OFFICIAL_WEIGHTS_REPO,
                            "--include",
                            "tokenizer/*",
                            "--local-dir",
                            str(uncensored_dir),
                        ],
                        timeout=3600,
                    )
                    step += 1

            # Apply the profile and refresh discovery while the current backend
            # keeps serving; this does not touch the GPU.
            self.job["step"] = step
            self._append_job_line("--- Apply uncensored profile ---")
            models_root, cache_root = self._default_roots()
            profile = uncensored_quick_template(tag, models_root, cache_root)
            profile_payload = profile_to_dict(profile)
            self.install_dirs(profile_payload)
            preview = self.dry_run(profile_payload)
            if not preview["ok"]:
                raise DeployError("; ".join(preview["errors"]))
            await self.apply(profile_payload)
            self.reload_discovery()
            step += 1

            # Switch onto uncensored: stop the active backend, free the GPU,
            # convert (if needed), then start uncensored. The convert runs as
            # the switch hook so it has the GPU to itself. On failure or cancel
            # the switch rolls back to the previous backend.
            self.job["step"] = step
            self._append_job_line("--- Stop active backend and start uncensored ---")
            image = f"ghcr.io/peonist-ai/halogen-flash-server:{tag}"

            async def convert_hook() -> None:
                if output.exists():
                    self._append_job_line("--- Conversion skipped; .hgn already exists ---")
                    return
                self._append_job_line("--- Convert GGUF to HGN ---")
                cmd = [
                    "podman", "run", "--rm",
                    "--device", "/dev/kfd",
                    "--device", "/dev/dri",
                    "--group-add", "keep-groups",
                    "--ipc=host",
                    "--ulimit", "memlock=-1:-1",
                    "-v", f"{uncensored_dir}:/models:Z",
                    image,
                    "convert",
                    f"/models/{gguf.name}",
                    f"/models/{output.name}",
                ]
                await self._run_stream(cmd, timeout=7200)

            await self._switch_with_heartbeat(
                profile.model_id, "waiting for uncensored", convert_hook
            )
            self.job["state"] = "done"
        except asyncio.CancelledError:
            self.job["state"] = "cancelled"
            self._append_job_line("--- Cancelled ---")
            raise
        except Exception as exc:
            self.job["state"] = "error"
            self.job["error"] = str(exc)

    # ------------------------------------------------------------ helpers

    def _require_enabled(self) -> None:
        deploy = getattr(self.config, "deploy", None)
        if deploy is None or not deploy.enabled:
            raise DeployError("Deployment is disabled in the configuration.")
        if sys.platform == "win32":
            raise DeployError("Deployment is only available on Linux/POSIX.")

    def _profile_from_payload(self, payload: dict) -> Profile:
        try:
            volumes = [tuple(v) for v in payload.get("volumes", [])]
            if not all(len(v) == 3 for v in volumes):
                raise DeployError("volumes: each entry requires host, container, and mode")
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
            raise DeployError(f"Invalid profile: {exc}")
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
            raise DeployError(f"Path must be absolute: {host_path}")
        resolved = path.resolve()
        roots = self._allowed_roots()
        if not any(_is_within(resolved, root) for root in roots):
            raise DeployError(
                f"Path is outside the allowed roots: {host_path} "
                f"(allowed: {', '.join(str(r) for r in roots)})"
            )
        if resolved == Path("/") or len(resolved.parts) < 2:
            raise DeployError(f"Path is too broad: {host_path}")
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
        """Hard conflicts: the same model id may not exist twice."""
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
                    f"Model ID '{profile.model_id}' is already used by profile "
                    f"'{other.profile_id}'"
                )
        return errors

    def _conflict_warnings(self, profile: Profile) -> list[str]:
        """Shared host ports are the intended design (one backend runs at a
        time), so a collision is only a warning, never a blocker."""
        warnings = []
        if not self.quadlet_dir.is_dir():
            return warnings
        for path in self.quadlet_dir.glob("halogen-*.container"):
            if path.name == f"{profile.container_name}.container":
                continue
            try:
                other = parse_quadlet(path.read_text(encoding="utf-8"), path)
            except ProfileError:
                continue
            if other.host_port == profile.host_port:
                warnings.append(
                    f"Port {profile.host_port} is shared by profile "
                    f"'{other.profile_id}' and this profile "
                    "(allowed because only one backend runs at a time)."
                )
        return warnings

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
            raise DeployError(f"Invalid service name: {service_name}")
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
            raise DeployError(f"Timeout running: {' '.join(args[:2])}")
        if process.returncode:
            raise DeployError(
                f"{' '.join(args[:2])} failed (exit {process.returncode}): "
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
                f"Generated unit for {profile.service_name} does not contain the "
                "expected image"
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
        # Windows can return the same value for both datetime.now() and
        # time.time_ns() across several rapid calls. Create the file exclusively
        # and add an ordered suffix so a rollback can never overwrite the backup
        # it is about to restore.
        stamp = (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-"
            f"{time.time_ns():020d}"
        )
        contents = target.read_bytes()
        sequence = 0
        while True:
            backup = self.backup_root / (
                f"profile-{target.stem}-{stamp}-{sequence:04d}.container"
            )
            try:
                with backup.open("xb") as handle:
                    handle.write(contents)
                return backup
            except FileExistsError:
                sequence += 1

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
            raise DeployError("Cross-origin request rejected")
        if request.headers.get("X-Halogen-Action") != "deploy":
            raise DeployError("Header X-Halogen-Action: deploy is required")

    async def _payload(self, request: web.Request) -> dict:
        if request.content_type != "application/json":
            raise DeployError("JSON body expected")
        try:
            text = await request.text()
        except Exception as exc:
            raise DeployError(f"Could not read body: {exc}")
        try:
            data = json.loads(text or "{}")
        except json.JSONDecodeError as exc:
            raise DeployError(f"Invalid JSON: {exc}")
        if not isinstance(data, dict):
            raise DeployError("JSON object expected")
        return data

    def _error(self, exc: Exception, status: int = 400) -> web.Response:
        if isinstance(exc, DeployError):
            logger.warning("deploy request rejected: %s", exc)
            message = str(exc) or "Deployment error"
        else:
            logger.exception("deploy request failed")
            message = f"{type(exc).__name__}: {exc}"
        return web.json_response({"error": message}, status=status)

    async def api_status(self, _: web.Request) -> web.Response:
        try:
            return web.json_response(await self.manager.status())
        except Exception as exc:
            return self._error(exc, 500)

    async def api_template(self, request: web.Request) -> web.Response:
        try:
            return web.json_response(self.manager.template(request.match_info["kind"]))
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_dry_run(self, request: web.Request) -> web.Response:
        try:
            self._guard(request)
            return web.json_response(self.manager.dry_run(await self._payload(request)))
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_apply(self, request: web.Request) -> web.Response:
        try:
            self._guard(request)
            return web.json_response(await self.manager.apply(await self._payload(request)))
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_install_dirs(self, request: web.Request) -> web.Response:
        try:
            self._guard(request)
            return web.json_response(self.manager.install_dirs(await self._payload(request)))
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_delete(self, request: web.Request) -> web.Response:
        try:
            self._guard(request)
            return web.json_response(
                await self.manager.delete(request.match_info["profile_id"])
            )
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_rollback(self, request: web.Request) -> web.Response:
        try:
            self._guard(request)
            return web.json_response(
                await self.manager.rollback(request.match_info["profile_id"])
            )
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_start(self, request: web.Request) -> web.Response:
        try:
            self._guard(request)
            return web.json_response(
                await self.manager.start(request.match_info["profile_id"])
            )
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_reload(self, _: web.Request) -> web.Response:
        try:
            return web.json_response(self.manager.reload_discovery())
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_job(self, _: web.Request) -> web.Response:
        try:
            return web.json_response(self.manager.job_status())
        except Exception as exc:
            return self._error(exc, 500)

    async def api_hf(self, _: web.Request) -> web.Response:
        try:
            return web.json_response(self.manager.hf_status())
        except Exception as exc:
            return self._error(exc, 500)

    async def api_hf_install(self, request: web.Request) -> web.Response:
        try:
            self._guard(request)
            return web.json_response(self.manager.hf_install())
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_hf_download(self, request: web.Request) -> web.Response:
        try:
            self._guard(request)
            return web.json_response(self.manager.hf_download(await self._payload(request)))
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_convert(self, request: web.Request) -> web.Response:
        try:
            self._guard(request)
            return web.json_response(self.manager.convert(await self._payload(request)))
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_verify(self, request: web.Request) -> web.Response:
        try:
            self._guard(request)
            return web.json_response(self.manager.verify(await self._payload(request)))
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_quick_official(self, request: web.Request) -> web.Response:
        try:
            self._guard(request)
            return web.json_response(await self.manager.quick_official())
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_quick_uncensored(self, request: web.Request) -> web.Response:
        try:
            self._guard(request)
            return web.json_response(await self.manager.quick_uncensored(await self._payload(request)))
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

    async def api_cancel_job(self, request: web.Request) -> web.Response:
        try:
            self._guard(request)
            return web.json_response(self.manager.cancel_job())
        except DeployError as exc:
            return self._error(exc, 400)
        except Exception as exc:
            return self._error(exc, 500)

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
        app.router.add_get("/dashboard/api/deploy/job", self.api_job)
        app.router.add_post("/dashboard/api/deploy/job/cancel", self.api_cancel_job)
        app.router.add_get("/dashboard/api/deploy/hf", self.api_hf)
        app.router.add_post("/dashboard/api/deploy/hf/install", self.api_hf_install)
        app.router.add_post("/dashboard/api/deploy/hf/download", self.api_hf_download)
        app.router.add_post("/dashboard/api/deploy/convert", self.api_convert)
        app.router.add_post("/dashboard/api/deploy/verify", self.api_verify)
        app.router.add_post("/dashboard/api/deploy/quick/official", self.api_quick_official)
        app.router.add_post(
            "/dashboard/api/deploy/quick/uncensored", self.api_quick_uncensored
        )
