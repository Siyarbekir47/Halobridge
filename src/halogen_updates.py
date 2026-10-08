"""Engine and official checkpoint updates, coordinated with routing.

Only stable upstream tags and the fixed GHCR repository are accepted. A durable
journal is written before changing Quadlets, allowing recovery on restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

from profiles import (
    CONTAINER_MODELS_PATH,
    OFFICIAL_WEIGHTS_REPO,
    ProfileError,
    parse_quadlet,
)

# Re-exported so existing callers/tests keep working; the definitions live in
# semver.py, shared with discovery.
from semver import normalize_version, version_tuple
from official_checkpoints import OFFICIAL_CHECKPOINTS, checkpoint_choice
from shared_assets import SharedAssets, AssetError


LOG = logging.getLogger("halogen-updates")
REPOSITORY = "peonist-ai/halogen-flash-server"
IMAGE_REPOSITORY = f"ghcr.io/{REPOSITORY}"
TAGS_URL = f"https://api.github.com/repos/{REPOSITORY}/tags"
CHECK_INTERVAL = 6 * 3600
CHECK_RETRY = 300
DRAIN_TIMEOUT = 7200
START_TIMEOUT = 600
TERMINAL_PHASES = {"succeeded", "failed", "rolled_back"}
JOB_PHASES = TERMINAL_PHASES | {"starting", "preparing", "preparing_npu", "pulling", "draining", "configuring", "restarting", "verifying", "rolling_back", "recovery_required"}

# Compatibility with the original w4b -> v2 API. Explicit dashboard choices
# prepare pinned weights before maintenance. 0.16.0 adds opt-in HT43 and keeps
# v2 as its default; engine updates never change checkpoint selection.
CHECKPOINT_V2_PATH = "/models/qwen38-flash-next-v2.hgn"
CHECKPOINT_V2_FILES = ("qwen38-flash-next-v2.hgn", "qwen38-flash-next-ngram.hgn")
LEGACY_W4B_PATH = "/models/qwen38-flash-next-w4b.hgn"
LEGACY_W4B_FILES = ("qwen38-flash-next-w4b.hgn",)
CHECKPOINT_MIN_IMAGE = (0, 15, 0)
CHECKPOINT_DOWNLOAD_GIB = 110
CHECKPOINT_MIN_FREE_GIB = 115
CHECKPOINT_START_TIMEOUT = 6 * 3600


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
        raise UpdateError("Each Quadlet must contain exactly one Image line in [Container].")
    prefix = (image_repo or IMAGE_REPOSITORY) + ":"
    if not found[0].startswith(prefix) or version_tuple(found[0][len(prefix):]) is None:
        raise UpdateError("The configured image is not a versioned Halogen image.")
    return found[0], "".join(lines).encode("utf-8")


def quadlet_checkpoint_upgrade(
    content: bytes,
    checkpoint: str,
    *,
    download_repo: str | None = None,
    writable_models: bool = False,
    models_container_path: str = CONTAINER_MODELS_PATH,
) -> bytes:
    """Set an official checkpoint, preserving unrelated settings.

    Sets HALOGEN_CHECKPOINT (inserting it after the Image line when the profile
    never declared one, as a quick-install profile does not), drops
    HALOGEN_FLASH_PIN_TRUNK (v2 refuses it at startup), and - when the weights
    still have to be fetched - adds HALOGEN_DOWNLOAD and makes the models
    volume writable so the backend can download into it.
    """
    text = content.decode("utf-8")
    has_checkpoint = re.search(
        r"^[ \t]*Environment[ \t]*=[ \t]*HALOGEN_CHECKPOINT[ \t]*=", text, re.MULTILINE
    ) is not None
    has_download = re.search(
        r"^[ \t]*Environment[ \t]*=[ \t]*HALOGEN_DOWNLOAD[ \t]*=", text, re.MULTILINE
    ) is not None
    lines = text.splitlines(keepends=True)
    section = ""
    placed = has_checkpoint
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        if section == "[Container]":
            checkpoint_match = re.fullmatch(
                r"([ \t]*Environment[ \t]*=[ \t]*)(HALOGEN_CHECKPOINT[ \t]*=[ \t]*)([^\r\n]*?)([ \t]*)(\r?\n?)",
                line,
            )
            if checkpoint_match:
                placed = True
                out.append(
                    checkpoint_match[1] + "HALOGEN_CHECKPOINT=" + checkpoint
                    + checkpoint_match[4] + checkpoint_match[5]
                )
                if download_repo and not has_download:
                    has_download = True
                    out.append(
                        checkpoint_match[1] + "HALOGEN_DOWNLOAD=" + download_repo
                        + checkpoint_match[5]
                    )
                continue
            if re.fullmatch(
                r"[ \t]*Environment[ \t]*=[ \t]*HALOGEN_FLASH_PIN_TRUNK[ \t]*=[^\r\n]*(\r?\n)?",
                line,
            ):
                continue
            if writable_models:
                volume_match = re.fullmatch(
                    r"([ \t]*Volume[ \t]*=[ \t]*)(\S+)([ \t]*)(\r?\n?)", line
                )
                if volume_match:
                    parts = volume_match[2].split(":")
                    if len(parts) >= 2 and parts[1] == models_container_path:
                        options = [
                            option
                            for option in (parts[2].split(",") if len(parts) > 2 else ["rw"])
                            if option and option != "ro"
                        ]
                        parts[2] = ",".join(options) if options else "rw"
                        out.append(
                            volume_match[1] + ":".join(parts)
                            + volume_match[3] + volume_match[4]
                        )
                        continue
            image_match = re.fullmatch(
                r"([ \t]*Image[ \t]*=[ \t]*)(\S+)([ \t]*)(\r?\n?)", line
            )
            if image_match and not placed:
                indent = re.match(r"[ \t]*", line).group()
                eol = image_match[4]
                out.append(line)
                out.append(indent + "Environment=HALOGEN_CHECKPOINT=" + checkpoint + eol)
                placed = True
                if download_repo and not has_download:
                    out.append(indent + "Environment=HALOGEN_DOWNLOAD=" + download_repo + eol)
                    has_download = True
                continue
        out.append(line)
    if not placed:
        raise UpdateError("Could not place a HALOGEN_CHECKPOINT line in [Container].")
    return "".join(out).encode("utf-8")


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
        self.checkpoint_assets = SharedAssets(
            Path(config.deploy.models_root) if config else Path.home() / "halogen/models",
            self.quadlet_dir, self.state_dir,
        )
        from npu import NpuAssets
        self.npu_assets = NpuAssets(self.checkpoint_assets, self.state_dir)

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def _paths(self) -> dict[str, Path]:
        return {model: self.quadlet_dir / service.replace(".service", ".container") for model, service in self.models.items()}

    def _configuration(self) -> dict[str, dict[str, Any]]:
        result = {}
        for model, path in self._paths().items():
            if path.is_symlink() or not path.is_file():
                raise UpdateError(f"Local Quadlet is missing or is a symlink: {path.name}")
            content = path.read_bytes()
            image, _ = quadlet_image(content, image_repo=self.image_repo)
            result[model] = {"image": image, "version": normalize_version(image.rsplit(":", 1)[1]), "content": content}
        return result

    def _support_error(self) -> str | None:
        if not self.enabled:
            return "Updates are disabled."
        if os.name != "posix" or not shutil.which("podman") or not shutil.which("systemctl"):
            return "Updates require a Linux host with Podman and systemd under the router's user account."
        try:
            self._configuration()
        except (OSError, UnicodeError, UpdateError) as error:
            return str(error)
        return None

    @staticmethod
    def _v2_files_present(models_path: Path) -> bool:
        return all((models_path / name).is_file() for name in CHECKPOINT_V2_FILES)

    @staticmethod
    def _w4b_files_present(models_path: Path) -> bool:
        return all((models_path / name).is_file() for name in LEGACY_W4B_FILES)

    @staticmethod
    def _free_gib(models_path: Path) -> float | None:
        try:
            return shutil.disk_usage(models_path).free / 2**30
        except OSError:
            return None

    def _legacy_checkpoint_info(self) -> dict[str, Any]:
        """Whether the active official profile still runs the 0.14 w4b
        checkpoint while the image is 0.15+, and what switching to v2 needs."""
        info: dict[str, Any] = {
            "available": False, "current": None, "target": CHECKPOINT_V2_PATH,
            "needs_download": False, "download_gib": 0, "free_gib": None,
            "blocked_reason": None, "can_install": False,
        }
        current_image = version_tuple(self.current_version)
        if current_image is None or current_image < CHECKPOINT_MIN_IMAGE:
            return info
        model = getattr(self.manager, "current_model", None)
        if model not in self.models:
            return info
        try:
            profile = parse_quadlet(self._configuration()[model]["content"].decode("utf-8"))
        except (OSError, UnicodeError, UpdateError, ProfileError):
            return info
        if model != "qwen3.8-flash" or profile.model_id != model:
            return info
        current = profile.env.get("HALOGEN_CHECKPOINT")
        models_host = next(
            (volume[0] for volume in profile.volumes if volume[1] == CONTAINER_MODELS_PATH),
            None,
        )
        if not models_host:
            return info
        models_path = Path(models_host)
        v2_present = self._v2_files_present(models_path)
        if profile.env.get("HALOGEN_NGRAM_TABLE"):
            from shared_assets import host_path
            table = host_path(profile, profile.env["HALOGEN_NGRAM_TABLE"])
            v2_present = ((models_path / "qwen38-flash-next-v2.hgn").is_file()
                          and table is not None and table.is_file())
        if current == LEGACY_W4B_PATH:
            on_legacy = True
        elif current is None:
            # Quick-install profile with no explicit checkpoint: on 0.15 the
            # backend serves v2 when present, otherwise the w4b it already has.
            # Only offer the upgrade when it is in fact still serving w4b.
            on_legacy = not v2_present and self._w4b_files_present(models_path)
            if on_legacy:
                current = LEGACY_W4B_PATH
        else:
            on_legacy = False
        info["current"] = current
        if not on_legacy:
            return info
        info["needs_download"] = not v2_present
        free_gib = self._free_gib(models_path)
        info["free_gib"] = round(free_gib, 1) if free_gib is not None else None
        if info["needs_download"]:
            shared_table = profile.env.get("HALOGEN_NGRAM_TABLE") and table is not None and table.is_file()
            download_gib = 63 if shared_table else CHECKPOINT_DOWNLOAD_GIB
            minimum_free = download_gib + 5 if shared_table else CHECKPOINT_MIN_FREE_GIB
            info["download_gib"] = download_gib
            if free_gib is None or free_gib < minimum_free:
                info["available"] = True
                info["blocked_reason"] = (
                    f"Not enough free disk space for the ~{download_gib} GiB checkpoint download."
                )
                return info
        info["available"] = True
        return info

    def _checkpoint_info(self, target: str | None = None) -> dict[str, Any]:
        """v2 stays the automatic migration target; HT43 requires an explicit choice."""
        if target is not None:
            choice = checkpoint_choice(target)
        else:
            choice = OFFICIAL_CHECKPOINTS["v2"]
        info = self._legacy_checkpoint_info() if target is None else {
            "available": False, "current": None, "target": "/models/" + choice["asset"].name,
            "needs_download": False, "download_gib": 0, "free_gib": None,
            "blocked_reason": None, "can_install": False,
        }
        info.update(target_key=target or "v2", label=choice["label"], optional=target == "ht43", selectable=False)
        model = getattr(self.manager, "current_model", None)
        if model != "qwen3.8-flash" or model not in self.models:
            return info
        try:
            profile = parse_quadlet(self._configuration()[model]["content"].decode("utf-8"))
        except (OSError, UnicodeError, UpdateError, ProfileError):
            return info
        models_host = next((v[0] for v in profile.volumes if v[1] == CONTAINER_MODELS_PATH), None)
        if profile.model_id != model or not models_host:
            return info
        root = Path(models_host).expanduser()
        current = profile.env.get("HALOGEN_CHECKPOINT")
        if not current:
            # The entrypoint selects by the checkpoint file, not its companions,
            # and never automatically selects HT43, even when it is on disk.
            current = (CHECKPOINT_V2_PATH if (root / Path(CHECKPOINT_V2_PATH).name).is_file()
                       else LEGACY_W4B_PATH if self._w4b_files_present(root) else CHECKPOINT_V2_PATH)
        known = {LEGACY_W4B_PATH, *("/models/" + c["asset"].name for c in OFFICIAL_CHECKPOINTS.values())}
        info["current"] = current
        if current not in known:
            return info  # A custom official checkpoint is never rewritten.
        running = version_tuple(self.current_version)
        configured = version_tuple(profile.image.rsplit(":", 1)[-1])
        if not running or running < CHECKPOINT_MIN_IMAGE:
            return info
        info["selectable"] = True
        if target is None:
            return info
        info["available"] = current != info["target"]
        if not configured or min(running, configured) < choice["minimum_engine"]:
            info["blocked_reason"] = "HT43 requires Halogen 0.16.0 or newer. Update the engine first."
            return info
        try:
            plan = self.checkpoint_assets.plan(checkpoint=choice["asset"], checkpoint_root=root)
        except (OSError, ValueError) as error:
            info["blocked_reason"] = str(error)
            return info
        info.update(plan=plan, needs_download=plan["download_bytes"] > 0,
                    download_gib=math.ceil(plan["download_bytes"] / 2**30),
                    unverified_gib=round(plan["unverified_bytes"] / 2**30, 1))
        info["repair"] = current == info["target"] and any(f["status"] != "verified" for f in plan["files"])
        info["available"] = info["available"] or info["repair"]
        if any(not fs["sufficient"] for fs in plan["filesystems"]):
            info["blocked_reason"] = "Not enough free disk space for the checkpoint plan and 5 GiB reserve."
        return info

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
            raise UpdateError(f"{' '.join(args[:2])} failed (exit {process.returncode}).")
        return stdout.decode("utf-8", errors="replace")

    async def _inspect_image(self, image: str) -> str:
        data = json.loads(await self._run("podman", "image", "inspect", image))
        image_id = data[0].get("Id")
        if not isinstance(image_id, str) or not image_id:
            raise UpdateError("Could not verify the image ID.")
        return image_id.removeprefix("sha256:")

    async def _container_image(self, model: str) -> str:
        name = self.models[model].removesuffix(".service")
        data = json.loads(await self._run("podman", "container", "inspect", name))
        image_id = data[0].get("Image")
        if not isinstance(image_id, str) or not image_id:
            raise UpdateError("Could not verify the running container image.")
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
                    raise UpdateError("GitHub rate limit reached. Try again later.")
                if response.status != 200:
                    raise UpdateError(f"GitHub releases unavailable (HTTP {response.status}).")
                tags = await response.json()
            if not isinstance(tags, list):
                raise UpdateError("Invalid release response from GitHub.")
            versions.extend(parsed for tag in tags if isinstance(tag, dict)
                            if (parsed := version_tuple(tag.get("name"))) is not None)
            if len(tags) < 100:
                break
        else:
            raise UpdateError("Tag list is too large for a complete version check.")
        if not versions:
            raise UpdateError("No stable Halogen version found.")
        return ".".join(map(str, max(versions)))

    async def _fetch_latest_registry(self) -> str:
        try:
            output = await self._run(
                "podman", "search", "--list-tags", "--limit", "1000",
                "--format", "{{.Tag}}", self.image_repo, timeout=120,
            )
        except UpdateError as error:
            raise UpdateError(f"Registry versions unavailable: {error}") from error
        versions = []
        for line in output.splitlines():
            parsed = version_tuple(line.strip())
            if parsed is not None:
                versions.append(parsed)
        if not versions:
            raise UpdateError("No stable Halogen version found in the registry.")
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
                self.check_error = str(error) if isinstance(error, UpdateError) else "Version check failed. Check your network connection."
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
            blocked_reason = "Version mismatch: a configuration is newer than the GitHub tag. Automatic downgrade is disabled."
        fresh = self.checked_at is not None and time.time() - self.checked_at < self.check_interval
        checkpoint = self._checkpoint_info()
        choices = [self._checkpoint_info(key) for key in OFFICIAL_CHECKPOINTS] if checkpoint["selectable"] else []
        for item in [checkpoint, *choices]:
            item["can_install"] = bool(
                item["available"] and item["blocked_reason"] is None
                and self.enabled and self.allow_install and support_error is None
                and self.check_error is None and not self.running and not self.recovery_required
                and not getattr(self.manager, "maintenance", False)
                and not getattr(self.manager, "switching", False)
            )
        checkpoint["choices"] = choices
        return {
            "latest_version": self.latest_version, "current_version": self.current_version,
            "configured_versions": configured, "checked_at": self.checked_at,
            "check_error": self.check_error, "supported": support_error is None,
            "support_error": support_error, "update_available": available,
            "up_to_date": up_to_date, "blocked_reason": blocked_reason,
            "checkpoint": checkpoint,
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
                raise web.HTTPForbidden(text="Updates are disabled.")
            if not self.allow_install:
                raise web.HTTPForbidden(text="Installation is disabled.")
            if self.running or self.recovery_required:
                raise web.HTTPConflict(text="An update or recovery is already running.")
            await self.check()
            status = self.status()
            if version != self.latest_version or not status["can_install"]:
                raise web.HTTPConflict(text=status["support_error"] or status["check_error"] or "Check versions again; this version cannot be installed.")
            self.job = {"version": version, "phase": "starting", "message": "Preparing update.",
                        "started_at": time.time(), "changed": False, "backup_dir": None}
            self._save_job()
            self.task = asyncio.create_task(self._update(version))

    async def start_checkpoint(self, target: str | None = None) -> None:
        """Migrate w4b to v2, or explicitly choose a supported official checkpoint."""
        if target is not None:
            try:
                checkpoint_choice(target)
            except ValueError as error:
                raise web.HTTPBadRequest(text=str(error)) from error
        async with self.start_lock:
            if not self.enabled:
                raise web.HTTPForbidden(text="Updates are disabled.")
            if not self.allow_install:
                raise web.HTTPForbidden(text="Installation is disabled.")
            if self.running or self.recovery_required:
                raise web.HTTPConflict(text="An update or recovery is already running.")
            support_error = self._support_error()
            if support_error:
                raise web.HTTPConflict(text=support_error)
            info = self._checkpoint_info(target)
            if not info["available"]:
                raise web.HTTPConflict(text="No checkpoint upgrade is available.")
            if info["blocked_reason"]:
                raise web.HTTPConflict(text=info["blocked_reason"])
            model = self.manager.current_model
            configurations = self._configuration()
            old_version = self.current_version
            old_image_id = await self._container_image(model)
            if await self._inspect_image(configurations[model]["image"]) != old_image_id:
                raise web.HTTPConflict(text="Running image does not match the active Quadlet.")
            if self.manager.current_model != model:
                raise web.HTTPConflict(text="Active model changed; check the checkpoint upgrade again.")
            self.job = {
                "kind": "checkpoint", "version": old_version, "phase": "starting",
                "message": "Preparing the checkpoint upgrade.", "started_at": time.time(),
                "changed": False, "backup_dir": None, "model": model,
                "old_version": old_version, "old_image_id": old_image_id,
                "needs_download": info["needs_download"],
                "target_key": target or "v2", "target": info["target"],
                "label": info["label"], "download_gib": info["download_gib"],
                "prepared_download": target is not None,
            }
            self._save_job()
            self.task = asyncio.create_task(self._update_checkpoint(info))

    async def _update_checkpoint(self, info: dict[str, Any]) -> None:
        maintenance = False
        try:
            assert self.job is not None
            model = self.job["model"]
            if model not in self.models or self.manager.current_model != model:
                raise UpdateError("Active model changed; check the checkpoint upgrade again.")
            configurations = self._configuration()
            health = await self.manager.backend_health()
            if not health or health.get("status") != "ok" or health.get("model") != model:
                raise UpdateError("Could not verify the active backend before the checkpoint upgrade.")
            locations = None
            selection = self.job["target_key"] if self.job.get("prepared_download") else None
            if selection:
                from shared_assets import host_path
                choice = checkpoint_choice(selection)
                profile = parse_quadlet(configurations[model]["content"].decode("utf-8"))
                root = host_path(profile, CONTAINER_MODELS_PATH)
                self._save_job(phase="preparing", message="Preparing verified checkpoint and shared assets. Requests continue to run.")
                locations = await self.checkpoint_assets.prepare(
                    None, self._download_checkpoint_asset,
                    lambda message: self._save_job(message=message),
                    checkpoint=choice["asset"], checkpoint_root=root,
                )
                if self.manager.current_model != model:
                    raise UpdateError("Active model changed during checkpoint preparation.")
            self._save_job(phase="draining", message="Waiting for model switches and active requests.")
            await asyncio.wait_for(self.manager.begin_maintenance(), self.drain_timeout)
            maintenance = True
            if self.manager.current_model != model:
                raise UpdateError("Active model changed; check the checkpoint upgrade again.")
            await asyncio.wait_for(self.manager.drain_maintenance(), self.drain_timeout)
            info = self._checkpoint_info(selection)
            if info["blocked_reason"] or (not info["available"] and
                    (not selection or info["current"] != info["target"])):
                raise UpdateError(info["blocked_reason"] or "No checkpoint upgrade is available.")
            paths = {model: self._paths()[model]}
            for name, path in paths.items():
                if path.is_symlink() or path.read_bytes() != configurations[name]["content"]:
                    raise UpdateError("Quadlet was changed externally during checkpoint preparation.")
            backup = self.backup_root / f"checkpoint-update-{time.time_ns()}"
            backup.mkdir(parents=True, mode=0o700)
            modes = {}
            for name, path in paths.items():
                modes[name] = stat.S_IMODE(path.stat().st_mode)
                atomic_write(backup / path.name, configurations[name]["content"], modes[name])
            self._save_job(phase="configuring", message=f"Official Quadlet backed up; switching to {self.job['label']}.",
                          backup_dir=str(backup), model=model, modes=modes,
                          changed_models=[model], changed=True)
            # changed is durable BEFORE the first write (including partial failure).
            for name, path in paths.items():
                if path.is_symlink() or path.read_bytes() != configurations[name]["content"]:
                    raise UpdateError("Quadlet was changed externally during the upgrade.")
                content = quadlet_checkpoint_upgrade(
                    configurations[name]["content"],
                    info["target"],
                    download_repo=OFFICIAL_WEIGHTS_REPO if info["needs_download"] and not locations else None,
                    writable_models=info["needs_download"] and not locations,
                )
                if locations:
                    from swift_install import shared_quadlet_content
                    content = shared_quadlet_content(content.decode("utf-8"), locations).encode("utf-8")
                    content = self._without_checkpoint_download(content)
                atomic_write(path, content, modes[name])
            await self._run("systemctl", "--user", "daemon-reload")
            self._save_job(
                phase="restarting",
                message=(
                    f"Starting {model} with {self.job['label']}. Downloading ~{info['download_gib']} GiB; "
                    "this can take a while."
                    if info["needs_download"]
                    else f"Starting {model} with {self.job['label']}."
                ),
            )
            await self._restart(model, timeout=CHECKPOINT_START_TIMEOUT)
            self._save_job(phase="verifying", message=f"Verifying {self.job['label']} is active.")
            await self._ready(model, self.job["old_version"], self.job["old_image_id"], timeout=CHECKPOINT_START_TIMEOUT)
            if selection:
                await self._verify_checkpoint(model, info["target"])
            self._save_job(
                phase="succeeded",
                message=f"The {self.job['label']} checkpoint is active. Previous checkpoint files are retained for switching back.",
                changed=False,
            )
        except (Exception, asyncio.CancelledError) as error:
            LOG.warning("Checkpoint upgrade failed: %s", type(error).__name__)
            reason = str(error) if isinstance(error, (UpdateError, AssetError)) else "Checkpoint upgrade cancelled or timed out."
            if self.job and self.job.get("changed"):
                try:
                    await self._rollback(reason)
                except (Exception, asyncio.CancelledError):
                    self.recovery_required = True
                    self._save_job(phase="recovery_required", message="Rollback incomplete. Retry recovery; the router remains in maintenance mode.")
            elif self.job:
                self._save_job(phase="failed", message=reason, changed=False)
        finally:
            if maintenance and not self.recovery_required:
                await self.manager.end_maintenance()

    async def _download_checkpoint_asset(self, asset, stage):
        from halogen_deploy import HF_CLI_PACKAGE
        hf = shutil.which("hf")
        local = Path(sys.executable).parent / ("hf.exe" if os.name == "nt" else "hf")
        if not hf and local.is_file():
            hf = str(local)
        if not hf:
            await self._run(sys.executable, "-m", "pip", "install", "--upgrade", HF_CLI_PACKAGE, timeout=900)
            hf = shutil.which("hf") or (str(local) if local.is_file() else None)
        if not hf:
            raise UpdateError("HF CLI missing after installation.")
        self._save_job(message=f"Downloading {asset.name} at pinned revision {asset.revision}. Requests continue to run.")
        await self._run(hf, "download", asset.repo, asset.name, "--revision", asset.revision,
                        "--local-dir", str(stage), timeout=CHECKPOINT_START_TIMEOUT)

    @staticmethod
    def _without_checkpoint_download(content: bytes) -> bytes:
        lines, section = [], ""
        for line in content.decode("utf-8").splitlines(keepends=True):
            if line.strip().startswith("["):
                section = line.strip()
            if section == "[Container]" and re.match(r"\s*Environment\s*=\s*HALOGEN_DOWNLOAD=", line):
                continue
            lines.append(line)
        return "".join(lines).encode("utf-8")

    async def _verify_checkpoint(self, model: str, target: str):
        from shared_assets import host_path
        data = json.loads(await self._run("podman", "container", "inspect", self.models[model].removesuffix(".service")))
        env = data[0].get("Config", {}).get("Env", [])
        checkpoints = [value for value in env if value.startswith("HALOGEN_CHECKPOINT=")]
        if checkpoints != [f"HALOGEN_CHECKPOINT={target}"]:
            raise UpdateError("The running container does not use the selected checkpoint.")
        profile = parse_quadlet(self._paths()[model].read_text(encoding="utf-8"))
        asset = checkpoint_choice(self.job["target_key"])["asset"]
        path = host_path(profile, target)
        if path is None or not await self.checkpoint_assets.verify(path, asset):
            raise UpdateError("The selected checkpoint changed during activation.")

    async def _verify_units(self, image: str) -> None:
        for service in self.models.values():
            command = await self._run("systemctl", "--user", "show", service, "--property=ExecStart", "--value")
            if image not in command.split():
                raise UpdateError(f"Generated service {service} does not use the expected image.")

    async def _ready(self, model: str, version: str, image_id: str, timeout: float | None = None) -> None:
        deadline = time.monotonic() + (timeout if timeout is not None else self.start_timeout)
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
                        raise UpdateError("The started container uses an unexpected image ID.")
                    from npu import profile_model_names
                    content = self._paths()[model].read_text(encoding="utf-8")
                    expected = profile_model_names(parse_quadlet(content)) if "HALOGEN_NPU_MODELS=" in content else set()
                    if expected and not expected.issubset({entry["id"] for entry in await self.manager.backend_model_entries()}):
                        await asyncio.sleep(2)
                        continue
                    return
            await asyncio.sleep(2)
        raise UpdateError("Backend did not become ready with the expected API/engine version and capability probe.")

    async def _restart(self, model: str, timeout: float | None = None) -> None:
        if getattr(self.manager, "asset_validator", None):
            await self.manager.asset_validator(model)
        # Stop + wait for GTT release avoids starting a second model over memory
        # still held by the driver. The router admission gate stays closed.
        start_timeout = timeout if timeout is not None else self.start_timeout
        await self._run("systemctl", "--user", "stop", self.models[model], timeout=self.stop_timeout)
        await self.manager.wait_for_gtt_release()
        await self._run("systemctl", "--user", "start", self.models[model], timeout=start_timeout)

    async def _prepare_npu_update(self, configurations, image):
        from npu import CUSTOM_TASKS, check_runtime, host_status, parse_models, patch_quadlet
        from shared_assets import host_path
        enabled = [parse_quadlet(value["content"].decode("utf-8"), self._paths()[name])
                   for name, value in configurations.items() if b"HALOGEN_NPU_MODELS=" in value["content"]]
        profiles = enabled
        if not enabled:
            return {}
        state = host_status()
        if not state["ready"]:
            raise UpdateError("; ".join(state["errors"]))
        # Cache the previous image's record before an update so rollback and
        # later offline starts can validate legacy NPU volumes too.
        for previous in dict.fromkeys(profile.image for profile in enabled):
            await self.npu_assets.manifest(previous, self._run, refresh=True)
        manifest = await self.npu_assets.manifest(image, self._run, refresh=True)
        wanted = list(dict.fromkeys(model for profile in enabled
                     for model in parse_models(profile.env["HALOGEN_NPU_MODELS"])))
        bases = [m for m, info in manifest.models.items() if info["task"] in CUSTOM_TASKS] if any(m.startswith("/") for m in wanted) else []
        selection = self.npu_assets.selection(manifest, wanted, profiles, bases)
        self._save_job(phase="preparing_npu", message="Preparing versioned NPU files; the current backend keeps serving.")
        await selection.prepare(None, self._download_checkpoint_asset,
                                lambda line: self._save_job(message=line))
        mounts = [tuple(v) for v in state["xrt_mounts"]]
        for profile in enabled:
            custom = []
            for model in parse_models(profile.env["HALOGEN_NPU_MODELS"]):
                if model.startswith("/"):
                    source = host_path(profile, model)
                    if source is None or not all((source / name).is_file() for name in
                                                 ("config.json", "model.safetensors", "tokenizer.json")):
                        raise UpdateError(f"Incomplete NPU fine-tune: {model}")
                    custom.append((str(source), model, "ro"))
            await check_runtime(self._run, image, mounts, custom)
        return {profile.quadlet_path: patch_quadlet(configurations[profile.model_id]["content"],
                shared_root=selection.root, mounts=mounts) for profile in enabled}

    async def _update(self, version: str) -> None:
        maintenance = False
        try:
            image = f"{self.image_repo}:{version}"
            self._save_job(phase="pulling", message=f"Downloading image {version}. Requests continue to run.")
            await self._run("podman", "pull", image, timeout=1800)
            image_id = await self._inspect_image(image)
            prepared_configurations = self._configuration()
            npu_writes = await self._prepare_npu_update(prepared_configurations, image)
            self._save_job(phase="draining", message="Waiting for model switches and active requests.")
            await asyncio.wait_for(self.manager.begin_maintenance(), self.drain_timeout)
            maintenance = True
            await asyncio.wait_for(self.manager.drain_maintenance(), self.drain_timeout)
            model = self.manager.current_model
            if model not in self.models:
                raise UpdateError("Active model is unknown.")
            configurations = self._configuration()
            if configurations != prepared_configurations:
                raise UpdateError("Quadlets changed while preparing the update; retry with the current configuration.")
            if any(version_tuple(value["version"]) > version_tuple(version) for value in configurations.values()):
                raise UpdateError("Configuration has changed; downgrade cancelled.")
            health = await self.manager.backend_health()
            if not health or health.get("status") != "ok" or health.get("model") != model:
                raise UpdateError("Could not verify the active backend before updating.")
            for other, service in self.models.items():
                if other != model:
                    active = await self._run("systemctl", "--user", "show", service, "--property=ActiveState", "--value")
                    if active.strip() not in {"inactive", "failed"}:
                        raise UpdateError("A second model service is active; update cancelled.")
            old_version = normalize_version(((health or {}).get("version") or {}).get("api"))
            if not old_version or version_tuple(old_version) > version_tuple(version):
                raise UpdateError("Running version is unknown or newer than the update.")
            old_image_id = await self._container_image(model)
            if await self._inspect_image(configurations[model]["image"]) != old_image_id:
                raise UpdateError("Running image does not match the active Quadlet.")
            backup = self.backup_root / f"dashboard-update-{version}-{time.time_ns()}"
            backup.mkdir(parents=True, mode=0o700)
            modes = {}
            for name, path in self._paths().items():
                modes[name] = stat.S_IMODE(path.stat().st_mode)
                atomic_write(backup / path.name, configurations[name]["content"], modes[name])
            self._save_job(phase="configuring", message="Quadlets backed up; updating both image versions.",
                           backup_dir=str(backup), model=model, old_version=old_version,
                           old_image_id=old_image_id, modes=modes, changed=True)
            # changed is durable BEFORE the first write (including partial failure).
            for name, path in self._paths().items():
                if path.read_bytes() != configurations[name]["content"]:
                    raise UpdateError("Quadlet was changed externally during the update.")
                _, content = quadlet_image(npu_writes.get(path, configurations[name]["content"]), image, image_repo=self.image_repo)
                atomic_write(path, content, modes[name])
            await self._run("systemctl", "--user", "daemon-reload")
            await self._verify_units(image)
            self._save_job(phase="restarting", message=f"Starting {model} with {version}.")
            await self._restart(model)
            self._save_job(phase="verifying", message="Verifying API, engine, model, capability probe, and container image.")
            await self._ready(model, version, image_id)
            self.current_version = version
            self._save_job(phase="succeeded", message=f"{version} is active. Both Quadlets updated; the inactive model will update on its next switch.", changed=False)
            self._prune_backups()
        except (Exception, asyncio.CancelledError) as error:
            LOG.warning("Container update failed: %s", type(error).__name__)
            reason = str(error) if isinstance(error, (UpdateError, AssetError)) else "Update cancelled or timed out."
            if self.job and self.job.get("changed"):
                try:
                    await self._rollback(reason)
                except (Exception, asyncio.CancelledError):
                    self.recovery_required = True
                    self._save_job(phase="recovery_required", message="Rollback incomplete. Retry recovery; the router remains in maintenance mode.")
            elif self.job:
                self._save_job(phase="failed", message=reason, changed=False)
        finally:
            if maintenance and not self.recovery_required:
                await self.manager.end_maintenance()

    async def _rollback(self, reason: str) -> None:
        assert self.job is not None
        self._save_job(phase="rolling_back", message=f"{reason} Restoring the previous configuration.")
        model = self.job["model"]
        backup = Path(self.job["backup_dir"])
        if model not in self.models or backup.resolve().parent != self.backup_root.resolve():
            raise UpdateError("Invalid recovery record.")
        paths = self._paths()
        if "changed_models" in self.job:
            changed_models = self.job["changed_models"]
            if (not isinstance(changed_models, list) or not changed_models
                    or any(not isinstance(name, str) or name not in paths for name in changed_models)
                    or model not in changed_models):
                raise UpdateError("Invalid recovery record.")
            paths = {name: paths[name] for name in changed_models}
        # Older journals changed every profile; new checkpoint jobs record only
        # the official profile. Validate all affected backups before restoring.
        originals = {name: (backup / path.name).read_bytes() for name, path in paths.items()}
        for content in originals.values():
            quadlet_image(content, image_repo=self.image_repo)
        for name, path in paths.items():
            atomic_write(path, originals[name], self.job["modes"][name])
        await self._run("systemctl", "--user", "daemon-reload")
        # A process restart could have occurred midway through startup. Stop both
        # managed units before bringing back exactly the originally active model.
        await self._run("systemctl", "--user", "stop", *self.models.values(), timeout=self.stop_timeout)
        await self.manager.wait_for_gtt_release()
        await self._run("systemctl", "--user", "reset-failed", self.models[model])
        if getattr(self.manager, "asset_validator", None):
            await self.manager.asset_validator(model)
        await self._run("systemctl", "--user", "start", self.models[model], timeout=self.start_timeout)
        await self._ready(model, self.job["old_version"], self.job["old_image_id"])
        self.manager.current_model = model
        self.current_version = self.job["old_version"]
        self.recovery_required = False
        self._save_job(phase="rolled_back", message=f"{reason} Previous version {self.current_version} is active again.", changed=False)

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
                self._save_job(phase="failed", message="Update interrupted by a router restart; Quadlets were not changed.")
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
            LOG.error("Automatic update recovery failed; maintenance mode is active.")

    async def _retry_recovery(self) -> None:
        try:
            await self._rollback("Wiederherstellung erneut angefordert.")
            await self.manager.end_maintenance()
        except (Exception, asyncio.CancelledError):
            self.recovery_required = True
            if self.job:
                self._save_job(phase="recovery_required", message="Recovery failed. Check the backup and router journal.")

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
            raise web.HTTPForbidden(text="Update actions require JSON from the local dashboard.")

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
            raise web.HTTPBadRequest(text="Invalid JSON")
        if not isinstance(body, dict) or normalize_version(body.get("version")) != body.get("version") or not body.get("version"):
            raise web.HTTPBadRequest(text="Specify a verified stable version.")
        await self.start(body["version"])
        return web.json_response(self.status(), status=202, headers={"Cache-Control": "no-store"})

    async def api_recover(self, request: web.Request) -> web.Response:
        self._require_same_origin(request)
        async with self.start_lock:
            if self.running or not self.recovery_required or not self.job or not self.job.get("backup_dir"):
                raise web.HTTPConflict(text="No automatically recoverable update is available.")
            self.task = asyncio.create_task(self._retry_recovery())
        return web.json_response(self.status(), status=202, headers={"Cache-Control": "no-store"})

    async def api_install_checkpoint(self, request: web.Request) -> web.Response:
        self._require_same_origin(request)
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            raise web.HTTPBadRequest(text="Invalid JSON")
        if not isinstance(body, dict) or ("target" in body and
                (not isinstance(body["target"], str) or body["target"] not in OFFICIAL_CHECKPOINTS)):
            raise web.HTTPBadRequest(text="Checkpoint target must be v2 or ht43")
        await self.start_checkpoint(body.get("target"))
        return web.json_response(self.status(), status=202, headers={"Cache-Control": "no-store"})

    def register_routes(self, app: web.Application) -> None:
        app.router.add_get("/dashboard/api/updates", self.api_status)
        app.router.add_post("/dashboard/api/updates/check", self.api_check)
        app.router.add_post("/dashboard/api/updates/install", self.api_install)
        app.router.add_post("/dashboard/api/updates/recover", self.api_recover)
        app.router.add_post("/dashboard/api/updates/checkpoint", self.api_install_checkpoint)
