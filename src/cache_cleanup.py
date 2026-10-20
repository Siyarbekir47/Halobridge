"""Disk-pressure cleanup of managed prompt caches, with the backend stopped.

Only Halogen's hexadecimal configuration subtrees inside configured cache
roots are eligible. Model downloads, arbitrary files and symlink targets are not.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
from pathlib import Path

from shared_assets import host_path

LOG = logging.getLogger(__name__)
CONFIG_DIR = re.compile(r"[0-9a-f]{16,64}")


def candidates(root: Path, models_root: Path, protected=()):
    """Return oldest cache configurations first; never follow links or mounts."""
    if root.is_symlink() or not root.is_dir():
        return []
    root = root.resolve()
    models_root = models_root.resolve()
    if root == root.parent or root == models_root or root in models_root.parents or models_root in root.parents:
        return []
    device = root.stat().st_dev
    result = []
    # A profile directory is only a boundary, never itself a deletion target.
    for parent in [root, *(p for p in root.iterdir() if p.is_dir() and not p.is_symlink())]:
        if parent.stat().st_dev != device:
            continue
        for group in parent.iterdir():
            if not CONFIG_DIR.fullmatch(group.name) or group.is_symlink() or not group.is_dir():
                continue
            if any(group.resolve() == p or group.resolve() in p.parents or p in group.resolve().parents for p in protected):
                continue
            files = []
            safe = True
            for directory, dirs, names in os.walk(group, followlinks=False):
                path = Path(directory)
                if path.stat().st_dev != device or any((path / d).is_symlink() for d in dirs):
                    safe = False
                    break
                for name in names:
                    file = path / name
                    if file.is_symlink() or not file.is_file() or file.stat().st_dev != device:
                        safe = False
                        break
                    if file.suffix.lower() in {".hgn", ".gguf", ".safetensors", ".pem", ".key"}:
                        safe = False
                        break
                    files.append(file)
            if safe and any(p.suffix == ".rec" or p.name.endswith(".rec.tmp") for p in files):
                result.append((max(p.stat().st_mtime for p in files), group))
    return sorted(result)


class CacheCleanup:
    def __init__(self, deploy, app_updater):
        self.deploy, self.app_updater = deploy, app_updater
        self.config = deploy.config.deploy
        self.task = None
        self.last_result = {}

    def start(self):
        if self.config.cache_cleanup_enabled and os.name == "posix" and sys.platform.startswith("linux"):
            self.task = asyncio.create_task(self._loop())

    async def close(self):
        if self.task:
            # Let any stop/cleanup/restart transaction finish before shutdown.
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def _loop(self):
        while True:
            try:
                await self.scan()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("Automatic prompt-cache cleanup failed")
            await asyncio.sleep(self.config.cache_cleanup_interval_s)

    async def scan(self):
        if not self.config.cache_cleanup_enabled:
            return
        roots = {Path(self.config.cache_root)}
        for profile in self.deploy.npu_assets.profiles():
            path = host_path(profile, profile.env.get("HALOGEN_CACHE_DIR", "/cache"))
            if path:
                roots.add(Path(path))
        devices = {}
        for root in sorted(roots):
            if root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)):
                devices.setdefault(root.stat().st_dev, []).append(root)
        for sources in devices.values():
            await self._scan_root(sources[0], sources)

    async def _scan_root(self, root, sources):
        if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
            return
        usage = shutil.disk_usage(root)
        if usage.free * 100 > usage.total * self.config.cache_cleanup_free_percent:
            return
        deploy, manager = self.deploy, self.deploy.manager
        if (deploy.deploy_lock.locked() or deploy.recovery_required or manager.maintenance
                or manager.active_requests or self.app_updater.running
                or (deploy.updater and deploy.updater.running)
                or (deploy.job and deploy.job.get("state") == "running")):
            return
        task = asyncio.create_task(self._clean(root, sources))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def _clean(self, root, sources):
        deploy, manager = self.deploy, self.deploy.manager
        async with deploy.deploy_lock:
            if (deploy.recovery_required or manager.maintenance or manager.active_requests
                    or self.app_updater.running or (deploy.updater and deploy.updater.running)
                    or (deploy.job and deploy.job.get("state") == "running")):
                return
            await asyncio.wait_for(manager.begin_maintenance(), manager.drain_timeout)
            stopped = False
            current = manager.current_model
            try:
                active = current in manager.models and await deploy._service_is_active(manager.models[current])
                if active:
                    await asyncio.wait_for(manager.drain_maintenance(), manager.drain_timeout)
                profiles = list(deploy.npu_assets.profiles())
                protected = []
                for profile in profiles:
                    for key in ("HALOGEN_CHECKPOINT", "HALOGEN_TOKENIZER", "HALOGEN_NGRAM_TABLE",
                                "HALOGEN_VISION_TOWER", "HALOGEN_CHAT_TEMPLATE", "HALOGEN_API_KEY_FILE", "HALOGEN_RECORD_DIR"):
                        path = host_path(profile, profile.env.get(key, "")) if profile.env.get(key) else None
                        if path:
                            protected.append(path.resolve())
                groups = {}
                for source in sources:
                    for stamp, path in await asyncio.to_thread(candidates, source, Path(self.config.models_root), protected):
                        groups[path] = (stamp, source)
                groups = sorted((stamp, path, source) for path, (stamp, source) in groups.items())
                if not groups:
                    LOG.warning("Disk space low: no recognized prompt-cache configurations under %s", root)
                    return
                # Never remove files used by another manually started backend.
                services = set(manager.models.values()) | {p.service_name for p in profiles}
                for service in services - {manager.models.get(current)}:
                    if await deploy._service_is_active(service):
                        LOG.warning("Cache cleanup deferred: another backend is active: %s", service)
                        return
                if active:
                    await deploy._run("systemctl", "--user", "stop", manager.models[current], timeout=manager.stop_timeout)
                    stopped = True
                    await manager.wait_for_gtt_release()
                removed = []
                before = shutil.disk_usage(root).free
                for _, path, source in groups:
                    usage = shutil.disk_usage(root)
                    if usage.free * 100 >= usage.total * self.config.cache_cleanup_target_percent:
                        break
                    # Revalidate before every deletion, including all ancestors.
                    if any(p.is_symlink() for p in (path, *path.parents)):
                        continue
                    checked = await asyncio.to_thread(candidates, source, Path(self.config.models_root), protected)
                    if not any(p == path for _, p in checked):
                        continue
                    await asyncio.to_thread(shutil.rmtree, path)
                    removed.append(str(path))
                    LOG.warning("Low disk space: removed prompt cache %s", path)
                    await asyncio.sleep(0)
                after = shutil.disk_usage(root)
                self.last_result = {"removed": removed, "freed_bytes": max(0, after.free - before),
                                    "free_percent": 100 * after.free / after.total}
                LOG.warning("Prompt-cache cleanup: %s", self.last_result)
                if after.free * 100 < after.total * self.config.cache_cleanup_target_percent:
                    LOG.warning("Disk remains low after prompt-cache cleanup; other files need attention")
            finally:
                try:
                    if stopped:
                        await manager.start_service(current)
                finally:
                    await manager.end_maintenance()
