"""Prepare NPU files while serving; activate with durable, targeted recovery."""
from __future__ import annotations

import asyncio
import base64
import json
import re
import stat
from pathlib import Path

from npu import (CUSTOM_TASKS, DOC_URL, MODEL_MIN_VERSION, NPU_MODELS, TASK_ROUTES, check_runtime, host_setup, host_status,
                 parse_models, patch_quadlet)
from profiles import parse_quadlet, validate_profile
from semver import normalize_version, version_tuple
from shared_assets import AssetError, atomic_json, host_path


class NpuInstallMixin:
    def _npu_profiles(self, requested=None):
        profiles = list(self.npu_assets.profiles())
        if requested is not None:
            wanted = set(requested)
            profiles = [p for p in profiles if p.profile_id in wanted]
            if {p.profile_id for p in profiles} != wanted:
                raise AssetError("Unknown NPU target profile")
        if not profiles:
            raise AssetError("Install a backend profile before configuring NPU models")
        return profiles

    async def npu_status(self):
        state = host_status()
        state.update({"models": [{"id": model, "task": task, "endpoint": TASK_ROUTES[task]}
                                  for model, task in NPU_MODELS.items()],
                      "profiles": [{"id": p.profile_id, "image": p.image,
                                    "models": p.env.get("HALOGEN_NPU_MODELS", "")}
                                   for p in self.npu_assets.profiles()],
                      "setup": host_setup(Path(self.config.dashboard.state_dir))})
        return state

    async def npu_plan(self, value: str, requested=None):
        models = parse_models(value) if value else []
        profiles = self._npu_profiles(requested)
        if {m.rsplit("/", 1)[-1] for m in models} & set(self.manager.models):
            raise AssetError("NPU model IDs must not conflict with backend profile model IDs")
        result = {"models": models, "profiles": [p.profile_id for p in profiles],
                  "host": host_status(), "documentation": DOC_URL, "plans": []}
        if not models:
            result["download_bytes"] = 0
            return result
        manifests = {}
        for image in dict.fromkeys(p.image for p in profiles):
            version = version_tuple(image.rsplit(":", 1)[-1]) or (0, 0, 0)
            if version < (0, 16, 2):
                raise AssetError("Update all selected profiles to Halogen 0.16.2 or newer before NPU Setup")
            for model in models:
                minimum = MODEL_MIN_VERSION.get(model, (0, 16, 2))
                if version < minimum:
                    raise AssetError(f"NPU model {model} requires Halogen {'.'.join(map(str, minimum))} or newer")
            manifest = await self.npu_assets.manifest(image)
            if manifest.key in manifests:
                manifests[manifest.key]["images"].append(image)
                continue
            bases = [m for m, info in manifest.models.items() if info["task"] in CUSTOM_TASKS] if any(m.startswith("/") for m in models) else []
            selection = self.npu_assets.selection(manifest, models, profiles, bases)
            plan = selection.plan()
            plan["image"] = image
            plan["images"] = [image]
            manifests[manifest.key] = plan
            result["plans"].append(plan)
        result["download_bytes"] = sum(p["download_bytes"] for p in result["plans"])
        result["custom_models"] = [m for m in models if m.startswith("/")]
        return result

    async def quick_npu(self, payload):
        from halogen_deploy import DeployError
        self._require_enabled()
        if self.recovery_required or self.deploy_lock.locked() or (self.updater and self.updater.running):
            raise DeployError("Another deployment or recovery is in progress")
        if self._job_task and not self._job_task.done():
            raise DeployError("A deployment job is already running")
        value = payload.get("models", "")
        requested = payload.get("profiles")
        if not isinstance(value, str) or (requested is not None and
                (not isinstance(requested, list) or not all(isinstance(p, str) for p in requested))):
            raise DeployError("NPU models must be a string and profiles must be a list of IDs")
        await self.npu_plan(value, requested)
        if value:
            state = host_status()
            if not state["ready"]:
                raise DeployError("; ".join(state["errors"]))
        self._start_pipeline_job("quick-npu", 4)
        self._job_task = asyncio.create_task(self._run_quick_npu(value, requested))
        return {"ok": True, "kind": "quick-npu"}

    async def _npu_runtime_check(self, image, mounts, custom_mounts=()):
        return await check_runtime(self._run, image, mounts, custom_mounts)

    def _npu_custom_mounts(self, profile, models):
        mounts = []
        for model in models:
            if not model.startswith("/"):
                continue
            source = host_path(profile, model)
            if source is None:
                raise AssetError(f"Custom NPU model is not mounted: {model}")
            source = self._validate_host_path(str(source))
            if not all((source / name).is_file() for name in ("config.json", "model.safetensors", "tokenizer.json")):
                raise AssetError(f"Incomplete NPU fine-tune: {source}; need config.json, model.safetensors and tokenizer.json")
            mounts.append((str(source), model, "ro,z"))
        return mounts

    async def _run_quick_npu(self, value, requested):
        record = None
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(stop, "preparing NPU assets"))
        try:
            profiles = self._npu_profiles(requested)
            originals = {p.quadlet_path: p.quadlet_path.read_bytes() for p in profiles}
            writes, locations = {}, {}
            models = parse_models(value) if value else []
            self.job["step"] = 1
            if models:
                mounts = [tuple(v) for v in host_status()["xrt_mounts"]]
                for image in dict.fromkeys(p.image for p in profiles):
                    await self._run_stream(["podman", "pull", image], timeout=1800)
                    active_profile = next((p for p in profiles if p.model_id == self.manager.current_model), None)
                    if self.updater and active_profile and active_profile.image == image:
                        if await self.updater._inspect_image(image) != await self.updater._container_image(active_profile.model_id):
                            raise AssetError("The pulled image differs from the running image; use the engine update first")
                    manifest = await self.npu_assets.manifest(image, self._run, refresh=True)
                    bases = [m for m, info in manifest.models.items() if info["task"] in CUSTOM_TASKS] if any(m.startswith("/") for m in models) else []
                    selection = self.npu_assets.selection(manifest, models, profiles, bases)
                    await selection.prepare(None, self._download_asset, self._append_job_line)
                    locations[image] = selection.root
                self.job["step"] = 2
                for profile in profiles:
                    custom = self._npu_custom_mounts(profile, models)
                    await self._npu_runtime_check(profile.image, mounts, custom)
                    writes[profile.quadlet_path] = patch_quadlet(originals[profile.quadlet_path], models=value,
                                                               shared_root=locations[profile.image], mounts=mounts,
                                                               custom_mounts=custom)
            else:
                writes = {path: patch_quadlet(content, models="") for path, content in originals.items()}
            for path, content in writes.items():
                profile = parse_quadlet(content.decode("utf-8"), path)
                errors = validate_profile(profile) + self._validate_host_paths(profile)
                if errors:
                    raise AssetError("; ".join(errors))
            if models and not host_status()["ready"]:
                raise AssetError("Host NPU prerequisites changed during preparation; run halobridge npu-check")
            async with self.deploy_lock:
                self.job["step"] = 3
                await asyncio.wait_for(self.manager.begin_maintenance(), self.manager.drain_timeout)
                await asyncio.wait_for(self.manager.drain_maintenance(), self.manager.drain_timeout)
                active = self.manager.current_model
                changes = []
                for path, after in writes.items():
                    if path.is_symlink() or path.read_bytes() != originals[path]:
                        raise AssetError(f"Quadlet changed during NPU preparation: {path}")
                    if originals[path] != after:
                        changes.append({"name": path.name,
                            "before": base64.b64encode(originals[path]).decode(), "after": base64.b64encode(after).decode(),
                            "mode": stat.S_IMODE(path.stat().st_mode)})
                active_changed = any(p.model_id == active and originals[p.quadlet_path] != writes[p.quadlet_path]
                                     for p in profiles)
                record = {"phase": "prepared", "previous_model": active,
                          "restart_previous": active_changed, "records": changes}
                atomic_json(self.npu_journal, record)
                from halogen_updates import atomic_write
                for item in record["records"]:
                    path = self.quadlet_dir / item["name"]
                    self._backup_existing(path)
                    atomic_write(path, base64.b64decode(item["after"]), item["mode"])
                await self._daemon_reload()
                for profile in profiles:
                    await self._verify_unit(parse_quadlet(writes[profile.quadlet_path].decode()))
                record["phase"] = "activating"
                atomic_json(self.npu_journal, record)
                if active_changed:
                    await self._run("systemctl", "--user", "stop", self.manager.models[active], timeout=self.manager.stop_timeout)
                    await self.manager.wait_for_gtt_release()
                    await self.manager.start_service(active)
                self.job["step"] = 4
                if any(p.model_id == active for p in profiles):
                    await self._npu_ready(active, {m.rsplit("/", 1)[-1] for m in models},
                                          next(p.image for p in profiles if p.model_id == active))
                record["phase"] = "complete"
                atomic_json(self.npu_journal, record)
                await self.manager.end_maintenance()
            self.job["state"] = "done"
        except (Exception, asyncio.CancelledError) as exc:
            if record:
                recovery = asyncio.create_task(self._restore_npu(record))
                try:
                    while True:
                        try:
                            await asyncio.shield(recovery)
                            break
                        except asyncio.CancelledError:
                            if recovery.cancelled():
                                raise AssetError("NPU recovery was interrupted; restart Halobridge")
                except Exception as error:
                    self.recovery_required = True
                    self.manager.maintenance = True
                    self._append_job_line(f"NPU recovery required: {error}; restart Halobridge to retry")
            elif self.manager.maintenance:
                await self.manager.end_maintenance()
            self.job["state"] = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            self.job["error"] = str(exc) or "Cancelled"
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            stop.set()
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _npu_ready(self, model, expected, image):
        if self.updater:
            await self.updater._ready(model, image.rsplit(":", 1)[-1], await self.updater._inspect_image(image))
        health = await self.manager.backend_health() or {}
        versions = health.get("version") or {}
        version = normalize_version(image.rsplit(":", 1)[-1])
        if (health.get("status") != "ok" or health.get("model") != model or health.get("capability_probe") != "ok"
                or not isinstance(versions, dict) or normalize_version(versions.get("api")) != version
                or normalize_version(versions.get("engine")) != version):
            raise AssetError("NPU activation: backend health/model/capability verification failed")
        entries = await self.manager.backend_model_entries()
        actual = {entry.get("id") for entry in entries}
        if actual != expected | {model}:
            raise AssetError("NPU activation: expected NPU models were not advertised by the backend")

    async def _restore_npu(self, record):
        from halogen_updates import atomic_write
        if not self.manager.maintenance:
            await self.manager.begin_maintenance()
        async with self.manager.condition:
            while self.manager.active_requests:
                await self.manager.condition.wait()
        restored = []
        for item in record["records"]:
            name = item["name"]
            if not re.fullmatch(r"halogen-[a-z0-9][a-z0-9-]{0,31}\.container", name):
                raise AssetError("Invalid NPU recovery filename")
            path = self.quadlet_dir / name
            before, after = (base64.b64decode(item[key], validate=True) for key in ("before", "after"))
            if path.is_symlink() or not path.exists() or path.read_bytes() not in (before, after):
                raise AssetError(f"Quadlet edited externally: {path}; NPU recovery paused")
            restored.append((path, before, item["mode"]))
        previous = record.get("previous_model")
        restart = record["phase"] == "activating" and record.get("restart_previous", True) and previous in self.manager.models
        if restart:
            await self._run("systemctl", "--user", "stop", self.manager.models[previous], timeout=self.manager.stop_timeout)
            await self.manager.wait_for_gtt_release()
        for path, before, mode in restored:
            atomic_write(path, before, mode)
        await self._daemon_reload()
        self.reload_discovery()
        if restart:
            await self.manager.start_service(previous)
            self.manager.current_model = previous
        record["phase"] = "rolled_back"
        atomic_json(self.npu_journal, record)
        self.recovery_required = False
        await self.manager.end_maintenance()

    async def recover_npu_on_startup(self):
        if not self.npu_journal.exists():
            return
        try:
            record = json.loads(self.npu_journal.read_text(encoding="utf-8"))
            if record.get("phase") in {"complete", "rolled_back"}:
                return
            if record.get("phase") not in {"prepared", "activating"}:
                raise AssetError("Invalid NPU recovery phase")
            await self._restore_npu(record)
        except Exception:
            self.recovery_required = True
            self.manager.maintenance = True
            import logging
            logging.getLogger("halobridge.deploy").exception("NPU recovery failed; maintenance active")
