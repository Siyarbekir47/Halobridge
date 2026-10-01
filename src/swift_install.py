"""Swift installation pipeline and durable, scoped Quadlet recovery."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import stat
import sys
import logging
from pathlib import Path

from profiles import bind_shared_assets, parse_quadlet, profile_to_dict, render_quadlet, swift_quick_template
from shared_assets import AssetError, atomic_json
from swift_catalog import variant_info


def shared_quadlet_content(content: str, locations: dict) -> str:
    """Preserve user service settings/comments; replace only shared bindings."""
    profile = bind_shared_assets(parse_quadlet(content), locations)
    keys = {"HALOGEN_NGRAM_TABLE", "HALOGEN_TOKENIZER", "HALOGEN_VISION_TOWER"}
    additions = [f"Environment={key}={profile.env[key]}" for key in sorted(keys) if key in profile.env]
    additions += [f"Volume={host}:{container}:{mode}" for host, container, mode in profile.volumes
                  if container in {"/shared/ngram", "/shared/vision", "/shared/tokenizer"}]
    lines, section = [], ""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped
        if section == "[Container]":
            if stripped == "[Container]":
                lines.extend([line, *additions])
                continue
            if any(re.match(rf'Environment=["\']?{key}=', stripped) for key in keys):
                continue
            if stripped.startswith("Volume=") and any(f":{alias}:" in stripped for alias in
                                                                       ("/shared/ngram", "/shared/vision", "/shared/tokenizer")):
                continue
        lines.append(line)
    return "\n".join(lines) + "\n"


class SwiftInstallMixin:
    def swift_plan(self, variant: str) -> dict:
        self._require_enabled()
        info = variant_info(variant)
        result = self.assets.plan(variant)
        result.update(label=info["label"], model_card=f"https://huggingface.co/{info['checkpoint'].repo}",
                      engine_version="0.15.1")
        return result

    def _swift_available(self):
        from halogen_deploy import DeployError
        self._require_enabled()
        if (self.recovery_required or self.deploy_lock.locked() or self.manager.maintenance
                or self.manager.switching or (self.updater and
                (self.updater.running or self.updater.recovery_required))):
            raise DeployError("Deployment, model switch, update or recovery is in progress")

    async def quick_swift(self, payload: dict) -> dict:
        self._swift_available()
        variant = str(payload.get("variant", ""))
        try:
            self.swift_plan(variant)
        except (ValueError, OSError) as exc:
            from halogen_deploy import DeployError
            raise DeployError(str(exc)) from exc
        self._start_pipeline_job("quick-" + variant, 4)
        self._job_task = asyncio.create_task(self._run_quick_swift(variant))
        return {"ok": True, "kind": "quick-" + variant}

    async def _download_asset(self, asset, stage):
        from halogen_deploy import HF_CLI_PACKAGE, DeployError, _hf_download_env
        hf = self._hf_executable()
        if not hf:
            await self._run_stream([sys.executable, "-m", "pip", "install", "--upgrade", HF_CLI_PACKAGE], timeout=900)
            hf = self._hf_executable()
        if not hf:
            raise DeployError("HF CLI missing after installation")
        self._append_job_line(f"Download {asset.repo}@{asset.revision}: {asset.name}")
        env = {**_hf_download_env(), "HF_HUB_OFFLINE": "0"}
        await self._run_stream([hf, "download", asset.repo, asset.name, "--revision", asset.revision,
                                "--local-dir", str(stage)], env, timeout=21600)

    def _installation_record(self, variant: str, writes: dict[Path, bytes]):
        records = []
        for path, content in writes.items():
            if path.is_symlink():
                raise AssetError(f"Refusing symlink Quadlet: {path}")
            before = path.read_bytes() if path.exists() else None
            if before == content:
                continue
            records.append({"name": path.name, "before": base64.b64encode(before).decode() if before is not None else None,
                            "after": base64.b64encode(content).decode(),
                            "mode": stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600})
        record = {"variant": variant, "previous_model": self.manager.current_model,
                  "phase": "prepared", "records": records}
        atomic_json(self.swift_journal, record)
        return record

    def _write_installation(self, record):
        self.quadlet_dir.mkdir(parents=True, exist_ok=True)
        for item in record["records"]:
            path = self.quadlet_dir / item["name"]
            expected = base64.b64decode(item["before"]) if item["before"] is not None else None
            actual = path.read_bytes() if path.exists() else None
            if actual != expected or path.is_symlink():
                raise AssetError(f"Quadlet changed during preparation: {path}")
            self._backup_existing(path)
            from halogen_updates import atomic_write
            atomic_write(path, base64.b64decode(item["after"]), item["mode"])

    async def _swift_ready(self, profile):
        from semver import normalize_version
        health = await self.manager.backend_health() or {}
        expected = normalize_version(profile.image.rsplit(":", 1)[-1])
        versions = health.get("version") or {}
        if (health.get("status") != "ok" or health.get("model") != profile.model_id
                or not isinstance(versions, dict) or normalize_version(versions.get("engine")) != expected
                or normalize_version(versions.get("api")) != expected):
            raise AssetError("Swift health/model/API/engine version verification failed")

    async def _run_quick_swift(self, variant):
        record = None
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(stop, "preparing Swift assets"))
        try:
            self.job["step"] = 1
            locations = await self.assets.prepare(variant, self._download_asset, self._append_job_line)
            models_root, cache_root = self._default_roots()
            target = self.quadlet_dir / f"halogen-{variant}.container"
            originals = {target: target.read_bytes() if target.exists() else None}
            existing = parse_quadlet(originals[target].decode("utf-8"), target) if originals[target] is not None else None
            profile = swift_quick_template(variant, models_root, cache_root, locations, existing)
            self.job["step"] = 2
            payload = profile_to_dict(profile)
            self.install_dirs(payload)
            preview = self.dry_run(payload)
            if not preview["ok"]:
                raise AssetError("; ".join(preview["errors"]))
            await self._run_stream(["podman", "pull", profile.image], timeout=1800)
            # The latest user requirement also connects Official/Orca to the
            # canonical shared tree. Preserve every unrelated setting verbatim.
            writes = {target: render_quadlet(profile).encode()}
            for other in self.assets.profiles():
                if other.profile_id == variant:
                    continue
                path = other.quadlet_path
                originals[path] = path.read_bytes()
                before = originals[path].decode("utf-8")
                writes[path] = shared_quadlet_content(before, locations).encode()
            async with self.deploy_lock:
                async def configure():
                    nonlocal record
                    # Admission is already closed by the existing switch gate;
                    # capture the actual previous backend only after draining.
                    for path, before in originals.items():
                        if (path.read_bytes() if path.exists() else None) != before:
                            raise AssetError(f"Quadlet changed during preparation: {path}")
                    record = self._installation_record(variant, writes)
                    self._write_installation(record)
                    await self._daemon_reload()
                    await self._verify_unit(profile)
                    self.reload_discovery()
                    record["phase"] = "activating"
                    atomic_json(self.swift_journal, record)
                self.job["step"] = 3
                self._append_job_line("Assets verified; drain requests and activate Swift")
                await self._switch_with_heartbeat(profile.model_id, "waiting for Swift", before_stop=configure)
                self.job["step"] = 4
                await self._swift_ready(profile)
                record["phase"] = "complete"
                atomic_json(self.swift_journal, record)
            self.job["state"] = "done"
        except (Exception, asyncio.CancelledError) as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            if record:
                try:
                    recovery = asyncio.create_task(self._restore_swift(record))
                    while True:
                        try:
                            await asyncio.shield(recovery)
                            break
                        except asyncio.CancelledError:
                            cancelled = True
                            if recovery.cancelled():
                                raise AssetError("Swift recovery was interrupted; restart Halobridge")
                except Exception as recovery_error:
                    self.recovery_required = True
                    self.manager.maintenance = True
                    self._append_job_line(f"Recovery required: {recovery_error}. Restart Halobridge to retry.")
            self.job["state"] = "cancelled" if cancelled else "error"
            self.job["error"] = str(exc) or "Cancelled"
            if cancelled:
                raise asyncio.CancelledError from exc
        finally:
            stop.set()
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _restore_swift(self, record):
        variant = record["variant"]
        variant_info(variant)
        target_service = f"halogen-{variant}.service"
        # Keep new requests from switching onto a profile during file recovery,
        # including a failed unit verification before the backend was stopped.
        await self.manager.begin_maintenance()
        async with self.manager.condition:
            while self.manager.active_requests > 0:
                await self.manager.condition.wait()
        if record["phase"] == "activating":
            health = await self.manager.backend_health()
            if health and health.get("status") == "ok":
                await self.manager.wait_for_backend_idle()
            if await self._service_is_active(target_service):
                await self._run("systemctl", "--user", "stop", target_service, timeout=120)
                await self.manager.wait_for_gtt_release()
        from halogen_updates import atomic_write
        restored = []
        for item in record["records"]:
            name = item["name"]
            if not re.fullmatch(r"halogen-[a-z0-9][a-z0-9-]{0,31}\.container", name):
                raise AssetError("Invalid Swift recovery filename")
            path = self.quadlet_dir / name
            if path.is_symlink():
                raise AssetError("Refusing symlink in Swift recovery")
            before = base64.b64decode(item["before"], validate=True) if item["before"] is not None else None
            after = base64.b64decode(item["after"], validate=True)
            actual = path.read_bytes() if path.exists() else None
            if actual not in (before, after):
                raise AssetError(f"Quadlet edited externally: {path}; recovery paused")
            restored.append((path, before, item["mode"]))
        for path, before, mode in restored:
            if before is None:
                path.unlink(missing_ok=True)
            else:
                atomic_write(path, before, mode)
        await self._daemon_reload()
        # Discovery can legitimately become empty after rolling back a fresh install.
        from discovery import discover
        specs, _ = discover(self.config.models.quadlet_dir, self.config.updates.image_repo)
        models = {s.model_id: s.service for s in specs.values()}
        models.update(self.config.models.explicit)
        self.manager.models = models
        if self.updater:
            self.updater.models = models
        if self.dashboard:
            self.dashboard.reload_models(models, specs)
        previous = record.get("previous_model")
        if record["phase"] == "activating":
            if previous in models:
                if not await self._service_is_active(models[previous]):
                    await self.manager.wait_for_gtt_release()
                    await self.manager.start_service(previous)
                else:
                    await self.manager.wait_for_backend(previous)
            self.manager.current_model = previous if previous in models else None
        record["phase"] = "rolled_back"
        atomic_json(self.swift_journal, record)
        self.recovery_required = False
        await self.manager.end_maintenance()

    async def recover_swift_on_startup(self):
        if not self.swift_journal.exists():
            return
        try:
            record = json.loads(self.swift_journal.read_text(encoding="utf-8"))
            if record.get("phase") in {"complete", "rolled_back"}:
                return
            if record.get("phase") not in {"prepared", "activating"}:
                raise AssetError("Invalid Swift recovery phase")
            await self._restore_swift(record)
        except Exception:
            self.recovery_required = True
            self.manager.maintenance = True
            logging.getLogger("halobridge.deploy").exception("Swift installation recovery failed; maintenance active")
