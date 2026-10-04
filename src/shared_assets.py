"""Verified shared assets, resumable preparation, and offline start checks.

Legacy files remain untouched. Verified files enter the canonical shared tree
via a hard link on the same filesystem, or a checked copy across filesystems.
Only fully verified downloads are atomically promoted into the shared tree.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import threading
from pathlib import Path, PurePosixPath

from profiles import parse_quadlet, ProfileError
from swift_catalog import SHARED_ASSETS, SWIFT_VARIANTS, SWIFT_ENGINE_VERSION, variant_info

SPACE_RESERVE = 5 * 1024 ** 3
KNOWN_MODELS = {"qwen3.8-flash", "qwen3.8-flash-uncensored",
                "halogen-swift15", "halogen-swift15-abliterated"}


class AssetError(ValueError):
    pass


def atomic_json(path: Path, value: dict) -> None:
    from halogen_updates import atomic_write
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(value, indent=2).encode(), mode=0o600)


def host_path(profile, container_path: str) -> Path | None:
    """Resolve by longest matching configured mount, never by guessed host roots."""
    wanted = PurePosixPath(container_path)
    for host, container, _mode in sorted(profile.volumes, key=lambda v: len(v[1]), reverse=True):
        try:
            suffix = wanted.relative_to(PurePosixPath(container))
        except ValueError:
            continue
        return Path(host).expanduser() / str(suffix)
    return None


def existing_parent(path: Path) -> Path:
    while not path.exists():
        parent = path.parent
        if parent == path:
            raise AssetError(f"No filesystem for {path}")
        path = parent
    return path


class SharedAssets:
    def __init__(self, models_root: Path, quadlet_dir: Path, state_dir: Path, validate_path=None):
        self.models_root = models_root
        self.quadlet_dir = quadlet_dir
        self.cache_path = state_dir / "asset-verification.json"
        self.validate_path = validate_path
        self.cache = {}
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                self.cache = value
        except (OSError, ValueError):
            pass
        self.lock = asyncio.Lock()

    def shared_path(self, asset) -> Path:
        return self.models_root / "shared" / "halogen-v2" / asset.revision / asset.name

    def staging_root(self, asset, variant=None, checkpoint_root=None) -> Path:
        if checkpoint_root is not None and asset not in SHARED_ASSETS:
            return checkpoint_root / ".downloads"
        root = (self.models_root / variant if variant and asset == variant_info(variant)["checkpoint"]
                else self.models_root / "shared" / "halogen-v2" / asset.revision)
        return root / ".downloads"

    def remaining_bytes(self, asset, variant=None, checkpoint_root=None) -> int:
        stage = self.staging_root(asset, variant, checkpoint_root)
        complete = stage / asset.name
        partial_dir = stage / ".cache" / "huggingface" / "download" / Path(asset.name).parent
        partials = list(partial_dir.glob(Path(asset.name).name + ".*.incomplete"))
        sizes = []
        for path in [complete, *partials]:
            if path.is_file():
                stat = path.stat()
                sizes.append(min(stat.st_size, stat.st_blocks * 512) if hasattr(stat, "st_blocks") else stat.st_size)
        return max(0, asset.size - min(max(sizes, default=0), asset.size))

    def locations(self) -> dict[str, str]:
        return {asset.name: str(self.shared_path(asset)) for asset in SHARED_ASSETS}

    def profile_ready(self, model_id: str) -> bool:
        """An active Quick Setup is complete only with verified canonical bindings."""
        profile = next((p for p in self.profiles() if p.model_id == model_id), None)
        if not profile:
            return False
        for asset in SHARED_ASSETS:
            if asset.name.startswith("tokenizer/"):
                container = profile.env.get("HALOGEN_TOKENIZER", "") + "/" + asset.name.split("/", 1)[1]
            elif "ngram" in asset.name:
                container = profile.env.get("HALOGEN_NGRAM_TABLE", "")
            else:
                container = profile.env.get("HALOGEN_VISION_TOWER", "")
                if container == "0":
                    continue
            path = host_path(profile, container)
            if path != self.shared_path(asset) or not self.cached(path, asset):
                return False
        return True

    def profiles(self):
        for path in sorted(self.quadlet_dir.glob("halogen-*.container")):
            try:
                profile = parse_quadlet(path.read_text(encoding="utf-8"), path)
            except (OSError, ProfileError):
                continue
            if profile.model_id in KNOWN_MODELS:
                yield profile

    def candidates(self, asset) -> list[Path]:
        paths = [self.shared_path(asset), self.staging_root(asset) / asset.name]
        for profile in self.profiles():
            if asset.name.startswith("tokenizer/"):
                container = profile.env.get("HALOGEN_TOKENIZER", "/models/tokenizer") + "/" + asset.name.split("/", 1)[1]
            elif "ngram" in asset.name:
                container = profile.env.get("HALOGEN_NGRAM_TABLE", "/models/" + asset.name)
            else:
                container = profile.env.get("HALOGEN_VISION_TOWER", "/models/" + asset.name)
            path = host_path(profile, container)
            if path and path not in paths:
                paths.append(path)
        return paths

    @staticmethod
    def fingerprint(path: Path):
        stat = path.stat()
        return [str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]

    def cached(self, path, asset) -> bool:
        try:
            return (path.is_file() and path.stat().st_size == asset.size
                    and self.cache.get(str(path.resolve())) == {
                        "fingerprint": self.fingerprint(path), "digest": asset.digest,
                        "algorithm": asset.algorithm})
        except OSError:
            return False

    async def verify(self, path, asset) -> bool:
        if self.cached(path, asset):
            return True
        try:
            before = self.fingerprint(path)
            if not path.is_file() or before[3] != asset.size:
                return False
        except OSError:
            return False
        stop = threading.Event()

        def compute():
            digest = hashlib.sha1() if asset.algorithm == "git-sha1" else hashlib.sha256()
            if asset.algorithm == "git-sha1":
                digest.update(f"blob {asset.size}\0".encode())
            with path.open("rb") as handle:
                while not stop.is_set():
                    chunk = handle.read(8 * 1024 * 1024)
                    if not chunk:
                        return digest.hexdigest()
                    digest.update(chunk)
            return None

        task = asyncio.create_task(asyncio.to_thread(compute))
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            stop.set()
            await task
            raise
        if result != asset.digest or before != self.fingerprint(path):
            return False
        self.cache[str(path.resolve())] = {
            "fingerprint": before, "digest": asset.digest, "algorithm": asset.algorithm}
        atomic_json(self.cache_path, self.cache)
        return True

    def entries(self, variant=None, *, checkpoint=None, checkpoint_root=None):
        if checkpoint is not None:
            dest = checkpoint_root / checkpoint.name
            yield checkpoint, dest, [dest, self.staging_root(checkpoint, checkpoint_root=checkpoint_root) / checkpoint.name]
        if variant:
            asset = variant_info(variant)["checkpoint"]
            yield asset, self.models_root / variant / asset.name, [self.models_root / variant / asset.name,
                                                                 self.staging_root(asset, variant) / asset.name]
        for asset in SHARED_ASSETS:
            yield asset, self.shared_path(asset), self.candidates(asset)

    def plan(self, variant=None, *, checkpoint=None, checkpoint_root=None) -> dict:
        files = []
        for asset, dest, candidates in self.entries(variant, checkpoint=checkpoint, checkpoint_root=checkpoint_root):
            if self.validate_path:
                self.validate_path(str(dest))
            present = next((p for p in candidates if p.is_file() and p.stat().st_size == asset.size), None)
            verified = next((p for p in candidates if self.cached(p, asset)), None)
            source = verified or present
            status = "verified" if verified else "unverified" if present else "download"
            files.append({"name": asset.name, "path": str(dest), "source_path": str(source) if source else None,
                          "status": status, "size": asset.size, "download_bytes": 0 if source else self.remaining_bytes(asset, variant, checkpoint_root),
                          "copy_bytes": asset.size if source and source.stat().st_dev != existing_parent(dest).stat().st_dev else 0,
                          "revision": asset.revision, "sha": asset.digest, "algorithm": asset.algorithm,
                          "url": f"https://huggingface.co/{asset.repo}/blob/{asset.revision}/{asset.name}"})
        return {"variant": variant, "files": files, "download_bytes": sum(f["download_bytes"] for f in files),
                "unverified_bytes": sum(f["size"] for f in files if f["status"] == "unverified"),
                "filesystems": self.space_plan(files), "shared_root": str(self.models_root / "shared" / "halogen-v2")}

    @staticmethod
    def space_plan(files):
        devices = {}
        for item in files:
            parent = existing_parent(Path(item["path"]))
            device = parent.stat().st_dev
            entry = devices.setdefault(device, {"path": str(parent), "free_bytes": shutil.disk_usage(parent).free,
                                                "required_bytes": 0, "reserve_bytes": SPACE_RESERVE})
            entry["required_bytes"] += item["download_bytes"] + item.get("copy_bytes", 0)
        for entry in devices.values():
            if not entry["required_bytes"]:
                entry["reserve_bytes"] = 0
            entry["sufficient"] = entry["free_bytes"] >= entry["required_bytes"] + entry["reserve_bytes"]
        return list(devices.values())

    @staticmethod
    def require_space(dest: Path, size: int):
        free = shutil.disk_usage(existing_parent(dest)).free
        if free < size + SPACE_RESERVE:
            raise AssetError(f"Insufficient disk space at {dest}: need {size + SPACE_RESERVE} bytes, have {free}")

    async def prepare(self, variant, download, log, *, checkpoint=None, checkpoint_root=None):
        """download(asset, staging_root) resumes exact pinned files. No backend work."""
        async with self.lock:
            prepared, requirements = [], []
            for asset, dest, candidates in self.entries(variant, checkpoint=checkpoint, checkpoint_root=checkpoint_root):
                if self.validate_path:
                    self.validate_path(str(dest))
                source = None
                for candidate in candidates:
                    if candidate.is_file():
                        log(f"Verify {asset.name}: {candidate}")
                        if await self.verify(candidate, asset):
                            source = candidate
                            break
                        log(f"Invalid or incomplete asset: {candidate}")
                prepared.append((asset, dest, source))
                cross_device = source and source.stat().st_dev != existing_parent(dest).stat().st_dev
                requirements.append({"path": str(dest), "download_bytes": 0 if source else self.remaining_bytes(asset, variant, checkpoint_root),
                                     "copy_bytes": asset.size if cross_device else 0})
            if any(not fs["sufficient"] for fs in self.space_plan(requirements)):
                raise AssetError("Insufficient disk space for the verified download plan and 5 GiB reserve")
            for asset, dest, source in prepared:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if source == dest:
                    log(f"Reuse verified {dest}")
                    continue
                if source:
                    temp = dest.with_name(dest.name + ".importing")
                    temp.unlink(missing_ok=True)
                    try:
                        os.link(source, temp)
                        # A new name for the verified inode needs no second 48 GiB read.
                        if (self.cached(source, asset) and
                                self.fingerprint(source)[1:] == self.fingerprint(temp)[1:]):
                            self.cache[str(temp.resolve())] = {"fingerprint": self.fingerprint(temp),
                                                              "digest": asset.digest, "algorithm": asset.algorithm}
                    except OSError:
                        self.require_space(dest, asset.size)
                        stop = threading.Event()
                        def copy_file():
                            with source.open("rb") as reader, temp.open("wb") as writer:
                                while not stop.is_set():
                                    chunk = reader.read(8 * 1024 * 1024)
                                    if not chunk:
                                        writer.flush()
                                        os.fsync(writer.fileno())
                                        return
                                    writer.write(chunk)
                        task = asyncio.create_task(asyncio.to_thread(copy_file))
                        try:
                            await asyncio.shield(task)
                        except asyncio.CancelledError:
                            stop.set()
                            await task
                            raise
                    if not await self.verify(temp, asset):
                        raise AssetError(f"Shared import checksum mismatch: {asset.name}")
                    await self.promote_verified(temp, dest)
                    log(f"Shared asset ready: {dest}")
                else:
                    self.require_space(dest, self.remaining_bytes(asset, variant, checkpoint_root))
                    # Keep Hugging Face's local-dir resume metadata after cancellation.
                    stage = self.staging_root(asset, variant, checkpoint_root)
                    stage.mkdir(parents=True, exist_ok=True)
                    await download(asset, stage)
                    downloaded = stage / asset.name
                    if not await self.verify(downloaded, asset):
                        # Do not let HF consider a complete corrupt file up to date next time.
                        downloaded.unlink(missing_ok=True)
                        metadata = stage / ".cache" / "huggingface" / "download" / (asset.name + ".metadata")
                        metadata.unlink(missing_ok=True)
                        raise AssetError(f"Checksum mismatch: {asset.name}; retry Quick Setup")
                    await self.promote_verified(downloaded, dest)
                # Record the final pathname fingerprint (promotion preserves inode/mtime).
                self.cache[str(dest.resolve())] = {"fingerprint": self.fingerprint(dest),
                                                    "digest": asset.digest, "algorithm": asset.algorithm}
                atomic_json(self.cache_path, self.cache)
            return self.locations()

    async def promote_verified(self, source, dest):
        task = asyncio.create_task(asyncio.to_thread(self.promote, source, dest))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    @staticmethod
    def promote(source: Path, dest: Path):
        # Persist data and the directory entry before saving its verification.
        with source.open("r+b" if os.name == "nt" else "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(source, dest)
        if os.name == "posix":
            directory = os.open(dest.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    async def validate_start(self, model_id):
        """Managed shared bindings are offline: fail before stopping a backend."""
        variant = model_id.removeprefix("halogen-")
        profile = next((p for p in self.profiles() if p.model_id == model_id), None)
        swift = variant in SWIFT_VARIANTS
        if not swift and (not profile or not profile.env.get("HALOGEN_NGRAM_TABLE", "").startswith("/shared/")):
            return
        if not profile:
            raise AssetError("Swift profile missing; repair it with Quick Setup")
        if model_id == "qwen3.8-flash" and profile.env.get("HALOGEN_CHECKPOINT") == "/models/qwen38-flash-next-ht43.hgn":
            from official_checkpoints import OFFICIAL_CHECKPOINTS
            from semver import version_tuple
            if (version_tuple(profile.image.rsplit(":", 1)[-1]) or ()) < (0, 16, 0):
                raise AssetError("HT43 requires Halogen 0.16.0 or newer")
            checkpoint = OFFICIAL_CHECKPOINTS["ht43"]["asset"]
            path = host_path(profile, profile.env["HALOGEN_CHECKPOINT"])
            if path is None or not await self.verify(path, checkpoint):
                raise AssetError("Missing or invalid HT43 checkpoint. Repair through the checkpoint selector (no startup download).")
        from semver import version_tuple
        if swift and (version_tuple(profile.image.rsplit(":", 1)[-1]) or (0, 0, 0)) < version_tuple(SWIFT_ENGINE_VERSION):
            raise AssetError("Swift requires Halogen 0.15.1 or newer; repair with Quick Setup")
        if swift and profile.env.get("HALOGEN_DOWNLOAD"):
            raise AssetError("Swift must not set HALOGEN_DOWNLOAD; repair with Quick Setup")
        checkpoint = variant_info(variant)["checkpoint"] if swift else None
        assets = ([checkpoint] if swift else []) + list(SHARED_ASSETS)
        for asset in assets:
            if asset == checkpoint:
                container = profile.env.get("HALOGEN_CHECKPOINT", "")
            elif asset.name.startswith("tokenizer/"):
                container = profile.env.get("HALOGEN_TOKENIZER", "") + "/" + asset.name.split("/", 1)[1]
            elif "ngram" in asset.name:
                container = profile.env.get("HALOGEN_NGRAM_TABLE", "")
            else:
                container = profile.env.get("HALOGEN_VISION_TOWER", "")
                if container == "0" and not swift:
                    continue
            path = host_path(profile, container)
            if path is None or not await self.verify(path, asset):
                raise AssetError(f"Missing or invalid shared/model asset: {asset.name}. Repair with Quick Setup (no startup download).")
