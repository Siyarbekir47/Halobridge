"""NPU models, image-owned manifests, versioned assets and host checks.

The image's own file record is authoritative. Preparing a new record never
changes the old tree, and containers mount the prepared tree read-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath

from shared_assets import AssetError, SharedAssets, atomic_json, host_path

NPU_MODELS = {
    "decider-0.8b": "decision",
    "decider-4b": "decision",
    "qwen3-embedding-0.6b": "embedding",
    "qwen3-embedding-4b": "embedding",
    "qwen3-reranker-0.6b": "score",
    "qwen3-reranker-4b": "score",
    "qwen3guard-gen-0.6b": "classify",
    "qwen3.5-2b": "generate",
    "flux2-klein-4b": "image",
    "qwen3-asr-0.6b": "transcribe",
    "qwen3-asr-1.7b": "transcribe",
    "embeddinggemma-2": "embedding",
    "wemm-embedding-4b": "embedding",
    "qwen3guard-stream-0.6b": "guard",
    "d3-lite": "decision",
    "d3-nano": "decision",
}
CUSTOM_TASKS = {"decision", "embedding", "score", "classify", "classifier"}
MODEL_MIN_VERSION = {"flux2-klein-4b": (0, 17, 0),
                     "decider-4b": (0, 17, 4), "qwen3-embedding-4b": (0, 17, 4),
                     "qwen3-reranker-4b": (0, 17, 4),
                     "qwen3-asr-0.6b": (0, 18, 0), "qwen3-asr-1.7b": (0, 18, 0),
                     "embeddinggemma-2": (0, 18, 3), "wemm-embedding-4b": (0, 18, 3),
                     "qwen3guard-stream-0.6b": (0, 18, 3),
                     "d3-lite": (0, 18, 11), "d3-nano": (0, 18, 11)}
# The 0.18.1 image's catalog was verified byte-for-byte against 0.18.0.
# Only explicitly verified versions share a bundle; later images are read locally.
# The 0.18.12 and 0.18.13 bundles are distinct models.txt files from their GHCR images.
BUNDLED_CATALOG_ALIASES = {"0.18.1": "0.18.0"}
BUNDLED_VERSIONS = {"0.16.2", "0.17.2", "0.17.3", "0.17.4", "0.17.5", "0.18.0", "0.18.2", "0.18.11", "0.18.12", "0.18.13"} | set(BUNDLED_CATALOG_ALIASES)
TASK_ROUTES = {
    "decision": "/v1/chat/completions", "generate": "/v1/chat/completions",
    "embedding": "/v1/embeddings", "score": "/v1/rerank",
    "classify": "/v1/moderations",
    "image": "/v1/images/generations",
    "transcribe": "/v1/audio/transcriptions",
    "classifier": "/classify",
    "guard": "/v1/chat/completions",
}
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
PATH_RE = re.compile(r"^/models/[A-Za-z0-9_./-]+$")
REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
XRT_LIBS = ("libxrt_coreutil.so.2", "libxrt_core.so.2", "libxrt_driver_xdna.so.2")
DOC_URL = "https://github.com/peonist-ai/halogen-flash-server/blob/v0.18.13/docs/NPU.md"


def parse_models(value: str) -> list[str]:
    values = value.split(",")
    if not value or len(values) > 32:
        raise ValueError("NPU models: select 1 to 32 model names or /models directories")
    names, result = set(), []
    for item in values:
        if item in NPU_MODELS:
            name = item
        elif (PATH_RE.fullmatch(item) and ".." not in PurePosixPath(item).parts
              and not item.endswith("/") and MODEL_RE.fullmatch(PurePosixPath(item).name)
              and not item.startswith("/models/npu/") and item != "/models/npu"):
            name = PurePosixPath(item).name
            if name in NPU_MODELS:
                raise ValueError("NPU custom directory name conflicts with a built-in model")
        else:
            raise ValueError(f"NPU models: unsupported name or unsafe path: {item}")
        if name in names:
            raise ValueError(f"NPU models: duplicate model ID: {name}")
        names.add(name)
        result.append(item)
    if sum(NPU_MODELS.get(item) == "transcribe" for item in result) > 1:
        raise ValueError("NPU models: select only one speech-to-text model")
    return result


def profile_model_names(profile) -> set[str]:
    value = profile.env.get("HALOGEN_NPU_MODELS", "")
    return {PurePosixPath(item).name for item in parse_models(value)} if value else set()


@dataclass(frozen=True)
class NpuFile:
    model: str
    name: str
    repo: str
    revision: str
    size: int
    digest: str
    algorithm: str = "sha256"


class NpuManifest:
    def __init__(self, text: str):
        if len(text.encode()) > 2 * 1024 * 1024:
            raise AssetError("NPU manifest is too large")
        self.text = text
        self.key = hashlib.sha256(text.encode()).hexdigest()
        self.models, records = {}, []
        for line in text.splitlines():
            bits = line.split()
            if not bits or bits[0].startswith("#"):
                continue
            if bits[0] == "model" and len(bits) >= 3 and MODEL_RE.fullmatch(bits[1]):
                if bits[1] in self.models:
                    raise AssetError("Duplicate NPU manifest model")
                info = dict(part.split("=", 1) for part in bits[2:] if "=" in part)
                if info.get("task") not in TASK_ROUTES:
                    raise AssetError("Unsupported NPU manifest task")
                if info.get("repo") and not REPO_RE.fullmatch(info["repo"]):
                    raise AssetError("Unsafe NPU manifest repository")
                if info.get("revision") and not re.fullmatch(r"[0-9a-f]{40}", info["revision"]):
                    raise AssetError("NPU manifest must pin full commit revisions")
                self.models[bits[1]] = info
            elif bits[0] == "file" and len(bits) == 5:
                records.append(bits[1:])
            else:
                raise AssetError("Invalid NPU manifest line")
        self.assets, seen = [], set()
        for model, name, size, digest in records:
            path = PurePosixPath(name)
            if (model not in self.models or path.is_absolute() or ".." in path.parts
                    or not re.fullmatch(r"[A-Za-z0-9_./-]+", name) or name in {".", ""}
                    or (model, name) in seen or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise AssetError("Unsafe or duplicate NPU manifest file")
            try:
                count = int(size)
            except ValueError as exc:
                raise AssetError("Invalid NPU file size") from exc
            if not 0 < count < 64 * 1024 ** 3:
                raise AssetError("Invalid NPU file size")
            seen.add((model, name))
            info = self.models[model]
            self.assets.append(NpuFile(model, name, info.get("repo", ""),
                                       info.get("revision", ""), count, digest))
        if not self.models or not self.assets:
            raise AssetError("Empty NPU manifest")
        for model, info in self.models.items():
            own = info.get("devices", model)
            paths = {asset.name for asset in self.assets if asset.model == model}
            programs = {asset.name for asset in self.assets if asset.model == own}
            if (own not in self.models or not info.get("repo") or not info.get("revision")
                    or f"{model}.hnpw" not in paths
                    or "tokenizer/tokenizer.json" not in paths
                    or "devices/devices.hnpm" not in programs
                    or (info["task"] == "image" and "devices/taef2.safetensors" not in programs)
                    or (info["task"] == "transcribe" and (
                        not {"enc/weights.hnpw", "enc/devices.hnpm"}.issubset(paths)
                        or not any(p.startswith("enc/") and p.endswith(".elf") for p in paths)))
                    or not any(p.startswith("devices/") and p.endswith(".elf") for p in programs)):
                raise AssetError(f"Incomplete NPU manifest for {model}")

    def select(self, models: list[str], custom_bases=()) -> list[NpuFile]:
        wanted = set()
        devices = set()
        for model in models:
            if model.startswith("/"):
                continue
            if model not in self.models:
                raise AssetError(f"This image does not provide NPU model {model}")
            wanted.add(model)
            devices.add(self.models[model].get("devices", model))
        for base in custom_bases:
            if base not in self.models or self.models[base]["task"] not in CUSTOM_TASKS:
                raise AssetError("Unsupported custom NPU base model")
            devices.add(self.models[base].get("devices", base))
        return [asset for asset in self.assets if asset.model in wanted
                or (asset.model in devices and asset.name.startswith("devices/"))]


def bundled_manifest(version="0.18.0") -> NpuManifest:
    if version not in BUNDLED_VERSIONS:
        raise AssetError("No bundled NPU catalog for this engine version")
    catalog_version = BUNDLED_CATALOG_ALIASES.get(version, version)
    return NpuManifest(files("halobridge_data").joinpath(f"npu/models-{catalog_version}.txt").read_text(encoding="utf-8"))


async def check_runtime(run, image, mounts, custom_mounts=()):
    """Load the actual image against host XRT before interrupting requests."""
    argv = ["podman", "run", "--rm", "--network=none", "--entrypoint", "/bin/sh"]
    for host, target, mode in mounts:
        argv += ["--volume", f"{host}:{target}:{mode}"]
    argv += [image, "-c", 'out=$(/usr/local/bin/halogen-npu 2>&1); code=$?; printf "%s\\n" "$out"; test "$code" -eq 1']
    output = await run(*argv, timeout=120)
    if "usage: halogen-npu" not in output:
        raise AssetError("The NPU engine cannot load with this host's XRT libraries")
    bases = []
    for host, target, mode in custom_mounts:
        output = await run("podman", "run", "--rm", "--network=none", "--entrypoint",
                           "/usr/local/bin/halogen-npu",
                           *(arg for h, t, m in (*mounts, (host, target, mode))
                             for arg in ("--volume", f"{h}:{t}:{m}")), image, "probe", target, timeout=120)
        info = dict(part.split("=", 1) for part in output.split() if "=" in part)
        if info.get("base") not in NPU_MODELS or info.get("task") not in CUSTOM_TASKS:
            raise AssetError(f"Unsupported NPU fine-tune: {target}")
        bases.append(info["base"])
    return bases


def xrt_mounts(root: Path = Path("/")) -> list[tuple[str, str, str]]:
    """Resolve distro symlinks on the host; never relabel system libraries."""
    amd = root / "opt/xilinx/xrt"
    if all((amd / "lib" / name).is_file() for name in XRT_LIBS):
        # Links into a distro library directory need the per-file form below.
        if all((amd / "lib" / name).resolve().is_relative_to(amd.resolve()) for name in XRT_LIBS):
            return [(str(amd), "/opt/xilinx/xrt", "ro")]
    for lib in (root / "usr/lib/x86_64-linux-gnu", root / "usr/lib64", root / "usr/lib"):
        if all((lib / name).is_file() for name in XRT_LIBS):
            result = []
            for name in XRT_LIBS:
                source = str((lib / name).resolve())
                result += [(source, f"/opt/xilinx/xrt/lib/{name}", "ro"),
                           (source, "/" + str((lib / name).relative_to(root)), "ro")]
            return result
    return []


def system_mount_allowed(host: str, container: str, mode: str) -> bool:
    return (host, container, mode) in xrt_mounts()


def fabric_devices(root: Path):
    devices = sorted((root / "sys/class/drm").glob("card[0-9]*/device"))
    for path in (root / "sys/class/kfd/kfd/topology/nodes").glob("*/properties"):
        try:
            props = dict(line.split(maxsplit=1) for line in path.read_text().splitlines() if len(line.split()) == 2)
            if not re.fullmatch(r"1105\d\d", props.get("gfx_target_version", "")):
                continue
            render = root / "sys/class/drm" / ("renderD" + props.get("drm_render_minor", "")) / "device"
            domain, location = int(props.get("domain", "-1")), int(props.get("location_id", "-1"))
            address = f"{domain:04x}:{location >> 8:02x}:{(location >> 3) & 31:02x}.{location & 7:x}"
            for device in devices:
                if (render.exists() and device.resolve() == render.resolve()) or device.resolve().name == address:
                    return [device]
        except (OSError, ValueError):
            continue
    return devices


def host_status(root: Path = Path("/")) -> dict:
    errors = []
    node = root / "dev/accel/accel0"
    if root == Path("/") and not sys.platform.startswith("linux"):
        errors.append("NPU setup requires Linux")
    if not node.exists():
        errors.append("NPU device missing: /dev/accel/accel0; install/load amdxdna and NPU firmware")
    elif not os.access(node, os.R_OK | os.W_OK):
        errors.append("NPU device access denied; add the service user to the device's group and log in again")
    try:
        command = (root / "proc/cmdline").read_text().split()
    except OSError:
        command = []
    if any(value in command for value in ("amd_iommu=off", "iommu=off")):
        errors.append("IOMMU is disabled; remove amd_iommu=off / iommu=off (iommu=pt is compatible)")
    mounts = xrt_mounts(root)
    if not mounts:
        errors.append("XRT and its NPU plugin are missing; install the host's matching XRT libraries")
    clocks = []
    for device in fabric_devices(root):
        try:
            levels = [line for line in (device / "pp_dpm_fclk").read_text().splitlines() if line.strip()]
            perf = (device / "power_dpm_force_performance_level").read_text().strip()
        except OSError:
            continue
        held = perf == "high" or (perf == "manual" and bool(levels)
                                  and "*" in levels[-1] and sum("*" in line for line in levels) == 1)
        clocks.append({"path": str(device), "held": held})
    if not clocks or not all(clock["held"] for clock in clocks):
        errors.append("GPU fabric clock is not held; install and start halogen-fabric-clock.service before enabling NPU")
    return {"ready": not errors, "errors": errors, "device": str(node),
            "xrt_mounts": [list(mount) for mount in mounts], "fabric_clocks": clocks, "documentation": DOC_URL}


def host_setup(state_dir: Path) -> dict:
    directory = state_dir / "npu-host"
    directory.mkdir(parents=True, exist_ok=True)
    # Download upstream's unmodified host helpers at setup time; no vendor
    # executable code is redistributed with Halobridge.
    revision = "002589b11480cf3422f209b336c19dcc208fc69f"
    helpers = [("halogen-fabric-clock", "0dbd47c15f1002e0a8997ec7070e96f3e009125068eab8eccbbd49424a594f5a", "755", "/usr/local/sbin/"),
               ("halogen-fabric-clock.service", "5f9f1cc4c85a1c027e907aa355bbabc87d40eba544512b51146b543a49bf8a71", "644", "/etc/systemd/system/")]
    commands = []
    for name, digest, mode, destination in helpers:
        path = shlex.quote(str(directory / name))
        url = f"https://raw.githubusercontent.com/peonist-ai/halogen-flash-server/{revision}/deploy/host/{name}"
        commands.append(f"curl -fL {shlex.quote(url)} -o {path} && "
                        f"printf '%s  %s\\n' {digest} {path} | sha256sum -c - && "
                        f"sudo install -m {mode} {path} {destination}")
    commands += ["sudo systemctl daemon-reload", "sudo systemctl enable --now halogen-fabric-clock.service",
                 "/usr/local/sbin/halogen-fabric-clock status"]
    return {"commands": commands, "undo": ["sudo systemctl disable --now halogen-fabric-clock.service",
                                            "sudo /usr/local/sbin/halogen-fabric-clock release"]}


def patch_quadlet(content: bytes, *, models: str | None = None, shared_root: Path | None = None,
                  mounts=(), custom_mounts=()) -> bytes:
    """Change only NPU bindings, retaining the rest of an imported Quadlet."""
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    lines, out, section = content.splitlines(), [], ""
    additions = []
    if models is not None and models:
        parse_models(models)
        additions.append("Environment=HALOGEN_NPU_MODELS=" + models)
    if shared_root is not None:
        additions.append(f"Volume={shared_root}:/models/npu:ro,z")
    enabled = bool(models) if models is not None else b"HALOGEN_NPU_MODELS=" in content
    if enabled:
        additions.append("AddDevice=/dev/accel/accel0")
        additions += [f"Volume={host}:{target}:{mode}" for host, target, mode in (*mounts, *custom_mounts)]
    destinations = {target for _, target, _ in (*mounts, *custom_mounts)}
    inserted = False
    def insert():
        nonlocal inserted
        if not inserted:
            out.extend(line.encode() for line in additions)
            inserted = True
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(b"["):
            if section == "Container":
                insert()  # Child mounts must follow an existing /models:Z mount.
            section = stripped[1:-1].decode()
        if section == "Container":
            if stripped.startswith(b"Environment=HALOGEN_NPU_MODELS=") and models is not None:
                continue
            if stripped == b"AddDevice=/dev/accel/accel0":
                continue
            if stripped.startswith(b"Volume="):
                parts = stripped[len(b"Volume="):].decode().rsplit(":", 2)
                if len(parts) == 3:
                    target = PurePosixPath(parts[1])
                    xrt = str(target) == "/opt/xilinx/xrt" or (
                        target.name in XRT_LIBS and str(target.parent) in
                        {"/opt/xilinx/xrt/lib", "/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"})
                    if parts[1] in destinations or parts[1] == "/models/npu" or (xrt and (mounts or models == "")):
                        continue
        out.append(line)
    if section == "Container":
        insert()
    if not inserted:
        raise AssetError("NPU target Quadlet has no Container section")
    return newline.join(out) + newline


class NpuSelection(SharedAssets):
    def __init__(self, owner, manifest, models, profiles, custom_bases=()):
        super().__init__(owner.models_root, owner.quadlet_dir, owner.state_dir, owner.validate_path)
        self.manifest, self.selected = manifest, manifest.select(models, custom_bases)
        self.source_profiles = profiles
        self.root = self.models_root / "shared" / "halogen-npu" / manifest.key
        self.cache, self.cache_path = owner.verifier.cache, owner.verifier.cache_path

    def staging_root(self, asset, variant=None, checkpoint_root=None):
        return self.root / ".downloads" / asset.model

    def locations(self):
        return {"root": str(self.root)}

    async def prepare(self, *args, **kwargs):
        locations = await super().prepare(*args, **kwargs)
        atomic_json(self.root / "halobridge-manifest.json", {"text": self.manifest.text,
                    "sha256": self.manifest.key})
        return locations

    def entries(self, variant=None, **kwargs):
        for asset in self.selected:
            dest = self.root / asset.model / asset.name
            candidates = [dest]
            candidates += sorted((self.root.parent).glob(f"*/{asset.model}/{asset.name}"))
            for profile in self.source_profiles:
                source = host_path(profile, f"/models/npu/{asset.model}/{asset.name}")
                if source:
                    candidates.append(source)
            candidates.append(self.staging_root(asset) / asset.name)
            yield asset, dest, list(dict.fromkeys(candidates))

    def plan(self, *args, **kwargs):
        value = super().plan(*args, **kwargs)
        value["shared_root"] = str(self.root)
        for item, asset in zip(value["files"], self.selected):
            item["name"] = asset.model + "/" + asset.name
        return value


class NpuAssets:
    def __init__(self, verifier: SharedAssets, state_dir: Path):
        self.verifier = verifier
        self.models_root, self.quadlet_dir = verifier.models_root, verifier.quadlet_dir
        self.validate_path, self.state_dir = verifier.validate_path, state_dir

    def profiles(self):
        from profiles import parse_quadlet, ProfileError
        for path in sorted(self.quadlet_dir.glob("halogen-*.container")):
            try:
                yield parse_quadlet(path.read_text(encoding="utf-8"), path)
            except (OSError, ProfileError):
                continue

    async def manifest(self, image, run=None, *, refresh=False):
        from semver import normalize_version
        version = normalize_version(image.rsplit(":", 1)[-1])
        if not version or not image.startswith("ghcr.io/peonist-ai/halogen-flash-server:"):
            raise AssetError("NPU assets require a stable official Halogen image")
        cache = self.state_dir / "npu-manifests" / (version + ".json")
        if not refresh:
            try:
                return NpuManifest(json.loads(cache.read_text(encoding="utf-8"))["text"])
            except (OSError, ValueError, KeyError):
                pass
            if version in BUNDLED_VERSIONS:
                return bundled_manifest(version)
        if run is None:
            raise AssetError(f"NPU catalog for Halogen {version} is unavailable; use NPU Setup to read the installed container image")
        text = await run("podman", "run", "--rm", "--pull=never", "--network=none", "--entrypoint", "/bin/cat",
                         image, "/opt/halogen/npu/models.txt", timeout=120)
        manifest = NpuManifest(text)
        atomic_json(cache, {"image": image, "text": text, "sha256": manifest.key})
        return manifest

    def selection(self, manifest, models, profiles, custom_bases=()):
        return NpuSelection(self, manifest, models, profiles, custom_bases)

    async def validate_profile_start(self, profile):
        models = parse_models(profile.env["HALOGEN_NPU_MODELS"])
        state = host_status()
        if not state["ready"]:
            raise AssetError("; ".join(state["errors"]))
        root = host_path(profile, "/models/npu")
        if root is None:
            raise AssetError("NPU shared mount is missing; use NPU Setup to prepare this profile")
        try:
            record = json.loads((root / "halobridge-manifest.json").read_text(encoding="utf-8"))
            manifest = NpuManifest(record["text"])
            if root.name != manifest.key:
                raise AssetError("NPU shared manifest does not match its versioned directory")
        except FileNotFoundError:
            manifest = await self.manifest(profile.image)
            # Legacy installs may contain just one fine-tune's programs.
            # Its image-owned record remains authoritative during rollback.
            legacy = True
        except (OSError, ValueError, KeyError) as exc:
            raise AssetError("Invalid NPU shared manifest; repair with NPU Setup") from exc
        else:
            legacy = False
        bases = [model for model, info in manifest.models.items() if info["task"] in CUSTOM_TASKS] if any(m.startswith("/") for m in models) else []
        for model in models:
            if model.startswith("/"):
                source = host_path(profile, model)
                if source is None or not all((source / name).is_file() for name in
                                             ("config.json", "model.safetensors", "tokenizer.json")):
                    raise AssetError(f"NPU fine-tune missing or incomplete: {model}; repair with NPU Setup")
        if legacy and bases:
            bases = [base for base in bases if (root / manifest.models[base].get("devices", base)
                                               / "devices/devices.hnpm").is_file()]
            if not bases:
                raise AssetError("NPU fine-tune device programs missing; repair with NPU Setup")
        for asset in manifest.select(models, bases):
            if not await self.verifier.verify(root / asset.model / asset.name, asset):
                raise AssetError(f"NPU file missing or invalid: {asset.model}/{asset.name}; repair with NPU Setup")
