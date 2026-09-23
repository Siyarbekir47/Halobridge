"""Typed Halogen backend profiles: allowlisted env fields, quadlet rendering, import.

A profile is the desired state for one halogen backend (one quadlet file).
Rendering is deterministic and importing a rendered quadlet round-trips to the
same profile. Safety rules enforced here:

- Every environment variable must appear in ENV_FIELDS with a value inside its
  declared type and range. Unknown variables are rejected, never passed through.
- Host paths are validated separately against configured roots (see
  halogen_deploy); nothing here accepts arbitrary shell text.
- Values are restricted to a safe character set; no quoting or shell semantics.
- Container/service names are always derived from the profile id
  (``halogen-<id>``), so a profile can never reference an arbitrary unit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from semver import normalize_version

PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9 ._:/=@+-]+$")
IN_CONTAINER_PATH_RE = re.compile(r"^/[A-Za-z0-9 ._/@+-]+$")
REPO_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

CONTAINER_MODELS_PATH = "/models"
CONTAINER_CACHE_PATH = "/cache"
CONTAINER_API_PORT = 8731


@dataclass(frozen=True)
class FieldSpec:
    kind: str  # int | float | flag | enum | path | repo | text
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None


ENV_FIELDS: dict[str, FieldSpec] = {
    "HALOGEN_MODEL_ID": FieldSpec("text"),
    "HALOGEN_CHECKPOINT": FieldSpec("path"),
    "HALOGEN_TOKENIZER": FieldSpec("path"),
    "HALOGEN_VISION_TOWER": FieldSpec("text"),
    "HALOGEN_MTP_HEAD": FieldSpec("path"),
    "HALOGEN_DOWNLOAD": FieldSpec("repo"),
    "HALOGEN_CK_OVERLAY": FieldSpec("text"),
    "HALOGEN_TEMPLATE_UNCHECKED": FieldSpec("flag"),
    "HALOGEN_API_PORT": FieldSpec("int", 1, 65535),
    "HALOGEN_KEEPALIVE_TIMEOUT": FieldSpec("int", 1, 86400),
    "HALOGEN_SSE_KEEPALIVE_S": FieldSpec("int", 0, 3600),
    "HALOGEN_QUEUE_TIMEOUT": FieldSpec("int", 1, 86400),
    "HALOGEN_TEMPERATURE": FieldSpec("float", 0, 2),
    "HALOGEN_TOP_P": FieldSpec("float", 0, 1),
    "HALOGEN_TOP_K": FieldSpec("int", 0, 1000),
    "HALOGEN_MIN_P": FieldSpec("float", 0, 1),
    "HALOGEN_PRESENCE_PENALTY": FieldSpec("float", -2, 2),
    "HALOGEN_FREQUENCY_PENALTY": FieldSpec("float", -2, 2),
    "HALOGEN_MAX_TOKENS_CAP": FieldSpec("int", 1, 1_048_576),
    "HALOGEN_MAX_TOKENS_DEFAULT": FieldSpec("int", 1, 1_048_576),
    "HALOGEN_REASONING_EFFORT": FieldSpec(
        "enum", choices=("minimal", "low", "medium", "high", "xhigh")
    ),
    "HALOGEN_ENABLE_THINKING": FieldSpec("flag"),
    "HALOGEN_MAX_THINKING_TOKENS": FieldSpec("int", 0, 1_048_576),
    "HALOGEN_THINKING_ANSWER_ROOM": FieldSpec("int", 0, 1_048_576),
    "HALOGEN_DRAFTER_DEFAULT": FieldSpec("flag"),
    "HALOGEN_PLD": FieldSpec("text"),
    "HALOGEN_SPEC_ADAPT": FieldSpec("text"),
    "HALOGEN_GRAMMAR": FieldSpec("flag"),
    "HALOGEN_VISION_MAX_PIXELS": FieldSpec("int", 65_536, 16_777_216),
    "HALOGEN_KV_SLOTS": FieldSpec("int", 1, 64),
    "HALOGEN_KV_POOL_POSITIONS": FieldSpec("int", 1024, 1_048_576),
    "HALOGEN_KV_POOL_FIT": FieldSpec("flag"),
    "HALOGEN_CTX": FieldSpec("int", 1024, 1_048_576),
    "HALOGEN_ADMIT_CHUNK": FieldSpec("int", -1, 1_048_576),
    "HALOGEN_MAX_TOK": FieldSpec("int", 512, 65536),
    "HALOGEN_HOST_RESERVE_GIB": FieldSpec("int", 0, 4096),
    "HALOGEN_CACHE_ENTRIES": FieldSpec("int", 1, 1024),
    "HALOGEN_CACHE_SNAP3": FieldSpec("flag"),
    "HALOGEN_CACHE_FULL": FieldSpec("flag"),
    "HALOGEN_CACHE_INPLACE": FieldSpec("flag"),
    "HALOGEN_CACHE_DIR": FieldSpec("path"),
    "HALOGEN_CACHE_DISK_GIB": FieldSpec("int", 0, 100_000),
    "HALOGEN_CACHE_PRUNE_OLD": FieldSpec("flag"),
    "HALOGEN_GGUF_CACHE": FieldSpec("text"),
    "HALOGEN_GGUF_THREADS": FieldSpec("int", 1, 64),
}


class ProfileError(ValueError):
    """Raised for an invalid profile; the message names the offending field."""


@dataclass
class Profile:
    profile_id: str
    image: str
    volumes: list[tuple[str, str, str]]  # (host_path, container_path, mode)
    host_port: int
    env: dict[str, str] = field(default_factory=dict)
    quadlet_path: Path | None = None

    @property
    def container_name(self) -> str:
        return f"halogen-{self.profile_id}"

    @property
    def service_name(self) -> str:
        return f"{self.container_name}.service"

    @property
    def model_id(self) -> str:
        return self.env.get("HALOGEN_MODEL_ID", self.profile_id)

    @property
    def downloads_weights(self) -> bool:
        return bool(self.env.get("HALOGEN_DOWNLOAD"))


def validate_profile(profile: Profile) -> list[str]:
    """Return human-readable validation errors; empty means valid."""
    errors: list[str] = []
    if not PROFILE_ID_RE.match(profile.profile_id):
        errors.append("profile_id: use lowercase letters, numbers, and hyphens (1–32)")
    parts = profile.image.rsplit(":", 1)
    if len(parts) != 2 or not parts[1] or ":" in parts[0]:
        errors.append("image: must be 'repository:tag'")
    elif normalize_version(parts[1]) is None:
        errors.append("image: tag must be a stable version X.Y.Z (not latest)")
    if not 1024 <= profile.host_port <= 65535:
        errors.append("host_port: must be between 1024 and 65535")
    container_paths = [v[1] for v in profile.volumes]
    if CONTAINER_MODELS_PATH not in container_paths:
        errors.append(f"volumes: models volume at {CONTAINER_MODELS_PATH} is missing")
    if CONTAINER_CACHE_PATH not in container_paths:
        errors.append(f"volumes: cache volume at {CONTAINER_CACHE_PATH} is missing")
    if len(set(container_paths)) != len(container_paths):
        errors.append("volumes: container paths must be unique")
    if profile.downloads_weights:
        models = next((v for v in profile.volumes if v[1] == CONTAINER_MODELS_PATH), None)
        if models is not None and ":ro" in f":{models[2]}":
            errors.append(
                "volumes: HALOGEN_DOWNLOAD requires a writable models volume (not ro)"
            )
    if "HALOGEN_MODEL_ID" not in profile.env:
        errors.append("env: HALOGEN_MODEL_ID is required")
    for key, value in profile.env.items():
        spec = ENV_FIELDS.get(key)
        if spec is None:
            errors.append(f"env: variable is not on the allowlist: {key}")
        elif not SAFE_VALUE_RE.match(value):
            errors.append(f"env {key}: invalid characters")
        else:
            error = _validate_value(value, spec)
            if error:
                errors.append(f"env {key}: {error}")
    default = profile.env.get("HALOGEN_MAX_TOKENS_DEFAULT")
    cap = profile.env.get("HALOGEN_MAX_TOKENS_CAP")
    if default and cap and int(default) > int(cap):
        errors.append("env: MAX_TOKENS_DEFAULT must not exceed MAX_TOKENS_CAP")
    return errors


def _validate_value(value: str, spec: FieldSpec) -> str | None:
    number: float
    if spec.kind == "int":
        try:
            number = int(value)
        except ValueError:
            return "integer expected"
    elif spec.kind == "float":
        try:
            number = float(value)
        except ValueError:
            return "number expected"
    else:
        number = 0.0
    if spec.kind in ("int", "float"):
        if spec.minimum is not None and number < spec.minimum:
            return f"Minimum is {spec.minimum:g}"
        if spec.maximum is not None and number > spec.maximum:
            return f"Maximum is {spec.maximum:g}"
    elif spec.kind == "flag":
        if value not in {"0", "1"}:
            return "0 or 1 expected"
    elif spec.kind == "enum":
        if spec.choices and value not in spec.choices:
            return "allowed: " + ", ".join(spec.choices)
    elif spec.kind == "path":
        if not IN_CONTAINER_PATH_RE.match(value) or value == "/":
            return "Container path expected (safe characters only)"
    elif spec.kind == "repo":
        if not REPO_ID_RE.match(value):
            return "'org/repo' expected"
    return None


def render_quadlet(profile: Profile) -> str:
    """Deterministic quadlet text for a profile."""
    lines = [
        "[Unit]",
        f"Description=Halogen backend profile {profile.profile_id}",
        "Wants=network-online.target",
        "After=network-online.target",
        "",
        "[Container]",
        f"Image={profile.image}",
        f"ContainerName={profile.container_name}",
        "Exec=all",
    ]
    for host_path, container_path, mode in profile.volumes:
        lines.append(f"Volume={host_path}:{container_path}:{mode}")
    lines += [
        "AddDevice=/dev/kfd",
        "AddDevice=/dev/dri",
        f"PublishPort=127.0.0.1:{profile.host_port}:{CONTAINER_API_PORT}",
    ]
    for key in sorted(profile.env):
        lines.append(f"Environment={key}={profile.env[key]}")
    lines += [
        "Ulimit=memlock=-1:-1",
        "PodmanArgs=--ipc=host",
        "PodmanArgs=--group-add=keep-groups",
        "",
        "[Service]",
        "Restart=on-failure",
        "RestartSec=10",
        "TimeoutStartSec=300",
        "TimeoutStopSec=120",
        "",
    ]
    return "\n".join(lines)


def parse_quadlet(text: str, quadlet_path: Path | None = None) -> Profile:
    """Parse a quadlet file into a profile. Raises ProfileError on anything odd."""
    section: str | None = None
    image = ""
    container_name = ""
    volumes: list[tuple[str, str, str]] = []
    host_port = 0
    env: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if section == "Container":
            if key == "Image":
                image = value
            elif key == "ContainerName":
                container_name = value
            elif key == "Volume":
                parts = value.split(":")
                if len(parts) >= 2:
                    mode = parts[2] if len(parts) > 2 else "rw"
                    volumes.append((parts[0], parts[1], mode))
            elif key == "PublishPort":
                parts = value.split(":")
                try:
                    host_port = int(parts[-2] if len(parts) > 1 else parts[0])
                except ValueError:
                    raise ProfileError(f"Invalid PublishPort: {value}")
            elif key == "Environment":
                if "=" not in value:
                    raise ProfileError(f"Invalid Environment: {value}")
                env_key, env_value = value.split("=", 1)
                env[env_key] = env_value
    if not image:
        raise ProfileError("Image is missing")
    if not container_name.startswith("halogen-"):
        raise ProfileError(f"ContainerName must start with 'halogen-': {container_name}")
    profile_id = container_name[len("halogen-"):]
    if not PROFILE_ID_RE.match(profile_id):
        raise ProfileError(f"Invalid profile ID in ContainerName: {profile_id}")
    return Profile(
        profile_id=profile_id,
        image=image,
        volumes=volumes,
        host_port=host_port,
        env=env,
        quadlet_path=quadlet_path,
    )


def profile_to_dict(profile: Profile) -> dict:
    return {
        "profile_id": profile.profile_id,
        "container_name": profile.container_name,
        "service_name": profile.service_name,
        "model_id": profile.model_id,
        "image": profile.image,
        "host_port": profile.host_port,
        "volumes": [list(v) for v in profile.volumes],
        "env": dict(profile.env),
        "quadlet_path": str(profile.quadlet_path) if profile.quadlet_path else None,
        "downloads_weights": profile.downloads_weights,
    }


OFFICIAL_IMAGE_REPO = "ghcr.io/peonist-ai/halogen-flash-server"
OFFICIAL_WEIGHTS_REPO = "peonist-ai/halogen-qwen3.8-flash-next"
UNCENSORED_HF_REPO = "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF"
UNCENSORED_DEFAULT_GGUF_NAME = "Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf"
UNCENSORED_DEFAULT_OUTPUT = "qwen3.8-flash-uncensored.hgn"

_COMMON_RUNTIME_ENV = {
    "HALOGEN_CTX": "262144",
    "HALOGEN_KV_SLOTS": "2",
    "HALOGEN_KV_POOL_POSITIONS": "524288",
    "HALOGEN_MAX_TOK": "32768",
    "HALOGEN_MAX_TOKENS_DEFAULT": "8192",
    "HALOGEN_MAX_TOKENS_CAP": "65536",
    "HALOGEN_REASONING_EFFORT": "xhigh",
    "HALOGEN_QUEUE_TIMEOUT": "3600",
    "HALOGEN_HOST_RESERVE_GIB": "18",
    "HALOGEN_CACHE_DIR": "/cache",
    "HALOGEN_CACHE_DISK_GIB": "750",
    "HALOGEN_CACHE_PRUNE_OLD": "1",
}


def official_template(image_tag: str, models_root: Path, cache_root: Path) -> Profile:
    """Template matching a provisioned official setup (weights on disk).

    The models volume is read-only. To install from scratch instead, fill the
    download field (HALOGEN_DOWNLOAD); the volume must then be writable and
    the first start fetches ~118 GiB.
    """
    env = {
        "HALOGEN_MODEL_ID": "qwen3.8-flash",
        "HALOGEN_CHECKPOINT": "/models/qwen38-flash-next-w4b.hgn",
        "HALOGEN_TOKENIZER": "/models/tokenizer",
        "HALOGEN_VISION_TOWER": "/models/qwen38-flash-next-vision.hgn",
        **_COMMON_RUNTIME_ENV,
    }
    return Profile(
        profile_id="official",
        image=f"{OFFICIAL_IMAGE_REPO}:{image_tag}",
        volumes=[
            (str(models_root / "official"), CONTAINER_MODELS_PATH, "ro,Z"),
            (str(cache_root / "official"), CONTAINER_CACHE_PATH, "Z"),
        ],
        host_port=8831,
        env=env,
    )


def uncensored_template(image_tag: str, models_root: Path, cache_root: Path) -> Profile:
    """Template for the OrcaRouter uncensored model converted to .hgn.

    Mounts the official models directory read-only (shared tokenizer and
    vision tower) and the uncensored directory at /uncensored. Shares the
    host port with the official profile: only one backend runs at a time.
    """
    env = {
        "HALOGEN_MODEL_ID": "qwen3.8-flash-uncensored",
        "HALOGEN_CHECKPOINT": "/uncensored/qwen3.8-flash-uncensored.hgn",
        "HALOGEN_TOKENIZER": "/models/tokenizer",
        "HALOGEN_VISION_TOWER": "/models/qwen38-flash-next-vision.hgn",
        **_COMMON_RUNTIME_ENV,
    }
    return Profile(
        profile_id="uncensored",
        image=f"{OFFICIAL_IMAGE_REPO}:{image_tag}",
        volumes=[
            (str(models_root / "official"), CONTAINER_MODELS_PATH, "ro,Z"),
            (str(models_root / "uncensored"), "/uncensored", "ro,Z"),
            (str(cache_root / "uncensored"), CONTAINER_CACHE_PATH, "Z"),
        ],
        host_port=8831,
        env=env,
    )


def official_quick_template(image_tag: str, models_root: Path, cache_root: Path) -> Profile:
    """One-click official install: first start downloads weights into /models."""
    env = {
        "HALOGEN_MODEL_ID": "qwen3.8-flash",
        "HALOGEN_DOWNLOAD": OFFICIAL_WEIGHTS_REPO,
        **_COMMON_RUNTIME_ENV,
    }
    return Profile(
        profile_id="official",
        image=f"{OFFICIAL_IMAGE_REPO}:{image_tag}",
        volumes=[
            (str(models_root), CONTAINER_MODELS_PATH, "Z"),
            (str(cache_root), CONTAINER_CACHE_PATH, "Z"),
        ],
        host_port=8831,
        env=env,
    )


def uncensored_quick_template(image_tag: str, models_root: Path, cache_root: Path) -> Profile:
    """One-click uncensored install after the converted .hgn and tokenizer are present."""
    env = {
        "HALOGEN_MODEL_ID": "qwen3.8-flash-uncensored",
        "HALOGEN_CHECKPOINT": f"/models/{UNCENSORED_DEFAULT_OUTPUT}",
        "HALOGEN_TOKENIZER": "/models/tokenizer",
        **_COMMON_RUNTIME_ENV,
    }
    return Profile(
        profile_id="uncensored",
        image=f"{OFFICIAL_IMAGE_REPO}:{image_tag}",
        volumes=[
            (str(models_root / "uncensored"), CONTAINER_MODELS_PATH, "ro,Z"),
            (str(cache_root / "uncensored"), CONTAINER_CACHE_PATH, "Z"),
        ],
        host_port=8831,
        env=env,
    )


def custom_template(image_tag: str, models_root: Path, cache_root: Path) -> Profile:
    """Template for a user-provided model (GGUF or .hgn); no auto-download."""
    return Profile(
        profile_id="custom",
        image=f"{OFFICIAL_IMAGE_REPO}:{image_tag}",
        volumes=[
            (str(models_root / "custom"), CONTAINER_MODELS_PATH, "ro,Z"),
            (str(cache_root / "custom"), CONTAINER_CACHE_PATH, "Z"),
        ],
        host_port=8832,
        env={
            "HALOGEN_MODEL_ID": "custom-model",
            "HALOGEN_CHECKPOINT": "/models/model.gguf",
            **_COMMON_RUNTIME_ENV,
        },
    )
