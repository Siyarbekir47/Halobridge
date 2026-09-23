"""Configuration layer: one TOML file, safe defaults, host-agnostic.

Everything that used to be hardcoded per-host (bind addresses, ports, backend
URL, directories, image repository, timeouts) is read here. Defaults let a
minimal setup run with no config file at all.

The product/brand name is isolated in ``APP`` so the trademark-safe rename is a
single edit. See docs/open-source-plan.md §0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any


# Product name. This drives the config filename, state dir and env-var prefix.
# The project is a companion for halogen, not a Peonist product.
APP = "halobridge"

CONFIG_ENV = f"{APP.upper()}_CONFIG"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / APP / f"{APP}.toml"
DEFAULT_STATE_DIR = Path.home() / ".local/state" / APP
LEGACY_STATE_DIR = Path.home() / ".local/state/halogen-dashboard"


class ConfigError(ValueError):
    """Raised for an invalid configuration value; message names the field."""


def _expand(value: str) -> Path:
    return Path(value).expanduser()


def _as_bool(section: dict[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _as_int(section: dict[str, Any], key: str, default: int, *, minimum: int = 1) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}")
    return value


def _as_str(section: dict[str, Any], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value.strip()


def _as_optional_str(section: dict[str, Any], key: str) -> str:
    """A string that may be empty or absent (e.g. an optional public URL)."""
    value = section.get(key, "")
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
    return value.strip()


def _as_str_list(section: dict[str, Any], key: str, default: list[str]) -> list[str]:
    value = section.get(key, default)
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise ConfigError(f"{key} must be a list of strings")
    return [v.strip() for v in value]


@dataclass
class RouterConfig:
    bind: list[str] = field(default_factory=lambda: ["127.0.0.1"])
    port: int = 8731
    backend_url: str = "http://127.0.0.1:8831"
    drain_timeout_s: int = 7200
    start_timeout_s: int = 600
    stop_timeout_s: int = 180
    gtt_path: str = "/sys/class/drm/card0/device/mem_info_gtt_used"
    gtt_limit_bytes: int = 1024 * 1024 * 1024

    @classmethod
    def from_section(cls, s: dict[str, Any]) -> "RouterConfig":
        return cls(
            bind=_as_str_list(s, "bind", ["127.0.0.1"]),
            port=_as_int(s, "port", 8731, minimum=1),
            backend_url=_as_str(s, "backend_url", "http://127.0.0.1:8831").rstrip("/"),
            drain_timeout_s=_as_int(s, "drain_timeout_s", 7200),
            start_timeout_s=_as_int(s, "start_timeout_s", 600),
            stop_timeout_s=_as_int(s, "stop_timeout_s", 180),
            gtt_path=_as_str(s, "gtt_path", "/sys/class/drm/card0/device/mem_info_gtt_used"),
            gtt_limit_bytes=_as_int(s, "gtt_limit_bytes", 1024 * 1024 * 1024, minimum=0),
        )


@dataclass
class ModelsConfig:
    auto_discover: bool = True
    quadlet_dir: Path = field(default_factory=lambda: Path.home() / ".config/containers/systemd")
    # Explicit override: model_id -> service name. Used when auto_discover is off
    # or merged on top of discovery for models not defined as quadlets.
    explicit: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_section(cls, s: dict[str, Any]) -> "ModelsConfig":
        explicit = s.get("explicit", {})
        if not isinstance(explicit, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
            for k, v in explicit.items()
        ):
            raise ConfigError("models.explicit must be a model-to-service table")
        return cls(
            auto_discover=_as_bool(s, "auto_discover", True),
            quadlet_dir=_expand(_as_str(s, "quadlet_dir", str(Path.home() / ".config/containers/systemd"))),
            explicit={k.strip(): v.strip() for k, v in explicit.items()},
        )


@dataclass
class UpdatesConfig:
    enabled: bool = True
    image_repo: str = "ghcr.io/peonist-ai/halogen-flash-server"
    tag_source: str = "github"
    github_repo: str = "peonist-ai/halogen-flash-server"
    check_interval_h: int = 6
    backup_dir: Path = field(default_factory=lambda: Path.home() / "halogen/backups")
    backup_keep: int = 5

    @classmethod
    def from_section(cls, s: dict[str, Any]) -> "UpdatesConfig":
        tag_source = _as_str(s, "tag_source", "github")
        if tag_source not in {"github", "registry"}:
            raise ConfigError("updates.tag_source must be 'github' or 'registry'")
        return cls(
            enabled=_as_bool(s, "enabled", True),
            image_repo=_as_str(s, "image_repo", "ghcr.io/peonist-ai/halogen-flash-server"),
            tag_source=tag_source,
            github_repo=_as_str(s, "github_repo", "peonist-ai/halogen-flash-server"),
            check_interval_h=_as_int(s, "check_interval_h", 6),
            backup_dir=_expand(_as_str(s, "backup_dir", str(Path.home() / "halogen/backups"))),
            backup_keep=_as_int(s, "backup_keep", 5, minimum=0),
        )


@dataclass
class DashboardConfig:
    state_dir: Path = field(default_factory=lambda: DEFAULT_STATE_DIR)
    retention_days: int = 365
    gpu_card: str = "auto"
    public_endpoint: str = ""

    @classmethod
    def from_section(cls, s: dict[str, Any]) -> "DashboardConfig":
        return cls(
            state_dir=_expand(_as_str(s, "state_dir", str(DEFAULT_STATE_DIR))),
            retention_days=_as_int(s, "retention_days", 365),
            gpu_card=_as_str(s, "gpu_card", "auto"),
            # Optional externally reachable API base shown in the dashboard.
            public_endpoint=_as_optional_str(s, "public_endpoint").rstrip("/"),
        )


@dataclass
class DeployConfig:
    """Profile deployment: writing quadlet files from the dashboard.

    allowed_roots bounds every host path a profile may mount; everything
    outside these roots is rejected before any file is written.
    """

    enabled: bool = True
    allowed_roots: list[Path] = field(default_factory=lambda: [Path.home()])
    models_root: Path = field(default_factory=lambda: Path.home() / "halogen/models")
    cache_root: Path = field(default_factory=lambda: Path.home() / "halogen/cache")

    @classmethod
    def from_section(cls, s: dict[str, Any]) -> "DeployConfig":
        raw_roots = s.get("allowed_roots") or [str(Path.home())]
        if not isinstance(raw_roots, list) or not raw_roots:
            raise ConfigError("deploy.allowed_roots must be a list of paths")
        return cls(
            enabled=_as_bool(s, "enabled", True),
            allowed_roots=[_expand(str(r)) for r in raw_roots],
            models_root=_expand(
                _as_str(s, "models_root", str(Path.home() / "halogen/models"))
            ),
            cache_root=_expand(
                _as_str(s, "cache_root", str(Path.home() / "halogen/cache"))
            ),
        )


@dataclass
class SecurityConfig:
    auth_token: str = ""
    allow_install: bool = True

    @classmethod
    def from_section(cls, s: dict[str, Any]) -> "SecurityConfig":
        return cls(
            auth_token=_as_optional_str(s, "auth_token"),
            allow_install=_as_bool(s, "allow_install", True),
        )


@dataclass
class Config:
    router: RouterConfig = field(default_factory=RouterConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    updates: UpdatesConfig = field(default_factory=UpdatesConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    source: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: Path | None = None) -> "Config":
        for unknown in set(data) - {
            "router",
            "models",
            "updates",
            "dashboard",
            "deploy",
            "security",
        }:
            raise ConfigError(f"Unknown configuration section: {unknown}")
        return cls(
            router=RouterConfig.from_section(data.get("router", {})),
            models=ModelsConfig.from_section(data.get("models", {})),
            updates=UpdatesConfig.from_section(data.get("updates", {})),
            dashboard=DashboardConfig.from_section(data.get("dashboard", {})),
            deploy=DeployConfig.from_section(data.get("deploy", {})),
            security=SecurityConfig.from_section(data.get("security", {})),
            source=source,
        )


def resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    env = os.environ.get(CONFIG_ENV)
    if env:
        return Path(env).expanduser()
    return DEFAULT_CONFIG_PATH


def load(path: str | os.PathLike[str] | None = None) -> Config:
    """Load config from an explicit path, $APP_CONFIG, or the default location.

    A missing file yields safe defaults. A malformed file or invalid value
    raises ConfigError naming the offending field.
    """
    resolved = resolve_config_path(path)
    if not resolved.exists():
        return Config(source=None)
    try:
        with resolved.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"Invalid TOML in {resolved}: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration in {resolved} must be an object")
    return Config.from_dict(data, source=resolved)


def app_version() -> str:
    """Installed package version, with a clear local-source fallback."""
    try:
        from importlib.metadata import version

        return version(APP)
    except Exception:
        return "0+local"
