"""Readiness checks for a Halobridge installation."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import platform
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout

import settings
from discovery import discover


OK = "OK"
WARN = "WARN"
FAIL = "FAIL"

LOCAL_BINDS = {"127.0.0.1", "::1", "localhost"}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []

    def add(self, level: str, message: str) -> None:
        self.rows.append((level, message))

    def ok(self, message: str) -> None:
        self.add(OK, message)

    def warn(self, message: str) -> None:
        self.add(WARN, message)

    def fail(self, message: str) -> None:
        self.add(FAIL, message)

    @property
    def exit_code(self) -> int:
        return 1 if any(level == FAIL for level, _ in self.rows) else 0

    def render(self) -> str:
        lines = ["Halobridge doctor", "=" * 40]
        for level, message in self.rows:
            lines.append(f"[{level:<4}] {message}")
        return "\n".join(lines)


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        probe = path / f".halobridge-doctor-{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _is_local_bind(address: str) -> bool:
    return address in LOCAL_BINDS or address.startswith("127.")


def _gpu_cards() -> list[Path]:
    base = Path("/sys/class/drm")
    if not base.is_dir():
        return []
    cards = []
    for path in base.iterdir():
        match = re.fullmatch(r"card(\d+)", path.name)
        if match and path.is_dir():
            cards.append((int(match.group(1)), path))
    return [path for _, path in sorted(cards)]


def _read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


async def run(config: settings.Config) -> int:
    report = Report()

    if sys.version_info >= (3, 11):
        report.ok(f"Python {platform.python_version()}")
    else:
        report.fail(f"Python >= 3.11 required, found {platform.python_version()}")

    if importlib.util.find_spec("aiohttp") is None:
        report.fail("aiohttp is not installed")
    else:
        report.ok("aiohttp installed")

    report.ok(f"Config: {config.source or 'Defaults (no file)'}")
    report.ok(
        f"Router: bind={', '.join(config.router.bind)} port={config.router.port} "
        f"backend={config.router.backend_url}"
    )

    specs: dict[str, Any] = {}
    skipped: list[str] = []
    if config.models.auto_discover:
        specs, skipped = discover(config.models.quadlet_dir, config.updates.image_repo)
        if specs:
            report.ok(
                "Discovery: "
                + ", ".join(
                    f"{model_id} -> {spec.service}" for model_id, spec in specs.items()
                )
            )
        else:
            report.warn(f"Discovery: no models found in {config.models.quadlet_dir}")
        for reason in skipped:
            report.warn(f"Discovery skipped: {reason}")
    else:
        report.ok("Discovery disabled")

    models = {spec.model_id: spec.service for spec in specs.values()}
    models.update(config.models.explicit)
    if models:
        report.ok(f"Model map: {', '.join(f'{m} -> {s}' for m, s in models.items())}")
    else:
        report.fail(
            "No models found. Check models.auto_discover/quadlet_dir "
            "or set models.explicit."
        )

    if config.models.explicit:
        report.ok("Explicit model overrides present")

    if _writable(config.dashboard.state_dir):
        report.ok(f"State directory writable: {config.dashboard.state_dir}")
    else:
        report.fail(f"State directory not writable: {config.dashboard.state_dir}")

    if config.updates.enabled:
        if _writable(config.updates.backup_dir):
            report.ok(f"Backup directory writable: {config.updates.backup_dir}")
        else:
            report.fail(
                f"Backup directory not writable: {config.updates.backup_dir}"
            )
    else:
        report.warn("Updates are disabled")

    podman = shutil.which("podman")
    systemctl = shutil.which("systemctl")
    if podman:
        report.ok(f"podman found: {podman}")
    else:
        report.warn("podman not found; model updates are not possible")

    if systemctl:
        report.ok(f"systemctl found: {systemctl}")
    else:
        report.warn("systemctl not found; model switching/updates are not possible")

    if shutil.which("journalctl"):
        report.ok("journalctl found; engine telemetry possible")
    else:
        report.warn("journalctl not found; engine telemetry limited")

    if config.dashboard.gpu_card == "auto":
        cards = _gpu_cards()
        if cards:
            report.ok(f"GPU card automatically detected: {cards[0].name}")
        else:
            report.warn("No GPU sysfs card found")
    else:
        path = (
            config.dashboard.gpu_card
            if config.dashboard.gpu_card.startswith("/")
            else f"/sys/class/drm/{config.dashboard.gpu_card}/device/gpu_busy_percent"
        )
        if _read_int(path) is not None:
            report.ok(f"GPU sysfs read: {path}")
        else:
            report.warn(f"GPU sysfs not readable: {path}")

    non_local = [address for address in config.router.bind if not _is_local_bind(address)]
    if non_local and not config.security.auth_token:
        report.warn(
            "Router is bound beyond localhost but security.auth_token is empty. "
            "Token gate recommended."
        )
    elif config.security.auth_token:
        report.ok("Token gate active")
    else:
        report.ok("Bind is local; token gate optional")

    if config.dashboard.public_endpoint:
        if config.dashboard.public_endpoint.startswith("https://"):
            report.ok(f"Public Endpoint: {config.dashboard.public_endpoint}")
        else:
            report.warn(
                f"Public Endpoint is not HTTPS: {config.dashboard.public_endpoint}"
            )

    session = ClientSession(timeout=ClientTimeout(total=3))
    try:
        async with session.get(f"{config.router.backend_url}/health") as response:
            if response.status == 200:
                health = await response.json()
                model = health.get("model") if isinstance(health, dict) else None
                report.ok(f"Backend reachable: {config.router.backend_url} ({model or 'no model reported'})")
            else:
                report.warn(
                    f"Backend not reachable: HTTP {response.status} ({config.router.backend_url})"
                )
    except Exception as error:
        report.warn(
            f"Backend not reachable: {type(error).__name__} ({config.router.backend_url})"
        )
    finally:
        await session.close()

    print(report.render())
    return report.exit_code
