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
        report.fail(f"Python >= 3.11 erforderlich, gefunden {platform.python_version()}")

    if importlib.util.find_spec("aiohttp") is None:
        report.fail("aiohttp ist nicht installiert")
    else:
        report.ok("aiohttp installiert")

    report.ok(f"Config: {config.source or 'Defaults (keine Datei)'}")
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
            report.warn(f"Discovery: keine Modelle in {config.models.quadlet_dir} gefunden")
        for reason in skipped:
            report.warn(f"Discovery übersprungen: {reason}")
    else:
        report.ok("Discovery deaktiviert")

    models = {spec.model_id: spec.service for spec in specs.values()}
    models.update(config.models.explicit)
    if models:
        report.ok(f"Modell-Map: {', '.join(f'{m} -> {s}' for m, s in models.items())}")
    else:
        report.fail(
            "Keine Modelle gefunden. models.auto_discover/quadlet_dir prüfen "
            "oder models.explicit setzen."
        )

    if config.models.explicit:
        report.ok("Explizite Modell-Overrides vorhanden")

    if _writable(config.dashboard.state_dir):
        report.ok(f"State-Verzeichnis beschreibbar: {config.dashboard.state_dir}")
    else:
        report.fail(f"State-Verzeichnis nicht beschreibbar: {config.dashboard.state_dir}")

    if config.updates.enabled:
        if _writable(config.updates.backup_dir):
            report.ok(f"Backup-Verzeichnis beschreibbar: {config.updates.backup_dir}")
        else:
            report.fail(
                f"Backup-Verzeichnis nicht beschreibbar: {config.updates.backup_dir}"
            )
    else:
        report.warn("Updates sind deaktiviert")

    podman = shutil.which("podman")
    systemctl = shutil.which("systemctl")
    if podman:
        report.ok(f"podman gefunden: {podman}")
    else:
        report.warn("podman nicht gefunden; Modell-Updates sind nicht möglich")

    if systemctl:
        report.ok(f"systemctl gefunden: {systemctl}")
    else:
        report.warn("systemctl nicht gefunden; Modellwechsel/Updates sind nicht möglich")

    if shutil.which("journalctl"):
        report.ok("journalctl gefunden; Engine-Telemetrie möglich")
    else:
        report.warn("journalctl nicht gefunden; Engine-Telemetrie eingeschränkt")

    if config.dashboard.gpu_card == "auto":
        cards = _gpu_cards()
        if cards:
            report.ok(f"GPU-Karte automatisch erkannt: {cards[0].name}")
        else:
            report.warn("Keine GPU-Sysfs-Karte gefunden")
    else:
        path = (
            config.dashboard.gpu_card
            if config.dashboard.gpu_card.startswith("/")
            else f"/sys/class/drm/{config.dashboard.gpu_card}/device/gpu_busy_percent"
        )
        if _read_int(path) is not None:
            report.ok(f"GPU-Sysfs gelesen: {path}")
        else:
            report.warn(f"GPU-Sysfs nicht lesbar: {path}")

    non_local = [address for address in config.router.bind if not _is_local_bind(address)]
    if non_local and not config.security.auth_token:
        report.warn(
            "Router ist nicht nur lokal gebunden, aber security.auth_token ist leer. "
            "Token-Gate empfohlen."
        )
    elif config.security.auth_token:
        report.ok("Token-Gate aktiv")
    else:
        report.ok("Bind ist lokal; Token-Gate optional")

    if config.dashboard.public_endpoint:
        if config.dashboard.public_endpoint.startswith("https://"):
            report.ok(f"Public Endpoint: {config.dashboard.public_endpoint}")
        else:
            report.warn(
                f"Public Endpoint ist nicht HTTPS: {config.dashboard.public_endpoint}"
            )

    session = ClientSession(timeout=ClientTimeout(total=3))
    try:
        async with session.get(f"{config.router.backend_url}/health") as response:
            if response.status == 200:
                health = await response.json()
                model = health.get("model") if isinstance(health, dict) else None
                report.ok(f"Backend erreichbar: {config.router.backend_url} ({model or 'kein Modell gemeldet'})")
            else:
                report.warn(
                    f"Backend nicht erreichbar: HTTP {response.status} ({config.router.backend_url})"
                )
    except Exception as error:
        report.warn(
            f"Backend nicht erreichbar: {type(error).__name__} ({config.router.backend_url})"
        )
    finally:
        await session.close()

    print(report.render())
    return report.exit_code
