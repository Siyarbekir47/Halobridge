"""Model auto-discovery from Podman Quadlet files.

A model is any ``*.container`` in the quadlet directory whose ``[Container]``
``Image`` belongs to the configured Halogen image repository. Everything the
router, dashboard and updater need is read from the quadlet itself:

- ``ContainerName``  -> systemd user unit ``<name>.service`` and container name
- ``HALOGEN_MODEL_ID`` -> the model id served on the OpenAI surface
- the ``:/cache`` volume -> the model's cache directory (dashboard display)
- the image tag -> the currently configured version

This removes the need for hand-maintained model/service/cache maps. An explicit
override in config is still supported for setups without quadlets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from semver import normalize_version

CACHE_TARGET = "/cache"
_VOLUME_CACHE_RE = re.compile(r"^([^:\s]+):" + re.escape(CACHE_TARGET) + r"(?::.*)?$")


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    service: str
    container: str
    cache_path: Path | None
    image: str
    version: str
    quadlet: Path


def _container_section(text: str) -> dict[str, list[str]]:
    """Return key -> values for the [Container] section only."""
    section: dict[str, list[str]] = {}
    current = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line
            continue
        if current != "[Container]" or "=" not in line:
            continue
        key, value = line.split("=", 1)
        section.setdefault(key.strip(), []).append(value.strip())
    return section


def parse_quadlet(path: Path, image_repo: str) -> tuple[ModelSpec | None, str | None]:
    """Parse one quadlet. Return (spec, None) or (None, skip_reason)."""
    if path.is_symlink():
        return None, f"{path.name}: symlink ignored"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return None, f"{path.name}: not readable ({type(error).__name__})"

    section = _container_section(text)
    images = section.get("Image", [])
    if len(images) != 1:
        return None, f"{path.name}: expected exactly one Image line, found {len(images)}"

    image = images[0]
    prefix = image_repo.rstrip("/") + ":"
    if not image.startswith(prefix):
        return None, f"{path.name}: image does not belong to {image_repo}"
    version = normalize_version(image[len(prefix):])
    if version is None:
        return None, f"{path.name}: no stable version in image tag"

    names = section.get("ContainerName", [])
    if len(names) != 1 or not names[0]:
        return None, f"{path.name}: expected exactly one ContainerName"
    container = names[0]

    model_id = None
    for value in section.get("Environment", []):
        if value.startswith("HALOGEN_MODEL_ID="):
            candidate = value.split("=", 1)[1].strip()
            if candidate:
                if model_id is not None and model_id != candidate:
                    return None, f"{path.name}: multiple different HALOGEN_MODEL_ID values"
                model_id = candidate
    if not model_id:
        return None, f"{path.name}: HALOGEN_MODEL_ID missing"

    cache_path = None
    for value in section.get("Volume", []):
        match = _VOLUME_CACHE_RE.match(value)
        if match:
            cache_path = Path(match.group(1))
            break

    return ModelSpec(
        model_id=model_id,
        service=f"{container}.service",
        container=container,
        cache_path=cache_path,
        image=image,
        version=version,
        quadlet=path,
    ), None


def discover(quadlet_dir: Path, image_repo: str) -> tuple[dict[str, ModelSpec], list[str]]:
    """Scan a directory for Halogen model quadlets.

    Returns (models keyed by model_id, list of human-readable skip reasons).
    Duplicate model_ids across files are reported as a skip, never silently
    merged, so a misconfiguration is visible rather than ambiguous.
    """
    models: dict[str, ModelSpec] = {}
    skipped: list[str] = []
    if not quadlet_dir.is_dir():
        return models, [f"Quadlet directory not found: {quadlet_dir}"]

    for path in sorted(quadlet_dir.glob("*.container")):
        spec, reason = parse_quadlet(path, image_repo)
        if spec is None:
            if reason:
                skipped.append(reason)
            continue
        if spec.model_id in models:
            skipped.append(
                f"{path.name}: model id {spec.model_id} already used by "
                f"{models[spec.model_id].quadlet.name}"
            )
            continue
        models[spec.model_id] = spec
    return models, skipped
