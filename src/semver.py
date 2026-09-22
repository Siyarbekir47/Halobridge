"""Minimal stable-version parsing shared by discovery and the updater.

Only plain ``X.Y.Z`` (optionally ``v``-prefixed) counts as a stable release.
Pre-release, build metadata and ``latest`` are deliberately rejected so an
update never targets a non-stable tag.
"""

from __future__ import annotations

import re
from typing import Any

VERSION_RE = re.compile(r"v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")


def version_tuple(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, str) or len(value) > 32:
        return None
    match = VERSION_RE.fullmatch(value)
    return tuple(int(part) for part in match.groups()) if match else None


def normalize_version(value: Any) -> str | None:
    parsed = version_tuple(value)
    return ".".join(map(str, parsed)) if parsed else None
