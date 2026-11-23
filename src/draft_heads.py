"""Pinned official draft heads, prepared before activation in shared storage."""
from pathlib import PurePosixPath

from profiles import ProfileError, parse_quadlet
from semver import version_tuple
from swift_catalog import Asset, OFFICIAL_REPO

REVISION = "05ec74b75cd68c4d6486d15825f2fc73b2ffe304"
DRAFT_HEADS = {
    (0, 20, 6): Asset("qwen38-flash-next-drafter-r2.hgn", OFFICIAL_REPO, REVISION,
                      1523568192, "07a7b5b358cb02737ecf85043db0d30b4938aed06ae6311772ea4e40b939696b",
                      family="halogen-drafter"),
    (0, 20, 7): Asset("qwen38-flash-next-drafter-r2-ht4.hgn", OFFICIAL_REPO, REVISION,
                      1488276864, "f9aa330800cc921fe20f4cf997d660f519c67b0052af0c8078b6b648972b07e6",
                      family="halogen-drafter"),
}


async def prepare_drafter(content, assets, image, download, log):
    version = version_tuple(image.rsplit(":", 1)[-1]) or ()
    asset = DRAFT_HEADS.get(version)
    if asset is None:
        return content
    try:
        profile = parse_quadlet(content.decode("utf-8"))
    except ProfileError:
        return content  # Minimal manually managed units keep native startup behavior.
    checkpoint = PurePosixPath(profile.env.get("HALOGEN_CHECKPOINT", "/models/qwen38-flash-next-v2.hgn"))
    if (profile.profile_id != "official" or checkpoint.name not in {
            "qwen38-flash-next-v2.hgn", "qwen38-flash-next-ht43.hgn"}
            or profile.env.get("HALOGEN_CK_OVERLAY") or profile.env.get("HALOGEN_MTP") == "0"):
        return content
    target = str(checkpoint.parent / asset.name)
    locations = await assets.prepare(None, download, log, files=(asset,))
    # Put the verified file beside the checkpoint, so the native entrypoint
    # selects it and its own draft-gating defaults. Third-party heads stay intact.
    mount = f"Volume={locations[asset.name]}:{target}:ro,z"
    newline = "\r\n" if b"\r\n" in content else "\n"
    lines, out, section = content.decode("utf-8").splitlines(keepends=True), [], ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            if section == "[Container]":
                out.append(mount + newline)
            section = stripped
        key, separator, value = stripped.partition("=")
        if section == "[Container]" and separator and key.strip() == "Volume":
            parts = value.strip().rsplit(":", 2)
            if len(parts) >= 2 and parts[1] == target:
                continue
        out.append(line)
    if section == "[Container]":
        if out and not out[-1].endswith("\n"):
            out[-1] += newline
        out.append(mount + newline)
    return "".join(out).encode("utf-8")
