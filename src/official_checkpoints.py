"""Explicit official checkpoint choices; engine releases never select HT43."""

from swift_catalog import Asset, OFFICIAL_REPO

REVISION = "d34cfa2c43f7565c99ad98135a5d5fb8b3f0f8b2"
OFFICIAL_CHECKPOINTS = {
    "v2": {
        "label": "v2",
        "minimum_engine": (0, 15, 0),
        "asset": Asset("qwen38-flash-next-v2.hgn", OFFICIAL_REPO, REVISION,
                       66687678432, "71246c6ab3fc1de2cf06326f18e275fe9c2a18366d646ed3357d194c884fc687"),
    },
    "ht43": {
        "label": "HT43",
        "minimum_engine": (0, 16, 0),
        "asset": Asset("qwen38-flash-next-ht43.hgn", OFFICIAL_REPO, REVISION,
                       57618253792, "12e8e91b06dfd3bad4b68cdd275138733da64542a254612b116e8bd9fe77fdd8"),
    },
}


def checkpoint_choice(value):
    if not isinstance(value, str) or value not in OFFICIAL_CHECKPOINTS:
        raise ValueError("Checkpoint target must be v2 or ht43")
    return OFFICIAL_CHECKPOINTS[value]
