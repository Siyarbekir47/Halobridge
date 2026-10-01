"""Pinned, public Swift checkpoints and the compatible official v2 assets."""

from dataclasses import dataclass

OFFICIAL_REPO = "peonist-ai/halogen-qwen3.8-flash-next"
NGRAM_REVISION = "5cc17cea1a10b502b4a5db41d8d2b1a0f22dba26"
ASSET_REVISION = "e053f488b120b99ed2525e6ac99f68c51b3b6179"
SWIFT_ENGINE_VERSION = "0.15.1"


@dataclass(frozen=True)
class Asset:
    name: str
    repo: str
    revision: str
    size: int
    digest: str
    algorithm: str = "sha256"


SHARED_ASSETS = (
    Asset("qwen38-flash-next-ngram.hgn", OFFICIAL_REPO, NGRAM_REVISION,
          51200246144, "3450acada94e19aabad88bd49b45eb70304820949c2240fc178067b20ae5dffe"),
    Asset("qwen38-flash-next-vision.hgn", OFFICIAL_REPO, ASSET_REVISION,
          897916416, "d62e0ae553fe88afd3833733d4a4c669f34d20fd8dfce4b9610525bed2134b10"),
    Asset("tokenizer/chat_template.jinja", OFFICIAL_REPO, ASSET_REVISION,
          8952, "c0c686f9c38d70d179fb7b5f5aa7530bc913dda3", "git-sha1"),
    Asset("tokenizer/generation_config.json", OFFICIAL_REPO, ASSET_REVISION,
          202, "023756cfadf88e5bf69eefeee3e172f38c448d64", "git-sha1"),
    Asset("tokenizer/merges.txt", OFFICIAL_REPO, ASSET_REVISION,
          3353259, "a494e019ca1502219fd0128658b979e5f05ae8e8", "git-sha1"),
    Asset("tokenizer/tokenizer.json", OFFICIAL_REPO, ASSET_REVISION,
          12809320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
    Asset("tokenizer/tokenizer_config.json", OFFICIAL_REPO, ASSET_REVISION,
          17928, "5de744b3fca2129d7186979ae47c06be33903243", "git-sha1"),
    Asset("tokenizer/vocab.json", OFFICIAL_REPO, ASSET_REVISION,
          6722759, "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176", "git-sha1"),
)

SWIFT_VARIANTS = {
    "swift15": {
        "label": "Swift 1.5",
        "checkpoint": Asset(
            "qwen38-flash-next-v2-swift15.hgn",
            "Quat3rnion/halogen-swift1.5-qwen3.8-flash-next-v2",
            "e5accd214c6c7a95fe27bd5799b532b62935ffdd", 68142577664,
            "a4ad500200982fcda77c3fe03367ebbbe120ec65724585e9aecda77435e55205"),
    },
    "swift15-abliterated": {
        "label": "Swift 1.5 Abliterated",
        "checkpoint": Asset(
            "qwen38-flash-next-v2-swift15-abliterated.hgn",
            "Quat3rnion/halogen-swift1.5-qwen3.8-flash-next-v2-abliterated",
            "69a85db2628c7444642e10aae4fed75943eebde3", 66687678464,
            "9c880b8ba48deabfa06172eff1e7572e4fe465e09e97faec6e50c3f6fdcac037"),
    },
}


def variant_info(variant: str) -> dict:
    if variant not in SWIFT_VARIANTS:
        raise ValueError("Variant must be swift15 or swift15-abliterated")
    return SWIFT_VARIANTS[variant]
