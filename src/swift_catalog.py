"""Pinned public quick models, with separate compatible asset families."""

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
    family: str = "halogen-v2"


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

CYBER_REPO = "Quat3rnion/CYBER-FROST-3.8-HGN"
CYBER_REVISION = "03b64836b1fbd4104fb258e1b1eb23e3567dfb96"


def _cyber_asset(name, size, digest, algorithm="sha256"):
    return Asset(name, CYBER_REPO, CYBER_REVISION, size, digest, algorithm, "cyber-frost")


# These sidecars are this conversion's weights, not the official sidecars.
CYBER_SHARED_ASSETS = (
    _cyber_asset("qwen38-flash-next-ngram.hgn", 51200246084,
                 "339f5c508ca592ee2613ac5c3082587a88991d5592c4160d49a32a6dbdb3f30b"),
    _cyber_asset("qwen38-flash-next-vision.hgn", 897916416,
                 "b6ca6716eaab601773f7155701bb09f39c72922caa21620bab577f0281f2ea2a"),
    _cyber_asset("tokenizer/chat_template.jinja", 9005,
                 "ba1946683f7615254fb246f0c0a652fd3aa02066ef8328ed4b8af08219749395"),
    _cyber_asset("tokenizer/config.json", 4028,
                 "c77cfd62621660b25e4c811bf44bf2ad4f63af98f23a874952a99e4e85f189f2"),
    _cyber_asset("tokenizer/generation_config.json", 202,
                 "023756cfadf88e5bf69eefeee3e172f38c448d64", "git-sha1"),
    _cyber_asset("tokenizer/merges.txt", 3353259,
                 "a494e019ca1502219fd0128658b979e5f05ae8e8", "git-sha1"),
    _cyber_asset("tokenizer/tokenizer.json", 12809320,
                 "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
    _cyber_asset("tokenizer/tokenizer_config.json", 16414,
                 "3bc90552399707778ca94ef93fb1da7aa3c4dfa5ba94ca15f52b18c2a38ec904"),
    _cyber_asset("tokenizer/vocab.json", 6722759,
                 "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176", "git-sha1"),
)
CYBER_MODEL_ASSETS = (
    _cyber_asset("config.json", 4028,
                 "c77cfd62621660b25e4c811bf44bf2ad4f63af98f23a874952a99e4e85f189f2"),
    _cyber_asset("preprocessor_config.json", 390,
                 "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
    _cyber_asset("video_preprocessor_config.json", 385,
                 "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"),
    _cyber_asset("conversion-manifest.json", 3451483,
                 "68cf9311dafab394ae162eb2171c674482b84b10", "git-sha1"),
    _cyber_asset("LICENSE", 3235,
                 "9557a8961782792814714b7488346732c043793d", "git-sha1"),
)
# Retain the existing catalog name and Swift API for compatibility.
SWIFT_VARIANTS["cyber-frost"] = {
    "label": "CYBER-FROST 3.8",
    "checkpoint": _cyber_asset("model.hgn", 66687678432,
                                "29225578c9e323ea98cc2912ede19a53007c4d40ca379e604f980547219639e3"),
    "shared_assets": CYBER_SHARED_ASSETS,
    "model_assets": CYBER_MODEL_ASSETS,
    "engine_version": "0.17.5",
    "template_unchecked": True,
    "runtime": {"HALOGEN_TEMPLATE_UNCHECKED": "1"},
}


def variant_info(variant: str) -> dict:
    if variant not in SWIFT_VARIANTS:
        raise ValueError("Unknown quick model variant: " + variant)
    return SWIFT_VARIANTS[variant]


def variant_assets(variant=None):
    return variant_info(variant).get("shared_assets", SHARED_ASSETS) if variant else SHARED_ASSETS


def variant_engine(variant):
    return variant_info(variant).get("engine_version", SWIFT_ENGINE_VERSION)
