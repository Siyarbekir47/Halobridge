"""Profile schema: allowlist validation, quadlet rendering and import round-trip."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from profiles import (
    ENV_FIELDS,
    Profile,
    ProfileError,
    custom_template,
    official_template,
    parse_quadlet,
    render_quadlet,
    uncensored_template,
    validate_profile,
)


def make_profile(**overrides) -> Profile:
    base = dict(
        profile_id="demo",
        image="ghcr.io/peonist-ai/halogen-flash-server:0.13.2",
        volumes=[
            ("/home/tester/halogen/models/demo", "/models", "ro,Z"),
            ("/home/tester/halogen/cache/demo", "/cache", "Z"),
        ],
        host_port=8899,
        env={
            "HALOGEN_MODEL_ID": "demo-model",
            "HALOGEN_KV_SLOTS": "2",
            "HALOGEN_KV_POOL_POSITIONS": "524288",
            "HALOGEN_MAX_TOK": "32768",
        },
    )
    base.update(overrides)
    return Profile(**base)


class ValidateProfileTests(unittest.TestCase):
    def test_valid_profile_has_no_errors(self):
        self.assertEqual(validate_profile(make_profile()), [])

    def test_rejects_bad_profile_id(self):
        errors = validate_profile(make_profile(profile_id="Bad ID!"))
        self.assertTrue(any("profile_id" in e for e in errors))

    def test_rejects_latest_tag(self):
        errors = validate_profile(
            make_profile(image="ghcr.io/peonist-ai/halogen-flash-server:latest")
        )
        self.assertTrue(any("image" in e for e in errors))

    def test_rejects_unknown_env_variable(self):
        errors = validate_profile(make_profile(env={"HALOGEN_MODEL_ID": "m", "EVIL_VAR": "1"}))
        self.assertTrue(any("Allowlist" in e for e in errors))

    def test_rejects_out_of_range_kv_slots(self):
        errors = validate_profile(
            make_profile(env={"HALOGEN_MODEL_ID": "m", "HALOGEN_KV_SLOTS": "999"})
        )
        self.assertTrue(any("Maximum" in e for e in errors))

    def test_rejects_shell_metacharacters_in_env_value(self):
        errors = validate_profile(
            make_profile(env={"HALOGEN_MODEL_ID": "m; rm -rf /", "HALOGEN_KV_SLOTS": "2"})
        )
        self.assertTrue(any("Zeichen nicht erlaubt" in e for e in errors))

    def test_rejects_missing_model_id(self):
        errors = validate_profile(make_profile(env={"HALOGEN_KV_SLOTS": "2"}))
        self.assertTrue(any("HALOGEN_MODEL_ID" in e for e in errors))

    def test_rejects_missing_models_volume(self):
        profile = make_profile(volumes=[("/x/cache", "/cache", "Z")])
        errors = validate_profile(profile)
        self.assertTrue(any("/models" in e for e in errors))

    def test_default_above_cap_rejected(self):
        errors = validate_profile(
            make_profile(
                env={
                    "HALOGEN_MODEL_ID": "m",
                    "HALOGEN_MAX_TOKENS_DEFAULT": "70000",
                    "HALOGEN_MAX_TOKENS_CAP": "65536",
                }
            )
        )
        self.assertTrue(any("CAP" in e for e in errors))

    def test_download_requires_writable_models_volume(self):
        profile = make_profile(
            env={"HALOGEN_MODEL_ID": "m", "HALOGEN_DOWNLOAD": "org/repo"}
        )
        errors = validate_profile(profile)
        self.assertTrue(any("beschreibbar" in e for e in errors))
        writable = make_profile(
            volumes=[
                ("/home/tester/halogen/models/demo", "/models", "Z"),
                ("/home/tester/halogen/cache/demo", "/cache", "Z"),
            ],
            env={"HALOGEN_MODEL_ID": "m", "HALOGEN_DOWNLOAD": "org/repo"},
        )
        self.assertEqual(validate_profile(writable), [])

    def test_enum_and_flag_kinds(self):
        errors = validate_profile(
            make_profile(
                env={
                    "HALOGEN_MODEL_ID": "m",
                    "HALOGEN_REASONING_EFFORT": "ultra",
                    "HALOGEN_GRAMMAR": "maybe",
                }
            )
        )
        self.assertEqual(len(errors), 2)


class RenderImportTests(unittest.TestCase):
    def test_render_contains_core_sections(self):
        text = render_quadlet(make_profile())
        for needle in (
            "[Unit]",
            "[Container]",
            "[Service]",
            "Image=ghcr.io/peonist-ai/halogen-flash-server:0.13.2",
            "ContainerName=halogen-demo",
            "PublishPort=127.0.0.1:8899:8731",
            "Environment=HALOGEN_KV_SLOTS=2",
            "PodmanArgs=--ipc=host",
        ):
            self.assertIn(needle, text)

    def test_render_is_deterministic(self):
        profile = make_profile(
            env={"HALOGEN_MODEL_ID": "m", "HALOGEN_TOP_K": "20", "HALOGEN_TOP_P": "0.95"}
        )
        self.assertEqual(render_quadlet(profile), render_quadlet(profile))

    def test_round_trip(self):
        original = make_profile()
        imported = parse_quadlet(render_quadlet(original))
        self.assertEqual(imported.profile_id, original.profile_id)
        self.assertEqual(imported.image, original.image)
        self.assertEqual(imported.host_port, original.host_port)
        self.assertEqual(imported.volumes, original.volumes)
        self.assertEqual(imported.env, original.env)
        self.assertEqual(render_quadlet(imported), render_quadlet(original))

    def test_parse_rejects_foreign_container_name(self):
        text = render_quadlet(make_profile()).replace(
            "ContainerName=halogen-demo", "ContainerName=nginx"
        )
        with self.assertRaises(ProfileError):
            parse_quadlet(text)

    def test_parse_rejects_missing_image(self):
        text = render_quadlet(make_profile()).replace(
            "Image=ghcr.io/peonist-ai/halogen-flash-server:0.13.2", ""
        )
        with self.assertRaises(ProfileError):
            parse_quadlet(text)


class TemplateTests(unittest.TestCase):
    def assert_common_runtime_env(self, profile: Profile) -> None:
        expected = {
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
        for key, value in expected.items():
            self.assertEqual(profile.env[key], value)

    def test_official_template_valid_and_matches_provisioned_setup(self):
        profile = official_template(
            "0.13.2", Path("/home/tester/halogen/models"), Path("/home/tester/halogen/cache")
        )
        self.assertEqual(validate_profile(profile), [])
        self.assertFalse(profile.downloads_weights)
        self.assertEqual(profile.env["HALOGEN_CHECKPOINT"], "/models/qwen38-flash-next-w4b.hgn")
        self.assertEqual(profile.env["HALOGEN_TOKENIZER"], "/models/tokenizer")
        self.assertEqual(
            profile.env["HALOGEN_VISION_TOWER"], "/models/qwen38-flash-next-vision.hgn"
        )
        self.assert_common_runtime_env(profile)
        models = next(v for v in profile.volumes if v[1] == "/models")
        self.assertIn("ro", models[2])

    def test_uncensored_template_valid(self):
        profile = uncensored_template(
            "0.13.2", Path("/home/tester/halogen/models"), Path("/home/tester/halogen/cache")
        )
        self.assertEqual(validate_profile(profile), [])
        self.assertEqual(profile.model_id, "qwen3.8-flash-uncensored")
        self.assertEqual(
            profile.env["HALOGEN_CHECKPOINT"], "/uncensored/qwen3.8-flash-uncensored.hgn"
        )
        self.assert_common_runtime_env(profile)
        container_paths = [v[1] for v in profile.volumes]
        self.assertEqual(container_paths, ["/models", "/uncensored", "/cache"])
        uncensored = next(v for v in profile.volumes if v[1] == "/uncensored")
        self.assertIn("ro", uncensored[2])

    def test_uncensored_round_trip(self):
        profile = uncensored_template(
            "0.13.2", Path("/home/tester/halogen/models"), Path("/home/tester/halogen/cache")
        )
        imported = parse_quadlet(render_quadlet(profile))
        self.assertEqual(render_quadlet(imported), render_quadlet(profile))

    def test_custom_template_valid_and_readonly(self):
        profile = custom_template(
            "0.13.2", Path("/home/tester/halogen/models"), Path("/home/tester/halogen/cache")
        )
        self.assertEqual(validate_profile(profile), [])
        self.assertFalse(profile.downloads_weights)
        self.assert_common_runtime_env(profile)
        models = next(v for v in profile.volumes if v[1] == "/models")
        self.assertIn("ro", models[2])

    def test_env_fields_cover_documented_variables(self):
        for key in (
            "HALOGEN_CTX",
            "HALOGEN_KV_SLOTS",
            "HALOGEN_KV_POOL_POSITIONS",
            "HALOGEN_MAX_TOK",
            "HALOGEN_MAX_TOKENS_DEFAULT",
            "HALOGEN_MAX_TOKENS_CAP",
            "HALOGEN_HOST_RESERVE_GIB",
            "HALOGEN_QUEUE_TIMEOUT",
            "HALOGEN_CACHE_DISK_GIB",
            "HALOGEN_DOWNLOAD",
            "HALOGEN_REASONING_EFFORT",
        ):
            self.assertIn(key, ENV_FIELDS)


if __name__ == "__main__":
    unittest.main()
