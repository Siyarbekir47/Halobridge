"""Profile schema: allowlist validation, quadlet rendering and import round-trip."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from profiles import (
    ENV_FIELDS,
    OFFICIAL_WEIGHTS_REPO,
    Profile,
    ProfileError,
    custom_template,
    official_quick_template,
    official_template,
    parse_quadlet,
    render_quadlet,
    uncensored_quick_template,
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
        self.assertTrue(any("allowlist" in e for e in errors))

    def test_rejects_out_of_range_kv_slots(self):
        errors = validate_profile(
            make_profile(env={"HALOGEN_MODEL_ID": "m", "HALOGEN_KV_SLOTS": "999"})
        )
        self.assertTrue(any("Maximum" in e for e in errors))

    def test_rejects_shell_metacharacters_in_env_value(self):
        errors = validate_profile(
            make_profile(env={"HALOGEN_MODEL_ID": "m; rm -rf /", "HALOGEN_KV_SLOTS": "2"})
        )
        self.assertTrue(any("invalid characters" in e for e in errors))

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
        self.assertTrue(any("writable" in e for e in errors))
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
    def test_01813_admission_cache_and_restart_controls_round_trip(self):
        profile = official_template("0.18.13", Path("/host/models"), Path("/host/cache"))
        profile.env.update(HALOGEN_ADMIT_SPLICE_MS="500", HALOGEN_ADMIT_SPLICE_FOR="all",
                           HALOGEN_ADMIT_NEST="1", HALOGEN_ADMIT_NEST_MAX="1024",
                           HALOGEN_CACHE_STAGE="2", HALOGEN_NPU_RESTARTS="3", HALOGEN_SPEC_GROUP="4")
        self.assertEqual(validate_profile(profile), [])
        self.assertEqual(parse_quadlet(render_quadlet(profile)).env, profile.env)
        self.assert_common_runtime_env(profile)

    def test_new_controls_require_their_engine_version(self):
        for name, value, old, supported in (
            ("HALOGEN_ADMIT_SPLICE_MS", "0", "0.18.11", "0.18.12"),
            ("HALOGEN_ADMIT_SPLICE_FOR", "h", "0.18.11", "0.18.12"),
            ("HALOGEN_CACHE_STAGE", "0", "0.18.11", "0.18.12"),
            ("HALOGEN_ADMIT_NEST", "0", "0.18.12", "0.18.13"),
            ("HALOGEN_ADMIT_NEST_MAX", "0", "0.18.12", "0.18.13"),
            ("HALOGEN_NPU_RESTARTS", "0", "0.18.12", "0.18.13"),
        ):
            with self.subTest(name=name):
                profile = official_template(old, Path("/host/models"), Path("/host/cache"))
                profile.env[name] = value
                self.assertTrue(any(f"{name}: requires Halogen {supported}" in e for e in validate_profile(profile)))
                profile.image = profile.image.rsplit(":", 1)[0] + ":" + supported
                self.assertEqual(validate_profile(profile), [])

    def test_new_controls_reject_invalid_values(self):
        for name, value in (("HALOGEN_ADMIT_SPLICE_MS", "-1"), ("HALOGEN_ADMIT_SPLICE_MS", "1.5"),
                            ("HALOGEN_ADMIT_SPLICE_FOR", "agents"), ("HALOGEN_ADMIT_NEST", "2"),
                            ("HALOGEN_ADMIT_NEST_MAX", "-1"), ("HALOGEN_NPU_RESTARTS", "-1"),
                            ("HALOGEN_CACHE_STAGE", "1"), ("HALOGEN_SPEC_GROUP", "5")):
            with self.subTest(name=name, value=value):
                profile = official_template("0.18.13", Path("/host/models"), Path("/host/cache"))
                profile.env[name] = value
                self.assertTrue(any(name in e for e in validate_profile(profile)))

    def test_01811_controls_and_incompatible_kv_modes(self):
        profile = official_template("0.18.11", Path("/host/models"), Path("/host/cache"))
        profile.env.update(HALOGEN_KV8="1", HALOGEN_CACHE_LEAD="1", HALOGEN_SPEC_GROUP="1",
                           HALOGEN_ACCESS_LOG="quiet", HALOGEN_SAMPLED_GATE="0.95,0.98",
                           HALOGEN_FORK="1", HALOGEN_FORK_MAP="auto")
        self.assertEqual(validate_profile(profile), [])
        self.assertEqual(parse_quadlet(render_quadlet(profile)).env, profile.env)
        profile.env["HALOGEN_COMPOSABLE_CONTEXT"] = "1"
        self.assertTrue(any("cannot combine" in e for e in validate_profile(profile)))
        profile.env.pop("HALOGEN_COMPOSABLE_CONTEXT")
        profile.env["HALOGEN_SAMPLED_GATE"] = "nan,2"
        self.assertTrue(any("probabilities" in e for e in validate_profile(profile)))
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
        self.assertEqual(profile.env["HALOGEN_CHECKPOINT"], "/models/qwen38-flash-next-v2.hgn")
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

    def test_official_quick_template_downloads_weights(self):
        profile = official_quick_template(
            "0.13.2", Path("/home/tester/halogen/models"), Path("/home/tester/halogen/cache")
        )
        self.assertEqual(validate_profile(profile), [])
        self.assertTrue(profile.downloads_weights)
        self.assertEqual(profile.env["HALOGEN_DOWNLOAD"], OFFICIAL_WEIGHTS_REPO)
        self.assertEqual(
            profile.env["HALOGEN_VISION_TOWER"],
            "/models/qwen38-flash-next-vision.hgn",
        )
        self.assert_common_runtime_env(profile)
        models = next(v for v in profile.volumes if v[1] == "/models")
        self.assertNotIn("ro", models[2])

    def test_uncensored_quick_template_uses_converted_hgn(self):
        profile = uncensored_quick_template(
            "0.13.2", Path("/home/tester/halogen/models"), Path("/home/tester/halogen/cache")
        )
        self.assertEqual(validate_profile(profile), [])
        self.assertEqual(profile.model_id, "qwen3.8-flash-uncensored")
        self.assertEqual(
            profile.env["HALOGEN_CHECKPOINT"], "/models/qwen3.8-flash-uncensored.hgn"
        )
        self.assertEqual(profile.env["HALOGEN_TOKENIZER"], "/models/tokenizer")
        self.assert_common_runtime_env(profile)
        models = next(v for v in profile.volumes if v[1] == "/models")
        self.assertIn("ro", models[2])

    def test_custom_template_valid_and_readonly(self):
        profile = custom_template(
            "0.13.2", Path("/home/tester/halogen/models"), Path("/home/tester/halogen/cache")
        )
        self.assertEqual(validate_profile(profile), [])
        self.assertFalse(profile.downloads_weights)
        self.assert_common_runtime_env(profile)
        models = next(v for v in profile.volumes if v[1] == "/models")
        self.assertIn("ro", models[2])

    def test_0180_features_validate_versions_paths_and_record_policy(self):
        profile = official_template("0.18.0", Path("/host/models"), Path("/host/cache"))
        profile.env.update(HALOGEN_FORK="1", HALOGEN_FORK_HANDLES="8", HALOGEN_FORK_TTL_S="600.5",
                           HALOGEN_FORK_GIB="2.5", HALOGEN_NEXT_TOKEN_S="990.5",
                           HALOGEN_RECORD_DIR="/cache/records", HALOGEN_RECORD_TEXT="prompt,reply",
                           HALOGEN_RECORD_GIB="2", HALOGEN_RECORD_DAYS="0",
                           HALOGEN_API_KEY="key-one,key-two", HALOGEN_NPU_MODELS="qwen3-asr-0.6b")
        self.assertEqual(validate_profile(profile), [])
        self.assertEqual(parse_quadlet(render_quadlet(profile)).env, profile.env)
        for key, value in (("HALOGEN_RECORD_TEXT", "none,prompt"), ("HALOGEN_RECORD_TEXT", "prompt,prompt"),
                           ("HALOGEN_RECORD_DIR", "/models/records"), ("HALOGEN_RECORD_DIR", "/unmounted/records"),
                           ("HALOGEN_API_KEY", "key,broken key"), ("HALOGEN_FORK_GIB", "NaN")):
            original = profile.env.get(key)
            profile.env[key] = value
            self.assertTrue(validate_profile(profile), (key, value))
            profile.env[key] = original
        profile.env["HALOGEN_API_KEY_FILE"] = "/cache/key"
        self.assertTrue(any("not both" in error for error in validate_profile(profile)))
        profile.env.pop("HALOGEN_API_KEY_FILE")
        profile.image = profile.image.replace("0.18.0", "0.17.5")
        errors = validate_profile(profile)
        self.assertTrue(any("HALOGEN_FORK" in error for error in errors))
        self.assertTrue(any("qwen3-asr" in error for error in errors))

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

    def test_0182_controls_validate_values_versions_and_session_dependency(self):
        profile = official_template("0.18.2", Path("/host/models"), Path("/host/cache"))
        additions = {"HALOGEN_NPU_DEC_RESUME": "2", "HALOGEN_NPU_DEC_ENTRIES": "6",
                     "HALOGEN_STRICT_TOOLS": "1", "HALOGEN_CLIENT_MARKERS": "text",
                     "HALOGEN_QSA_FORCE_SEL": "1", "HALOGEN_RETRY_GRACE_S": "20.5",
                     "HALOGEN_FORK": "1", "HALOGEN_FORK_MAP": "claude-code"}
        profile.env.update(additions)
        self.assertEqual(validate_profile(profile), [])
        self.assertEqual(parse_quadlet(render_quadlet(profile)).env, profile.env)
        for key, invalid in (("HALOGEN_NPU_DEC_RESUME", "3"), ("HALOGEN_NPU_DEC_ENTRIES", "0"),
                             ("HALOGEN_RETRY_GRACE_S", "121"), ("HALOGEN_CLIENT_MARKERS", "unknown"),
                             ("HALOGEN_FORK_MAP", "unknown"), ("HALOGEN_FORK", "0")):
            profile.env[key] = invalid
            self.assertTrue(validate_profile(profile), key)
            profile.env[key] = additions[key]
        profile.image = profile.image.replace("0.18.2", "0.18.1")
        errors = validate_profile(profile)
        for key in ("HALOGEN_NPU_DEC_RESUME", "HALOGEN_NPU_DEC_ENTRIES", "HALOGEN_STRICT_TOOLS", "HALOGEN_CLIENT_MARKERS"):
            self.assertTrue(any(key in error and "0.18.2" in error for error in errors), key)
        self.assertFalse(any("HALOGEN_FORK_MAP" in error for error in errors))
        profile.image = profile.image.replace("0.18.1", "0.18.0")
        errors = validate_profile(profile)
        self.assertTrue(any("HALOGEN_FORK_MAP" in error and "0.18.1" in error for error in errors))
        self.assertTrue(any("HALOGEN_RETRY_GRACE_S" in error and "0.18.1" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
