"""Config + discovery tests: python -m unittest discover -s tests -v"""

import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import settings
from settings import Config, ConfigError, load, resolve_config_path
from discovery import discover, parse_quadlet
from semver import normalize_version, version_tuple

REPO = "ghcr.io/peonist-ai/halogen-flash-server"


def good_quadlet(version="0.13.1", container="halogen-official", model="qwen3.8-flash",
                cache="/home/u/halogen/cache/official"):
    return (
        "[Unit]\nDescription=x\n[Container]\n"
        f"Image={REPO}:{version}\nContainerName={container}\nExec=all\n"
        f"Volume=/models:/models:ro,Z\nVolume={cache}:/cache:Z\n"
        f"Environment=HALOGEN_MODEL_ID={model}\nEnvironment=HALOGEN_KV_SLOTS=2\n"
        "[Service]\nRestart=on-failure\n"
    )


class TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=os.environ.get("HALOGEN_TEST_TMP"))
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self._old_env = os.environ.get(settings.CONFIG_ENV)

    def tearDown(self):
        if self._old_env is None:
            os.environ.pop(settings.CONFIG_ENV, None)
        else:
            os.environ[settings.CONFIG_ENV] = self._old_env


class SettingsTests(TempCase):
    def test_defaults_without_file(self):
        config = load(self.root / "missing.toml")
        self.assertIsNone(config.source)
        self.assertEqual(config.router.bind, ["127.0.0.1"])
        self.assertEqual(config.router.port, 8731)
        self.assertEqual(
            config.router.gtt_path,
            "/sys/class/drm/card0/device/mem_info_gtt_used",
        )
        self.assertEqual(config.router.gtt_limit_bytes, 1024 * 1024 * 1024)
        self.assertTrue(config.models.auto_discover)
        self.assertTrue(config.updates.enabled)
        self.assertEqual(config.updates.backup_keep, 5)
        self.assertEqual(config.dashboard.retention_days, 365)
        self.assertEqual(config.security.auth_token, "")
        self.assertTrue(config.security.allow_install)

    def test_full_file_parses_and_expands_home(self):
        path = self.root / "c.toml"
        path.write_text(
            "[router]\nbind = ['127.0.0.1', '10.0.0.5']\nport = 9000\n"
            "backend_url = 'http://127.0.0.1:8831/'\n"
            "gtt_path = '/tmp/gtt'\ngtt_limit_bytes = 123\n"
            "[models]\nauto_discover = false\nquadlet_dir = '~/qd'\n"
            "[models.explicit]\n'model-a' = 'svc-a.service'\n"
            "[updates]\ntag_source = 'registry'\nbackup_dir = '~/bk'\ncheck_interval_h = 12\nbackup_keep = 2\n"
            "[dashboard]\nstate_dir = '~/st'\nretention_days = 30\ngpu_card = 'card1'\n"
            "public_endpoint = 'http://host:9000/'\n"
            "[security]\nauth_token = 'secret'\nallow_install = false\n",
            encoding="utf-8",
        )
        config = load(path)
        self.assertEqual(config.router.bind, ["127.0.0.1", "10.0.0.5"])
        self.assertEqual(config.router.port, 9000)
        self.assertEqual(config.router.backend_url, "http://127.0.0.1:8831")
        self.assertEqual(config.router.gtt_path, "/tmp/gtt")
        self.assertEqual(config.router.gtt_limit_bytes, 123)
        self.assertFalse(config.models.auto_discover)
        self.assertEqual(config.models.quadlet_dir, Path.home() / "qd")
        self.assertEqual(config.models.explicit, {"model-a": "svc-a.service"})
        self.assertEqual(config.updates.tag_source, "registry")
        self.assertEqual(config.updates.backup_dir, Path.home() / "bk")
        self.assertEqual(config.updates.backup_keep, 2)
        self.assertEqual(config.dashboard.state_dir, Path.home() / "st")
        self.assertEqual(config.dashboard.public_endpoint, "http://host:9000")
        self.assertEqual(config.security.auth_token, "secret")
        self.assertFalse(config.security.allow_install)

    def test_env_var_selects_path(self):
        path = self.root / "env.toml"
        path.write_text("[router]\nport = 7000\n", encoding="utf-8")
        os.environ[settings.CONFIG_ENV] = str(path)
        self.assertEqual(resolve_config_path(), path)
        self.assertEqual(load().router.port, 7000)

    def test_explicit_path_beats_env(self):
        os.environ[settings.CONFIG_ENV] = str(self.root / "ignored.toml")
        path = self.root / "explicit.toml"
        path.write_text("[router]\nport = 7001\n", encoding="utf-8")
        self.assertEqual(load(path).router.port, 7001)

    def test_validation_errors_name_the_field(self):
        cases = {
            "[router]\nport = 0\n": "port",
            "[router]\nport = 'x'\n": "port",
            "[router]\nbind = 'notalist'\n": "bind",
            "[models]\nauto_discover = 'yes'\n": "auto_discover",
            "[updates]\ntag_source = 'gitlab'\n": "tag_source",
            "[dashboard]\nretention_days = -1\n": "retention_days",
            "[nope]\nx = 1\n": "nope",
        }
        for text, needle in cases.items():
            with self.subTest(needle=needle):
                path = self.root / f"bad-{needle}.toml"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ConfigError) as error:
                    load(path)
                self.assertIn(needle, str(error.exception))

    def test_malformed_toml_is_reported(self):
        path = self.root / "broken.toml"
        path.write_text("[router\nport = 1", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load(path)

    def test_explicit_section_type_validation(self):
        path = self.root / "e.toml"
        path.write_text("[models.explicit]\n'a' = 5\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load(path)


class SemverTests(unittest.TestCase):
    def test_stable_only(self):
        self.assertEqual(version_tuple("v0.13.1"), (0, 13, 1))
        self.assertEqual(normalize_version("0.13.10"), "0.13.10")
        for bad in ("latest", "0.13.2-rc1", "v0.13.1;rm", "0.01.2", "1.2"):
            self.assertIsNone(version_tuple(bad))


class DiscoveryTests(TempCase):
    def _write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_parses_all_fields_from_quadlet(self):
        spec, reason = parse_quadlet(self._write("a.container", good_quadlet()), REPO)
        self.assertIsNone(reason)
        self.assertEqual(spec.model_id, "qwen3.8-flash")
        self.assertEqual(spec.service, "halogen-official.service")
        self.assertEqual(spec.container, "halogen-official")
        self.assertEqual(spec.cache_path, Path("/home/u/halogen/cache/official"))
        self.assertEqual(spec.version, "0.13.1")

    def test_skips_foreign_image_and_invalid_version(self):
        self.assertIsNone(parse_quadlet(self._write("x.container", good_quadlet().replace(
            REPO, "ghcr.io/other/img")), REPO)[0])
        self.assertIsNone(parse_quadlet(self._write("y.container", good_quadlet(version="latest")), REPO)[0])

    def test_skips_missing_container_name_or_model_id(self):
        no_name = good_quadlet().replace("ContainerName=halogen-official\n", "")
        self.assertIsNone(parse_quadlet(self._write("n.container", no_name), REPO)[0])
        no_model = good_quadlet().replace("Environment=HALOGEN_MODEL_ID=qwen3.8-flash\n", "")
        self.assertIsNone(parse_quadlet(self._write("m.container", no_model), REPO)[0])

    def test_cache_path_optional(self):
        spec, reason = parse_quadlet(self._write("c.container", good_quadlet().replace(
            "Volume=/home/u/halogen/cache/official:/cache:Z\n", "")), REPO)
        self.assertIsNone(reason)
        self.assertIsNone(spec.cache_path)

    def test_discover_multiple_and_duplicate_skip(self):
        self._write("a.container", good_quadlet(container="halogen-official", model="m-a"))
        self._write("b.container", good_quadlet(container="halogen-uncensored", model="m-b",
                                              cache="/home/u/halogen/cache/uncensored"))
        self._write("foreign.container", good_quadlet(container="other", model="m-c").replace(REPO, "ghcr.io/x/y"))
        models, skipped = discover(self.root, REPO)
        self.assertEqual(set(models), {"m-a", "m-b"})
        self.assertEqual(models["m-b"].service, "halogen-uncensored.service")
        self.assertTrue(any("foreign" in s for s in skipped))

        # Duplicate model id in a second file is reported, not merged.
        self._write("dup.container", good_quadlet(container="halogen-dup", model="m-a"))
        models, skipped = discover(self.root, REPO)
        self.assertEqual(set(models), {"m-a", "m-b"})
        self.assertTrue(any("bereits" in s for s in skipped))

    def test_discover_missing_dir(self):
        models, skipped = discover(self.root / "nope", REPO)
        self.assertEqual(models, {})
        self.assertTrue(skipped)


if __name__ == "__main__":
    unittest.main()
