"""Pinned NPU records and user-supplied checkpoints for Halogen 0.18.21/22."""
from pathlib import Path
import tempfile
import unittest
from npu import (CUSTOM_PROGRAMS, NPU_MODELS, NpuManifest, bundled_manifest, check_runtime,
                 custom_model_path, model_name, parse_models, profile_model_names)
from profiles import official_template, validate_profile
from shared_assets import AssetError


class CatalogTests(unittest.TestCase):
    def test_pinned_records_and_unchanged_previous_bundles(self):
        for version, digest in (
            ('0.18.21', '51a5a669fddcb6336919c388196fbc6d83b81e21ec2d3174a1ed3baed9632507'),
            ('0.18.22', '97557b927561b822885c893076f6fbdf2f88e3576ff911e170e2cd55825dbec5')):
            manifest = bundled_manifest(version)
            self.assertEqual(manifest.key, digest)
            self.assertEqual(set(manifest.models), set(NPU_MODELS) | set(CUSTOM_PROGRAMS))
            for model in NPU_MODELS:
                self.assertEqual(manifest.select([model]), bundled_manifest('0.18.14').select([model]))
        for version in ('0.18.19', '0.18.20'):
            self.assertEqual(bundled_manifest(version).key, bundled_manifest('0.18.14').key)

    def test_lfm_and_d1_select_only_requested_programs_including_vision(self):
        manifest = bundled_manifest('0.18.22')
        for base in CUSTOM_PROGRAMS:
            selected = manifest.select(['/models/own-model'], [base])
            self.assertTrue(selected)
            self.assertEqual({asset.model for asset in selected}, {base})
            self.assertTrue(all(asset.name.startswith(('devices/', 'vis/')) for asset in selected))
            self.assertFalse(any(asset.name.endswith('model.safetensors') for asset in selected))
        self.assertIn('vis/devices.hnpm', {a.name for a in manifest.select(['/models/own-d1'], ['d1-2b'])})
        with self.assertRaisesRegex(AssetError, 'published programs'):
            bundled_manifest('0.18.21').select(['/models/own-d1'], ['d1-2b'])
        with self.assertRaisesRegex(AssetError, 'programs only'):
            manifest.select(['lfm3-1.7b'])

    def test_program_only_exception_does_not_weaken_other_manifest_checks(self):
        text = bundled_manifest('0.18.22').text
        for missing in ('file d1-2b vis/devices.hnpm ', 'file lfm3-1.7b devices/devices.hnpm ',
                        'file qwen3.5-2b qwen3.5-2b.hnpw '):
            broken = '\n'.join(line for line in text.splitlines() if not line.startswith(missing))
            with self.assertRaisesRegex(AssetError, 'Incomplete NPU manifest'):
                NpuManifest(broken)

    def test_fast_paths_have_correct_unique_model_ids_and_version_validation(self):
        values = '/models/LFM3-1.7B,/models/LFM3-1.7B:fast,/models/d1-2B'
        self.assertEqual(parse_models(values), values.split(','))
        profile = official_template('0.18.22', Path('/models'), Path('/cache'))
        profile.env['HALOGEN_NPU_MODELS'] = values
        self.assertEqual(profile_model_names(profile), {'LFM3-1.7B', 'LFM3-1.7B-fast', 'd1-2B'})
        self.assertFalse(validate_profile(profile))
        profile.image = profile.image.replace('0.18.22', '0.18.20')
        self.assertTrue(any('0.18.21' in e for e in validate_profile(profile)))
        self.assertEqual(custom_model_path('/models/LFM3:fast'), '/models/LFM3')
        self.assertEqual(model_name('/models/LFM3:fast'), 'LFM3-fast')
        for value in ('/models/../x:fast', '/models/npu/x:fast', '/models/x:fast:fast',
                      '/models/lfm3-1.7b', '/models/qwen3.5-2b:fast',
                      '/models/x:fast,/models/x-fast', '/models/' + 'a' * 95 + ':fast'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_models(value)


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_uses_read_only_directory_once_for_both_lfm_formats(self):
        calls = []
        async def run(*args, **kwargs):
            calls.append(args)
            return 'task=generate base=lfm3-1.7b' if 'probe' in args else 'usage: halogen-npu'
        result = await check_runtime(run, 'ghcr.io/peonist-ai/halogen-flash-server:0.18.22', [],
            [('/host/lfm', '/models/LFM3', 'ro,z')], custom_models=['/models/LFM3', '/models/LFM3:fast'])
        self.assertEqual(result, ['lfm3-1.7b', 'lfm3-1.7b-fast'])
        self.assertEqual(len(calls), 2)
        self.assertIn('/host/lfm:/models/LFM3:ro,z', calls[1])
        self.assertEqual(calls[1][-1], '/models/LFM3')

    async def test_d1_requires_calibration_and_new_engine_and_refuses_fast(self):
        async def run(*args, **kwargs):
            return 'task=sets base=d1-2b' if 'probe' in args else 'usage: halogen-npu'
        with tempfile.TemporaryDirectory() as directory:
            mounts = [(directory, '/models/d1-2B', 'ro,z')]
            image = 'ghcr.io/peonist-ai/halogen-flash-server:0.18.22'
            with self.assertRaisesRegex(AssetError, 'calibration.json'):
                await check_runtime(run, image, [], mounts, custom_models=['/models/d1-2B'])
            Path(directory, 'calibration.json').write_text('{}')
            self.assertEqual(await check_runtime(run, image, [], mounts, custom_models=['/models/d1-2B']), ['d1-2b'])
            with self.assertRaisesRegex(AssetError, 'only for LFM3'):
                await check_runtime(run, image, [], mounts, custom_models=['/models/d1-2B:fast'])
            with self.assertRaisesRegex(AssetError, 'newer engine'):
                await check_runtime(run, image.replace('22', '21'), [], mounts, custom_models=['/models/d1-2B'])

    async def test_custom_mounts_deduplicate_formats_without_exposing_suffix_as_a_path(self):
        from npu_install import NpuInstallMixin
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('config.json', 'model.safetensors', 'tokenizer.json'):
                (root / name).write_text('{}')
            profile = official_template('0.18.22', root, root / 'cache')
            profile.volumes = [(str(root), '/models/LFM3', 'ro,z')]
            instance = NpuInstallMixin()
            instance._validate_host_path = lambda value: Path(value)
            self.assertEqual(instance._npu_custom_mounts(profile, ['/models/LFM3', '/models/LFM3:fast']),
                             [(str(root), '/models/LFM3', 'ro,z')])
