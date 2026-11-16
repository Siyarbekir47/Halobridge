"""Published 0.20.x catalogs, guard dependencies and reversible settings."""
from pathlib import Path
import unittest

from npu import NpuManifest, bundled_manifest, profile_models, profile_model_names
from profiles import official_template, parse_quadlet, render_quadlet, validate_profile
from shared_assets import AssetError


class CatalogTests(unittest.TestCase):
    def test_exact_catalogs_and_program_updates_keep_other_assets(self):
        hashes = {
            '0.20.1': 'ffb92f3e407af1613b6b5e3f683e39e2a5df46608286901327128a8339bb8ffe',
            '0.20.2': 'ffd15d40a6d990ddaa94a13918a0ab56a4d151d769aea4dc8cc1fe206de6a172',
            '0.20.3': '2bd3520c95285159cae4d5007548fa62a986de76df201a92a07672a01307412f',
            '0.20.4': '2bd3520c95285159cae4d5007548fa62a986de76df201a92a07672a01307412f',
            '0.20.5': '004923db1eb4703ad720322ff520ed2aea3aadcd4968fef814facf4d050ca81d',
        }
        old = bundled_manifest('0.20.0')
        for version, digest in hashes.items():
            current = bundled_manifest(version)
            self.assertEqual(current.key, digest)
            self.assertEqual(set(current.models), set(old.models))
            changed = ({'d3-lite'} if version == '0.20.2' else
                       {'d3-lite', 'd3-nano'} if version in {'0.20.3', '0.20.4'} else
                       {'d3-lite', 'd3-nano', 'lfm3-1.7b', 'lfm3-1.7b-fast'} if version == '0.20.5' else set())
            self.assertEqual([a for a in current.assets if a.model not in changed],
                             [a for a in old.assets if a.model not in changed])
            # The blank-revision records themselves are unchanged; only exact
            # image hashes may receive the previously verified HF pins.
            for model in ('halogen-spotter-0.6b', 'ovisocr3'):
                lines = lambda m: [s for s in m.text.splitlines() if s.startswith((f'model {model} ', f'file {model} '))]
                self.assertEqual(lines(current), lines(old))
        current = bundled_manifest('0.20.5')
        self.assertEqual(current.models['lfm3-1.7b']['revision'], 'b9b15f09c8a8ecb2550791e5fb806674a310c8da')
        self.assertEqual(current.models['lfm3-1.7b-fast']['revision'], 'aa6207e4d84fc10c669629da29c133248a473429')
        selected = current.select(['/models/LFM3-VL-2B:fast'], ['lfm3-vl-2b-fast'])
        self.assertEqual({a.model for a in selected}, {'lfm3-1.7b-fast', 'lfm3-vl-2b'})
        self.assertTrue(any(a.name == 'devices/u7.elf' for a in selected))
        with self.assertRaises(AssetError):
            NpuManifest(current.text + '# unknown catalog\n')


class ProfileTests(unittest.TestCase):
    def test_watch_dependencies_follow_engine_version_and_keep_explicit_decider(self):
        for version in ('0.19.0', '0.20.0', '0.20.4', '0.20.5', '0.21.0'):
            profile = official_template(version, Path('/models'), Path('/cache'))
            profile.env['HALOGEN_AGENT_GUARD'] = 'watch'
            expected = {'halogen-spotter-0.6b'}
            if version in ('0.19.0', '0.20.0', '0.20.4'):
                expected.add('decider-4b')
            self.assertEqual(profile_model_names(profile), expected)
            self.assertEqual(set(profile_models(profile, guard='hold')),
                             {'halogen-spotter-0.6b', 'decider-4b'})
            self.assertEqual(profile_models(profile, guard='off'), [])
            self.assertEqual(set(profile_models(profile, 'decider-4b', 'watch')),
                             {'halogen-spotter-0.6b', 'decider-4b'})

    def test_new_settings_round_trip_and_reject_invalid_or_older_engine_values(self):
        profile = official_template('0.20.5', Path('/models'), Path('/cache'))
        values = {'HALOGEN_DN_STATE': 'fp32', 'HALOGEN_TOKENIZER_NATIVE': '0',
                  'HALOGEN_NPU_KILL_S': '1800', 'HALOGEN_SERVE_TRACE': '1'}
        profile.env.update(values)
        self.assertEqual(validate_profile(profile), [])
        self.assertEqual(parse_quadlet(render_quadlet(profile)).env, profile.env)
        for key, bad in (('HALOGEN_DN_STATE', 'fp16'), ('HALOGEN_TOKENIZER_NATIVE', 'yes'),
                         ('HALOGEN_NPU_KILL_S', '-1'), ('HALOGEN_NPU_KILL_S', '1.5'),
                         ('HALOGEN_SERVE_TRACE', 'verbose')):
            profile.env[key] = bad
            self.assertTrue(any(key in error for error in validate_profile(profile)))
            profile.env[key] = values[key]
        profile.env['HALOGEN_NPU_KILL_S'] = '0'
        self.assertEqual(validate_profile(profile), [])
        profile.image = profile.image.rsplit(':', 1)[0] + ':0.20.4'
        for key in values:
            self.assertTrue(any(key in error and '0.20.5' in error for error in validate_profile(profile)))

    def test_retired_auto_mapping_is_preserved_without_requiring_named_keeps(self):
        profile = official_template('0.20.5', Path('/models'), Path('/cache'))
        profile.env.update(HALOGEN_FORK_MAP='auto', HALOGEN_FORK_AUTO_KEEPS='4')
        self.assertEqual(validate_profile(profile), [])
        self.assertEqual(parse_quadlet(render_quadlet(profile)).env, profile.env)
        profile.env['HALOGEN_FORK_MAP'] = 'claude-code,auto'
        self.assertTrue(any('HALOGEN_FORK=1' in error for error in validate_profile(profile)))
        profile.env['HALOGEN_FORK'] = '1'
        self.assertEqual(validate_profile(profile), [])


if __name__ == '__main__':
    unittest.main()
