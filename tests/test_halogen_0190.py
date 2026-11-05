"""Published 0.19.0 catalog and multimodal program dependencies."""
from pathlib import Path
import unittest
from npu import (AGENT_GUARD_MODELS, NpuManifest, bundled_manifest,
                 check_runtime, profile_model_names, patch_quadlet)
from profiles import official_template, render_quadlet, parse_quadlet, validate_profile
from shared_assets import AssetError


class CatalogTests(unittest.TestCase):
    def test_image_catalogs_and_spotter_revision_are_pinned(self):
        previous = bundled_manifest('0.18.22')
        for version, digest in [('0.18.23','d15c0dab59162e6ccaff81f0b5708f3ef667be2f1bba47a56e2a017f43f4ca58'),
                                ('0.19.0','351ab0519d16d4d5442562246b49df34635ec656fe2970c952cd2ef30a76553d')]:
            current = bundled_manifest(version)
            self.assertEqual(current.key, digest)
            for name in previous.models:
                self.assertEqual([a for a in current.assets if a.model == name],
                                 [a for a in previous.assets if a.model == name])
        manifest = bundled_manifest('0.19.0')
        files = manifest.select(list(AGENT_GUARD_MODELS))
        self.assertTrue(all(a.revision and a.digest for a in files))
        self.assertEqual(manifest.models['halogen-spotter-0.6b']['revision'], '827eb67e9631fae618ed66b1b4d1b8ba74be8fca')
        # The exception applies only to the verified byte-identical catalog.
        with self.assertRaisesRegex(AssetError, 'Incomplete NPU manifest'):
            NpuManifest(manifest.text + '\n')

    def test_composite_models_select_only_matching_text_vision_and_audio(self):
        manifest = bundled_manifest('0.19.0')
        for base, expected in [
            ('lfm3-vl-2b-fast', {('lfm3-1.7b-fast','devices'),('lfm3-vl-2b','vis')}),
            ('lfm3-audio-2b', {('lfm3-1.7b','devices'),('lfm3-audio-2b','aud')}),
            ('lfm3-omni-2b-fast', {('lfm3-1.7b-fast','devices'),('lfm3-vl-2b','vis'),('lfm3-audio-2b','aud')})]:
            selected = manifest.select(['/models/own-model'], [base])
            self.assertEqual({(a.model,a.name.split('/')[0]) for a in selected}, expected)
            self.assertTrue(all(a.name.endswith(('.elf','.hnpm','.hnpb','.md')) for a in selected))
        for missing in ['file lfm3-vl-2b vis/devices.hnpm ', 'file lfm3-audio-2b aud/devices.hnpm ']:
            broken = '\n'.join(s for s in manifest.text.splitlines() if not s.startswith(missing))
            # Supply the existing verified Spotter revision so this checks the missing program.
            broken = broken.replace('revision= task=classify', 'revision=827eb67e9631fae618ed66b1b4d1b8ba74be8fca task=classify')
            with self.assertRaisesRegex(AssetError, 'Incomplete NPU manifest'):
                NpuManifest(broken)

    def test_guard_only_profile_keeps_npu_device_and_implicit_names(self):
        profile = official_template('0.19.0',Path('/host/models'),Path('/host/cache'))
        for guard in ('watch','hold'):
            profile.env['HALOGEN_AGENT_GUARD'] = guard
            self.assertEqual(validate_profile(profile), [])
            self.assertEqual(profile_model_names(profile), set(AGENT_GUARD_MODELS))
            original = render_quadlet(profile).encode()
            updated = patch_quadlet(original, models='', shared_root=Path('/host/shared'))
            self.assertEqual(updated.count(b'AddDevice=/dev/accel/accel0'),1)
            self.assertEqual(profile_model_names(parse_quadlet(updated.decode())),set(AGENT_GUARD_MODELS))
            disabled = patch_quadlet(updated, models='', guard='off')
            self.assertNotIn(b'AddDevice=/dev/accel/accel0',disabled)
            self.assertFalse(profile_model_names(parse_quadlet(disabled.decode())))
        profile.image = profile.image.replace('0.19.0','0.18.23')
        self.assertTrue(any('0.19.0' in x for x in validate_profile(profile)))
        profile.env['HALOGEN_AGENT_GUARD'] = 'invalid'
        self.assertTrue(validate_profile(profile))


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_multimodal_formats_probe_then_choose_their_programs(self):
        for base in ('lfm3-vl-2b','lfm3-audio-2b','lfm3-omni-2b'):
            async def run(*args,**kwargs):
                return f'task=generate base={base}' if 'probe' in args else 'usage: halogen-npu'
            selected = await check_runtime(run,'ghcr.io/peonist-ai/halogen-flash-server:0.19.0',[],
                [('/host/model','/models/own','ro,z')],custom_models=['/models/own','/models/own:fast'])
            self.assertEqual(selected,[base,base+'-fast'])
            with self.assertRaisesRegex(AssetError,'newer engine'):
                await check_runtime(run,'ghcr.io/peonist-ai/halogen-flash-server:0.18.22',[],
                    [('/host/model','/models/own','ro,z')],custom_models=['/models/own'])
