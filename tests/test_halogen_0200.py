"""Image-owned XRT and verified OvisOCR3 document programs."""
from pathlib import Path
import tempfile
import unittest
from npu import (XRT_LIBS, NpuManifest, bundled_manifest, check_runtime,
                 host_status, runtime_mounts, parse_models, patch_quadlet)
from profiles import official_template, render_quadlet
from shared_assets import AssetError


class CatalogTests(unittest.TestCase):
    def test_image_catalog_pins_ocr_and_reuses_all_previous_assets(self):
        old = bundled_manifest('0.19.0')
        self.assertEqual(bundled_manifest('0.19.1').key,old.key)
        current = bundled_manifest('0.20.0')
        self.assertEqual(current.key,'ffb92f3e407af1613b6b5e3f683e39e2a5df46608286901327128a8339bb8ffe')
        self.assertEqual(set(current.models),set(old.models)|{'ovisocr3'})
        self.assertEqual([a for a in current.assets if a.model!='ovisocr3'],old.assets)
        selected = current.select(['/models/OvisOCR3'],['ovisocr3'])
        self.assertEqual(len(selected),20)
        self.assertEqual({a.model for a in selected},{'ovisocr3'})
        self.assertEqual({a.revision for a in selected},{'588f22c6909e7c69d440072d96bb0d646f4f1998'})
        self.assertEqual({a.name.split('/')[0] for a in selected},{'devices','vis'})
        self.assertEqual(parse_models('/models/OvisOCR3'),['/models/OvisOCR3'])
        with self.assertRaises(ValueError): parse_models('ovisocr3')
        for missing in ('file ovisocr3 vis/devices.hnpm ', 'file ovisocr3 devices/devices.hnpm '):
            text='\n'.join(s for s in current.text.splitlines() if not s.startswith(missing))
            text=text.replace('revision= task=classify','revision=827eb67e9631fae618ed66b1b4d1b8ba74be8fca task=classify')
            text=text.replace('revision= task=generate source=ATH-MaaS/','revision=588f22c6909e7c69d440072d96bb0d646f4f1998 task=generate source=ATH-MaaS/')
            with self.assertRaisesRegex(AssetError,'Incomplete NPU manifest'): NpuManifest(text)


class RuntimeTests(unittest.TestCase):
    def test_bundled_xrt_removes_only_the_host_library_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            node=root/'dev/accel/accel0'; node.parent.mkdir(parents=True); node.touch()
            proc=root/'proc'; proc.mkdir(); (proc/'cmdline').write_text('iommu=pt')
            device=root/'sys/class/drm/card0/device'; device.mkdir(parents=True)
            (device/'pp_dpm_fclk').write_text('0: 400Mhz\n1: 1200Mhz *')
            (device/'power_dpm_force_performance_level').write_text('high')
            prefix='ghcr.io/peonist-ai/halogen-flash-server:'
            for version in ('0.19.1','0.20.0'):
                state=host_status(root,image=prefix+version)
                self.assertTrue(state['ready'],state)
                self.assertEqual(state['xrt_source'],'image')
                self.assertEqual(state['xrt_mounts'],[])
            state=host_status(root,image=prefix+'0.19.0')
            self.assertFalse(state['ready'])
            self.assertTrue(any('XRT' in e for e in state['errors']))
            (proc/'cmdline').write_text('iommu=off'); node.unlink()
            (device/'power_dpm_force_performance_level').write_text('auto')
            state=host_status(root,image=prefix+'0.20.0')
            self.assertEqual(len(state['errors']),3)

    def test_new_profiles_use_image_xrt_and_complete_overrides_are_preserved(self):
        profile=official_template('0.20.0',Path('/models'),Path('/cache'))
        host=[('/host/xrt','/opt/xilinx/xrt','ro')]
        state={'xrt_mounts':host}
        self.assertEqual(runtime_mounts(profile,state),[])
        self.assertEqual(runtime_mounts(profile,state,profile.image.replace('0.20.0','0.19.0')),host)
        profile.volumes+=host
        self.assertEqual(runtime_mounts(profile,state),host)
        profile.volumes=[(f'/host/{name}',f'/opt/xilinx/xrt/lib/{name}','ro') for name in XRT_LIBS]
        mounts=runtime_mounts(profile,state)
        self.assertEqual(len(mounts),6)
        for name in XRT_LIBS:
            self.assertIn((f'/host/{name}',f'/opt/xilinx/lib/x86_64-linux-gnu/{name}','ro'),mounts)
        content=render_quadlet(profile).encode()
        content=patch_quadlet(content,models='qwen3.5-2b',mounts=mounts)
        self.assertNotIn(b'/opt/xilinx/lib/',patch_quadlet(content,models=''))
        profile.volumes.pop()
        with self.assertRaisesRegex(AssetError,'all three'): runtime_mounts(profile,state)
        profile.volumes=host+[(f'/host/{XRT_LIBS[0]}',f'/opt/xilinx/xrt/lib/{XRT_LIBS[0]}','ro')]
        with self.assertRaisesRegex(AssetError,'all three'): runtime_mounts(profile,state)


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_ocr_probe_uses_image_runtime_and_has_no_fast_format(self):
        calls=[]
        async def run(*args,**kwargs):
            calls.append(args)
            return 'task=generate base=ovisocr3' if 'probe' in args else 'usage: halogen-npu'
        image='ghcr.io/peonist-ai/halogen-flash-server:0.20.0'
        mounts=[('/host/ocr','/models/OvisOCR3','ro,z')]
        self.assertEqual(await check_runtime(run,image,[],mounts,custom_models=['/models/OvisOCR3']),['ovisocr3'])
        self.assertFalse(any('/opt/xilinx' in arg for call in calls for arg in call))
        with self.assertRaisesRegex(AssetError,'newer engine'):
            await check_runtime(run,image.replace('0.20.0','0.19.1'),[],mounts)
        with self.assertRaisesRegex(AssetError,'only for LFM3'):
            await check_runtime(run,image,[],mounts,custom_models=['/models/OvisOCR3:fast'])
