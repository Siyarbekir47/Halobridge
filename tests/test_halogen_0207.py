"""0.20.7 draft preparation, QAD formats and native concurrency controls."""
import asyncio
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import draft_heads
from draft_heads import prepare_drafter
from npu import (bundled_manifest, check_runtime, custom_files_ready,
                 custom_model_path, model_name, parse_models)
from profiles import (official_template, uncensored_template, swift_quick_template,
                      parse_quadlet, render_quadlet, validate_profile)
from shared_assets import AssetError, SharedAssets, host_path
import test_updates as transactions

IMAGE = 'ghcr.io/peonist-ai/halogen-flash-server:0.20.7'
DATA = b'HGN1verified draft head fixture'
HEAD = replace(draft_heads.DRAFT_HEADS[(0,20,7)], size=len(DATA), digest=hashlib.sha256(DATA).hexdigest())


class CatalogTests(unittest.TestCase):
    def test_catalog_adds_only_qad_and_reuses_previous_programs(self):
        old, current = bundled_manifest('0.20.5'), bundled_manifest('0.20.7')
        self.assertEqual(current.key, 'eb434269616c65d6fc79d94c4f9ad8976e90cbff08f6a77c29e0a9229c5bc597')
        self.assertEqual(bundled_manifest('0.20.6').key, current.key)
        self.assertEqual(set(current.models), set(old.models)|{'lfm3-1.7b-qad'})
        self.assertEqual([a for a in current.assets if a.model != 'lfm3-1.7b-qad'], old.assets)
        selection = current.select(['/models/Liquid:qad'], ['lfm3-1.7b-qad'])
        self.assertEqual(len(selection), 11)
        self.assertEqual({a.revision for a in selection}, {'eb9877084a8017ebaa9eac384ebe7ce0bff67538'})
        self.assertTrue(all(a.name.startswith('devices/') for a in selection))

    def test_qad_ids_are_distinct_and_suffixes_cannot_be_stacked(self):
        models = parse_models('/models/Liquid,/models/Liquid:fast,/models/Liquid:qad')
        self.assertEqual([model_name(m) for m in models], ['Liquid','Liquid-fast','Liquid-qad'])
        self.assertEqual({custom_model_path(m) for m in models}, {'/models/Liquid'})
        for value in ('/models/Liquid:qad:fast', '/models/Liquid:fast:qad',
                      '/models/../Liquid:qad', '/models/Liquid:qad,/models/Liquid-qad'):
            with self.assertRaises(ValueError): parse_models(value)

    def test_controls_validate_without_changing_capacity(self):
        profile = official_template('0.20.7', Path('/models'), Path('/cache'))
        profile.env.update(HALOGEN_CONC_MODE='1', HALOGEN_PROMPT_CACHE='1')
        before = dict(profile.env)
        self.assertEqual(validate_profile(profile), [])
        self.assertEqual(parse_quadlet(render_quadlet(profile)).env, before)
        for key, value in (('HALOGEN_MTP','0'), ('HALOGEN_CHECKPOINT','/models/own.gguf'),
                           ('HALOGEN_CHECKPOINT','/models/qwen38-flash-next-w4b.hgn')):
            profile.env = {**before,key:value}
            self.assertTrue(any('HALOGEN_CONC_MODE' in e for e in validate_profile(profile)))
        profile.env = {**before,'HALOGEN_NPU_MODELS':'/models/Liquid:qad'}
        self.assertEqual(validate_profile(profile), [])
        profile.image = IMAGE.replace('0.20.7','0.20.5')
        errors = validate_profile(profile)
        self.assertTrue(any('CONC_MODE' in e for e in errors))
        self.assertTrue(any('Liquid:qad' in e for e in errors))


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_directory_probes_qad_separately_from_normal_and_fast(self):
        calls=[]
        async def run(*args, **kwargs):
            calls.append(args)
            return ('task=generate base=lfm3-1.7b-qad' if '--form' in args else
                    'task=generate base=lfm3-1.7b' if 'probe' in args else 'usage: halogen-npu')
        models=parse_models('/models/Liquid,/models/Liquid:fast,/models/Liquid:qad')
        result=await check_runtime(run, IMAGE, [], [('/host/liquid','/models/Liquid','ro,z')], custom_models=models)
        self.assertEqual(result,['lfm3-1.7b','lfm3-1.7b-fast','lfm3-1.7b-qad'])
        probes=[c for c in calls if 'probe' in c]
        self.assertEqual(len(probes),2)
        self.assertEqual(probes[1][-4:],('probe','/models/Liquid','--form','qad'))
        with self.assertRaisesRegex(AssetError,'newer engine'):
            await check_runtime(run, IMAGE.replace('0.20.7','0.20.5'), [],
                                [('/host/liquid','/models/Liquid','ro,z')], custom_models=[models[-1]])
        bad=AsyncMock(side_effect=lambda *a,**k: 'task=generate base=ovisocr3' if 'probe' in a else 'usage: halogen-npu')
        with self.assertRaisesRegex(AssetError,'Unsupported'):
            await check_runtime(bad, IMAGE, [], [('/host/ocr','/models/OCR','ro')],custom_models=['/models/OCR:qad'])

    async def test_qad_only_directory_needs_no_safetensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('config.json','tokenizer.json','LFM3-1.7B-QAD-Q4_0.gguf'):
                (root/name).write_text('{}')
            self.assertTrue(custom_files_ready(root,'/models/Liquid:qad'))
            self.assertTrue(custom_files_ready(root,'/models/Liquid'))
            self.assertFalse(custom_files_ready(root,'/models/Liquid:fast'))
            (root/'LFM3-1.7B-QAD-Q4_0.gguf').unlink()
            self.assertFalse(custom_files_ready(root,'/models/Liquid:qad'))


class DraftTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.quadlets=self.root/'quadlets'; self.quadlets.mkdir()
        self.assets=SharedAssets(self.root/'models',self.quadlets,self.root/'state')
        self.profile=official_template('0.20.5',self.root/'models',self.root/'cache')
        self.content=render_quadlet(self.profile).replace('\n','\r\n').encode()
        (self.quadlets/'halogen-official.container').write_bytes(self.content)
        self.pin=patch.dict(draft_heads.DRAFT_HEADS,{(0,20,7):HEAD}); self.pin.start()
        self.downloads=[]
        async def download(asset,stage):
            self.downloads.append(asset.name)
            (stage/asset.name).write_bytes(DATA)
        self.download=download

    async def asyncTearDown(self):
        self.pin.stop(); self.temp.cleanup()

    async def test_only_head_downloads_and_readonly_mounts_keep_all_settings(self):
        result=await prepare_drafter(self.content,self.assets,IMAGE,self.download,lambda _:None)
        after=parse_quadlet(result.decode())
        self.assertEqual(self.downloads,[HEAD.name])
        self.assertEqual(after.env,self.profile.env)
        self.assertEqual(after.volumes[:-1],self.profile.volumes)
        self.assertEqual(after.volumes[-1],(str(self.assets.shared_path(HEAD)),'/models/'+HEAD.name,'ro,z'))
        self.assertNotIn(b'\n',result.replace(b'\r\n',b''))
        self.assertEqual((self.quadlets/'halogen-official.container').read_bytes(),self.content)
        again=await prepare_drafter(result,self.assets,IMAGE,self.download,lambda _:None)
        self.assertEqual(again,result)
        self.assertEqual(self.downloads,[HEAD.name])

    async def test_existing_head_is_verified_and_imported_without_download(self):
        own=host_path(self.profile,'/models/'+HEAD.name); own.parent.mkdir(parents=True)
        own.write_bytes(DATA)
        await prepare_drafter(self.content,self.assets,IMAGE,self.download,lambda _:None)
        self.assertEqual(self.downloads,[])
        self.assertEqual(self.assets.shared_path(HEAD).read_bytes(),DATA)
        self.assertEqual(own.read_bytes(),DATA)

    async def test_legacy_file_mount_spacing_is_replaced_without_duplicates(self):
        first=await prepare_drafter(self.content,self.assets,IMAGE,self.download,lambda _:None)
        text=first.decode().replace('Volume='+str(self.assets.shared_path(HEAD)),
                                  '  Volume = '+str(self.assets.shared_path(HEAD)))
        text=text.replace('/models/'+HEAD.name+':ro,z','/models/'+HEAD.name)
        result=await prepare_drafter(text.encode(),self.assets,IMAGE,self.download,lambda _:None)
        mounts=[v for v in parse_quadlet(result.decode()).volumes if v[1]=='/models/'+HEAD.name]
        self.assertEqual(len(mounts),1)
        self.assertEqual(mounts[0][2],'ro,z')

    async def test_optouts_and_other_models_never_receive_an_official_head(self):
        profiles=[uncensored_template('0.20.7',self.root/'models',self.root/'cache')]
        profiles += [swift_quick_template(v,self.root/'models',self.root/'cache',self.assets.locations(v))
                     for v in ('swift15','swift15-abliterated','cyber-frost')]
        for key,value in (('HALOGEN_MTP','0'),('HALOGEN_CK_OVERLAY','none'),
                          ('HALOGEN_CK_OVERLAY','/models/custom.hgn'),('HALOGEN_CHECKPOINT','/models/custom.hgn')):
            profiles.append(replace(self.profile,env={**self.profile.env,key:value}))
        for profile in profiles:
            content=render_quadlet(profile).encode()
            self.assertEqual(await prepare_drafter(content,self.assets,IMAGE,self.download,lambda _:None),content)
        self.assertEqual(self.downloads,[])

    async def test_corrupt_head_does_not_modify_configuration(self):
        async def corrupt(asset,stage): (stage/asset.name).write_bytes(b'invalid')
        with self.assertRaisesRegex(AssetError,'Checksum mismatch'):
            await prepare_drafter(self.content,self.assets,IMAGE,corrupt,lambda _:None)
        self.assertFalse(self.assets.shared_path(HEAD).exists())
        self.assertEqual((self.quadlets/'halogen-official.container').read_bytes(),self.content)

    async def test_missing_managed_head_blocks_start_without_downloading(self):
        prepared=await prepare_drafter(self.content,self.assets,IMAGE,self.download,lambda _:None)
        (self.quadlets/'halogen-official.container').write_bytes(prepared)
        self.assets.shared_path(HEAD).unlink()
        with self.assertRaisesRegex(AssetError,'draft head'):
            await self.assets.validate_start(self.profile.model_id)
        self.assertEqual(self.downloads,[HEAD.name])


class UpdateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.versions=patch.multiple(transactions,OLD='0.20.5',NEW='0.20.7'); self.versions.start()
        self.fixture=transactions.TransactionTests(); await self.fixture.asyncSetUp()
        self.updater=self.fixture.updater
        self.updater.checkpoint_assets=SharedAssets(self.fixture.root/'models',self.fixture.quadlets,self.fixture.root/'state')
        self.pin=patch.dict(draft_heads.DRAFT_HEADS,{(0,20,7):HEAD}); self.pin.start()
        self.before={}
        for maker in (official_template,uncensored_template):
            profile=maker('0.20.5',self.fixture.root/'models',self.fixture.root/'cache')
            path=self.fixture.quadlets/(profile.container_name+'.container')
            path.write_text(render_quadlet(profile)); self.before[path]=path.read_bytes()
        self.downloads=[]
        async def download(asset,stage):
            self.assertFalse(self.fixture.manager.maintenance)
            self.downloads.append(asset.name); (stage/asset.name).write_bytes(DATA)
        self.updater._download_checkpoint_asset=download

    async def asyncTearDown(self):
        self.pin.stop(); await self.fixture.asyncTearDown(); self.versions.stop()

    async def test_engine_update_prepares_head_before_drain_and_backs_up_old_bindings(self):
        await self.fixture.execute()
        self.assertEqual(self.updater.job['phase'],'succeeded',self.updater.job)
        self.assertEqual(self.downloads,[HEAD.name])
        for path,original in self.before.items():
            self.assertEqual((Path(self.updater.job['backup_dir'])/path.name).read_bytes(),original)
            profile=parse_quadlet(path.read_text())
            self.assertEqual(profile.env,parse_quadlet(original.decode()).env)
            self.assertTrue(profile.image.endswith(':0.20.7'))

    async def test_activation_failure_restores_both_profiles_and_keeps_prepared_file(self):
        self.fixture.fail_new_start=True
        await self.fixture.execute()
        self.assertEqual(self.updater.job['phase'],'rolled_back',self.updater.job)
        for path,original in self.before.items(): self.assertEqual(path.read_bytes(),original)
        self.assertEqual(self.updater.checkpoint_assets.shared_path(HEAD).read_bytes(),DATA)
        self.assertFalse(self.fixture.manager.maintenance)

    async def test_cancel_during_head_download_keeps_serving_and_resumable_files(self):
        entered=asyncio.Event()
        async def paused(asset,stage):
            (stage/'resumable.partial').write_bytes(b'partial download')
            entered.set()
            await asyncio.Event().wait()
        self.updater._download_checkpoint_asset=paused
        await self.updater.start('0.20.7')
        await asyncio.wait_for(entered.wait(),2)
        self.assertFalse(self.fixture.manager.maintenance)
        self.updater.task.cancel()
        await self.updater.task
        for path,original in self.before.items(): self.assertEqual(path.read_bytes(),original)
        self.assertTrue((self.updater.checkpoint_assets.staging_root(HEAD)/'resumable.partial').exists())
        self.assertFalse(any(c[:3]==('systemctl','--user','stop') for c in self.fixture.commands))
        self.assertFalse(self.fixture.manager.maintenance)


if __name__=='__main__': unittest.main()
