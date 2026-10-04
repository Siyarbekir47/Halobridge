"""Local UI fixture. No system services, model files, or real backend are used.

Run: python tests/preview_dashboard.py
"""
import time
from pathlib import Path, PurePosixPath
import sys

from aiohttp import web
from test_dashboard import database, insert

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from profiles import official_template, custom_template, swift_quick_template, profile_to_dict, ENV_FIELDS
from swift_catalog import SHARED_ASSETS, variant_info
from official_checkpoints import OFFICIAL_CHECKPOINTS
from halogen_router import _login_response, _request_lang
from npu import NPU_MODELS, TASK_ROUTES, bundled_manifest, parse_models


def create_preview():
    dashboard = database()
    for i in range(32):
        insert(dashboard, request_id=f'preview-{i}', completed_at=time.time() - 3600 - i * 2200,
               client='OpenCode' if i % 3 else 'Open WebUI', input_tokens=1000+i*90,
               output_tokens=100+i*10, cached_tokens=800 if i % 2 else 0,
               status=200 if i % 9 else 499, duration_ms=3200+i*50)
    insert(dashboard, request_id='preview-missing', completed_at=time.time() - 600,
           input_tokens=None, output_tokens=None, cached_tokens=None, reasoning_tokens=None)
    async def snapshot():
        return {'generated_at': time.time(), 'app': {'version': app_updates['current_version']},
                'api': {'active_model': 'qwen3.8-flash', 'status': 'ok'},
                'backend': {'status': 'ok', 'in_flight': 0, 'queued': 0, 'context': 262144,
                            'slots': 4, 'max_tokens_default': 8192, 'max_tokens_cap': 65536,
                            'reasoning_effort_default': 'xhigh', 'version': {'api': '0.16.2', 'engine': '0.16.2'}},
                'system': {'gpu_busy_percent': 8, 'memory': {'used_bytes': 62*1024**3, 'total_bytes': 128*1024**3},
                           'disk_used_bytes': 420*1024**3, 'disk_total_bytes': 2000*1024**3},
                'cache': {'pool': {'usage_ratio': .3}, 'model_bytes': {},
                          'counters': {'hits': 32, 'evicted': 2, 'dropped': 7}}, 'active_requests': []}
    dashboard.snapshot = snapshot
    original_page = dashboard.page
    async def fixture_page(request):
        response = await original_page(request)
        response.text = response.text.replace('</main>', '''
          <div class="controls" aria-label="Test fixture controls">
            <form method="post" action="/preview/request/complete"><button>Add complete test request</button></form>
            <form method="post" action="/preview/request/incomplete"><button>Add incomplete test request</button></form>
          </div></main>''')
        return response
    dashboard.page = fixture_page
    app = web.Application()
    dashboard.register_routes(app)
    async def fixture_request(request):
        kind = request.match_info['kind']
        if kind not in {'complete', 'incomplete'}:
            raise web.HTTPBadRequest()
        fields = {} if kind == 'complete' else dict(input_tokens=None, output_tokens=None,
                                                   cached_tokens=None, reasoning_tokens=None)
        insert(dashboard, request_id=f'fixture-{time.time_ns()}', completed_at=time.time(), **fields)
        raise web.HTTPSeeOther(location='/dashboard')
    app.router.add_post('/preview/request/{kind}', fixture_request)
    choices = []
    for key, choice in OFFICIAL_CHECKPOINTS.items():
        files = []
        for asset in [choice['asset'], *SHARED_ASSETS]:
            checkpoint = asset == choice['asset']
            missing = checkpoint and key == 'ht43'
            path = '/home/demo/models/' + ('official/' if checkpoint else 'shared/halogen-v2/' + asset.revision + '/') + asset.name
            files.append({'name': asset.name, 'path': path, 'status': 'download' if missing else 'verified',
                          'size': asset.size, 'download_bytes': asset.size if missing else 0,
                          'url': f'https://huggingface.co/{asset.repo}/blob/{asset.revision}/{asset.name}'})
        choices.append({'target_key': key, 'label': choice['label'], 'target': '/models/' + choice['asset'].name,
                        'available': key == 'ht43', 'can_install': key == 'ht43', 'optional': key == 'ht43',
                        'needs_download': key == 'ht43', 'download_gib': 54 if key == 'ht43' else 0,
                        'plan': {'files': files, 'filesystems': [{'path': '/home/demo/models', 'free_bytes': 200 * 2**30,
                                 'required_bytes': choice['asset'].size if key == 'ht43' else 0,
                                 'reserve_bytes': 5 * 2**30 if key == 'ht43' else 0}]}})
    updates = {'current_version': '0.16.2', 'latest_version': '0.16.2', 'checked_at': time.time(),
               'configured_versions': {'qwen3.8-flash': '0.16.2'}, 'up_to_date': True, 'running': False,
               'supported': True, 'support_error': None, 'checkpoint': {
                   'current': choices[0]['target'], 'target': choices[0]['target'], 'target_key': 'v2',
                   'label': 'v2', 'available': False, 'choices': choices}}
    checkpoint_polls = 0
    async def update_status(request):
        nonlocal checkpoint_polls
        if updates['running']:
            checkpoint_polls += 1
            updates['job']['updated_at'] = time.time()
            updates['job']['phase'] = 'preparing' if checkpoint_polls < 2 else 'verifying'
            if checkpoint_polls >= 3:
                updates['running'] = False
                updates['job']['phase'] = 'succeeded'
                updates['checkpoint']['current'] = updates['job']['target']
                for choice in choices:
                    choice.update(available=choice['target'] != updates['job']['target'],
                                  can_install=choice['target'] != updates['job']['target'])
                    if choice['target'] == updates['job']['target']:
                        choice.update(needs_download=False, download_gib=0)
                        for file in choice['plan']['files']:
                            file.update(status='verified', download_bytes=0)
        return web.json_response(updates)
    async def install_checkpoint(request):
        nonlocal checkpoint_polls
        payload = await request.json()
        choice = next(c for c in choices if c['target_key'] == payload['target'])
        checkpoint_polls = 0
        updates.update(running=True, job={'kind': 'checkpoint', 'target_key': choice['target_key'],
                       'target': choice['target'], 'label': choice['label'], 'needs_download': choice['needs_download'],
                       'download_gib': choice['download_gib'], 'prepared_download': True,
                       'started_at': time.time(), 'updated_at': time.time(), 'phase': 'preparing'})
        return web.json_response(updates, status=202)
    app.router.add_get('/dashboard/api/updates', update_status)
    app.router.add_post('/dashboard/api/updates/check', update_status)
    app.router.add_post('/dashboard/api/updates/checkpoint', install_checkpoint)
    # The self-update interaction is simulated; no installation or restart occurs.
    app_updates = {'current_version': '0.1.25', 'latest_version': '0.1.25',
                   'checked_at': time.time(), 'source_ref': 'develop',
                   'update_available': False, 'up_to_date': True, 'running': False,
                   'supported': True, 'can_install': False, 'job': None,
                   'release_url': 'https://github.com/Siyarbekir47/Halobridge/blob/develop/CHANGELOG.md'}
    app_poll_count = 0
    async def app_update_status(request):
        nonlocal app_poll_count
        if request.match_info.get('action') == 'install':
            data = await request.json()
            assert data['version'] == '0.1.24'
            assert request.headers.get('X-Halogen-Action') == 'update'
            app_updates.update(running=True, can_install=False,
                               job={'phase': 'installing', 'message': 'Installing the verified Halobridge version.', 'target_version': '0.1.24'})
        elif app_updates['running']:
            app_poll_count += 1
            if app_poll_count >= 3:
                app_updates.update(current_version='0.1.24', update_available=False, up_to_date=True,
                                   running=False, job={'phase': 'succeeded', 'message': 'Halobridge was updated and the user service restarted.', 'target_version': '0.1.24'})
        return web.json_response(app_updates)
    app.router.add_get('/dashboard/api/app-updates', app_update_status)
    app.router.add_post('/dashboard/api/app-updates/{action}', app_update_status)
    profile = profile_to_dict(official_template('0.16.2', PurePosixPath('/home/demo/models'), PurePosixPath('/home/demo/cache')))
    profile['env'].update(HALOGEN_CACHE_EVICT='0', HALOGEN_NPU_EMB_BATCH='0', HALOGEN_NPU_VERIFY='1')
    profile['volumes'].append(['/opt/xilinx/xrt', '/opt/xilinx/xrt', 'ro'])
    npu_ready = True
    npu_space = True
    async def npu_fixture(request):
        nonlocal npu_ready, npu_space
        data = await request.json()
        npu_ready = data.get('ready', npu_ready)
        npu_space = data.get('space', npu_space)
        return web.json_response({'ok': True})
    async def npu_status(request):
        return web.json_response({'ready': npu_ready,
            'errors': [] if npu_ready else ['GPU fabric clock is not held; install and start halogen-fabric-clock.service before enabling NPU'],
            'setup': {'commands': ['halobridge npu-host-setup', 'halobridge npu-check']},
            'models': [{'id': model, 'task': task, 'endpoint': TASK_ROUTES[task]} for model, task in NPU_MODELS.items()],
            'profiles': [{'id': 'official', 'image': profile['image'], 'models': profile['env'].get('HALOGEN_NPU_MODELS', '')}]})
    async def npu_plan(request):
        models = parse_models(request.query['models']) if request.query.get('models') else []
        manifest = bundled_manifest()
        files = [{'name': asset.model + '/' + asset.name,
                  'path': '/home/demo/models/shared/halogen-npu/' + manifest.key + '/' + asset.model + '/' + asset.name,
                  'size': asset.size, 'status': 'download', 'download_bytes': asset.size,
                  'url': f'https://huggingface.co/{asset.repo}/blob/{asset.revision}/{asset.name}'}
                 for asset in manifest.select(models)]
        size = sum(file['size'] for file in files)
        return web.json_response({'models': models, 'profiles': ['official'], 'download_bytes': size,
            'plans': [{'files': files, 'filesystems': [{'path': '/home/demo/models', 'free_bytes': 80*2**30 if npu_space else 0,
                      'required_bytes': size, 'reserve_bytes': 5*2**30, 'sufficient': npu_space}]}]})
    async def npu_install(request):
        nonlocal swift_job
        data = await request.json()
        assert request.headers.get('X-Halogen-Action') == 'deploy'
        profile['env']['HALOGEN_NPU_MODELS'] = data['models']
        swift_job = {'active': True, 'kind': 'quick-npu', 'state': 'running',
                     'lines': ['Preparing NPU files (UI fixture; no downloads).'], 'step': 1, 'total_steps': 4, 'elapsed': 0}
        return web.json_response({'ok': True}, status=202)
    async def deploy(request):
        return web.json_response({'enabled': True, 'posix': True, 'quadlet_dir': '/home/demo/.config/containers/systemd',
                                 'allowed_roots': ['/home/demo'], 'env_keys': list(ENV_FIELDS),
                                 'profiles': [{**profile, 'model_id': 'qwen3.8-flash', 'service_active': True, 'has_backup': True}]})
    async def template(request):
        kind = request.match_info['kind']
        if kind in {'swift15-quick', 'swift15-abliterated-quick'}:
            locations = {a.name: '/home/demo/models/shared/halogen-v2/' + a.revision + '/' + a.name for a in SHARED_ASSETS}
            p = swift_quick_template(kind.removesuffix('-quick'), Path('/home/demo/models'), Path('/home/demo/cache'), locations)
            p.volumes = [(host.replace('\\', '/'), container, mode) for host, container, mode in p.volumes]
            return web.json_response({'profile': profile_to_dict(p)})
        factory = custom_template if request.match_info['kind'] == 'custom' else official_template
        p = factory('0.13.2', PurePosixPath('/home/demo/models'), PurePosixPath('/home/demo/cache'))
        return web.json_response({'profile': profile_to_dict(p)})
    async def hf(request):
        return web.json_response({'installed': True, 'default_repo': 'demo/model', 'default_file': 'model.gguf'})
    swift_job = None
    async def job(request):
        return web.json_response(swift_job or {'active': False})
    async def swift_plan(request):
        variant = request.query.get('variant', 'swift15')
        info = variant_info(variant)
        files = []
        for i, asset in enumerate([info['checkpoint'], *SHARED_ASSETS]):
            path = '/home/demo/models/' + (variant if i == 0 else 'shared/halogen-v2/' + asset.revision) + '/' + asset.name
            files.append({'name': asset.name, 'path': path, 'source_path': path if i else None,
                          'status': 'download' if i == 0 else 'verified' if i < 3 else 'unverified',
                          'size': asset.size, 'download_bytes': asset.size if i == 0 else 0,
                          'url': f'https://huggingface.co/{asset.repo}/blob/{asset.revision}/{asset.name}'})
        return web.json_response({'variant': variant, 'files': files,
                                  'shared_root': '/home/demo/models/shared/halogen-v2',
                                  'model_card': 'https://huggingface.co/' + info['checkpoint'].repo,
                                  'download_bytes': info['checkpoint'].size,
                                  'unverified_bytes': sum(f['size'] for f in files if f['status'] == 'unverified'),
                                  'filesystems': [{'path': '/home/demo/models', 'free_bytes': 300 * 1024**3,
                                                   'required_bytes': info['checkpoint'].size,
                                                   'reserve_bytes': 5 * 1024**3, 'sufficient': True}]})
    async def swift_install(request):
        nonlocal swift_job
        data = await request.json()
        variant_info(data['variant'])
        assert request.headers.get('X-Halogen-Action') == 'deploy'
        swift_job = {'active': True, 'kind': 'quick-' + data['variant'], 'state': 'running',
                     'lines': ['Preparing verified shared assets (UI fixture; no downloads).'],
                     'step': 1, 'total_steps': 4, 'elapsed': 0}
        return web.json_response({'ok': True}, status=202)
    async def cancel_job(request):
        if swift_job:
            swift_job.update(state='cancelled')
        return web.json_response({'ok': True})
    async def preview(request):
        from profiles import Profile, validate_profile, render_quadlet
        data = await request.json()
        p = Profile(**data)
        errors = validate_profile(p)
        return web.json_response({'ok': not errors, 'errors': errors, 'warnings': [], 'diff': render_quadlet(p) if not errors else ''})
    async def login(request):
        return _login_response(error=request.method == 'POST', lang=_request_lang(request))
    app.router.add_get('/dashboard/api/deploy', deploy)
    app.router.add_post('/preview/npu', npu_fixture)
    app.router.add_get('/dashboard/api/deploy/npu', npu_status)
    app.router.add_get('/dashboard/api/deploy/npu/plan', npu_plan)
    app.router.add_post('/dashboard/api/deploy/quick/npu', npu_install)
    app.router.add_get('/dashboard/api/deploy/template/{kind}', template)
    app.router.add_get('/dashboard/api/deploy/hf', hf)
    app.router.add_get('/dashboard/api/deploy/job', job)
    app.router.add_get('/dashboard/api/deploy/quick/swift/plan', swift_plan)
    app.router.add_post('/dashboard/api/deploy/quick/swift', swift_install)
    app.router.add_post('/dashboard/api/deploy/job/cancel', cancel_job)
    app.router.add_post('/dashboard/api/deploy/dry-run', preview)
    app.router.add_route('*', '/dashboard/login', login)
    async def cleanup(_):
        await dashboard.close()
    app.on_cleanup.append(cleanup)
    return app


if __name__ == '__main__':
    web.run_app(create_preview(), host='127.0.0.1', port=8732, access_log=None)
