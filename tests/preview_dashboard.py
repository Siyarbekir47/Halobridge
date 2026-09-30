"""Local UI fixture. No system services, model files, or real backend are used.

Run: python tests/preview_dashboard.py
"""
import time
from pathlib import Path, PurePosixPath
import sys

from aiohttp import web
from test_dashboard import database, insert

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from profiles import official_template, custom_template, profile_to_dict, ENV_FIELDS
from halogen_router import _login_response, _request_lang


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
                            'reasoning_effort_default': 'xhigh', 'version': {'api': '0.13.2'}},
                'system': {'gpu_busy_percent': 8, 'memory': {'used_bytes': 62*1024**3, 'total_bytes': 128*1024**3},
                           'disk_used_bytes': 420*1024**3, 'disk_total_bytes': 2000*1024**3},
                'cache': {'pool': {'usage_ratio': .3}, 'model_bytes': {}}, 'active_requests': []}
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
    updates = {'current_version': '0.13.2', 'latest_version': '0.13.2', 'checked_at': time.time(),
               'configured_versions': {'qwen3.8-flash': '0.13.2'}, 'up_to_date': True, 'running': False,
               'supported': False, 'support_error': 'Updates require a Linux host with Podman and systemd under the router\'s user account.'}
    async def update_status(request):
        return web.json_response(updates)
    app.router.add_get('/dashboard/api/updates', update_status)
    app.router.add_post('/dashboard/api/updates/check', update_status)
    # The self-update interaction is simulated; no installation or restart occurs.
    app_updates = {'current_version': '0.1.19', 'latest_version': '0.1.20',
                   'checked_at': time.time(), 'source_ref': 'develop',
                   'update_available': True, 'up_to_date': False, 'running': False,
                   'supported': True, 'can_install': True, 'job': None,
                   'release_url': 'https://github.com/Siyarbekir47/Halobridge/blob/develop/CHANGELOG.md'}
    app_poll_count = 0
    async def app_update_status(request):
        nonlocal app_poll_count
        if request.match_info.get('action') == 'install':
            data = await request.json()
            assert data['version'] == '0.1.20'
            assert request.headers.get('X-Halogen-Action') == 'update'
            app_updates.update(running=True, can_install=False,
                               job={'phase': 'installing', 'message': 'Installing the verified Halobridge version.', 'target_version': '0.1.20'})
        elif app_updates['running']:
            app_poll_count += 1
            if app_poll_count >= 3:
                app_updates.update(current_version='0.1.20', update_available=False, up_to_date=True,
                                   running=False, job={'phase': 'succeeded', 'message': 'Halobridge was updated and the user service restarted.', 'target_version': '0.1.20'})
        return web.json_response(app_updates)
    app.router.add_get('/dashboard/api/app-updates', app_update_status)
    app.router.add_post('/dashboard/api/app-updates/{action}', app_update_status)
    profile = profile_to_dict(official_template('0.13.2', PurePosixPath('/home/demo/models'), PurePosixPath('/home/demo/cache')))
    async def deploy(request):
        return web.json_response({'enabled': True, 'posix': True, 'quadlet_dir': '/home/demo/.config/containers/systemd',
                                 'allowed_roots': ['/home/demo'], 'env_keys': list(ENV_FIELDS),
                                 'profiles': [{**profile, 'model_id': 'qwen3.8-flash', 'service_active': True, 'has_backup': True}]})
    async def template(request):
        factory = custom_template if request.match_info['kind'] == 'custom' else official_template
        p = factory('0.13.2', PurePosixPath('/home/demo/models'), PurePosixPath('/home/demo/cache'))
        return web.json_response({'profile': profile_to_dict(p)})
    async def hf(request):
        return web.json_response({'installed': True, 'default_repo': 'demo/model', 'default_file': 'model.gguf'})
    async def job(request):
        return web.json_response({'active': False})
    async def preview(request):
        from profiles import Profile, validate_profile, render_quadlet
        data = await request.json()
        p = Profile(**data)
        errors = validate_profile(p)
        return web.json_response({'ok': not errors, 'errors': errors, 'warnings': [], 'diff': render_quadlet(p) if not errors else ''})
    async def login(request):
        return _login_response(error=request.method == 'POST', lang=_request_lang(request))
    app.router.add_get('/dashboard/api/deploy', deploy)
    app.router.add_get('/dashboard/api/deploy/template/{kind}', template)
    app.router.add_get('/dashboard/api/deploy/hf', hf)
    app.router.add_get('/dashboard/api/deploy/job', job)
    app.router.add_post('/dashboard/api/deploy/dry-run', preview)
    app.router.add_route('*', '/dashboard/login', login)
    async def cleanup(_):
        await dashboard.close()
    app.on_cleanup.append(cleanup)
    return app


if __name__ == '__main__':
    web.run_app(create_preview(), host='127.0.0.1', port=8732, access_log=None)
