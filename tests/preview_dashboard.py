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
    async def snapshot():
        return {'generated_at': time.time(), 'app': {'version': '0.1.8'},
                'api': {'active_model': 'qwen3.8-flash', 'status': 'ok'},
                'backend': {'status': 'ok', 'in_flight': 0, 'queued': 0, 'context': 262144,
                            'slots': 4, 'max_tokens_default': 8192, 'max_tokens_cap': 65536,
                            'reasoning_effort_default': 'xhigh', 'version': {'api': '0.13.2'}},
                'system': {'gpu_busy_percent': 8, 'memory': {'used_bytes': 62*1024**3, 'total_bytes': 128*1024**3},
                           'disk_used_bytes': 420*1024**3, 'disk_total_bytes': 2000*1024**3},
                'cache': {'pool': {'usage_ratio': .3}, 'model_bytes': {}}, 'active_requests': []}
    dashboard.snapshot = snapshot
    app = web.Application()
    dashboard.register_routes(app)
    updates = {'current_version': '0.13.2', 'latest_version': '0.13.2', 'checked_at': time.time(),
               'configured_versions': {'qwen3.8-flash': '0.13.2'}, 'up_to_date': True, 'running': False,
               'supported': False, 'support_error': 'Updates require a Linux host with Podman and systemd under the router\'s user account.'}
    async def update_status(request):
        return web.json_response(updates)
    app.router.add_get('/dashboard/api/updates', update_status)
    app.router.add_post('/dashboard/api/updates/check', update_status)
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
