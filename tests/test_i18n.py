"""Request-scoped localization must not mutate shared jobs or user data."""
import asyncio
import json
from pathlib import Path
import sys
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from halobridge_i18n import dashboard_locale_middleware, localize_payload, request_language, translate_message
from halogen_router import _login_response, _request_lang


class MessageTests(unittest.TestCase):
    def test_defaults_and_weighted_language_preferences(self):
        self.assertEqual(request_language(make_mocked_request('GET', '/')), 'en')
        for header, expected in [('fr', 'en'), ('de-CH', 'de'), ('de;q=0.2,en;q=0.9', 'en'), ('de;q=0,en', 'en')]:
            self.assertEqual(request_language(make_mocked_request('GET', '/', headers={'Accept-Language': header})), expected)

    def test_persisted_german_jobs_are_normalized_for_english(self):
        self.assertEqual(translate_message('Image 0.13.2 wird geladen. Anfragen laufen weiter.', 'en'),
                         'Downloading image 0.13.2. Requests continue to run.')
        self.assertEqual(translate_message('Downloading image 0.13.2. Requests continue to run.', 'de'),
                         'Image 0.13.2 wird geladen. Anfragen laufen weiter.')

    def test_nested_errors_and_parameters(self):
        self.assertEqual(translate_message('env HALOGEN_CTX: integer expected', 'de'), 'env HALOGEN_CTX: ganze Zahl erwartet')
        self.assertEqual(translate_message('Program not found: /home/Unknown/current', 'de'),
                         'Programm nicht gefunden: /home/Unknown/current')
        self.assertIn('ganze Zahl erwartet', translate_message('ERROR: env HALOGEN_CTX: integer expected', 'de'))
        self.assertIn('models volume at /models is missing', translate_message('volumes: Models-Volume auf /models fehlt', 'en'))

    def test_translation_does_not_mutate_profiles_or_raw_output(self):
        payload = {'profile_id': 'Unknown', 'env': {'message': 'Unknown'}, 'quadlet': 'Updates are disabled.',
                   'job': {'message': 'Preparing update.', 'lines': ['podman output: test', '--- Apply official profile ---']}}
        localized = localize_payload(payload, 'de')
        self.assertEqual(localized['profile_id'], 'Unknown')
        self.assertEqual(localized['env'], payload['env'])
        self.assertEqual(localized['quadlet'], 'Updates are disabled.')
        self.assertEqual(localized['job']['message'], 'Update wird vorbereitet.')
        self.assertEqual(payload['job']['message'], 'Preparing update.')
        self.assertEqual(localized['job']['lines'][0], 'podman output: test')

    def test_login_defaults_to_english_and_keeps_explicit_language(self):
        self.assertIn('<html lang="en">', _login_response().text)
        self.assertIn('action="/dashboard/login?lang=de"', _login_response(lang='de').text)
        self.assertEqual(_request_lang(make_mocked_request('GET', '/', headers={'Accept-Language': 'de'})), 'en')
        self.assertEqual(_request_lang(make_mocked_request('GET', '/?lang=de')), 'de')

    def test_catalog_placeholders_round_trip(self):
        from halobridge_i18n import PLACEHOLDER
        catalog = json.loads((Path(__file__).resolve().parents[1] / 'src/halobridge_data/locales/server.de.json').read_text(encoding='utf-8'))
        for en, de in catalog.items():
            self.assertEqual(set(PLACEHOLDER.findall(en)), set(PLACEHOLDER.findall(de)), en)
            # Distinct placeholders make ambiguous template regressions visible.
            original = PLACEHOLDER.sub(lambda m: 'sample_' + m.group()[1:-1], en)
            german = PLACEHOLDER.sub(lambda m: 'sample_' + m.group()[1:-1], de)
            self.assertEqual(translate_message(original, 'de'), german, en)
            self.assertEqual(translate_message(german, 'en'), original, de)


class ResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_browsers_localize_the_same_background_job(self):
        job = {'message': 'Preparing update.', 'phase': 'starting'}
        async def status(request):
            await asyncio.sleep(0)
            return web.json_response({'job': job})
        async def error(request):
            raise web.HTTPBadRequest(text='Telemetry is not available yet')
        app = web.Application(middlewares=[dashboard_locale_middleware])
        app.router.add_get('/dashboard/api/job', status)
        app.router.add_get('/dashboard/api/error', error)
        app.router.add_get('/v1/status', status)
        async with TestClient(TestServer(app)) as client:
            en, de = await asyncio.gather(client.get('/dashboard/api/job', headers={'Accept-Language': 'en'}),
                                         client.get('/dashboard/api/job', headers={'Accept-Language': 'de'}))
            self.assertEqual((await en.json())['job']['message'], 'Preparing update.')
            self.assertEqual((await de.json())['job']['message'], 'Update wird vorbereitet.')
            self.assertEqual(de.headers['Content-Language'], 'de')
            self.assertEqual(job['message'], 'Preparing update.')
            bad = await client.get('/dashboard/api/error', headers={'Accept-Language': 'de'})
            self.assertEqual(bad.status, 400)
            self.assertEqual(await bad.text(), 'Telemetrie noch nicht verfügbar')
            raw = await client.get('/v1/status', headers={'Accept-Language': 'de'})
            self.assertEqual((await raw.json())['job']['message'], 'Preparing update.')


if __name__ == '__main__':
    unittest.main()
