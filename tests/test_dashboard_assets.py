"""Verify the shipped dashboard assets, translations, and resource boundary."""
from html.parser import HTMLParser
import json
from pathlib import Path
import unittest

from aiohttp.test_utils import TestClient, TestServer
from aiohttp import web
from test_dashboard import database
from halogen_router import make_auth_middleware


class Markup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.keys = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        for key in ('data-i18n', 'data-i18n-html'):
            if key in attrs:
                self.keys.append(attrs[key])
        if 'data-i18n-attr' in attrs:
            self.keys.extend(item.split(':')[1] for item in attrs['data-i18n-attr'].split(';'))


class AssetsTests(unittest.IsolatedAsyncioTestCase):
    async def test_assets_remain_behind_the_dashboard_token_gate(self):
        dashboard = database()
        app = web.Application(middlewares=[make_auth_middleware('test-secret')])
        dashboard.register_routes(app)
        try:
            async with TestClient(TestServer(app)) as client:
                for name in ('dashboard.js', 'dashboard.css'):
                    response = await client.get('/dashboard/assets/' + name, allow_redirects=False)
                    self.assertEqual(response.status, 302)
                    response = await client.get('/dashboard/assets/' + name, headers={'Authorization': 'Bearer test-secret'})
                    self.assertEqual(response.status, 200)
        finally:
            await dashboard.close()

    async def test_assets_are_served_and_unknown_resources_are_rejected(self):
        dashboard = database()
        app = web.Application()
        dashboard.register_routes(app)
        try:
            async with TestClient(TestServer(app)) as client:
                page = await client.get('/dashboard')
                html = await page.text()
                self.assertIn('<html lang="en">', html)
                self.assertIn('script-src \'self\'', page.headers['Content-Security-Policy'])
                for name, content_type in [('dashboard.css', 'text/css'), ('dashboard.js', 'application/javascript')]:
                    self.assertIn('/dashboard/assets/' + name, html)
                    response = await client.get('/dashboard/assets/' + name)
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.content_type, content_type)
                    self.assertTrue(await response.text())
                response = await client.get('/dashboard/assets/server.de.json')
                self.assertEqual(response.status, 404)
        finally:
            await dashboard.close()

    def test_unique_ids_and_complete_locale_catalogs(self):
        root = Path(__file__).resolve().parents[1] / 'src/halobridge_data'
        markup = Markup()
        markup.feed((root / 'dashboard.html').read_text(encoding='utf-8'))
        self.assertEqual(len(markup.ids), len(set(markup.ids)))
        en, de = [json.loads((root / f'locales/{lang}.json').read_text(encoding='utf-8')) for lang in ['en', 'de']]
        self.assertEqual(set(en), set(de))
        self.assertFalse(set(markup.keys) - set(en))


if __name__ == '__main__':
    unittest.main()
