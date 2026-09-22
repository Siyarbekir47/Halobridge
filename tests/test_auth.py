"""Token-gate tests: python -m unittest discover -s tests -v"""

from pathlib import Path
import sys
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import halogen_router as router


def make_app(token: str) -> web.Application:
    app = web.Application()
    middleware = router.make_auth_middleware(token)
    if middleware:
        app.middlewares.append(middleware)

    login_page, login_post, logout = router.make_login_handlers(token, False)
    app.router.add_get("/dashboard/login", login_page)
    app.router.add_post("/dashboard/login", login_post)
    app.router.add_post("/dashboard/logout", logout)

    async def dashboard(request: web.Request) -> web.Response:
        return web.Response(text="dashboard")

    async def api(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def status(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def inference(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    app.router.add_get("/dashboard", dashboard)
    app.router.add_get("/dashboard/", dashboard)
    app.router.add_get("/dashboard/api/snapshot", api)
    app.router.add_get("/router/status", status)
    app.router.add_get("/v1/models", inference)
    return app


class AuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_token_disables_middleware(self):
        self.assertIsNone(router.make_auth_middleware(""))

    async def test_unauthenticated_page_redirects_and_api_returns_401(self):
        async with TestClient(TestServer(make_app("secret"))) as client:
            response = await client.get("/dashboard", allow_redirects=False)
            self.assertEqual(response.status, 302)
            self.assertEqual(response.headers["Location"], "/dashboard/login")

            response = await client.get("/dashboard/api/snapshot")
            self.assertEqual(response.status, 401)
            self.assertEqual(response.headers["WWW-Authenticate"], "Bearer")

            response = await client.get("/router/status")
            self.assertEqual(response.status, 401)

            response = await client.get("/v1/models")
            self.assertEqual(response.status, 200)

    async def test_login_sets_session_cookie_and_grants_access(self):
        async with TestClient(TestServer(make_app("secret"))) as client:
            response = await client.post(
                "/dashboard/login", data={"token": "wrong"}, allow_redirects=False
            )
            self.assertEqual(response.status, 401)

            response = await client.post(
                "/dashboard/login", data={"token": "secret"}, allow_redirects=False
            )
            self.assertEqual(response.status, 302)
            self.assertEqual(response.headers["Location"], "/dashboard")
            self.assertIn(router.SESSION_COOKIE, response.cookies)
            self.assertNotEqual(response.cookies[router.SESSION_COOKIE].value, "secret")

            response = await client.get("/dashboard")
            self.assertEqual(response.status, 200)
            self.assertEqual(await response.text(), "dashboard")

    async def test_json_login_and_bearer_header_work(self):
        async with TestClient(TestServer(make_app("secret"))) as client:
            response = await client.post(
                "/dashboard/login", json={"token": "secret"}, allow_redirects=False
            )
            self.assertEqual(response.status, 302)

            response = await client.get(
                "/dashboard/api/snapshot", headers={"Authorization": "Bearer secret"}
            )
            self.assertEqual(response.status, 200)

            response = await client.get(
                "/dashboard/api/snapshot", headers={"X-Halogen-Token": "secret"}
            )
            self.assertEqual(response.status, 200)

    async def test_cross_origin_login_is_rejected(self):
        async with TestClient(TestServer(make_app("secret"))) as client:
            response = await client.post(
                "/dashboard/login",
                data={"token": "secret"},
                headers={"Origin": "https://evil.example"},
                allow_redirects=False,
            )
            self.assertEqual(response.status, 403)

    async def test_logout_clears_cookie(self):
        async with TestClient(TestServer(make_app("secret"))) as client:
            await client.post("/dashboard/login", data={"token": "secret"}, allow_redirects=False)
            response = await client.get("/dashboard")
            self.assertEqual(response.status, 200)

            response = await client.post("/dashboard/logout", allow_redirects=False)
            self.assertEqual(response.status, 302)
            self.assertEqual(response.cookies[router.SESSION_COOKIE].value, "")

            response = await client.get("/dashboard", allow_redirects=False)
            self.assertEqual(response.status, 302)
            self.assertEqual(response.headers["Location"], "/dashboard/login")


if __name__ == "__main__":
    unittest.main()
