"""Exercise backend credentials through real HTTP and mounted secret files."""
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer

from backend_auth import BackendAuth, BackendAuthError
from halogen_router import ModelManager, list_models, make_auth_middleware, proxy_request
from profiles import official_template, profile_to_dict, validate_profile
from settings import Config, ConfigError, RouterConfig
from test_dashboard import database
from test_deploy import make_manager


class AuthTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.profile = official_template("0.18.0", self.root / "models", self.root / "cache")
        self.key_file = self.root / "backend.key"
        self.key_file.write_text("private-backend,second-key\n")
        self.profile.volumes.append((str(self.key_file), "/secrets/key", "ro,z"))
        self.profile.env["HALOGEN_API_KEY_FILE"] = "/secrets/key"
        self.seen = []

        async def backend(request):
            self.seen.append(dict(request.headers))
            key = self.key_file.read_text().strip().split(",")[0] if self.key_file.exists() else "private-backend"
            if request.path != "/health" and request.headers.get("Authorization") != "Bearer " + key:
                return web.json_response({"error": "key required"}, status=401)
            if request.path == "/v1/models":
                return web.json_response({"data": [{"id": self.profile.model_id},
                                                  {"id": "qwen3-embedding-0.6b", "task": "embedding"}]})
            if request.path == "/cache":
                return web.json_response({"hits": 12})
            return web.json_response({"usage": {"prompt_tokens": 3, "completion_tokens": 2}})

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", backend)
        self.backend = TestServer(app)
        await self.backend.start_server()
        self.session = ClientSession()
        self.manager = ModelManager(self.session, {self.profile.model_id: self.profile.service_name},
                                    backend_url=str(self.backend.make_url("")).rstrip("/"))
        self.manager.current_model = self.profile.model_id
        self.manager.npu_names = lambda _: {"qwen3-embedding-0.6b"}
        self.config = Config()
        self.manager.backend_auth = BackendAuth(self.config.router, lambda: [self.profile])
        self.dashboard = database()
        self.dashboard.manager = self.manager
        self.dashboard.session = self.session
        self.dashboard.backend_url = self.manager.backend_url
        await self.start_router("router-login")

    async def start_router(self, token):
        self.config.security.auth_token = token
        middleware = make_auth_middleware(token)
        app = web.Application(middlewares=[middleware] if middleware else [])
        app["manager"], app["session"], app["dashboard"] = self.manager, self.session, self.dashboard
        app["backend_url"], app["config"] = self.manager.backend_url, self.config
        app.router.add_get("/v1/models", list_models)
        app.router.add_route("*", "/{tail:.*}", proxy_request)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.backend.close()
        await self.session.close()
        await self.dashboard.close()
        self.temp.cleanup()

    async def test_discovery_and_dashboard_use_mounted_backend_key(self):
        self.assertEqual(len(await self.manager.backend_model_entries()), 2)
        self.assertEqual((await self.dashboard._get_json("/cache"))["hits"], 12)
        response = await self.client.get("/v1/models", headers={"Authorization": "Bearer router-login"})
        self.assertEqual(response.status, 200)
        self.assertIn("qwen3-embedding-0.6b", {x["id"] for x in (await response.json())["data"]})
        self.assertTrue(all(h["Authorization"] == "Bearer private-backend" for h in self.seen))

    async def test_router_login_is_replaced_only_after_authentication(self):
        response = await self.client.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(response.status, 401)
        self.assertNotIn("Authorization", self.seen[-1])
        response = await self.client.post("/v1/chat/completions", json={"messages": []},
            headers={"Authorization": "Bearer router-login", "x-api-key": "wrong-client-key"})
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(self.seen[-1]["Authorization"], "Bearer private-backend")
        self.assertNotIn("x-api-key", {name.lower() for name in self.seen[-1]})
        self.assertEqual(self.manager.active_requests, 0)

    async def test_without_router_login_backend_key_is_still_required(self):
        await self.client.close()
        await self.start_router("")
        for headers, expected in (({}, 401), ({"Authorization": "Bearer private-backend"}, 200)):
            response = await self.client.post("/v1/chat/completions", json={"messages": []}, headers=headers)
            self.assertEqual(response.status, expected)
            await response.read()
        self.assertEqual(self.manager.active_requests, 0)

    async def test_missing_key_fails_without_leaking_or_leaving_a_reservation(self):
        self.key_file.unlink()
        response = await self.client.post("/v1/chat/completions", json={"messages": []},
                                          headers={"Authorization": "Bearer router-login"})
        self.assertEqual(response.status, 503)
        self.assertNotIn("private-backend", await response.text())
        self.assertEqual(self.manager.active_requests, 0)
        self.assertEqual(self.seen, [])

    async def test_key_rotation_and_inline_list_are_read_from_target_profile(self):
        self.key_file.write_text("rotated-key\n")
        self.assertEqual(self.manager.backend_headers(), {"Authorization": "Bearer rotated-key"})
        self.profile.env.pop("HALOGEN_API_KEY_FILE")
        self.profile.env["HALOGEN_API_KEY"] = "inline-key,other-key"
        self.assertEqual(self.manager.backend_headers(), {"Authorization": "Bearer inline-key"})
        self.manager.backend_auth = BackendAuth(RouterConfig(backend_api_key_file=str(self.key_file)), lambda: [])
        self.assertEqual(self.manager.backend_headers(), {"Authorization": "Bearer rotated-key"})

    async def test_profile_preview_redacts_inline_keys_and_file_mounts_stay_files(self):
        deploy = make_manager(self.root)
        self.profile.env.pop("HALOGEN_API_KEY_FILE")
        self.profile.env["HALOGEN_API_KEY"] = "never-display-this"
        payload = profile_to_dict(self.profile)
        preview = deploy.dry_run(payload)
        self.assertTrue(preview["ok"], preview["errors"])
        self.assertNotIn("never-display-this", preview["quadlet"] + preview["diff"])
        self.profile.env.pop("HALOGEN_API_KEY")
        self.profile.env["HALOGEN_API_KEY_FILE"] = "/secrets/key"
        with patch("halogen_deploy.sys.platform", "linux"):
            deploy.install_dirs(profile_to_dict(self.profile))
        self.assertTrue(self.key_file.is_file())
        self.assertEqual(validate_profile(self.profile), [])

    async def test_config_rejects_ambiguous_credentials_without_printing_them(self):
        with self.assertRaises(ConfigError):
            Config.from_dict({"router": {"backend_api_key": "secret", "backend_api_key_file": str(self.key_file)}})
        with self.assertRaises(ConfigError) as caught:
            Config.from_dict({"router": {"backend_api_key": "secret\r\nInjected: value"}})
        self.assertNotIn("secret", str(caught.exception))
        self.assertNotIn("secret", repr(RouterConfig(backend_api_key="secret")))
        self.key_file.write_text("x" * 4097)
        with self.assertRaises(BackendAuthError):
            BackendAuth(RouterConfig(backend_api_key_file=str(self.key_file)), lambda: []).headers("gpu")
