"""create_application wiring tests: python -m unittest discover -s tests -v"""

from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import halogen_router as router
from discovery import ModelSpec
from settings import Config, DashboardConfig, ModelsConfig, SecurityConfig


class CreateApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_models_raises_clear_error(self):
        config = Config(
            models=ModelsConfig(auto_discover=False, explicit={}),
        )
        with self.assertRaises(router.RouterError):
            await router.create_application(config)

    async def test_explicit_models_are_wired_into_app(self):
        config = Config(
            models=ModelsConfig(
                auto_discover=False,
                explicit={"model-a": "svc-a.service"},
            ),
        )
        with patch.object(router.ModelManager, "initialize", new=AsyncMock()), \
             patch.object(router.ContainerUpdater, "recover_on_startup", new=AsyncMock()), \
             patch.object(router.Dashboard, "initialize", new=AsyncMock()), \
             patch.object(router.ContainerUpdater, "start_checks", new=lambda self: None):
            app = await router.create_application(config)
        await app["session"].close()

        self.assertEqual(app["models"], {"model-a": "svc-a.service"})
        self.assertEqual(app["backend_url"], config.router.backend_url)
        self.assertIs(app["manager"].models, app["models"])
        self.assertIs(app["updater"].models, app["models"])
        self.assertEqual(app["manager"].default_model, "model-a")
        self.assertEqual(len(app.middlewares), 1)

    async def test_discovery_and_explicit_override_are_merged(self):
        spec = ModelSpec(
            model_id="model-a",
            service="svc-a.service",
            container="svc-a",
            cache_path=Path("/cache/a"),
            image="ghcr.io/peonist-ai/halogen-flash-server:0.13.1",
            version="0.13.1",
            quadlet=Path("a.container"),
        )
        config = Config(
            models=ModelsConfig(
                auto_discover=True,
                explicit={"model-b": "svc-b.service", "model-a": "svc-override.service"},
            ),
        )
        with patch.object(router, "discover", return_value=({"model-a": spec}, ["skip"])), \
             patch.object(router.ModelManager, "initialize", new=AsyncMock()), \
             patch.object(router.ContainerUpdater, "recover_on_startup", new=AsyncMock()), \
             patch.object(router.Dashboard, "initialize", new=AsyncMock()), \
             patch.object(router.ContainerUpdater, "start_checks", new=lambda self: None):
            app = await router.create_application(config)
        await app["session"].close()

        self.assertEqual(
            app["models"],
            {"model-a": "svc-override.service", "model-b": "svc-b.service"},
        )
        self.assertEqual(app["dashboard"].cache_paths, {"model-a": Path("/cache/a")})
        self.assertEqual(
            app["dashboard"].service_models,
            {"svc-override.service": "model-a", "svc-b.service": "model-b"},
        )

    async def test_auth_token_installs_middleware_and_login_routes(self):
        config = Config(
            models=ModelsConfig(
                auto_discover=False,
                explicit={"model-a": "svc-a.service"},
            ),
            security=SecurityConfig(auth_token="secret"),
            dashboard=DashboardConfig(public_endpoint="https://example.test/"),
        )
        with patch.object(router.ModelManager, "initialize", new=AsyncMock()), \
             patch.object(router.ContainerUpdater, "recover_on_startup", new=AsyncMock()), \
             patch.object(router.Dashboard, "initialize", new=AsyncMock()), \
             patch.object(router.ContainerUpdater, "start_checks", new=lambda self: None):
            app = await router.create_application(config)
        await app["session"].close()

        self.assertEqual(len(app.middlewares), 2)
        routes = {route.resource.canonical for route in app.router.routes()}
        self.assertIn("/dashboard/login", routes)
        self.assertIn("/dashboard/logout", routes)


if __name__ == "__main__":
    unittest.main()
