"""ModelManager.switch_with_hook: takeover switch that frees the GPU, runs a
one-shot hook (e.g. a convert container) and then starts the target model.

All systemd/podman interaction is mocked; only the lock and ordering logic is
exercised.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from halogen_router import ModelManager, RouterError

OFFICIAL = "qwen3.8-flash"
UNCENSORED = "qwen3.8-flash-uncensored"


class SwitchWithHookTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = ModelManager(None)
        self.manager.models = {
            OFFICIAL: "halogen-official.service",
            UNCENSORED: "halogen-uncensored.service",
        }
        self.calls = []
        self.manager.wait_for_backend_idle = AsyncMock(
            side_effect=lambda: self.calls.append("idle")
        )
        self.manager.stop_service = AsyncMock(
            side_effect=lambda m: self.calls.append(("stop", m))
        )
        self.manager.stop_all_model_services = AsyncMock(
            side_effect=lambda: self.calls.append("stop_all")
        )
        self.manager.wait_for_gtt_release = AsyncMock(
            side_effect=lambda: self.calls.append("gtt")
        )
        self.manager.start_service = AsyncMock(
            side_effect=lambda m, start_timeout=None: self.calls.append(("start", m))
        )
        self.manager.service_is_active = AsyncMock(return_value=False)

    async def test_stops_old_then_hook_then_starts_new(self):
        self.manager.current_model = OFFICIAL

        async def hook():
            self.calls.append("hook")

        await self.manager.switch_with_hook(UNCENSORED, hook)
        self.assertEqual(
            self.calls,
            ["idle", ("stop", OFFICIAL), "gtt", "hook", ("start", UNCENSORED)],
        )
        self.assertEqual(self.manager.current_model, UNCENSORED)
        self.assertFalse(self.manager.switching)
        self.assertIsNone(self.manager.switch_target)

    async def test_no_current_model_stops_all(self):
        self.manager.current_model = None
        await self.manager.switch_with_hook(UNCENSORED)
        self.assertIn("stop_all", self.calls)
        self.assertNotIn(("stop", OFFICIAL), self.calls)
        self.assertEqual(self.manager.current_model, UNCENSORED)

    async def test_hook_runs_only_after_gpu_release(self):
        self.manager.current_model = OFFICIAL
        order = []
        self.manager.wait_for_gtt_release = AsyncMock(side_effect=lambda: order.append("gtt"))

        async def hook():
            order.append("hook")

        await self.manager.switch_with_hook(UNCENSORED, hook)
        self.assertEqual(order, ["gtt", "hook"])

    async def test_error_resets_switching_and_keeps_current_model(self):
        self.manager.current_model = OFFICIAL
        self.manager.start_service = AsyncMock(side_effect=RuntimeError("boom"))
        with self.assertRaises(RouterError):
            await self.manager.switch_with_hook(UNCENSORED)
        self.assertFalse(self.manager.switching)
        self.assertIsNone(self.manager.switch_target)
        self.assertEqual(self.manager.current_model, OFFICIAL)

    async def test_rollback_restarts_old_model_on_start_failure(self):
        self.manager.current_model = OFFICIAL
        self.manager.service_is_active = AsyncMock(return_value=True)

        async def start_side(model, start_timeout=None):
            self.calls.append(("start", model))
            if model == UNCENSORED:
                raise RuntimeError("boom")

        self.manager.start_service = AsyncMock(side_effect=start_side)
        with self.assertRaises(RouterError):
            await self.manager.switch_with_hook(UNCENSORED)
        # target start failed, then old model was restarted as rollback
        self.assertIn(("start", OFFICIAL), self.calls)
        self.assertEqual(self.calls.count(("start", UNCENSORED)), 1)
        self.assertEqual(self.manager.current_model, OFFICIAL)
        self.assertFalse(self.manager.switching)

    async def test_waits_for_active_requests_to_drain(self):
        self.manager.current_model = OFFICIAL
        async with self.manager.condition:
            self.manager.active_requests = 1
        task = asyncio.create_task(self.manager.switch_with_hook(UNCENSORED))
        await asyncio.sleep(0.05)
        self.assertFalse(task.done())
        async with self.manager.condition:
            self.manager.active_requests = 0
            self.manager.condition.notify_all()
        await asyncio.wait_for(task, 2)
        self.assertEqual(self.manager.current_model, UNCENSORED)

    async def test_maintenance_blocks_switch(self):
        self.manager.current_model = OFFICIAL
        self.manager.maintenance = True
        with self.assertRaises(Exception):
            await self.manager.switch_with_hook(UNCENSORED)

    async def test_initialize_adopts_active_service_without_waiting_for_health(self):
        self.manager.backend_health = AsyncMock(return_value=None)
        self.manager.service_is_active = AsyncMock(
            side_effect=lambda service: service == "halogen-official.service"
        )
        self.manager.wait_for_backend = AsyncMock()

        await self.manager.initialize()

        self.assertEqual(self.manager.current_model, OFFICIAL)
        self.manager.wait_for_backend.assert_not_awaited()
        self.manager.start_service.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
