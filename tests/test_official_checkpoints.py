"""Real small files exercise official checkpoint preparation and transactions."""

import asyncio
import hashlib
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import official_checkpoints
import shared_assets
import test_updates
from halogen_updates import ContainerUpdater, UpdateError, CHECKPOINT_V2_PATH
from profiles import parse_quadlet, render_quadlet, swift_quick_template
from shared_assets import SharedAssets, AssetError


class OfficialCheckpointTests(unittest.IsolatedAsyncioTestCase):
    VERSION = "0.16.0"
    uncensored_quadlet = test_updates.CheckpointUpgradeTests.uncensored_quadlet
    make_updater = test_updates.CheckpointUpgradeTests.make_updater
    health = test_updates.CheckpointUpgradeTests.health
    asyncTearDown = test_updates.CheckpointUpgradeTests.asyncTearDown

    def checkpoint_quadlet(self):
        return test_updates.CheckpointUpgradeTests.checkpoint_quadlet(self).replace(
            b"qwen38-flash-next-w4b.hgn", b"qwen38-flash-next-v2.hgn"
        ).replace(b"/home/tester/halogen/models", str(self.models).encode())

    async def asyncSetUp(self):
        await test_updates.CheckpointUpgradeTests.asyncSetUp(self)
        self.updater.models = dict(self.updater.models)
        self.manager.models = dict(self.manager.models)
        self.bytes = {}

        def tiny(asset):
            content = (asset.name + " checked bytes\n").encode()
            self.bytes[asset.name] = content
            digest = (hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                      if asset.algorithm == "git-sha1" else hashlib.sha256(content).hexdigest())
            return replace(asset, size=len(content), digest=digest)

        assets = tuple(tiny(a) for a in shared_assets.SHARED_ASSETS)
        choices = {key: {**c, "asset": tiny(c["asset"])} for key, c in official_checkpoints.OFFICIAL_CHECKPOINTS.items()}
        for target, value in (("shared_assets.SHARED_ASSETS", assets),
                              ("official_checkpoints.OFFICIAL_CHECKPOINTS", choices),
                              ("halogen_updates.OFFICIAL_CHECKPOINTS", choices)):
            p = patch(target, value)
            p.start()
            self.addCleanup(p.stop)
        self.assets = self.updater.checkpoint_assets = SharedAssets(self.models / "storage", self.quadlets, self.root / "state")
        self.manager.asset_validator = self.assets.validate_start
        self.path = self.quadlets / "halogen-official.container"
        self.downloads = []
        self.updater._download_checkpoint_asset = self.download
        (self.models / "qwen38-flash-next-v2.hgn").write_bytes(self.bytes["qwen38-flash-next-v2.hgn"])

    async def command(self, *args, **kwargs):
        result = await test_updates.CheckpointUpgradeTests.command(self, *args, **kwargs)
        if args[:3] == ("podman", "container", "inspect"):
            value = json.loads(result)
            profile = parse_quadlet(self.path.read_text())
            value[0]["Config"] = {"Env": [f"{k}={v}" for k, v in profile.env.items()]}
            return json.dumps(value)
        return result

    async def download(self, asset, stage):
        self.assertFalse(self.manager.maintenance, "Downloads must finish before draining")
        self.assertFalse(any(c[:3] == ("systemctl", "--user", "stop") for c in self.commands))
        self.downloads.append(asset.name)
        path = stage / asset.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.bytes[asset.name])

    async def select(self, target="ht43"):
        await self.updater.start_checkpoint(target)
        await self.updater.task
        return self.updater.job

    async def test_v2_default_is_retained_and_ht43_is_explicit(self):
        cp = self.updater.status()["checkpoint"]
        self.assertEqual(cp["current"], CHECKPOINT_V2_PATH)
        self.assertFalse(cp["available"])
        self.assertEqual([c["target_key"] for c in cp["choices"]], ["v2", "ht43"])
        with self.assertRaises(web.HTTPConflict):
            await self.updater.start_checkpoint()
        self.assertEqual(self.downloads, [])

    async def test_fresh_templates_use_016_engine_and_keep_v2_default(self):
        from halogen_deploy import DeployManager
        from test_deploy import make_config
        deploy = DeployManager(self.manager, make_config(self.root))
        for kind in ("official", "official-quick", "uncensored", "uncensored-quick"):
            template = deploy.template(kind)["profile"]
            self.assertTrue(template["image"].endswith(":0.17.2"))
        self.assertEqual(deploy.template("official")["profile"]["env"]["HALOGEN_CHECKPOINT"], CHECKPOINT_V2_PATH)

    async def test_download_verify_shared_activate_and_switch_back(self):
        uncensored = (self.quadlets / "halogen-uncensored.container").read_bytes()
        original_v2 = (self.models / "qwen38-flash-next-v2.hgn").read_bytes()
        job = await self.select()
        self.assertEqual(job["phase"], "succeeded", job)
        self.assertEqual(len(self.downloads), 9)
        content = self.path.read_text()
        self.assertIn("HALOGEN_CHECKPOINT=/models/qwen38-flash-next-ht43.hgn", content)
        self.assertNotIn("HALOGEN_DOWNLOAD", content)
        self.assertIn("/shared/ngram:ro,z", content)
        self.assertEqual((self.quadlets / "halogen-uncensored.container").read_bytes(), uncensored)
        self.assertEqual((self.models / "qwen38-flash-next-v2.hgn").read_bytes(), original_v2)
        self.commands.clear()
        self.downloads.clear()
        job = await self.select("v2")
        self.assertEqual(job["phase"], "succeeded", job)
        self.assertEqual(self.downloads, [])
        self.assertIn("HALOGEN_CHECKPOINT=" + CHECKPOINT_V2_PATH, self.path.read_text())

    async def test_reused_shared_assets_download_only_ht43(self):
        await self.assets.prepare(None, self.download, lambda _: None)
        self.downloads.clear()
        job = await self.select()
        self.assertEqual(job["phase"], "succeeded", job)
        self.assertEqual(self.downloads, ["qwen38-flash-next-ht43.hgn"])

    async def test_015_rejects_ht43_before_any_download(self):
        self.updater.current_version = "0.15.3"
        self.path.write_bytes(self.path.read_bytes().replace(b":0.16.0", b":0.15.3"))
        info = self.updater._checkpoint_info("ht43")
        self.assertIn("0.16.0", info["blocked_reason"])
        with self.assertRaises(web.HTTPConflict):
            await self.updater.start_checkpoint("ht43")
        self.assertEqual(self.downloads, [])

    async def test_configured_old_engine_also_blocks_ht43(self):
        self.path.write_bytes(self.path.read_bytes().replace(b":0.16.0", b":0.15.3"))
        with self.assertRaises(web.HTTPConflict):
            await self.updater.start_checkpoint("ht43")

    async def test_disk_failure_does_not_interrupt_backend(self):
        original = self.path.read_bytes()
        with patch("shared_assets.shutil.disk_usage", return_value=type("Disk", (), {"free": 1})()):
            with self.assertRaises(web.HTTPConflict):
                await self.updater.start_checkpoint("ht43")
        self.assertEqual(original, self.path.read_bytes())
        self.assertEqual(self.downloads, [])
        self.assertFalse(self.manager.maintenance)

    async def test_checksum_failure_keeps_old_backend_and_removes_bad_stage(self):
        original = self.path.read_bytes()
        self.bytes["qwen38-flash-next-ht43.hgn"] = b"corrupt"
        job = await self.select()
        self.assertEqual(job["phase"], "failed", job)
        self.assertIn("Checksum mismatch", job["message"])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse((self.models / ".downloads/qwen38-flash-next-ht43.hgn").exists())
        self.assertFalse(self.manager.maintenance)
        self.assertFalse(any(c[:3] == ("systemctl", "--user", "stop") for c in self.commands))

    async def test_cancelled_download_keeps_quadlets_and_resume_files(self):
        entered = asyncio.Event()
        original = self.path.read_bytes()
        async def download(asset, stage):
            (stage / "resume.incomplete").write_bytes(b"partial")
            entered.set()
            await asyncio.Event().wait()
        self.updater._download_checkpoint_asset = download
        await self.updater.start_checkpoint("ht43")
        await entered.wait()
        await self.updater.close()
        self.assertEqual(self.updater.job["phase"], "failed")
        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue((self.models / ".downloads/resume.incomplete").exists())
        self.assertFalse(self.manager.maintenance)

    async def test_failed_start_restores_only_official(self):
        originals = {p: p.read_bytes() for p in self.updater._paths().values()}
        self.fail_start = True
        job = await self.select()
        self.assertEqual(job["phase"], "rolled_back", job)
        for p, original in originals.items():
            self.assertEqual(p.read_bytes(), original)
        self.assertTrue((self.models / "qwen38-flash-next-ht43.hgn").exists())
        self.assertFalse(self.manager.maintenance)

    async def test_wrong_running_checkpoint_triggers_rollback(self):
        self.updater._verify_checkpoint = AsyncMock(side_effect=UpdateError("Wrong checkpoint"))
        original = self.path.read_bytes()
        job = await self.select()
        self.assertEqual(job["phase"], "rolled_back", job)
        self.assertEqual(self.path.read_bytes(), original)

    async def test_checkpoint_changed_during_activation_triggers_rollback(self):
        original = self.path.read_bytes()
        run = self.updater._run
        async def corrupt_after_start(*args, **kwargs):
            result = await run(*args, **kwargs)
            if args[:3] == ("systemctl", "--user", "start") and self.updater.job["phase"] == "restarting":
                (self.models / "qwen38-flash-next-ht43.hgn").write_bytes(b"changed")
            return result
        self.updater._run = corrupt_after_start
        job = await self.select()
        self.assertEqual(job["phase"], "rolled_back", job)
        self.assertEqual(self.path.read_bytes(), original)

    async def test_cancellation_during_activation_restores_backend(self):
        entered = asyncio.Event()
        original = self.path.read_bytes()
        ready = self.updater._ready
        async def wait_for_cancel(*args, **kwargs):
            if self.updater.job["phase"] == "verifying":
                entered.set()
                await asyncio.Event().wait()
            return await ready(*args, **kwargs)
        self.updater._ready = wait_for_cancel
        await self.updater.start_checkpoint("ht43")
        await entered.wait()
        await self.updater.close()
        self.assertEqual(self.updater.job["phase"], "rolled_back", self.updater.job)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(self.manager.maintenance)

    async def test_startup_recovers_new_checkpoint_journal_without_touching_other_profiles(self):
        original = self.path.read_bytes()
        self.assertEqual((await self.select())["phase"], "succeeded")
        self.updater._save_job(phase="verifying", changed=True)
        uncensored = self.quadlets / "halogen-uncensored.container"
        changed = uncensored.read_bytes() + b"# changed later\n"
        uncensored.write_bytes(changed)
        recovered = self.make_updater()
        await recovered.recover_on_startup()
        self.assertEqual(recovered.job["phase"], "rolled_back", recovered.job)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(uncensored.read_bytes(), changed)
        self.assertFalse(self.manager.maintenance)

    async def test_pinned_hf_download_uses_only_named_asset(self):
        self.updater.job = {"phase": "preparing", "changed": False}
        self.updater._run = AsyncMock()
        asset = official_checkpoints.OFFICIAL_CHECKPOINTS["ht43"]["asset"]
        stage = self.models / ".downloads"
        with patch("halogen_updates.shutil.which", return_value="/test/hf"):
            await ContainerUpdater._download_checkpoint_asset(self.updater, asset, stage)
        self.updater._run.assert_awaited_once_with(
            "/test/hf", "download", asset.repo, asset.name, "--revision", asset.revision,
            "--local-dir", str(stage), timeout=6 * 3600,
        )

    async def test_shared_selinux_mounts_follow_private_parent_mount(self):
        self.assertEqual((await self.select())["phase"], "succeeded")
        mounts = parse_quadlet(self.path.read_text()).volumes
        private = next(i for i, (_, container, _) in enumerate(mounts) if container == "/models")
        shared = [(i, mode) for i, (_, container, mode) in enumerate(mounts) if container.startswith("/shared/")]
        self.assertEqual(len(shared), 3)
        self.assertTrue(all(i > private and mode == "ro,z" for i, mode in shared))

    async def test_external_edit_during_preparation_is_retained(self):
        original_download = self.download
        async def edit(asset, stage):
            await original_download(asset, stage)
            self.path.write_bytes(self.path.read_bytes() + b"# external edit\n")
        self.updater._download_checkpoint_asset = edit
        job = await self.select()
        self.assertEqual(job["phase"], "failed", job)
        self.assertIn("# external edit", self.path.read_text())
        self.assertFalse(any(c[:3] == ("systemctl", "--user", "stop") for c in self.commands))

    async def test_missing_checkpoint_after_activation_fails_offline_and_can_be_repaired(self):
        self.assertEqual((await self.select())["phase"], "succeeded")
        (self.models / "qwen38-flash-next-ht43.hgn").unlink()
        with self.assertRaisesRegex(AssetError, "no startup download"):
            await self.assets.validate_start("qwen3.8-flash")
        cp = self.updater._checkpoint_info("ht43")
        self.assertTrue(cp["available"])
        self.assertTrue(cp["repair"])
        self.commands.clear()
        self.assertEqual((await self.select())["phase"], "succeeded")

    async def test_custom_official_and_uncensored_never_receive_choices(self):
        self.path.write_bytes(self.path.read_bytes().replace(b"qwen38-flash-next-v2.hgn", b"my-model.hgn"))
        self.assertEqual(self.updater.status()["checkpoint"]["choices"], [])
        with self.assertRaises(web.HTTPConflict):
            await self.updater.start_checkpoint("ht43")
        self.manager.current_model = "qwen3.8-flash-uncensored"
        with self.assertRaises(web.HTTPConflict):
            await self.updater.start_checkpoint("ht43")

    async def test_swift_quadlets_stay_byte_identical(self):
        originals = {}
        for variant in ("swift15", "swift15-abliterated"):
            model = "halogen-" + variant
            service = model + ".service"
            self.updater.models[model] = self.manager.models[model] = service
            p = self.quadlets / (model + ".container")
            originals[p] = render_quadlet(swift_quick_template(variant, self.models, self.root / "cache", self.assets.locations())).encode()
            p.write_bytes(originals[p])
        self.assertEqual((await self.select())["phase"], "succeeded")
        for p, original in originals.items():
            self.assertEqual(p.read_bytes(), original)

    async def test_api_validates_explicit_target_and_origin(self):
        app = web.Application()
        self.updater.register_routes(app)
        async with TestClient(TestServer(app)) as client:
            headers = {"Origin": str(client.make_url("")).rstrip("/"), "X-Halogen-Action": "update"}
            for value in ("unknown", None, [], {}):
                response = await client.post("/dashboard/api/updates/checkpoint", json={"target": value}, headers=headers)
                self.assertEqual(response.status, 400)
            response = await client.post("/dashboard/api/updates/checkpoint", json={"target": "ht43"})
            self.assertEqual(response.status, 403)
            response = await client.post("/dashboard/api/updates/checkpoint", json={"target": "ht43"}, headers=headers)
            self.assertEqual(response.status, 202)
            await self.updater.task
            self.assertEqual(self.updater.job["phase"], "succeeded")


if __name__ == "__main__":
    unittest.main()
