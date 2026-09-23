"""Optional real-browser checks using synthetic in-memory telemetry.

Requires playwright and a Chromium browser. Nothing contacts the real backend.
python tests/browser_check.py --browser "/path/to/chrome" --screenshots /tmp/halogen
"""

import argparse
import asyncio
import logging
from pathlib import Path
import sys
import time

from aiohttp import web
from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright, expect

from test_dashboard import database, insert


async def main(args):
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    dashboard = database()
    now = time.time()
    local = time.localtime(now)
    midnight = time.mktime(local[:3] + (0, 0, 0) + local[6:])
    span = min(900, now - midnight)
    for i in range(32):
        insert(
            dashboard, request_id=f'browser-{i}', completed_at=now - (i + 1) * span / 40,
            client="OpenCode" if i % 3 else "Open WebUI", input_tokens=1000 + i * 90,
            output_tokens=100 + i * 10, cached_tokens=800 if i % 2 else 0,
            status=200 if i % 9 else 499, duration_ms=3200 + i * 50,
            telemetry_version=2 if i else 0,
        )
    insert(dashboard, completed_at=now - span * .9, input_tokens=None, output_tokens=None, cached_tokens=None, reasoning_tokens=None)
    updates = {
        "current_version": "0.13.1", "latest_version": "0.13.1", "checked_at": now,
        "configured_versions": {"qwen3.8-flash": "0.13.1", "qwen3.8-flash-uncensored": "0.13.1"},
        "supported": True, "update_available": False, "can_install": False, "up_to_date": True,
        "running": False, "recovery_required": False, "can_recover": False, "job": None,
        "release_url": "https://github.com/peonist-ai/halogen-flash-server/blob/v0.13.1/CHANGELOG.md",
    }
    update_actions = []

    async def update_status(request):
        if request.method == "POST":
            assert request.headers.get("Origin") == f"{request.scheme}://{request.host}"
            assert request.headers.get("X-Halogen-Action") == "update"
            action = request.match_info["action"]
            update_actions.append(action)
            if action == "install":
                assert (await request.json())["version"] == updates["latest_version"]
                updates.update(running=True, can_install=False, job={"phase": "pulling", "message": "Image wird geladen."})
            elif action == "recover":
                updates.update(running=True, can_recover=False, job={"phase": "rolling_back", "message": "Vorherige Version wird wiederhergestellt."})
        return web.json_response(updates)

    async def snapshot():
        return {
            "generated_at": time.time(),
            "api": {"active_model": "qwen3.8-flash", "status": "maintenance" if updates["running"] else "ok", "models": ["qwen3.8-flash"]},
            "backend": {"status": "ok", "in_flight": 0, "queued": 0, "context": 262144,
                        "slots": 4, "max_tokens_default": 8192, "max_tokens_cap": 65536,
                        "reasoning_effort_default": "xhigh", "version": {"api": "test"}},
            "system": {"gpu_busy_percent": 8, "memory": {"used_bytes": 62 * 1024**3, "total_bytes": 128 * 1024**3},
                       "disk_used_bytes": 420 * 1024**3, "disk_total_bytes": 2000 * 1024**3},
            "cache": {"pool": {"usage_ratio": .3}, "model_bytes": {}}, "active_requests": [],
        }

    dashboard.snapshot = snapshot
    app = web.Application()
    dashboard.register_routes(app)
    app.router.add_get("/dashboard/api/updates", update_status)
    app.router.add_post("/dashboard/api/updates/{action}", update_status)
    async with TestServer(app) as server, async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=args.browser, headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 1050}, timezone_id="Europe/Berlin")
        page = await context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(str(server.make_url("/dashboard")))
        await expect(page.locator('#connection')).to_have_text('Ready')
        await expect(page.locator('html')).to_have_attribute('lang', 'en')
        await page.locator('#langSelect').select_option('de')
        await expect(page.locator("#analytics")).to_have_attribute("aria-busy", "false")
        await expect(page.locator("#requestCount")).to_have_text("33")
        await expect(page.locator("#connection")).to_have_text("Bereit")
        await expect(page.locator("#updateTitle")).to_contain_text("Server 0.13.1")
        await expect(page.locator("#installUpdate")).to_be_hidden()
        await expect(page.locator("#coverageNotice")).to_contain_text("31 von 33")
        await expect(page.locator("#coverageNotice")).to_contain_text("1 Altbestände")
        assert await page.locator("#tokenChart rect").count() > 0
        await expect(page.locator("#tokenChart")).to_be_visible()
        assert (await page.locator("#tokenChart").bounding_box())["height"] >= 150
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        await page.get_by_role('link', name='Anfragen', exact=True).click()
        await page.locator(".row-toggle").first.click()
        await expect(page.locator(".request-detail").first).to_be_visible()
        await expect(page.locator(".request-detail").first).to_contain_text("Altbestand")
        await page.locator(".row-toggle").first.click()

        if args.screenshots:
            await page.get_by_role('link', name='Übersicht', exact=True).click()
            await page.screenshot(path=str(args.screenshots / "dashboard-desktop.png"), full_page=True)
            await page.get_by_role('link', name='Anfragen', exact=True).click()

        # Pagination pins the range so incoming requests do not shift older pages.
        await page.locator("#nextPage").click()
        await expect(page.locator("#pageInfo")).to_contain_text("Seite 2 von 4")
        insert(dashboard, completed_at=time.time(), request_id="new-request")
        await page.locator("#refresh").click()
        await expect(page.locator("#analytics")).to_have_attribute("aria-busy", "false")
        await expect(page.locator("#requestCount")).to_have_text("33")
        await page.locator("#prevPage").click()
        await expect(page.locator("#requestCount")).to_have_text("34")

        await page.locator("#period").select_option("7d")
        await expect(page.locator("#chartNote")).to_contain_text("Täglich")
        await page.locator("#period").select_option("custom")
        await expect(page.locator("#customRange")).to_be_visible()
        # Compare datetime-local with the local clock, not UTC.
        offset = await page.evaluate("Math.abs(new Date(document.getElementById('toDate').value).getTime() - Date.now())")
        assert offset < 120000
        await page.locator("#fromDate").fill("2030-01-01T12:00")
        await page.locator("#toDate").fill("2029-01-01T12:00")
        await page.get_by_role("button", name="Anwenden", exact=True).click()
        await expect(page.locator("#rangeHint")).to_contain_text("gültigen")

        # Pick an empty time range between the historical records and the new one.
        start, end = await page.evaluate("""() => {
            const local = d => new Date(d.getTime() - d.getTimezoneOffset()*60000).toISOString().slice(0,16);
            return [local(new Date(Date.now()-3600000)), local(new Date(Date.now()-1800000))];
        }""")
        await page.locator("#fromDate").fill(start)
        await page.locator("#toDate").fill(end)
        await page.get_by_role("button", name="Anwenden", exact=True).click()
        await expect(page.locator("#requestCount")).to_have_text("0")
        await expect(page.locator("#inputTokens")).to_have_text("–")
        await expect(page.locator("#chartEmpty")).to_have_text("Keine Anfragen in diesem Zeitraum.")
        await expect(page.locator("#nextPage")).to_be_disabled()

        # True zero usage is rendered as zero, not as missing data.
        insert(dashboard, completed_at=now - 2700, input_tokens=0, output_tokens=0, cached_tokens=0, reasoning_tokens=0)
        await page.locator("#refresh").click()
        await expect(page.locator("#inputTokens")).to_have_text("0")
        await expect(page.locator("#outputTokens")).to_have_text("0")
        await expect(page.locator("#cacheRatio")).to_have_text("–")

        await page.route("**/dashboard/api/analytics?**", lambda route: route.abort())
        await page.locator("#refresh").click()
        await expect(page.locator("#analyticsError")).to_be_visible()
        await page.unroute("**/dashboard/api/analytics?**")
        await page.locator("#refresh").click()
        await expect(page.locator("#analyticsError")).to_be_hidden()

        await page.route("**/dashboard/api/snapshot", lambda route: route.abort())
        await page.locator("#refresh").click()
        await expect(page.locator("#connection")).to_have_text("Verbindung unterbrochen")
        await page.unroute("**/dashboard/api/snapshot")
        await page.locator("#refresh").click()
        await expect(page.locator("#connection")).to_have_text("Bereit")

        # A superseded slow request must not overwrite the newly selected range.
        async def slow_response(route):
            response = await route.fetch()
            await asyncio.sleep(.3)
            await route.fulfill(response=response)
        await page.route("**/dashboard/api/analytics?period=7d*", slow_response)
        await page.locator("#period").select_option("7d")
        await page.locator("#period").select_option("24h")
        await expect(page.locator("#chartNote")).to_contain_text("Stündlich")
        await page.wait_for_timeout(400)
        await expect(page.locator("#chartNote")).to_contain_text("Stündlich")
        await page.unroute("**/dashboard/api/analytics?period=7d*")

        await page.get_by_role('link', name='System', exact=True).click()
        # One click starts the server-side job; status survives a page reload.
        updates.update(latest_version="0.13.2", update_available=True, can_install=True)
        await page.locator("#checkUpdate").click()
        await expect(page.locator("#installUpdate")).to_have_text("Auf 0.13.2 aktualisieren")
        await page.locator("#installUpdate").click()
        await expect(page.locator("#updateMessage")).to_have_text("Image wird geladen.")
        await expect(page.locator("#installUpdate")).to_be_disabled()
        await expect(page.locator("#connection")).to_have_text("Wartung")
        assert update_actions.count("install") == 1
        await page.reload()
        await expect(page.locator("#updateTitle")).to_contain_text("Update läuft")
        updates.update(running=False, current_version="0.13.2", update_available=False,
                       configured_versions={"qwen3.8-flash": "0.13.2", "qwen3.8-flash-uncensored": "0.13.2"},
                       job={"phase": "succeeded", "message": "Beide Quadlets aktualisiert.", "backup_dir": "/synthetic/backup"})
        await page.evaluate("refreshUpdates()")
        await expect(page.locator("#installUpdate")).to_be_hidden()
        await expect(page.locator("#updateMessage")).to_have_text("Beide Quadlets aktualisiert.")

        updates.update(recovery_required=True, can_recover=True,
                       job={"phase": "recovery_required", "message": "Wiederherstellung erforderlich."})
        await page.evaluate("refreshUpdates()")
        await expect(page.locator("#recoverUpdate")).to_be_enabled()
        await page.locator("#recoverUpdate").click()
        await expect(page.locator("#recoverUpdate")).to_be_disabled()
        assert update_actions.count("recover") == 1
        updates.update(recovery_required=False, can_recover=False, running=False,
                       job={"phase": "rolled_back", "message": "Vorherige Version ist wieder aktiv."})
        await page.evaluate("refreshUpdates()")
        await expect(page.locator("#recoverUpdate")).to_be_hidden()
        await page.route("**/dashboard/api/updates/check", lambda route: route.fulfill(status=503, body="Versionsprüfung fehlgeschlagen."))
        await page.locator("#checkUpdate").click()
        await expect(page.locator("#updateError")).to_have_text("Versionsprüfung fehlgeschlagen.")
        await page.unroute("**/dashboard/api/updates/check")
        updates.update(job=None)
        await page.locator("#checkUpdate").click()
        await expect(page.locator("#updateError")).to_be_hidden()
        await expect(page.locator("#analytics")).to_have_attribute("aria-busy", "false")
        await expect(page.locator("#context")).to_have_text("262.144 / 4")
        await page.get_by_role('link', name='Übersicht', exact=True).click()
        for width in (768, 390, 320):
            await page.set_viewport_size({"width": width, "height": 844})
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"Page overflows at {width}px"
            await expect(page.locator("#period")).to_be_visible()
            await expect(page.locator("#tokenChart")).to_be_visible()
            await page.get_by_role('link', name='Anfragen', exact=True).click()
            await expect(page.locator("#nextPage")).to_be_visible()
            await page.get_by_role('link', name='Übersicht', exact=True).click()
            if args.screenshots and width == 390:
                await page.screenshot(path=str(args.screenshots / "dashboard-mobile.png"), full_page=True)
        assert not errors, errors
        await browser.close()
    await dashboard.close()
    print("Browser checks passed: desktop/tablet/mobile, history, time ranges, usage, connection recovery, one-click updates/reload/recovery/errors, no JavaScript errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", help="Path to an existing Chromium/Chrome executable")
    parser.add_argument("--screenshots", type=Path, help="Existing directory for screenshots")
    args = parser.parse_args()
    if args.screenshots and not args.screenshots.is_dir():
        sys.exit("Screenshot directory must already exist")
    asyncio.run(main(args))
