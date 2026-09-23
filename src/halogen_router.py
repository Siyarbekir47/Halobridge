#!/usr/bin/env python3

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from halogen_dashboard import Dashboard, INFERENCE_ENDPOINTS
from halogen_deploy import DeployManager, DeployRoutes
from halogen_updates import ContainerUpdater
import settings
from discovery import discover


BACKEND_URL = "http://127.0.0.1:8831"

MODELS = {
    "qwen3.8-flash": "halogen-official.service",
    "qwen3.8-flash-uncensored": "halogen-uncensored.service",
}

DEFAULT_MODEL = "qwen3.8-flash"

DRAIN_TIMEOUT_SECONDS = 7200
START_TIMEOUT_SECONDS = 600
STOP_TIMEOUT_SECONDS = 180

GTT_PATH = Path(
    "/sys/class/drm/card0/device/mem_info_gtt_used"
)
GTT_LIMIT_BYTES = 1024 * 1024 * 1024

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
}

SESSION_COOKIE = f"{settings.APP}_session"
SESSION_MAX_AGE_S = 8 * 3600


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

LOG = logging.getLogger("halogen-router")


class RouterError(RuntimeError):
    pass


class ModelManager:
    def __init__(
        self,
        session: ClientSession,
        models: Optional[dict[str, str]] = None,
        *,
        backend_url: Optional[str] = None,
        default_model: Optional[str] = None,
        drain_timeout: Optional[float] = None,
        start_timeout: Optional[float] = None,
        stop_timeout: Optional[float] = None,
        gtt_path: Optional[str] = None,
        gtt_limit_bytes: Optional[int] = None,
    ) -> None:
        self.session = session
        self.condition = asyncio.Condition()

        self.models = models if models is not None else MODELS
        self.backend_url = backend_url
        self.default_model = default_model or DEFAULT_MODEL
        if self.default_model not in self.models:
            self.default_model = next(iter(self.models), self.default_model)
        self.drain_timeout = (
            drain_timeout if drain_timeout is not None else DRAIN_TIMEOUT_SECONDS
        )
        self.start_timeout = (
            start_timeout if start_timeout is not None else START_TIMEOUT_SECONDS
        )
        self.stop_timeout = (
            stop_timeout if stop_timeout is not None else STOP_TIMEOUT_SECONDS
        )
        self.gtt_path = Path(gtt_path) if gtt_path else GTT_PATH
        self.gtt_limit_bytes = (
            gtt_limit_bytes if gtt_limit_bytes is not None else GTT_LIMIT_BYTES
        )

        self.current_model: Optional[str] = None
        self.switch_target: Optional[str] = None
        self.switching = False
        self.active_requests = 0
        self.maintenance = False

    async def begin_maintenance(self) -> None:
        """Close admission atomically with model switching and reservation."""
        async with self.condition:
            while self.switching:
                await self.condition.wait()
            if self.maintenance:
                raise RouterError("Maintenance mode is already active")
            self.maintenance = True
            self.condition.notify_all()

    async def end_maintenance(self) -> None:
        async with self.condition:
            self.maintenance = False
            self.condition.notify_all()

    async def drain_maintenance(self) -> None:
        async with self.condition:
            while self.active_requests > 0:
                await self.condition.wait()
        # Probe the backend directly: routing this probe would reserve a new
        # request and would now correctly be rejected by the maintenance gate.
        while True:
            health = await self.backend_health()
            if not health or health.get("status") != "ok":
                raise RouterError("Backend is unreachable while draining the queue")
            if health.get("in_flight") == 0 and health.get("queued") == 0:
                return
            await asyncio.sleep(2)

    async def run_command(
        self,
        *command: str,
        check: bool = True,
        timeout: float = 600,
    ) -> tuple[int, str, str]:
        LOG.info("Running: %s", " ".join(command))

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RouterError(
                f"Timed out running: {' '.join(command)}"
            )

        stdout = stdout_bytes.decode(errors="replace").strip()
        stderr = stderr_bytes.decode(errors="replace").strip()

        if stdout:
            LOG.info("stdout: %s", stdout)

        if stderr:
            LOG.info("stderr: %s", stderr)

        if check and process.returncode != 0:
            raise RouterError(
                f"Command failed ({process.returncode}): "
                f"{' '.join(command)}; {stderr or stdout}"
            )

        return process.returncode, stdout, stderr

    async def service_is_active(self, service: str) -> bool:
        return_code, _, _ = await self.run_command(
            "systemctl",
            "--user",
            "is-active",
            "--quiet",
            service,
            check=False,
            timeout=30,
        )

        return return_code == 0

    async def backend_health(self) -> Optional[dict]:
        try:
            async with self.session.get(
                f"{self.backend_url or BACKEND_URL}/health",
                timeout=ClientTimeout(total=5),
            ) as response:
                if response.status != 200:
                    return None

                return await response.json()
        except (
            ClientError,
            asyncio.TimeoutError,
            json.JSONDecodeError,
        ):
            return None

    async def wait_for_backend(
        self,
        expected_model: str,
        timeout: Optional[float] = None,
    ) -> dict:
        if timeout is None:
            timeout = self.start_timeout
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            health = await self.backend_health()

            if (
                health
                and health.get("status") == "ok"
                and health.get("model") == expected_model
            ):
                LOG.info(
                    "Backend %s is ready",
                    expected_model,
                )
                return health

            await asyncio.sleep(2)

        raise RouterError(
            f"Backend {expected_model} was not ready within "
            f"{timeout:.0f} seconds"
        )

    async def wait_for_backend_idle(self) -> None:
        deadline = time.monotonic() + self.drain_timeout

        while time.monotonic() < deadline:
            health = await self.backend_health()

            if health is None:
                return

            in_flight = int(health.get("in_flight", 0) or 0)
            queued = int(health.get("queued", 0) or 0)

            if in_flight == 0 and queued == 0:
                LOG.info("Backend has no active requests")
                return

            LOG.info(
                "Waiting for backend: in_flight=%d queued=%d",
                in_flight,
                queued,
            )
            await asyncio.sleep(2)

        raise RouterError(
            "Timed out waiting for active requests"
        )

    def read_gtt_used(self) -> int:
        try:
            return int(self.gtt_path.read_text().strip())
        except (OSError, ValueError) as error:
            raise RouterError(
                f"Could not read GTT counter: {error}"
            ) from error

    async def wait_for_gtt_release(self) -> None:
        deadline = time.monotonic() + self.stop_timeout

        while time.monotonic() < deadline:
            used = self.read_gtt_used()

            if used <= self.gtt_limit_bytes:
                LOG.info(
                    "GTT released: %.1f MiB",
                    used / 1024 / 1024,
                )
                return

            LOG.info(
                "Waiting for GTT release: %.2f GiB",
                used / 1024 / 1024 / 1024,
            )
            await asyncio.sleep(2)

        used = self.read_gtt_used()

        raise RouterError(
            "GTT was not released: "
            f"{used / 1024 / 1024 / 1024:.2f} GiB remaining"
        )

    async def stop_service(self, model: str) -> None:
        service = self.models[model]

        await self.run_command(
            "systemctl",
            "--user",
            "stop",
            service,
            timeout=self.stop_timeout,
        )

    async def start_service(self, model: str, start_timeout: Optional[float] = None) -> None:
        service = self.models[model]

        await self.run_command(
            "systemctl",
            "--user",
            "reset-failed",
            service,
            check=False,
            timeout=30,
        )

        await self.run_command(
            "systemctl",
            "--user",
            "start",
            service,
            timeout=self.start_timeout,
        )

        await self.wait_for_backend(model, timeout=start_timeout)

    async def stop_all_model_services(self) -> None:
        for model, service in self.models.items():
            if await self.service_is_active(service):
                LOG.warning(
                    "Stopping unexpectedly active service %s",
                    service,
                )
                await self.stop_service(model)

        await self.wait_for_gtt_release()

    async def initialize(self) -> None:
        health = await self.backend_health()

        if health and health.get("model") in self.models:
            self.current_model = str(health["model"])
            LOG.info(
                "Detected an already active backend: %s",
                self.current_model,
            )
            return

        active_models = []

        for model, service in self.models.items():
            if await self.service_is_active(service):
                active_models.append(model)

        if len(active_models) == 1:
            candidate = active_models[0]

            try:
                await self.wait_for_backend(candidate)
                self.current_model = candidate
                return
            except RouterError:
                LOG.exception(
                    "Active service %s did not become ready",
                    candidate,
                )
                await self.stop_service(candidate)
                await self.wait_for_gtt_release()

        elif len(active_models) > 1:
            LOG.error(
                "Multiple backend services are active at the same time: %s",
                active_models,
            )
            await self.stop_all_model_services()

        LOG.info(
            "Starte Boot-Standardmodell %s",
            self.default_model,
        )

        await self.start_service(self.default_model)
        self.current_model = self.default_model

    async def perform_switch(self, target_model: str) -> None:
        old_model = self.current_model

        if old_model == target_model:
            return

        LOG.info(
            "Model switch: %s -> %s",
            old_model,
            target_model,
        )

        await self.wait_for_backend_idle()

        if old_model in self.models:
            await self.stop_service(old_model)
        else:
            await self.stop_all_model_services()

        await self.wait_for_gtt_release()

        try:
            await self.start_service(target_model)
        except Exception as start_error:
            LOG.exception(
                "Failed to start %s",
                target_model,
            )

            try:
                if await self.service_is_active(
                    self.models[target_model]
                ):
                    await self.stop_service(target_model)

                await self.wait_for_gtt_release()

                if old_model in self.models:
                    LOG.warning(
                        "Attempting rollback to %s",
                        old_model,
                    )
                    await self.start_service(old_model)
            except Exception:
                LOG.exception("Rollback also failed")

            raise RouterError(
                f"Model switch to {target_model} failed: "
                f"{start_error}"
            ) from start_error

        LOG.info(
            "Model switch completed: %s",
            target_model,
        )

    async def switch_with_hook(
        self,
        target_model: str,
        hook: Optional[Callable[[], Awaitable[None]]] = None,
        start_timeout: Optional[float] = None,
    ) -> None:
        """Stop the current backend and release the GPU, run an optional async
        ``hook`` (e.g. a one-shot convert container), then start
        ``target_model`` and make it current.

        Uses the same switch lock as :meth:`reserve_request`, so request
        admission stays closed for the whole operation and in-flight requests
        are drained first. The hook runs after the GPU is free and before the
        target service is started.

        If the target fails to start (or the task is cancelled), the host is
        rolled back to the previously active model so a failed takeover never
        leaves the machine without a backend. ``start_timeout`` bounds how long
        we wait for the target to become healthy before giving up and rolling
        back.
        """
        async with self.condition:
            self._check_maintenance()
            while self.switching:
                await self.condition.wait()
                self._check_maintenance()
            self.switching = True
            self.switch_target = target_model
            while self.active_requests > 0:
                await self.condition.wait()

        old_model = self.current_model
        try:
            await self.wait_for_backend_idle()
            if old_model in self.models:
                await self.stop_service(old_model)
            else:
                await self.stop_all_model_services()
            await self.wait_for_gtt_release()
            if hook is not None:
                await hook()
            await self.start_service(target_model, start_timeout=start_timeout)
        except BaseException as exc:
            cancelled = isinstance(exc, asyncio.CancelledError)
            LOG.warning(
                "Switch to %s failed (%s); rolling back to %s",
                target_model,
                "abgebrochen" if cancelled else exc,
                old_model,
            )
            try:
                if target_model in self.models and await self.service_is_active(
                    self.models[target_model]
                ):
                    await self.stop_service(target_model)
                await self.wait_for_gtt_release()
                if old_model in self.models:
                    await self.start_service(old_model)
            except Exception:
                LOG.exception("Rollback to %s failed", old_model)
            async with self.condition:
                self.switching = False
                self.switch_target = None
                self.condition.notify_all()
            if cancelled:
                raise
            raise RouterError(
                f"Switch to {target_model} failed: {exc}"
            ) from exc

        async with self.condition:
            self.current_model = target_model
            self.switching = False
            self.switch_target = None
            self.condition.notify_all()

    def _check_maintenance(self) -> None:
        if self.maintenance:
            raise web.HTTPServiceUnavailable(
                text=json.dumps({"error": {
                    "type": "maintenance",
                    "message": "Container update in progress. Try again later.",
                }}),
                content_type="application/json",
                headers={"Retry-After": "30"},
            )

    async def reserve_request(
        self,
        requested_model: Optional[str],
    ) -> str:
        self._check_maintenance()
        target_model = requested_model or self.current_model

        if target_model not in self.models:
            raise web.HTTPNotFound(
                text=json.dumps(
                    {
                        "error": {
                            "message": (
                                f"Unknown model: {target_model}"
                            ),
                            "type": "invalid_request_error",
                            "available_models": list(self.models),
                        }
                    }
                ),
                content_type="application/json",
            )

        while True:
            async with self.condition:
                self._check_maintenance()
                while self.switching:
                    await self.condition.wait()
                    self._check_maintenance()

                if self.current_model == target_model:
                    self.active_requests += 1
                    return target_model

                self.switching = True
                self.switch_target = target_model

                while self.active_requests > 0:
                    LOG.info(
                        "Waiting for %d router request(s)",
                        self.active_requests,
                    )
                    await self.condition.wait()

                break

        try:
            await self.perform_switch(target_model)
        except Exception:
            async with self.condition:
                self.switching = False
                self.switch_target = None
                self.condition.notify_all()
            raise

        async with self.condition:
            self.current_model = target_model
            self.switching = False
            self.switch_target = None
            self.active_requests += 1
            self.condition.notify_all()

        return target_model

    async def release_request(self) -> None:
        async with self.condition:
            self.active_requests = max(
                0,
                self.active_requests - 1,
            )
            self.condition.notify_all()

    async def status(self) -> dict:
        async with self.condition:
            state = {
                "status": (
                    "maintenance" if self.maintenance
                    else "switching" if self.switching else "ok"
                ),
                "active_model": self.current_model,
                "switch_target": self.switch_target,
                "active_requests": self.active_requests,
                "models": list(self.models),
            }

        health = await self.backend_health()
        state["backend_health"] = health

        return state


async def list_models(request: web.Request) -> web.Response:
    created = int(time.time())
    models = request.app.get("models", MODELS)

    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": created,
                    "owned_by": settings.APP,
                }
                for model in models
            ],
        }
    )


async def router_status(request: web.Request) -> web.Response:
    manager: ModelManager = request.app["manager"]
    return web.json_response(await manager.status())


def filtered_request_headers(
    request: web.Request,
) -> dict[str, str]:
    return {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }


def filtered_response_headers(
    headers,
) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }


async def proxy_request(request: web.Request) -> web.StreamResponse:
    manager: ModelManager = request.app["manager"]
    session: ClientSession = request.app["session"]
    dashboard: Dashboard = request.app["dashboard"]

    body = await request.read()
    requested_model: Optional[str] = None
    payload = {}

    if body:
        try:
            decoded = json.loads(body)
            if isinstance(decoded, dict):
                payload = decoded
                model_value = payload.get("model")
                if isinstance(model_value, str):
                    requested_model = model_value
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    reserved_model = await manager.reserve_request(requested_model)
    track_usage = request.method == "POST" and request.path in INFERENCE_ENDPOINTS
    trace = await dashboard.begin_request(
        request,
        payload,
        reserved_model,
    ) if track_usage else None
    headers = filtered_request_headers(request)
    if track_usage:
        # Telemetry inspects the wire body; request an uncompressed response.
        headers["Accept-Encoding"] = "identity"
        for name in list(headers):
            if name.lower() == "accept-encoding" and name != "Accept-Encoding":
                del headers[name]
        if request.path in {"/v1/chat/completions", "/v1/completions"} and payload.get("stream") is True:
            options = payload.get("stream_options")
            if options is None or isinstance(options, dict):
                options = dict(options or {})
                # Respect an explicit opt-out. Missing usage remains unknown.
                if "include_usage" not in options:
                    options["include_usage"] = True
                    payload["stream_options"] = options
                    body = json.dumps(payload).encode("utf-8")
                    headers = {name: value for name, value in headers.items()
                               if name.lower() not in {"content-length", "content-encoding"}}
    status = 500
    error_text = None
    backend_url = f"{request.app.get('backend_url') or BACKEND_URL}{request.rel_url}"

    try:
        try:
            upstream = await session.request(
                method=request.method,
                url=backend_url,
                headers=headers,
                data=body if body else None,
                allow_redirects=False,
            )
        except (ClientError, asyncio.TimeoutError) as error:
            status = 502
            error_text = "backend_unreachable"
            raise web.HTTPBadGateway(
                text=json.dumps(
                    {
                        "error": {
                            "message": (
                                "Active Halogen backend is "
                                f"unreachable: {error}"
                            ),
                            "type": "backend_error",
                        }
                    }
                ),
                content_type="application/json",
            ) from error

        status = upstream.status
        response = web.StreamResponse(
            status=upstream.status,
            reason=upstream.reason,
            headers=filtered_response_headers(
                upstream.headers
            ),
        )

        await response.prepare(request)

        try:
            content_type = upstream.headers.get("Content-Type", "")
            async for chunk in upstream.content.iter_chunked(
                64 * 1024
            ):
                if trace is not None:
                    dashboard.observe_chunk(trace, chunk, content_type)
                await response.write(chunk)

            await response.write_eof()
        finally:
            upstream.release()

        return response

    except (ConnectionResetError, asyncio.CancelledError):
        status = 499
        error_text = "client_disconnected"
        raise
    except Exception:
        if status < 400:
            status = 500
        error_text = error_text or "proxy_error"
        raise
    finally:
        try:
            if trace is not None:
                await dashboard.finish_request(trace, status, error_text)
        finally:
            await manager.release_request()


def _session_value(token: str) -> str:
    return hmac.new(
        token.encode("utf-8"),
        f"{settings.APP}-session".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _authenticated(request: web.Request, token: str) -> bool:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], token):
        return True
    if hmac.compare_digest(request.headers.get("X-Halogen-Token", ""), token):
        return True
    return hmac.compare_digest(request.cookies.get(SESSION_COOKIE, ""), _session_value(token))


LOGIN_TEMPLATE = """<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Halobridge · {login_suffix}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font: 14px/1.6 "Segoe UI", sans-serif; background: #f5f6f8; color: #202a35; display: grid; place-items: center; min-height: 100vh; margin: 0; }}
main {{ width: min(420px, calc(100vw - 40px)); padding-block: 32px; }}
form {{ display: grid; gap: 20px; padding-top: 28px; margin-top: 24px; border-top: 1px solid #dce1e7; }}
h1 {{ margin: 0; font-size: 30px; letter-spacing: -.04em; }}
.mark {{ color: #176650; font-weight: 700; font-size: 12px; letter-spacing: .08em; margin-bottom: 12px; }}
label {{ display: grid; gap: 8px; font-size: 13px; color: #5f6b79; }}
input, button {{ font: inherit; width: 100%; border: 1px solid #bfc8d2; border-radius: 5px; background: #fff; color: #202a35; padding: 11px 14px; }}
button {{ cursor: pointer; border-color: #176650; background: #176650; color: white; font-weight: 600; }}
button:hover {{ background: #104e3d; }}
:focus-visible {{ outline: 3px solid #345da7; outline-offset: 3px; }}
.err {{ color: #b02d3d; background: #fff0f1; border-left: 3px solid #b02d3d; padding: 12px; font-size: 13px; margin: 0; }}
.lang {{ margin: 8px 0 0; font-size: 12px; }}
.lang a {{ color: #345da7; text-underline-offset: 3px; }}
.lang a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<main>
<div class="mark">HALOBRIDGE</div>
<h1>{title}</h1>
<p style="color:#5f6b79;margin:12px 0 0">{prompt}</p>
<form method="post" action="/dashboard/login?lang={lang}">
{error}
<label>Token<input type="password" name="token" required autofocus autocomplete="current-password"></label>
<button type="submit">{submit}</button>
<p class="lang"><a href="?lang=de" hreflang="de">Deutsch</a> · <a href="?lang=en" hreflang="en">English</a></p>
</form>
</main>
</body>
</html>
"""


def _load_locale(lang: str) -> dict:
    if lang not in {"de", "en"}:
        lang = "en"
    try:
        from importlib.resources import files

        content = files("halobridge_data").joinpath("locales", f"{lang}.json").read_text(encoding="utf-8")
    except Exception:
        content = (
            Path(__file__).with_name("halobridge_data") / "locales" / f"{lang}.json"
        ).read_text(encoding="utf-8")
    return json.loads(content)


def _request_lang(request: web.Request) -> str:
    lang = request.query.get("lang", "")
    if lang in {"de", "en"}:
        return lang
    saved = request.cookies.get("halobridge_lang")
    return saved if saved in {"de", "en"} else "en"


def _login_response(error: bool = False, status: int = 200, lang: str = "en") -> web.Response:
    strings = _load_locale(lang)
    html = LOGIN_TEMPLATE.format(
        lang=lang,
        title=strings["login_page_title"],
        login_suffix=strings["login_page_title"],
        prompt=strings["login_prompt"],
        error=f'<p class="err" role="alert">{strings["login_error"]}</p>' if error else "",
        submit=strings["login_submit"],
    )
    response = web.Response(
        text=html,
        content_type="text/html",
        status=status,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'self'; style-src 'unsafe-inline'; script-src 'none'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
    )
    response.set_cookie("halobridge_lang", lang, path="/dashboard", samesite="Lax")
    return response


def make_auth_middleware(token: str):
    if not token:
        return None

    @web.middleware
    async def middleware(request: web.Request, handler):
        path = request.path
        if path in {"/dashboard/login", "/dashboard/logout"}:
            return await handler(request)
        protected = (
            path == "/dashboard"
            or path.startswith("/dashboard/")
            or path == "/router/status"
        )
        if not protected:
            return await handler(request)
        if _authenticated(request, token):
            return await handler(request)
        if path.startswith("/dashboard/api/") or path == "/router/status":
            return web.json_response(
                {
                    "error": {
                        "type": "unauthorized",
                        "message": "Authentication required.",
                    }
                },
                status=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        raise web.HTTPFound("/dashboard/login")

    return middleware


def make_login_handlers(token: str, secure: bool):
    async def login_page(request: web.Request) -> web.Response:
        if _authenticated(request, token):
            raise web.HTTPFound("/dashboard")
        return _login_response(
            error=request.query.get("error") == "1", lang=_request_lang(request)
        )

    async def login_post(request: web.Request) -> web.Response:
        expected_origin = f"{request.scheme}://{request.host}"
        origin = request.headers.get("Origin")
        if origin and origin != expected_origin:
            raise web.HTTPForbidden(text="Login requires the same origin.")

        if request.content_type == "application/json":
            try:
                body = await request.json()
            except (ValueError, UnicodeError):
                body = {}
            provided = body.get("token", "") if isinstance(body, dict) else ""
        else:
            form = await request.post()
            provided = form.get("token", "")

        if not isinstance(provided, str) or not hmac.compare_digest(provided, token):
            return _login_response(error=True, status=401, lang=_request_lang(request))

        response = web.HTTPFound("/dashboard")
        response.set_cookie(
            SESSION_COOKIE,
            _session_value(token),
            httponly=True,
            samesite="Lax",
            max_age=SESSION_MAX_AGE_S,
            secure=secure,
        )
        raise response

    async def logout(request: web.Request) -> web.Response:
        response = web.HTTPFound("/dashboard/login")
        response.del_cookie(
            SESSION_COOKIE,
            httponly=True,
            samesite="Lax",
            secure=secure,
        )
        raise response

    return login_page, login_post, logout


def resolve_models(
    config: settings.Config,
) -> tuple[dict, list[str], dict[str, str]]:
    specs: dict = {}
    skipped: list[str] = []
    if config.models.auto_discover:
        specs, skipped = discover(config.models.quadlet_dir, config.updates.image_repo)
    models = {spec.model_id: spec.service for spec in specs.values()}
    models.update(config.models.explicit)
    return specs, skipped, models


async def create_application(config: Optional[settings.Config] = None) -> web.Application:
    config = config or settings.load()

    # Preserve existing telemetry if a host still uses the pre-Halobridge state dir.
    if (
        config.dashboard.state_dir == settings.DEFAULT_STATE_DIR
        and not config.dashboard.state_dir.exists()
        and settings.LEGACY_STATE_DIR.exists()
    ):
        config.dashboard.state_dir = settings.LEGACY_STATE_DIR
        LOG.warning(
            "Legacy state directory found; dashboard uses %s",
            settings.LEGACY_STATE_DIR,
        )

    specs, skipped, models = resolve_models(config)
    if not models:
        raise RouterError(
            "No models found. Check models.auto_discover/quadlet_dir "
            "or set models.explicit."
        )

    default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else next(iter(models))

    timeout = ClientTimeout(
        total=None,
        connect=15,
        sock_connect=15,
        sock_read=None,
    )

    session = ClientSession(
        timeout=timeout,
        auto_decompress=False,
    )

    manager = ModelManager(
        session,
        models,
        backend_url=config.router.backend_url,
        default_model=default_model,
        drain_timeout=config.router.drain_timeout_s,
        start_timeout=config.router.start_timeout_s,
        stop_timeout=config.router.stop_timeout_s,
        gtt_path=config.router.gtt_path,
        gtt_limit_bytes=config.router.gtt_limit_bytes,
    )
    updater = ContainerUpdater(manager, session, models, config=config)
    await updater.recover_on_startup()
    if not updater.recovery_required:
        await manager.initialize()
    dashboard = Dashboard(
        manager=manager,
        session=session,
        backend_url=config.router.backend_url,
        models=models,
        specs=specs,
        state_dir=config.dashboard.state_dir,
        retention_days=config.dashboard.retention_days,
        gpu_card=config.dashboard.gpu_card,
        public_endpoint=config.dashboard.public_endpoint,
    )
    await dashboard.initialize()

    deploy_manager = DeployManager(manager, config, updater=updater, dashboard=dashboard)
    deploy_routes = DeployRoutes(deploy_manager)

    app = web.Application(
        client_max_size=128 * 1024 * 1024,
    )

    app["session"] = session
    app["manager"] = manager
    app["dashboard"] = dashboard
    app["updater"] = updater
    app["deploy"] = deploy_manager
    app["models"] = models
    app["backend_url"] = config.router.backend_url
    app["config"] = config

    if skipped:
        LOG.warning("Model discovery skipped entries: %s", "; ".join(skipped))

    if config.security.auth_token:
        secure = config.dashboard.public_endpoint.startswith("https://")
        middleware = make_auth_middleware(config.security.auth_token)
        if middleware:
            app.middlewares.append(middleware)
        login_page, login_post, logout = make_login_handlers(
            config.security.auth_token, secure
        )
        app.router.add_get("/dashboard/login", login_page)
        app.router.add_post("/dashboard/login", login_post)
        app.router.add_post("/dashboard/logout", logout)

    dashboard.register_routes(app)
    updater.register_routes(app)
    deploy_routes.register_routes(app)
    updater.start_checks()
    app.router.add_get("/v1/models", list_models)
    app.router.add_get("/router/status", router_status)

    app.router.add_route("*", "/{tail:.*}", proxy_request)

    async def close_session(_: web.Application) -> None:
        await updater.close()
        await dashboard.close()
        await session.close()

    app.on_cleanup.append(close_session)

    return app


def app_version() -> str:
    """Single source of truth: the installed package metadata (pyproject.toml).

    Falls back to '0+local' when running from a bare source checkout that was
    never pip-installed, so --version never lies about a released number.
    """
    return settings.app_version()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=settings.APP,
        description="Router & Dashboard for halogen",
    )
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration and exit",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{settings.APP} {app_version()}",
    )

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("router", help="Start router and dashboard")
    sub.add_parser("dashboard", help="Alias for router")
    sub.add_parser("doctor", help="Check setup")
    sub.add_parser("update-check", help="Check update status without installing")

    return parser.parse_args(argv)


async def run_router(config: settings.Config) -> None:
    try:
        app = await create_application(config)
    except RouterError as error:
        print(f"Router error: {error}", file=sys.stderr)
        raise SystemExit(1)

    runner = web.AppRunner(app)
    await runner.setup()

    sites = []

    for address in config.router.bind:
        site = web.TCPSite(
            runner,
            host=address,
            port=config.router.port,
            reuse_address=True,
        )
        await site.start()
        sites.append(site)

        LOG.info(
            "Router listening on http://%s:%d",
            address,
            config.router.port,
        )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for signal_name in (
        signal.SIGTERM,
        signal.SIGINT,
    ):
        try:
            loop.add_signal_handler(
                signal_name,
                stop_event.set,
            )
        except NotImplementedError:
            pass

    await stop_event.wait()
    await runner.cleanup()


async def run_update_check(config: settings.Config) -> int:
    _, _, models = resolve_models(config)
    if not models:
        print("No models found; update check is unavailable.", file=sys.stderr)
        return 1

    session = ClientSession(
        timeout=ClientTimeout(total=20),
        auto_decompress=False,
    )
    try:
        manager = ModelManager(
            session,
            models,
            backend_url=config.router.backend_url,
        )
        updater = ContainerUpdater(manager, session, models, config=config)
        status = await updater.check(force=True)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0 if not status.get("check_error") else 1
    finally:
        await session.close()


def cli(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    try:
        config = settings.load(args.config)
    except settings.ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        raise SystemExit(2)

    if args.check_config:
        print("Configuration is valid.")
        return

    command = args.command or "router"

    if command in {"router", "dashboard"}:
        asyncio.run(run_router(config))
    elif command == "doctor":
        from doctor import run as run_doctor

        raise SystemExit(asyncio.run(run_doctor(config)))
    elif command == "update-check":
        raise SystemExit(asyncio.run(run_update_check(config)))
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        raise SystemExit(2)


async def main(argv: Optional[list[str]] = None) -> None:
    """Backwards-compatible async entry point."""
    args = parse_args(argv)
    try:
        config = settings.load(args.config)
    except settings.ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        raise SystemExit(2)
    if args.check_config:
        print("Configuration is valid.")
        return
    await run_router(config)


if __name__ == "__main__":
    cli()
