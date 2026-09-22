#!/usr/bin/env python3
"""Local operational telemetry. Request and response content is never persisted."""

from __future__ import annotations

import asyncio
import datetime
import json
import math
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from aiohttp import ClientSession, ClientTimeout, web


STATE_DIR = Path.home() / ".local/state/halogen-dashboard"
RETENTION_DAYS = 365
TELEMETRY_VERSION = 2
PERIOD_SECONDS = {"24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400, "1y": 365 * 86400}
INFERENCE_ENDPOINTS = {"/v1/chat/completions", "/v1/completions", "/v1/responses"}
SERVE_HEAD_RE = re.compile(
    r"serve_api: (?P<mode>\w+) (?P<tokens>\d+) tok in "
    r"(?P<seconds>[\d.]+)s = (?P<tps>[\d.]+|n/a) t/s"
)
REQUEST_SCOPE = "endpoint IN ('/v1/chat/completions','/v1/completions','/v1/responses')"

# Every aggregate uses this same population. Legacy counters cannot be repaired
# from stored metadata: the original usage object was deliberately not retained.
MEASUREMENTS_CTE = f"""
WITH measured AS (
    SELECT *,
        CASE WHEN telemetry_version >= {TELEMETRY_VERSION}
                  AND input_tokens >= 0 THEN input_tokens END AS measured_input,
        CASE WHEN telemetry_version >= {TELEMETRY_VERSION}
                  AND output_tokens >= 0 THEN output_tokens END AS measured_output,
        CASE WHEN telemetry_version >= {TELEMETRY_VERSION}
                  AND cached_tokens BETWEEN 0 AND input_tokens
             THEN cached_tokens END AS measured_cache,
        CASE WHEN telemetry_version >= {TELEMETRY_VERSION}
                  AND reasoning_tokens BETWEEN 0 AND output_tokens
             THEN reasoning_tokens END AS measured_reasoning
    FROM requests
    WHERE completed_at >= ? AND completed_at <= ? AND {REQUEST_SCOPE}
)
"""
TOKEN_AGGREGATES = """
    COUNT(*) AS requests,
    SUM(measured_input) AS input_tokens,
    SUM(measured_output) AS output_tokens,
    SUM(measured_cache) AS cached_tokens,
    SUM(measured_reasoning) AS reasoning_tokens,
    SUM(CASE WHEN measured_cache IS NOT NULL
             THEN measured_input - measured_cache END) AS new_input_tokens,
    SUM(CASE WHEN measured_cache IS NOT NULL
             THEN measured_input END) AS cache_input_tokens,
    SUM(measured_input + measured_output) AS total_tokens,
    COUNT(measured_input) AS input_reported,
    COUNT(measured_output) AS output_reported,
    COUNT(measured_cache) AS cache_reported,
    COUNT(measured_reasoning) AS reasoning_reported,
    COUNT(measured_input + measured_output) AS usage_reported,
    SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END) AS errors
"""


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _int(value: Any) -> Optional[int]:
    number = _num(value)
    # SQLite uses signed 64-bit integers. Never truncate malformed token counts.
    if number is None or not number.is_integer() or number >= 2**63:
        return None
    return int(value)


def _safe_label(value: str, maximum: int = 64) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 ._:/@+-]", "", value or "").strip()
    return cleaned[:maximum] or "Unbekannt"


def _read_int(path: str) -> Optional[int]:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _gpu_cards() -> list[Path]:
    base = Path("/sys/class/drm")
    if not base.is_dir():
        return []
    cards = []
    for path in base.iterdir():
        match = re.fullmatch(r"card(\d+)", path.name)
        if match and path.is_dir():
            cards.append((int(match.group(1)), path))
    return [path for _, path in sorted(cards)]


def _memory_info() -> dict[str, Any]:
    try:
        values = {
            key: int(raw.strip().split()[0]) * 1024
            for key, raw in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        }
    except (OSError, ValueError, IndexError):
        return {}
    total, available = values.get("MemTotal"), values.get("MemAvailable")
    used = max(0, total - available) if total is not None and available is not None else None
    return {
        "total_bytes": total, "used_bytes": used,
        "usage_ratio": used / total if total and used is not None else None,
    }


def _client_name(request: web.Request) -> str:
    explicit = request.headers.get("X-Halogen-Client", "")
    if explicit:
        return _safe_label(explicit)
    agent = request.headers.get("User-Agent", "")
    # A generic SDK user agent does not identify the application using it.
    for marker, name in (("opencode", "OpenCode"), ("openclaw", "OpenClaw"), ("open-webui", "Open WebUI"), ("curl", "curl")):
        if marker in agent.lower():
            return name
    return _safe_label(agent.split(" ", 1)[0] if agent else "Unbekannt")


def _thinking(payload: dict[str, Any]) -> tuple[str, str]:
    effort = payload.get("reasoning_effort")
    if not isinstance(effort, str):
        reasoning = payload.get("reasoning")
        thinking = payload.get("thinking")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
        elif isinstance(thinking, dict):
            effort = thinking.get("type")
    effort_text = _safe_label(str(effort), 20) if effort is not None else "default"
    enabled = payload.get("enable_thinking")
    if enabled is False or effort_text in {"none", "disabled", "off"}:
        return "off", effort_text
    if enabled is True or effort_text != "default":
        return "on", effort_text
    return "unknown", "default"


class Trace:
    """Extract cumulative API usage without guessing tokens from response bytes."""

    def __init__(self, request: web.Request, payload: dict[str, Any], model: str) -> None:
        self.request_id = f"r-{time.time_ns():x}"
        self.started_epoch = time.time()
        self.started_mono = time.monotonic()
        self.first_byte_mono: Optional[float] = None
        self.client = _client_name(request)
        self.client_ip = _safe_label(request.remote or "lokal", 64)
        self.model = model
        self.endpoint = request.path
        self.stream = bool(payload.get("stream", False))
        self.thinking, self.reasoning_effort = _thinking(payload)
        self.max_tokens = next((
            _int(payload[key]) for key in ("max_completion_tokens", "max_output_tokens", "max_tokens")
            if _int(payload.get(key)) is not None
        ), None)
        self.response_bytes = 0
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.cached_tokens: Optional[int] = None
        self.reasoning_tokens: Optional[int] = None
        self.tps: Optional[float] = None
        self.prompt_tps: Optional[float] = None
        self._json_buffer = bytearray()
        self._line_buffer = bytearray()
        self._event_data: list[bytes] = []
        self._event_size = 0
        self._discard_event = False
        self._discard_json = False

    def _extract(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        # Responses streams wrap final usage in response.completed/incomplete.
        response = data.get("response")
        if isinstance(response, dict):
            self._extract(response)
        usage = data.get("usage")
        if isinstance(usage, dict):
            for attr, keys in (
                ("input_tokens", ("prompt_tokens", "input_tokens")),
                ("output_tokens", ("completion_tokens", "output_tokens")),
            ):
                for key in keys:
                    count = _int(usage.get(key))
                    if count is not None:
                        setattr(self, attr, count)
                        break
            for attr, detail_keys, key in (
                ("cached_tokens", ("prompt_tokens_details", "input_tokens_details"), "cached_tokens"),
                ("reasoning_tokens", ("completion_tokens_details", "output_tokens_details"), "reasoning_tokens"),
            ):
                for details_key in detail_keys:
                    details = usage.get(details_key)
                    count = _int(details.get(key)) if isinstance(details, dict) else None
                    if count is not None:
                        setattr(self, attr, count)
                        break
            if self.reasoning_tokens is not None and self.reasoning_tokens > 0:
                self.thinking = "on"

        timings = data.get("timings")
        if isinstance(timings, dict):
            for attr, key in (("tps", "predicted_per_second"), ("prompt_tps", "prompt_per_second")):
                value = _num(timings.get(key))
                if value is not None:
                    setattr(self, attr, value)
            # prompt_n may count only uncached prefill tokens. It must never
            # overwrite usage.prompt_tokens (the full logical input).

    def _parse_json(self, raw: bytes) -> None:
        if not raw or raw.strip() == b"[DONE]":
            return
        try:
            self._extract(json.loads(raw))
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
            pass

    def _flush_event(self) -> None:
        if not self._discard_event:
            self._parse_json(b"\n".join(self._event_data))
        self._event_data.clear()
        self._event_size = 0
        self._discard_event = False

    def _sse_line(self, line: bytes) -> None:
        line = line.rstrip(b"\r")
        if not line:
            self._flush_event()
        elif line.startswith(b"data:") and not self._discard_event:
            raw = line[5:]
            if raw.startswith(b" "):
                raw = raw[1:]
            self._event_size += len(raw)
            if self._event_size > 2 * 1024 * 1024:
                self._event_data.clear()
                self._discard_event = True
            else:
                self._event_data.append(raw)

    def observe(self, chunk: bytes, content_type: str) -> None:
        if self.first_byte_mono is None and chunk:
            self.first_byte_mono = time.monotonic()
        self.response_bytes += len(chunk)
        if "text/event-stream" in content_type.lower():
            self._line_buffer.extend(chunk)
            while b"\n" in self._line_buffer:
                line, _, remainder = self._line_buffer.partition(b"\n")
                self._line_buffer = bytearray(remainder)
                self._sse_line(bytes(line))
            if len(self._line_buffer) > 2 * 1024 * 1024:
                self._line_buffer.clear()
                self._discard_event = True
        elif not self._discard_json:
            if len(self._json_buffer) + len(chunk) > 4 * 1024 * 1024:
                self._json_buffer.clear()
                self._discard_json = True
            else:
                self._json_buffer.extend(chunk)

    def finalize_parsing(self) -> None:
        if self._json_buffer:
            self._parse_json(bytes(self._json_buffer))
        if self._line_buffer:
            self._sse_line(bytes(self._line_buffer))
        self._flush_event()
        self._json_buffer.clear()
        self._line_buffer.clear()
        if self.cached_tokens is not None and (
            self.input_tokens is None or self.cached_tokens > self.input_tokens
        ):
            self.cached_tokens = None
        if self.reasoning_tokens is not None and (
            self.output_tokens is None or self.reasoning_tokens > self.output_tokens
        ):
            self.reasoning_tokens = None


class Dashboard:
    def __init__(
        self,
        manager: Any,
        session: ClientSession,
        backend_url: str,
        models: dict[str, str],
        specs: Optional[dict[str, Any]] = None,
        *,
        state_dir: Optional[Path] = None,
        retention_days: int = RETENTION_DAYS,
        gpu_card: str = "auto",
        public_endpoint: str = "",
    ) -> None:
        self.manager, self.session = manager, session
        self.backend_url, self.models = backend_url, models
        self.specs = specs or {}
        # The final routing map is authoritative. A discovered service that was
        # explicitly overridden must not keep claiming engine telemetry.
        self.service_models = {service: model for model, service in models.items()}
        self.cache_paths = {
            spec.model_id: spec.cache_path
            for spec in self.specs.values()
            if spec.cache_path
        }
        self.journal_units = sorted(set(self.service_models))
        self.state_dir = Path(state_dir) if state_dir else STATE_DIR
        self.db_path = self.state_dir / "telemetry.sqlite3"
        self.retention_days = retention_days
        self.gpu_card = gpu_card
        self.public_endpoint = public_endpoint
        self.db: Optional[sqlite3.Connection] = None
        self.active: dict[str, Trace] = {}
        self.cache_sizes: dict[str, int] = {}
        self.cache_sizes_updated = 0.0
        self.cache_refresh_task: Optional[asyncio.Task] = None
        self.journal_task: Optional[asyncio.Task] = None
        self.journal_process: Optional[asyncio.subprocess.Process] = None
        self.write_count = 0

    def _initialize_schema(self) -> None:
        assert self.db is not None
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL,
                started_at REAL NOT NULL, completed_at REAL NOT NULL,
                client TEXT NOT NULL, client_ip TEXT NOT NULL, model TEXT NOT NULL,
                endpoint TEXT NOT NULL, status INTEGER NOT NULL, duration_ms REAL NOT NULL,
                ttft_ms REAL, stream INTEGER NOT NULL, thinking TEXT NOT NULL,
                reasoning_effort TEXT NOT NULL, max_tokens INTEGER, input_tokens INTEGER,
                output_tokens INTEGER, cached_tokens INTEGER, reasoning_tokens INTEGER,
                tps REAL, prompt_tps REAL, response_bytes INTEGER NOT NULL, error TEXT
            );
            CREATE INDEX IF NOT EXISTS requests_completed_idx ON requests(completed_at DESC);
            CREATE TABLE IF NOT EXISTS engine_requests (
                cursor TEXT PRIMARY KEY, completed_at REAL NOT NULL, model TEXT NOT NULL,
                mode TEXT NOT NULL, output_tokens INTEGER NOT NULL, generation_s REAL NOT NULL,
                tps REAL, rounds INTEGER, commit_per_round REAL, prompt_tokens INTEGER,
                cached_tokens INTEGER, cache_percent REAL, new_tokens INTEGER,
                prefill_s REAL, prefill_tps REAL, detok_us REAL, beside_tokens INTEGER,
                head_off_tokens INTEGER, pld_rounds INTEGER, pld_accept_per_round REAL,
                pool_used INTEGER, pool_total INTEGER, pool_percent REAL, thinking TEXT,
                closed_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS engine_completed_idx ON engine_requests(completed_at DESC);
        """)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(requests)")}
        if "telemetry_version" not in columns:
            self.db.execute("ALTER TABLE requests ADD COLUMN telemetry_version INTEGER NOT NULL DEFAULT 0")
        self.db.commit()

    async def initialize(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.state_dir, 0o700)
        self.db = sqlite3.connect(self.db_path)
        os.chmod(self.db_path, 0o600)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self._initialize_schema()
        self._prune()
        if self.journal_units and Path("/usr/bin/journalctl").exists():
            self.journal_task = asyncio.create_task(self._journal_loop())

    async def close(self) -> None:
        tasks = [task for task in (self.cache_refresh_task, self.journal_task) if task]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.journal_process and self.journal_process.returncode is None:
            self.journal_process.terminate()
            try:
                await asyncio.wait_for(self.journal_process.wait(), timeout=3)
            except asyncio.TimeoutError:
                self.journal_process.kill()
                await self.journal_process.wait()
        if self.db:
            self.db.close()
            self.db = None

    async def begin_request(self, request: web.Request, payload: dict[str, Any], model: str) -> Trace:
        trace = Trace(request, payload, model)
        self.active[trace.request_id] = trace
        return trace

    def observe_chunk(self, trace: Trace, chunk: bytes, content_type: str) -> None:
        trace.observe(chunk, content_type)

    async def finish_request(self, trace: Trace, status: int, error: Optional[str] = None) -> None:
        trace.finalize_parsing()
        self.active.pop(trace.request_id, None)
        if not self.db:
            return
        duration_ms = max(0.0, (time.monotonic() - trace.started_mono) * 1000)
        # Historical column name retained for compatibility. It measures the
        # first response BYTE, not the first generated token (SSE sends headers).
        ttfb_ms = (trace.first_byte_mono - trace.started_mono) * 1000 if trace.first_byte_mono is not None else None
        self.db.execute("""
            INSERT INTO requests (
                request_id, started_at, completed_at, client, client_ip, model,
                endpoint, status, duration_ms, ttft_ms, stream, thinking,
                reasoning_effort, max_tokens, input_tokens, output_tokens,
                cached_tokens, reasoning_tokens, tps, prompt_tps, response_bytes,
                error, telemetry_version
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            trace.request_id, trace.started_epoch, time.time(), trace.client,
            trace.client_ip, trace.model, trace.endpoint, int(status), duration_ms,
            ttfb_ms, int(trace.stream), trace.thinking, trace.reasoning_effort,
            trace.max_tokens, trace.input_tokens, trace.output_tokens,
            trace.cached_tokens, trace.reasoning_tokens, trace.tps, trace.prompt_tps,
            trace.response_bytes, _safe_label(error, 100) if error else None,
            TELEMETRY_VERSION,
        ))
        self.db.commit()
        self.write_count += 1
        if self.write_count % 50 == 0:
            self._prune()

    def _prune(self) -> None:
        if self.db:
            cutoff = time.time() - self.retention_days * 86400
            for table in ("requests", "engine_requests"):
                self.db.execute(f"DELETE FROM {table} WHERE completed_at < ?", (cutoff,))
            self.db.commit()

    @staticmethod
    def _parse_engine_line(message: str) -> Optional[dict[str, Any]]:
        head = SERVE_HEAD_RE.search(message)
        if not head:
            return None
        result: dict[str, Any] = {
            "mode": head.group("mode"), "output_tokens": int(head.group("tokens")),
            "generation_s": float(head.group("seconds")),
            "tps": float(head.group("tps")) if head.group("tps") != "n/a" else None,
        }
        for part in (item.strip() for item in message.split("|")):
            # Anchor MTP to avoid accidentally reading the PLD counter as MTP.
            match = re.search(r"(?:^|\s)(\d+) rounds, commit ([\d.]+)/round", part)
            if match:
                result.update(rounds=int(match[1]), commit_per_round=float(match[2]))
            match = re.search(
                r"prompt (\d+)(?: \((\d+) cached, ([\d.]+)%\))?, "
                r"prefill ([\d.]+)s(?: = ([\d.]+) t/s| \((\d+) new\))?", part,
            )
            if match:
                result.update(
                    prompt_tokens=int(match[1]), cached_tokens=int(match[2]) if match[2] else 0,
                    cache_percent=float(match[3]) if match[3] else 0.0,
                    prefill_s=float(match[4]), prefill_tps=float(match[5]) if match[5] else None,
                    new_tokens=int(match[6]) if match[6] else None,
                )
            for pattern, key in (
                (r"detok ([\d.]+)us/tok", "detok_us"),
                (r"(\d+) tok beside other streams", "beside_tokens"),
                (r"(\d+) tok with the head off", "head_off_tokens"),
            ):
                match = re.search(pattern, part)
                if match:
                    result[key] = float(match[1]) if key == "detok_us" else int(match[1])
            match = re.search(r"pld (\d+) rounds, ([\d.]+) acc/round", part)
            if match:
                result.update(pld_rounds=int(match[1]), pld_accept_per_round=float(match[2]))
            match = re.search(r"pool (\d+)/(\d+) (\d+)%", part)
            if match:
                result.update(pool_used=int(match[1]), pool_total=int(match[2]), pool_percent=float(match[3]))
            match = re.search(r"think (on|off)(?:, (.+))?", part)
            if match:
                result.update(thinking=match[1], closed_reason=match[2])
        if result.get("new_tokens") is None and result.get("prompt_tokens") is not None:
            result["new_tokens"] = max(0, result["prompt_tokens"] - result["cached_tokens"])
        if result.get("prefill_tps") is None and result.get("prefill_s", 0) > 0 and result.get("new_tokens") is not None:
            result["prefill_tps"] = result["new_tokens"] / result["prefill_s"]
        return result

    def _store_engine_line(self, item: dict[str, Any]) -> None:
        if not self.db:
            return
        parsed = self._parse_engine_line(str(item.get("MESSAGE") or ""))
        model = self.service_models.get(str(item.get("_SYSTEMD_USER_UNIT") or ""))
        cursor = str(item.get("__CURSOR") or "")
        if not parsed or not model or not cursor:
            return
        try:
            completed_at = int(item["__REALTIME_TIMESTAMP"]) / 1_000_000
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(completed_at) or completed_at < time.time() - RETENTION_DAYS * 86400:
            return
        columns = list(parsed)
        self.db.execute(
            f"INSERT OR IGNORE INTO engine_requests (cursor, completed_at, model, {','.join(columns)}) "
            f"VALUES ({','.join('?' for _ in range(3 + len(columns)))})",
            [cursor, completed_at, model] + [parsed[key] for key in columns],
        )
        self.db.commit()

    async def _journal_loop(self) -> None:
        units = [arg for unit in self.journal_units for arg in ("-u", unit)]
        while True:
            try:
                self.journal_process = await asyncio.create_subprocess_exec(
                    "/usr/bin/journalctl", "--user", "-f", "-n", "5000", "-o", "json",
                    *units,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                )
                assert self.journal_process.stdout is not None
                while line := await self.journal_process.stdout.readline():
                    try:
                        item = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if isinstance(item, dict):
                        self._store_engine_line(item)
                await self.journal_process.wait()
            except asyncio.CancelledError:
                raise
            except (OSError, ValueError):
                pass
            await asyncio.sleep(2)

    async def _get_json(self, path: str) -> dict[str, Any]:
        try:
            async with self.session.get(f"{self.backend_url}{path}", timeout=ClientTimeout(total=4)) as response:
                if response.status != 200:
                    return {}
                value = await response.json()
                return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    async def _refresh_cache_sizes(self) -> None:
        sizes: dict[str, int] = {}
        du = shutil.which("du")
        for model, path in self.cache_paths.items():
            if not path.exists():
                continue
            if du:
                process = None
                try:
                    process = await asyncio.create_subprocess_exec(
                        du, "-sb", str(path), stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=20)
                    if process.returncode == 0:
                        sizes[model] = int(stdout.split()[0])
                except (OSError, ValueError, IndexError, asyncio.TimeoutError):
                    continue
                finally:
                    if process and process.returncode is None:
                        process.kill()
                        await process.wait()
            else:
                try:
                    sizes[model] = await asyncio.to_thread(_dir_size, path)
                except OSError:
                    continue
        self.cache_sizes = sizes
        self.cache_sizes_updated = time.time()

    def _summary(self, start: float, end: float) -> dict[str, Any]:
        assert self.db is not None
        row = self.db.execute(MEASUREMENTS_CTE + f"""
            SELECT {TOKEN_AGGREGATES},
                SUM(CASE WHEN status < 400 THEN 1 ELSE 0 END) AS successes,
                SUM(CASE WHEN telemetry_version < {TELEMETRY_VERSION} THEN 1 ELSE 0 END) AS legacy_requests,
                AVG(duration_ms) AS avg_duration_ms, AVG(ttft_ms) AS avg_ttfb_ms
            FROM measured
        """, (start, end)).fetchone()
        result = dict(row)
        for key in ("errors", "successes", "legacy_requests"):
            result[key] = result[key] or 0
        denominator = result["cache_input_tokens"]
        result["cache_ratio"] = result["cached_tokens"] / denominator if denominator else None
        result["source"] = "api_usage"
        return result

    def _breakdown(self, start: float, end: float, column: str) -> list[dict[str, Any]]:
        assert self.db is not None
        if column not in {"client", "model"}:
            return []
        rows = self.db.execute(MEASUREMENTS_CTE + f"""
            SELECT {column} AS name, {TOKEN_AGGREGATES} FROM measured
            GROUP BY {column} ORDER BY requests DESC, name
        """, (start, end)).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _add_months(moment: datetime.datetime, months: int) -> datetime.datetime:
        year, month = divmod(moment.year * 12 + moment.month - 1 + months, 12)
        return moment.replace(year=year, month=month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)

    def _timeline(self, start: float, end: float, period: str) -> tuple[int, str, list[dict[str, Any]]]:
        assert self.db is not None
        if period == "24h":
            unit, fmt, step = "hour", "%Y-%m-%d %H", datetime.timedelta(hours=1)
        elif period in {"7d", "30d"}:
            unit, fmt, step = "day", "%Y-%m-%d", datetime.timedelta(days=1)
        elif period == "1y":
            unit, fmt, step = "month", "%Y-%m", None
        else:
            span = end - start
            if span <= 2 * 86400:
                unit, fmt, step = "hour", "%Y-%m-%d %H", datetime.timedelta(hours=1)
            elif span <= 14 * 86400:
                unit, fmt, step = "6hour", None, 6 * 3600
            elif span <= 90 * 86400:
                unit, fmt, step = "day", "%Y-%m-%d", datetime.timedelta(days=1)
            else:
                unit, fmt, step = "week", None, 7 * 86400
        zero = {
            "requests": 0, "errors": 0,
            "input_tokens": 0, "output_tokens": 0, "input_reported": 0, "output_reported": 0,
        }
        if fmt is None:
            bucket = 6 * 3600 if unit == "6hour" else 7 * 86400
            rows = self.db.execute(MEASUREMENTS_CTE + f"""
                SELECT CAST(completed_at / ? AS INTEGER) * ? AS bucket_start,
                    {TOKEN_AGGREGATES}
                FROM measured GROUP BY CAST(completed_at / ? AS INTEGER) ORDER BY bucket_start
            """, (start, end, bucket, bucket, bucket)).fetchall()
            by_time = {int(row["bucket_start"]): dict(row) for row in rows}
            timeline = []
            for timestamp in range(int(start // bucket) * bucket, int(end // bucket) * bucket + 1, bucket):
                # Empty time buckets are real zero activity. Requests without usage
                # stay null and are visually distinguishable from an empty bucket.
                item = by_time.get(timestamp, dict(zero))
                item["bucket_start"] = timestamp
                item["bucket_end"] = timestamp + bucket
                item["partial"] = item["input_reported"] < item["requests"] or item["output_reported"] < item["requests"]
                timeline.append(item)
            return bucket, unit, timeline
        rows = self.db.execute(MEASUREMENTS_CTE + f"""
            SELECT strftime(?, completed_at, 'unixepoch', 'localtime') AS bucket_key,
                {TOKEN_AGGREGATES}
            FROM measured GROUP BY bucket_key ORDER BY bucket_key
        """, (start, end, fmt)).fetchall()
        by_key = {row["bucket_key"]: dict(row) for row in rows}
        # Naive local datetimes match SQLite's 'localtime' modifier exactly
        # (both use the OS timezone database, including historical offsets).
        current = datetime.datetime.fromtimestamp(start)
        if unit == "hour":
            current = current.replace(minute=0, second=0, microsecond=0)
        elif unit == "day":
            current = current.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            current = current.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_local = datetime.datetime.fromtimestamp(end)
        timeline = []
        bucket_seconds = 0
        while current < end_local:
            nxt = self._add_months(current, 1) if unit == "month" else current + step
            item = by_key.get(current.strftime(fmt), dict(zero))
            item["bucket_start"] = current.timestamp()
            item["bucket_end"] = nxt.timestamp()
            item["partial"] = item["input_reported"] < item["requests"] or item["output_reported"] < item["requests"]
            if not bucket_seconds:
                bucket_seconds = int(item["bucket_end"] - item["bucket_start"])
            timeline.append(item)
            current = nxt
        return bucket_seconds or 3600, unit, timeline

    def _engine_summary(self, start: float, end: float) -> dict[str, Any]:
        assert self.db is not None
        # Independent engine population; never join by approximate timestamps.
        row = self.db.execute("""
            SELECT COUNT(*) AS requests,
                SUM(CASE WHEN generation_s > 0 THEN output_tokens END) * 1.0
                    / NULLIF(SUM(CASE WHEN generation_s > 0 THEN generation_s END), 0) AS weighted_tps,
                SUM(CASE WHEN prefill_s > 0 AND new_tokens IS NOT NULL THEN new_tokens END) * 1.0
                    / NULLIF(SUM(CASE WHEN prefill_s > 0 AND new_tokens IS NOT NULL THEN prefill_s END), 0) AS weighted_prefill_tps,
                SUM(CASE WHEN commit_per_round IS NOT NULL THEN commit_per_round * rounds END)
                    / NULLIF(SUM(CASE WHEN commit_per_round IS NOT NULL THEN rounds END), 0) AS avg_commit_per_round,
                SUM(CASE WHEN pld_accept_per_round IS NOT NULL THEN pld_accept_per_round * pld_rounds END)
                    / NULLIF(SUM(CASE WHEN pld_accept_per_round IS NOT NULL THEN pld_rounds END), 0) AS avg_pld_accept_per_round,
                MAX(pool_percent) AS max_pool_percent
            FROM engine_requests WHERE completed_at >= ? AND completed_at <= ?
        """, (start, end)).fetchone()
        return dict(row)

    def _history(self, start: float, end: float, page: int = 1, page_size: int = 10) -> dict[str, Any]:
        assert self.db is not None
        page_size = max(1, min(page_size, 100))
        total = self.db.execute(
            f"SELECT COUNT(*) FROM requests WHERE completed_at >= ? AND completed_at <= ? AND {REQUEST_SCOPE}",
            (start, end),
        ).fetchone()[0]
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(max(1, page), pages)
        rows = self.db.execute(f"""
            SELECT request_id, started_at, completed_at, client, model, endpoint,
                status, duration_ms, ttft_ms AS ttfb_ms, stream, thinking,
                reasoning_effort, input_tokens, output_tokens, cached_tokens,
                reasoning_tokens, tps, prompt_tps, error, telemetry_version
            FROM requests WHERE completed_at >= ? AND completed_at <= ? AND {REQUEST_SCOPE}
            ORDER BY completed_at DESC, id DESC LIMIT ? OFFSET ?
        """, (start, end, page_size, (page - 1) * page_size)).fetchall()
        return {"items": [dict(row) for row in rows], "page": page, "page_size": page_size, "pages": pages, "total": total}

    def _resolve_range(self, request: web.Request) -> tuple[str, float, float]:
        now = time.time()
        earliest = now - self.retention_days * 86400
        period = request.query.get("period", "24h")
        if period == "custom":
            try:
                start, end = float(request.query["from"]), float(request.query["to"])
                if not math.isfinite(start) or not math.isfinite(end):
                    raise ValueError
                start, end = max(earliest, start), min(now, end)
                if start >= end:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                raise web.HTTPBadRequest(text="Ungültiger Zeitraum: from und to als Unix-Zeit angeben.")
            return period, start, end
        if period not in PERIOD_SECONDS:
            raise web.HTTPBadRequest(text="Unbekannter Zeitraum")
        midnight = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if period == "24h":
            # The whole current calendar day; data fills up to "now".
            start, end = midnight.timestamp(), (midnight + datetime.timedelta(days=1)).timestamp()
        elif period == "7d":
            # The current week, Monday through Sunday.
            week_start = midnight - datetime.timedelta(days=midnight.weekday())
            start, end = week_start.timestamp(), (week_start + datetime.timedelta(days=7)).timestamp()
        elif period == "30d":
            # The whole current calendar month.
            month_start = midnight.replace(day=1)
            start, end = month_start.timestamp(), self._add_months(month_start, 1).timestamp()
        else:  # 1y -> the whole current calendar year, January through December
            year_start = midnight.replace(month=1, day=1)
            start, end = year_start.timestamp(), year_start.replace(year=year_start.year + 1).timestamp()
        return period, max(earliest, start), end

    def analytics(self, request: web.Request) -> dict[str, Any]:
        period, start, end = self._resolve_range(request)
        if not self.db:
            raise web.HTTPServiceUnavailable(text="Telemetrie noch nicht verfügbar")
        try:
            page = max(1, int(request.query.get("page", "1")))
        except ValueError:
            page = 1
        bucket, bucket_unit, timeline = self._timeline(start, end, period)
        return {
            "period": period, "from": start, "to": end, "generated_at": time.time(),
            "bucket_seconds": bucket, "bucket_unit": bucket_unit, "summary": self._summary(start, end), "timeline": timeline,
            "engine": self._engine_summary(start, end), "history": self._history(start, end, page),
            "by_client": self._breakdown(start, end, "client"), "by_model": self._breakdown(start, end, "model"),
        }

    def _disk_usage_path(self) -> str:
        paths = [str(path) for path in self.cache_paths.values() if path]
        if not paths:
            return str(self.state_dir)
        try:
            return os.path.commonpath(paths)
        except ValueError:
            return paths[0]

    def _gpu_busy_percent(self) -> Optional[int]:
        if self.gpu_card == "auto":
            for card in _gpu_cards():
                value = _read_int(str(card / "device/gpu_busy_percent"))
                if value is not None:
                    return value
            return None
        if self.gpu_card.startswith("/"):
            return _read_int(self.gpu_card)
        return _read_int(f"/sys/class/drm/{self.gpu_card}/device/gpu_busy_percent")

    async def snapshot(self) -> dict[str, Any]:
        if time.time() - self.cache_sizes_updated >= 60 and (
            not self.cache_refresh_task or self.cache_refresh_task.done()
        ):
            self.cache_refresh_task = asyncio.create_task(self._refresh_cache_sizes())
        state, cache = await asyncio.gather(self.manager.status(), self._get_json("/cache"))
        health = state.get("backend_health") or {}
        try:
            disk = shutil.disk_usage(self._disk_usage_path())
            disk_info = {"disk_total_bytes": disk.total, "disk_used_bytes": disk.used}
        except OSError:
            disk_info = {}
        return {
            "generated_at": time.time(),
            "api": {"models": list(self.models), "active_model": state.get("active_model"),
                    "switch_target": state.get("switch_target"), "status": state.get("status")},
            "backend": {
                key: health.get(key) for key in (
                    "status", "version", "context", "slots", "in_flight", "queued",
                    "max_tokens_default", "max_tokens_cap", "reasoning_effort_default",
                )
            },
            "cache": {"pool": cache.get("pool") or {}, "model_bytes": self.cache_sizes},
            "system": {
                "memory": _memory_info(),
                "gpu_busy_percent": self._gpu_busy_percent(),
                **disk_info,
            },
            "active_requests": [
                {"request_id": trace.request_id, "client": trace.client, "model": trace.model,
                 "elapsed_ms": (time.monotonic() - trace.started_mono) * 1000}
                for trace in self.active.values()
            ],
        }

    async def page(self, _: web.Request) -> web.Response:
        try:
            from importlib.resources import files

            html = files("halobridge_data").joinpath("dashboard.html").read_text(encoding="utf-8")
        except Exception:
            html = Path(__file__).with_name("halogen_dashboard.html").read_text(encoding="utf-8")
        return web.Response(
            text=html,
            content_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
                "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
            },
        )

    async def api_snapshot(self, _: web.Request) -> web.Response:
        return web.json_response(await self.snapshot(), headers={"Cache-Control": "no-store"})

    async def api_analytics(self, request: web.Request) -> web.Response:
        return web.json_response(self.analytics(request), headers={"Cache-Control": "no-store"})

    async def api_requests(self, request: web.Request) -> web.Response:
        if not self.db:
            raise web.HTTPServiceUnavailable()
        try:
            limit = int(request.query.get("limit", "100"))
        except ValueError:
            limit = 100
        now = time.time()
        history = self._history(now - self.retention_days * 86400, now, page_size=limit)
        return web.json_response({"requests": history["items"]}, headers={"Cache-Control": "no-store"})

    async def api_locale(self, request: web.Request) -> web.Response:
        lang = request.match_info.get("lang", "")
        if lang not in {"de", "en"}:
            raise web.HTTPNotFound()
        try:
            from importlib.resources import files

            content = files("halobridge_data").joinpath("locales", f"{lang}.json").read_text(encoding="utf-8")
        except Exception:
            content = (
                Path(__file__).with_name("halobridge_data") / "locales" / f"{lang}.json"
            ).read_text(encoding="utf-8")
        return web.json_response(json.loads(content), headers={"Cache-Control": "no-store"})

    async def api_reset(self, request: web.Request) -> web.Response:
        if request.headers.get("X-Halogen-Action") != "reset":
            raise web.HTTPBadRequest(text="Bestätigungsheader X-Halogen-Action: reset fehlt.")
        if not self.db:
            raise web.HTTPServiceUnavailable(text="Telemetrie noch nicht verfügbar")
        deleted = {}
        for table in ("requests", "engine_requests"):
            deleted[table] = self.db.execute(f"DELETE FROM {table}").rowcount
        self.db.commit()
        self.db.execute("VACUUM")
        return web.json_response(
            {"deleted": deleted, "reset_at": time.time()},
            headers={"Cache-Control": "no-store"},
        )

    def register_routes(self, app: web.Application) -> None:
        app.router.add_get("/dashboard", self.page)
        app.router.add_get("/dashboard/", self.page)
        app.router.add_get("/dashboard/api/snapshot", self.api_snapshot)
        app.router.add_get("/dashboard/api/requests", self.api_requests)
        app.router.add_get("/dashboard/api/analytics", self.api_analytics)
        app.router.add_get("/dashboard/api/locale/{lang}", self.api_locale)
        app.router.add_post("/dashboard/api/reset", self.api_reset)
