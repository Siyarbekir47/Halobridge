"""Localize dashboard messages at the response boundary.

Jobs and exceptions retain English source text. Each browser gets its own
translation, including for jobs started before a language change or restart.
Only message fields are translated: model IDs, paths, commands, and raw
third-party output remain untouched. The catalog also recognizes messages
from older German job journals.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import json
import re

from aiohttp import web

MESSAGE_FIELDS = frozenset({
    "message", "error", "errors", "warnings", "support_error", "check_error",
    "blocked_reason", "reason", "lines",
})
PLACEHOLDER = re.compile(r"\{p\d+\}")


def request_language(request: web.Request) -> str:
    explicit = request.query.get("lang")
    if explicit in {"en", "de"}:
        return explicit
    # Dashboard fetches send an explicit language even if the browser differs.
    header = request.headers.get("Accept-Language", "")
    preferences = []
    for index, item in enumerate(header.split(",")):
        language, _, weight = item.strip().partition(";")
        try:
            quality = float(weight.strip().removeprefix("q=")) if weight else 1.0
        except ValueError:
            continue
        base = language.lower().split("-", 1)[0]
        if base in {"en", "de"} and 0 < quality <= 1:
            preferences.append((quality, -index, base))
    if preferences:
        return max(preferences)[2]
    return "en"


@lru_cache(maxsize=1)
def _catalog():
    data = json.loads(files("halobridge_data").joinpath("locales/server.de.json").read_text(encoding="utf-8"))
    exact, patterns = {}, []
    for english, german in data.items():
        for source in {english, german}:
            if not PLACEHOLDER.search(source):
                exact[source] = {"en": english, "de": german}
                continue
            parts, end = [], 0
            for match in PLACEHOLDER.finditer(source):
                parts.append(re.escape(source[end:match.start()]))
                parts.append(f"(?P<{match.group()[1:-1]}>.+?)")
                end = match.end()
            parts.append(re.escape(source[end:]))
            patterns.append((len(PLACEHOLDER.sub('', source)), re.compile(''.join(parts), re.S), english, german))
    # More specific templates win over wrappers such as "env {p0}: {p1}".
    patterns.sort(key=lambda item: item[0], reverse=True)
    return exact, patterns


def translate_message(message: str, lang: str, depth: int = 0) -> str:
    if depth > 6:
        return message
    lang = lang if lang in {"en", "de"} else "en"
    exact, patterns = _catalog()
    if message in exact:
        return exact[message][lang]
    # Deploy validation joins multiple independent errors with semicolons.
    if "; " in message:
        pieces = message.split("; ")
        translated = [translate_message(piece, lang, depth + 1) for piece in pieces]
        if translated != pieces:
            return "; ".join(translated)
    for _, pattern, english, german in patterns:
        match = pattern.fullmatch(message)
        if match is None:
            continue
        values = match.groupdict()
        # These slots contain another application message, not a user identifier.
        nested = {"p1"} if english == "env {p0}: {p1}" else {"p0"} if (
            english.startswith(("ERROR: ", "... ", "{p0} Restoring", "{p0} Previous", "Invalid profile: "))
        ) else set()
        for key in nested & values.keys():
            values[key] = translate_message(values[key], lang, depth + 1)
        template = german if lang == "de" else english
        return PLACEHOLDER.sub(lambda slot: values[slot.group()[1:-1]], template)
    return message


def localize_payload(value, lang: str, message_field: bool = False):
    if isinstance(value, dict):
        return {key: item if key in {"env", "volumes"} else localize_payload(item, lang, key in MESSAGE_FIELDS)
                for key, item in value.items()}
    if isinstance(value, list):
        return [localize_payload(item, lang, message_field) for item in value]
    if isinstance(value, str) and message_field:
        return translate_message(value, lang)
    return value


@web.middleware
async def dashboard_locale_middleware(request: web.Request, handler):
    if not request.path.startswith("/dashboard/api/") or request.path.startswith("/dashboard/api/locale/"):
        return await handler(request)
    lang = request_language(request)
    try:
        response = await handler(request)
    except web.HTTPException as error:
        error.text = translate_message(error.text, lang)
        error.headers["Content-Language"] = lang
        error.headers["Vary"] = "Accept-Language"
        raise
    if isinstance(response, web.Response) and response.text:
        if response.content_type == "application/json":
            response.text = json.dumps(localize_payload(json.loads(response.text), lang), ensure_ascii=False)
        elif response.content_type == "text/plain":
            response.text = translate_message(response.text, lang)
        response.headers["Content-Language"] = lang
        response.headers["Vary"] = "Accept-Language"
    return response
