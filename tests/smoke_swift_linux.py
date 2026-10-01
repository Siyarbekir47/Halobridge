"""Opt-in GPU smoke check against an already installed Linux Halobridge.

Run on the Strix Halo host: python3 tests/smoke_swift_linux.py
Requires all four installed profiles and aiohttp (or run with pipx runpip's
Halobridge interpreter). HALOBRIDGE_TOKEN supplies optional authentication.
Sends real requests and switches models; it never installs/downloads weights.
"""

import argparse
import asyncio
import base64
import json
import os
import struct
import sys
import zlib

from aiohttp import ClientSession, ClientTimeout

MODELS = ("qwen3.8-flash", "qwen3.8-flash-uncensored", "halogen-swift15", "halogen-swift15-abliterated")


def test_image():
    """Small deterministic RGB PNG: a solid red square, no external asset."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    pixels = (b"\0" + b"\xff\0\0" * 64) * 64
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


async def main(args):
    if sys.platform != "linux":
        raise RuntimeError("Run this smoke test on the Linux GPU host")
    headers = {"Authorization": "Bearer " + os.environ["HALOBRIDGE_TOKEN"]} if os.environ.get("HALOBRIDGE_TOKEN") else {}
    async with ClientSession(headers=headers, timeout=ClientTimeout(total=900)) as session:
        async def request(path, payload=None):
            method = session.get if payload is None else session.post
            async with method(args.url.rstrip("/") + path, **({"json": payload} if payload is not None else {})) as response:
                value = await response.json()
                if response.status >= 400:
                    raise RuntimeError(f"{path}: HTTP {response.status}: {value}")
                return value
        installed = {entry["id"] for entry in (await request("/v1/models"))["data"]}
        missing = set(args.models) - installed
        if missing:
            raise RuntimeError("Install these profiles in Quick Setup first: " + ", ".join(sorted(missing)))
        original = (await request("/router/status")).get("active_model")
        try:
            for model in args.models:
                for kind in ("text", "image"):
                    content = "Reply briefly: What is 2 + 2?"
                    if kind == "image":
                        content = [{"type": "text", "text": "Name the main color in this image. Reply briefly."},
                                   {"type": "image_url", "image_url": {"url": test_image()}}]
                    reply = await request("/v1/chat/completions", {
                        "model": model, "messages": [{"role": "user", "content": content}],
                        "max_tokens": 256, "reasoning_effort": "minimal", "stream": False})
                    if not reply.get("choices"):
                        raise RuntimeError(f"{model} {kind}: missing choices")
                    message = reply["choices"][0].get("message") or {}
                    if not message.get("content"):
                        raise RuntimeError(f"{model} {kind}: empty answer")
                    health = await request("/health")
                    if health.get("status") != "ok" or health.get("model") != model:
                        raise RuntimeError(f"{model}: unexpected health after switch")
                    if model.startswith("halogen-swift"):
                        version = health.get("version") or {}
                        if not version.get("engine") or tuple(map(int, version["engine"].lstrip("v").split("."))) < (0, 15, 1):
                            raise RuntimeError(f"{model}: incompatible engine version")
                    print(json.dumps({"model": model, "kind": kind, "answer": message["content"],
                                      "usage": reply.get("usage"), "version": health.get("version")}))
        finally:
            if original and original in installed:
                await request("/v1/chat/completions", {"model": original, "messages": [{"role": "user", "content": "Reply OK."}],
                                                      "max_tokens": 32, "reasoning_effort": "minimal"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8731")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    asyncio.run(main(parser.parse_args()))
