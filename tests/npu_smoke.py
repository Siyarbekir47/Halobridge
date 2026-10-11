"""Opt-in Linux NPU API smoke test. Installs nothing; sends small requests."""
import argparse
import base64
import concurrent.futures
import json
import math
import os
import uuid
from pathlib import Path
import time
import urllib.error
import urllib.request


class Api:
    def __init__(self, url, timeout):
        self.url, self.timeout = url.rstrip("/"), timeout
        self.headers = {"Content-Type": "application/json"}
        if os.environ.get("HALOBRIDGE_TOKEN"):
            self.headers["Authorization"] = "Bearer " + os.environ["HALOBRIDGE_TOKEN"]

    def call(self, route, payload=None, *, method=None, headers=None):
        request = urllib.request.Request(self.url + route, headers={**self.headers, **(headers or {})}, method=method,
                    data=json.dumps(payload).encode() if payload is not None else None)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if payload and payload.get("stream"):
                    return response.read().decode()
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{route}: HTTP {error.code}: {error.read(2048).decode(errors='replace')}") from error

    def gpu(self, model):
        return self.call("/v1/chat/completions", {"model": model, "max_tokens": 16,
                         "messages": [{"role": "user", "content": "Say hello in one sentence."}]})


def check_tasks(api):
    response = api.call("/v1/chat/completions", {"model": api.decider,
        "messages": [{"role": "user", "content": "I need help resetting my password."}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "topic", "description": "Which support team should handle this request?",
            "schema": {"enum": ["accounts", "billing", "shipping"]}}},
        "logprobs": True, "top_logprobs": 3})
    choice = response["choices"][0]
    assert json.loads(choice["message"]["content"]) in {"accounts", "billing", "shipping"}
    assert len(choice["logprobs"]["content"][0]["top_logprobs"]) == 3
    print("PASS decisions and option probabilities")

    response = api.call("/v1/systemone", {"model": api.decider, "state": "I was charged twice.",
        "questions": {"team": {"type": "choice", "instructions": "Which support team handles this?",
            "criteria": {"billing": "Payment issues", "technical": "Software bugs"}},
            "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Routine", "Urgent"]},
            "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"}}})
    answers = response["answers"]
    assert answers["team"]["choice"] in {"billing", "technical"}
    assert abs(sum(answers["team"]["probabilities"].values()) - 1) < .01
    assert 0 <= answers["team"]["confidence"] <= 1
    assert 0 <= answers["urgency"]["score"] <= 1 and 0 <= answers["refund"]["noul"] <= 1
    print("PASS System One choice, score and yes/no probability")

    response = api.call("/v1/embeddings", {"model": api.embedding, "dimensions": 256,
        "input": ["Instruct: Retrieve documentation\nQuery:How do I enable the NPU?",
                  "Install the NPU driver, firmware and XRT."]})
    assert len(response["data"]) == 2
    for row in response["data"]:
        vector = row["embedding"]
        assert len(vector) == 256 and all(math.isfinite(v) for v in vector)
        assert abs(sum(v*v for v in vector) - 1) < .01
    print("PASS embeddings and dimensions")

    response = api.call("/v1/rerank", {"model": api.reranker,
        "query": "How do I enable the NPU?", "documents": [
            "Install the NPU driver and XRT.", "The dashboard supports a dark theme."],
        "top_n": 2, "return_documents": True})
    results = response["results"]
    assert {row["index"] for row in results} == {0, 1}
    scores = [row["relevance_score"] for row in results]
    assert all(math.isfinite(score) and 0 <= score <= 1 for score in scores)
    assert scores == sorted(scores, reverse=True)
    print("PASS reranking")

    response = api.call("/v1/moderations", {"model": "qwen3guard-gen-0.6b",
                                          "input": ["How do I bake bread?"], "strict": True})
    result = response["results"][0]
    assert isinstance(result["flagged"], bool)
    assert result["label"] in {"Safe", "Unsafe", "Controversial"}
    assert set(result["label_scores"]) == {"Safe", "Unsafe", "Controversial"}
    print("PASS moderation labels and scores")

    payload = {"model": "qwen3.5-2b", "max_tokens": 32,
               "messages": [{"role": "user", "content": "Explain a local inference router in one sentence."}]}
    response = api.call("/v1/chat/completions", payload)
    assert response["choices"][0]["message"]["content"]
    assert response["usage"]["prompt_tokens"] > 0 and response["usage"]["completion_tokens"] > 0
    stream = api.call("/v1/chat/completions", {**payload, "stream": True, "stream_options": {"include_usage": True}})
    assert "data: [DONE]" in stream
    chunks = [json.loads(line[6:]) for line in stream.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert any(chunk.get("usage", {}).get("completion_tokens", 0) > 0 for chunk in chunks)
    print("PASS generation, streaming and usage")


def check_images(api, size="256x256"):
    response = api.call("/v1/images/generations", {"model": "flux2-klein-4b", "size": size, "steps": 4,
        "n": 1, "seed": 7, "prompt": "A green cloud icon on white", "response_format": "b64_json"})
    assert len(response["data"]) == 1
    png = base64.b64decode(response["data"][0]["b64_json"], validate=True)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and png[12:16] == b"IHDR"
    assert (int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")) == tuple(map(int, size.split("x")))
    print("PASS Flux NPU image generation and PNG dimensions")


def check_audio(api, path, model):
    assert path.stat().st_size <= 25 * 1024**2, "WAV file exceeds 25 MiB"
    boundary = "halobridge-" + uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample.wav"\r\n'
            'Content-Type: audio/wav\r\n\r\n').encode() + path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    headers = {**api.headers, "Content-Type": "multipart/form-data; boundary=" + boundary}
    request = urllib.request.Request(api.url + "/v1/audio/transcriptions", data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=api.timeout) as response:
        result = json.load(response)
    assert isinstance(result.get("text"), str)
    print("PASS WAV transcription (transcript omitted)")


def check_sessions(api, model):
    names = ["halobridge-smoke-" + uuid.uuid4().hex for _ in range(2)]
    prompt = {"role": "user", "content": "Say hello in one sentence."}
    payload = {"model": model, "messages": [prompt], "enable_thinking": False, "max_tokens": 32}
    try:
        reply = api.call("/v1/chat/completions", payload, headers={"X-Halogen-Keep": names[0]})
        assert names[0] in {h["name"] for h in api.call("/halogen/handles")["handles"]}, "Session was not kept"
        payload["messages"] += [reply["choices"][0]["message"], {"role": "user", "content": "Say it again."}]
        reply = api.call("/v1/chat/completions", payload,
                        headers={"X-Halogen-From": names[0], "X-Halogen-Keep": names[1]})
        assert reply["choices"] and set(names) <= {h["name"] for h in api.call("/halogen/handles")["handles"]}
        print("PASS session keep, branch and listing")
    finally:
        for name in names:
            try:
                api.call("/halogen/handles/" + name, method="DELETE")
            except RuntimeError as error:
                if "HTTP 404" not in str(error):
                    raise
    assert not set(names) & {h["name"] for h in api.call("/halogen/handles")["handles"]}
    print("PASS session release")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8731")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--gpu-switches", nargs="*", default=[])
    parser.add_argument("--images", action="store_true", help="Also test Flux; enable it first (about 8 GB memory)")
    parser.add_argument("--image-size", default="256x256", help="Flux WxH; larger images may not fit with ASR")
    parser.add_argument("--audio", type=Path, help="Also transcribe this WAV file (max 25 MiB / 20 min)")
    parser.add_argument("--asr-model", choices=["qwen3-asr-0.6b", "qwen3-asr-1.7b"], default="qwen3-asr-0.6b")
    parser.add_argument("--sessions", action="store_true", help="Also test GPU keep/fork/release; enable HALOGEN_FORK first")
    parser.add_argument("--decider", choices=["decider-0.8b", "decider-4b"], default="decider-0.8b")
    parser.add_argument("--embedding", choices=["qwen3-embedding-0.6b", "qwen3-embedding-4b"], default="qwen3-embedding-0.6b")
    parser.add_argument("--reranker", choices=["qwen3-reranker-0.6b", "qwen3-reranker-4b"], default="qwen3-reranker-0.6b")
    args = parser.parse_args()
    api = Api(args.url, args.timeout)
    api.decider, api.embedding, api.reranker = args.decider, args.embedding, args.reranker
    initial = api.call("/health").get("model")
    assert initial, "Backend health did not report the active model"
    expected = {api.decider, api.embedding, api.reranker, "qwen3guard-gen-0.6b", "qwen3.5-2b"}
    if args.images:
        expected.add("flux2-klein-4b")
    if args.audio:
        expected.add(args.asr_model)
    names = {row["id"] for row in api.call("/v1/models")["data"]}
    assert expected <= names, f"Enable the required NPU models first; missing: {expected - names}"
    started = time.monotonic()
    try:
        check_tasks(api)
        if args.images:
            check_images(api, args.image_size)
        if args.audio:
            check_audio(api, args.audio, args.asr_model)
        if args.sessions:
            check_sessions(api, initial)
        assert api.call("/health")["model"] == initial, "NPU requests changed the GPU model"
        for model in args.gpu_switches:
            api.gpu(model)
            assert api.call("/health")["model"] == model
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                gpu = pool.submit(api.gpu, model)
                embedding = pool.submit(api.call, "/v1/embeddings",
                     {"model": api.embedding, "input": "Concurrent GPU and NPU smoke test."})
                assert embedding.result()["data"] and gpu.result()["choices"]
            assert api.call("/health")["model"] == model
            print(f"PASS GPU/NPU concurrency after switch to {model}")
    finally:
        if args.gpu_switches and api.call("/health").get("model") != initial:
            api.gpu(initial)
            assert api.call("/health")["model"] == initial, "Could not restore the initial GPU model"
    print(f"PASS NPU smoke test ({time.monotonic() - started:.1f}s); active GPU model: {initial}")


if __name__ == "__main__":
    main()
