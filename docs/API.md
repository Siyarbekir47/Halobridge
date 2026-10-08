# Halogen endpoints through Halobridge

Halobridge 0.1.26 supports Halogen 0.17.2. Install the Halobridge update, then
update the backend profiles in **System → Engine updates**. For NPU endpoints,
complete the host setup and select the required models in **Models → NPU models
alongside your GPU model**. NPU requests keep the active GPU model loaded.

Use `http://127.0.0.1:8731` in these examples, or your configured router address.
Get your actual GPU model ID from `GET /v1/models`: Official is normally
`qwen3.8-flash`, Swift is `halogen-swift15` or `halogen-swift15-abliterated`.
A request for another installed GPU profile switches the GPU backend.

| Endpoint | Model / requirement | What it is useful for |
| --- | --- | --- |
| `POST /v1/chat/completions` | Installed GPU model | Chat, reasoning, tools, structured JSON and image understanding with the vision tower. |
| `POST /v1/responses` | Installed GPU model | OpenAI Responses clients and agents; tools, reasoning and streaming. |
| `POST /v1/completions` | Installed GPU model | Older clients that send a text prompt rather than chat messages. |
| `POST /v1/messages` | Installed GPU model | Anthropic SDKs and clients that use Messages, including tools, thinking and streaming. |
| `POST /v1/messages/count_tokens` | Installed GPU model | Check a Messages prompt's token count before generation. |
| `POST /v1/chat/completions` | NPU `decider-0.8b` | Choose one of 2–10 schema options, with probabilities; classification and routing. |
| `POST /v1/systemone` **new in 0.16.3** | Loaded NPU decision model | Ask several typed questions about the same state: category, score or yes/no probability. |
| `POST /v1/embeddings` | NPU `qwen3-embedding-0.6b` | Turn documents and queries into vectors for semantic search or a RAG database. |
| `POST /v1/rerank` | NPU `qwen3-reranker-0.6b` | Sort retrieved documents by relevance before sending the best ones to the GPU model. |
| `POST /v1/moderations` | NPU `qwen3guard-gen-0.6b` | Check prompts or replies for the Safe, Unsafe and Controversial labels. |
| `POST /v1/chat/completions` | NPU `qwen3.5-2b` | Short text jobs and summaries beside the GPU model; no thinking, tools or vision. |
| `POST /v1/images/generations` **new in 0.17.0** | NPU `flux2-klein-4b` | Generate icons, illustrations, placeholders or simple diagrams from a text prompt. |
| `GET /v1/models` | No generation | Discover GPU profile IDs and currently loaded NPU IDs, tasks and context limits. |
| `GET /health` | Running backend | Check engine health and priority admission counters. |
| `GET /metrics` | Running backend | Scrape Prometheus metrics, including priority-slot gauges. |
| `GET /cache` | Running backend | Inspect prompt-cache hits, dropped/replaced entries, evictions and pool usage. |

The first five NPU models run on 0.16.2. System One needs 0.16.3+, Flux needs
0.17.0+, and 0.17.2 is the recommended engine for all of them. Enable a model
on every profile where it should remain available after a GPU model switch.

## System One: classify a ticket and estimate urgency

Enable `decider-0.8b`, then send:

```bash
curl --fail-with-body -sS localhost:8731/v1/systemone \
  -H 'Content-Type: application/json' -d '{
  "model":"decider-0.8b",
  "state":"Customer: I was charged twice and nobody answers my emails.",
  "questions":{
    "team":{"type":"choice","instructions":"Which team should handle this?",
      "criteria":{"billing":"Payment issues","technical":"Bugs","other":null}},
    "urgency":{"type":"score","instructions":"How urgent is this ticket?",
      "criteria":["Routine","Needs attention","Urgent"]},
    "refund":{"type":"noul","instructions":"Does the customer request a refund?"}
  }}'
```

`state` can be text, an object or an array. Every question takes one NPU pass.
`choice` returns the selected category, option probabilities and confidence;
`score` returns an expected level, its legend, probabilities and confidence;
`noul` returns the probability of yes. Choices and scores accept 2–10 options.
This is useful for ticket routing, document classification, agent decisions or
sorting jobs by urgency without a long generated reasoning response.

A System One client may send any model name when exactly one decision model
is loaded; Halobridge preserves that name and routes the request to the decider.
If multiple decision fine-tunes are loaded, select the exact ID. Invalid question
shapes receive an upstream 422 response.

## Images: generate and save a PNG

Enable `flux2-klein-4b`. Its verified files need about **7.5 GiB on disk** and
the loaded model needs about **8 GB of host memory**, alongside the GPU model.
It requires the same NPU driver, firmware, XRT and held fabric clock as the
other NPU tasks.

```bash
curl --fail-with-body -sS localhost:8731/v1/images/generations \
  -H 'Content-Type: application/json' -d '{
  "model":"flux2-klein-4b",
  "prompt":"A clean flat illustration of a green cloud with a lightning bolt on white",
  "size":"512x512","n":1,"seed":7,"response_format":"b64_json"
  }' > /tmp/halobridge-image.json

python3 -c 'import base64,json,pathlib; r=json.load(open("/tmp/halobridge-image.json")); pathlib.Path("/tmp/halobridge-image.png").write_bytes(base64.b64decode(r["data"][0]["b64_json"]))'
```

Supported sizes are `256x256` and `512x512`; `auto` means `512x512`. `n` is
1–4, drawn sequentially. Output is PNG. `response_format=url` returns an inline
`data:image/png;base64,...` URL, not a hosted download. A fixed seed makes a
request repeatable; image `i` uses `seed + i`. The prompt is truncated to 512
tokens. `quality=low` is currently refused; the other quality values produce
the same image. Only one image request runs on the NPU at a time.

Images are generated locally. These are small illustrations, rather than a
high-resolution image-editing endpoint. Image bodies pass through Halobridge;
request history stores duration, status, model and response size, not the image.

## Priority slots: keep interactive chat responsive

Set `HALOGEN_ADMISSION_RESERVE=1` in the backend profile's **Engine scheduling &
compatibility** section and apply the profile. The engine keeps at least one
background slot; for example, with two slots one is reserved for priority work.
Send the header on a request a person is waiting for:

```bash
curl --fail-with-body -sS localhost:8731/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Halogen-Priority: 1' -d '{
  "model":"qwen3.8-flash","max_tokens":128,
  "messages":[{"role":"user","content":"Explain the NPU in two sentences."}]}'
```

Background agent jobs omit the header and wait when only reserved slots are
free. Nothing is preempted and output is unchanged. This is a GPU admission
policy; it does not give image generation its own parallel NPU slot. The System
view and `/health.admission` show `reserve`, `priority_in_flight` and
`waiting_for_reserve`; `/metrics` exports the same gauges. The default reserve
is 0. Halobridge passes the priority header through unchanged.

## Anthropic Messages and token counting

These upstream endpoints existed before 0.17.0; Halobridge now tracks their
streamed and non-streamed usage, including cached input, and supplies Anthropic
model-list fields. Use your installed GPU model ID, not a hosted Claude ID:

```bash
curl --fail-with-body -sS localhost:8731/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' -d '{
  "model":"qwen3.8-flash","max_tokens":128,
  "messages":[{"role":"user","content":"Explain semantic search in two sentences."}]}'

curl --fail-with-body -sS localhost:8731/v1/messages/count_tokens \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' -d '{
  "model":"qwen3.8-flash",
  "messages":[{"role":"user","content":"Explain semantic search in two sentences."}]}'
```

For the Anthropic SDK, set the base URL to `http://127.0.0.1:8731`, provide any
API-key string the SDK requires, and select a Halobridge GPU model ID. Thinking,
tools, streaming and images use the upstream Messages shapes; images require
the vision tower. Token counting does not generate an answer.

## Changes that work after an engine update

0.17.2 chooses MTP draft depth per conversation when `HALOGEN_MTP_DEPTH` is
unset; leave the editor field empty to use it. Two concurrent conversations
speculate together automatically. Faster decode and prompt processing, improved
disk-prefix cache reuse, streamed tool-call cleanup and schema/tool fixes work
without a new endpoint. Late system/developer turns in Chat Completions are
treated as user text at their position; images in those late turns are refused.
GPU checkpoints remain unchanged from 0.16.2.

Existing embeddings, reranking, moderation, decision and NPU text-generation
examples remain in the [README](../README.md#api-examples). For exact request
limits and upstream behavior, see the pinned [NPU guide](https://github.com/peonist-ai/halogen-flash-server/blob/v0.17.2/docs/NPU.md)
and [changelog](https://github.com/peonist-ai/halogen-flash-server/blob/v0.17.2/CHANGELOG.md).
