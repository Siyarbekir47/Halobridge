# Halogen endpoints through Halobridge

Halobridge 0.1.29 supports Halogen 0.18.0 and 0.18.1. Install the Halobridge update, then
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
| `POST /v1/audio/transcriptions` **new in 0.18.0** | NPU `qwen3-asr-0.6b` or `qwen3-asr-1.7b` | Convert a WAV recording to text for meeting notes, dictation or searchable audio. |
| `POST /classify` | Custom sequence classifier | Label inputs with the labels trained into your classifier. |
| `GET /halogen/handles` | Sessions enabled on the active GPU profile | Inspect kept conversation states. |
| `DELETE /halogen/handles/<name>` | Sessions enabled on the active GPU profile | Release one kept state. |
| `GET /v1/models` | No generation | Discover GPU profile IDs and currently loaded NPU IDs, tasks and context limits. |
| `GET /health` | Running backend | Check engine health and priority admission counters. |
| `GET /metrics` | Running backend | Scrape Prometheus metrics, including priority-slot gauges. |
| `GET /cache` | Running backend | Inspect prompt-cache hits, dropped/replaced entries, evictions and pool usage. |

The first five NPU models run on 0.16.2. System One needs 0.16.3+, Flux needs
0.17.0+, the decision/embedding/reranker 4B variants need 0.17.4+, and ASR needs 0.18.0+. Use 0.18.0 for all features. Enable a model
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

Supported sizes are `256x256`, `512x512`, `768x768`, `1024x1024`, `768x512`, `512x768`, `1024x768`, `768x1024`, `1024x576`, `576x1024`, `640x512`, `512x640`, `896x512`, `512x896`, `1024x448`, `448x1024`, `1024x512`, `512x1024`, `1024x640`, `640x1024`; `auto` means `512x512`. Set `steps` (or `num_inference_steps`) to **4**, or omit it; every other step count is rejected. `n` is
1–4, drawn sequentially. Output is PNG. `response_format=url` returns an inline
`data:image/png;base64,...` URL, not a hosted download. A fixed seed makes a
request repeatable; image `i` uses `seed + i`. The prompt is truncated to 512
tokens. `quality=low` is currently refused; the other quality values produce
the same image. Only one image request runs on the NPU at a time.

Images are generated locally. With ASR loaded, sizes above 512x512 may exceed shared NPU memory and return an error; the server keeps running. After `qwen3-asr-1.7b` has transcribed, larger sizes no longer fit. Use 256x256 or 512x512 with ASR, or disable ASR through NPU Setup for larger images. Image bodies pass through Halobridge;
request history stores duration, status, model and response size, not the image.

## Speech: turn a WAV recording into text

Update the target GPU profiles to 0.18.0 and select **one** speech model in
NPU Setup: `qwen3-asr-0.6b` (about 1.6 GB host RAM) or `qwen3-asr-1.7b`
(about 4 GB, more accurate). Halobridge prepares the model, tokenizer, device
programs and audio encoder together; they are shared across profiles.

```bash
curl --fail-with-body -sS localhost:8731/v1/audio/transcriptions \
  -F model=qwen3-asr-0.6b -F file=@meeting.wav \
  -F language=en -F response_format=json
```

The response is `{"text":"..."}`. Use `response_format=text` for plain text.
`language` is optional; `prompt` may supply names or terminology. Clients may
send `model=whisper-1`; the request still uses the single loaded ASR model.
Only WAV is accepted: PCM 8–32 bit or floating point, any sample rate,
mono or multiple channels, up to **20 minutes or 25 MiB**, whichever is first.
Other formats and streaming are rejected by Halogen. Audio uploads and response
text pass through unchanged. History records the loaded model, duration, status
and response size; it stores neither audio nor transcript, and invents no token
counts. Transcription queues with the other NPU work and keeps the GPU loaded.

## Larger NPU models and custom classification

Select `decider-4b`, `qwen3-embedding-4b` or `qwen3-reranker-4b` in NPU Setup
on 0.17.4+. They use the same endpoints as their smaller counterparts; change
the request's model ID. The embedding model supports up to 2560 dimensions.
If several decision models are loaded, `/v1/systemone` needs the exact model ID.

A sequence-classifier fine-tune (`Qwen3ForSequenceClassification`) of Qwen3-0.6B,
Qwen3-Embedding-0.6B or Qwen3-Embedding-4B can be mounted and selected by path
in NPU Setup. Halogen determines its task and base programs. Send its directory
basename as the model:

```bash
curl --fail-with-body -sS localhost:8731/classify \
  -H 'Content-Type: application/json' -d '{
  "model":"my-classifier","input":["Win a free prize now","Lunch at noon?"]}'
```

Each entry in `data` includes `index`, `label`, `probs` and `num_classes`.
`usage` counts prompt tokens. Use this for labels learned by your classifier,
such as spam or ticket categories. Moderation remains `/v1/moderations`.

## Sessions: reuse and branch an exact conversation

In the GPU profile editor, enable `HALOGEN_FORK=1` and apply/restart the profile.
The optional limits default to 8 handles, 600 seconds idle expiry and 2 GiB.
Keep `/cache` mounted and `HALOGEN_CACHE_DIR` set for restart persistence.

```bash
curl --fail-with-body -sS localhost:8731/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Halogen-Keep: example-start' \
  -d '{"model":"qwen3.8-flash","max_tokens":128,"enable_thinking":false,
       "messages":[{"role":"user","content":"Suggest three names for a local search tool."}]}' \
  > /tmp/halobridge-session.json

# Reuse the exact assistant message returned by the first request.
python3 - <<'PY' > /tmp/halobridge-session-branch.json
import json
with open('/tmp/halobridge-session.json') as f:
    reply = json.load(f)['choices'][0]['message']
print(json.dumps({'model': 'qwen3.8-flash', 'max_tokens': 128,
                  'enable_thinking': False, 'messages': [
    {'role': 'user', 'content': 'Suggest three names for a local search tool.'},
    reply,
    {'role': 'user', 'content': 'Explain the first name in one sentence.'}]}))
PY

curl --fail-with-body -sS localhost:8731/v1/chat/completions \
  -H 'Content-Type: application/json' -H 'X-Halogen-From: example-start' \
  -H 'X-Halogen-Keep: example-branch' \
  --data-binary @/tmp/halobridge-session-branch.json

curl --fail-with-body -sS localhost:8731/halogen/handles
curl --fail-with-body -sS -X DELETE localhost:8731/halogen/handles/example-start
```

Start several requests with the same `X-Halogen-From` to fork. A request that
also supplies `X-Halogen-Keep` saves its resulting branch. These headers work
on Chat Completions, Messages and Responses. Response header `X-Halogen-From`
reports `used`, `unmatched` or `unknown`; an unmatched/expired state becomes an
ordinary request. This accelerates repeated context, rather than replacing the
conversation in the request. Handles belong to the active GPU profile's cache;
GET/DELETE manage that backend, so select the intended GPU profile first.
Clients of that backend share names. Changing `HALOGEN_FORK`, `FORK_GIB` or
`FORK_TTL_S` invalidates its disk cache configuration.

## Prompt progress and stream silence

On 0.17.5+, send `"stream":true,"return_progress":true` on Chat Completions
or Completions. The SSE stream includes `prompt_progress` with `total`, `cache`,
`processed` and `time_ms` before generated tokens. This lets a client show
progress while a long prompt is processed. Halobridge forwards these chunks
and still reads final usage. Dashboard latency is **time to first response byte**,
so progress or keepalive events can arrive before the first generated token.

Leave `HALOGEN_NEXT_TOKEN_S` empty to use Halogen's watchdog-derived silence
limit (990 seconds at its default settings). Set it in the Engine section only
when a different limit is required. Halobridge imposes no fixed stream read
timeout. The engine's memory-stall and cache fixes work after an engine update.

## Optional backend keys and request records

Halobridge's dashboard login and Halogen's backend API key are separate.
Set `HALOGEN_API_KEY_FILE` in the profile editor to a container path backed by
an existing secret file, for example `/secrets/backend.key`, and add a read-only
mount for its host directory. The key file accepts comma-separated or newline-
separated ASCII keys. Inline `HALOGEN_API_KEY` is also supported; set one option,
not both. Inline values are hidden in the editor and redacted in Quadlet previews.

Halobridge resolves that file using the profile's actual mounts for discovery
and cache monitoring. Clients can send a backend key directly. Clients that
authenticate with Halobridge's configured login token can use the profile's
backend credential through the bridge. Unauthenticated clients never receive
the bridge's credential. `/health` stays open on the backend; other backend
routes require a key when this option is enabled. Configure each GPU profile
that should enforce it. For externally managed backends without a matching
Quadlet, `[router].backend_api_key_file` is an optional **host** file path.

To keep Halogen's own diagnostic request records, set `HALOGEN_RECORD_DIR` to
a writable mounted path such as `/cache/request-records`. Set
`HALOGEN_RECORD_TEXT=none` for metadata only, or `all` / a comma list of
`prompt,reply,reasoning,args,results`. If omitted, text defaults to **all**.
`HALOGEN_RECORD_GIB` defaults to 2 and `HALOGEN_RECORD_DAYS` to 90; 0 days
keeps all days. These are optional local files, separate from Halobridge's
usage database. They have no read/delete API. Full or failing record storage
drops records while inference continues.

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
GPU checkpoints remain unchanged from 0.16.2 through 0.18.0. The 0.18.0 prompt-cache fixes, Flux tokenizer correction, NPU health listing and clean container stops take effect automatically after the engine update.

Existing embeddings, reranking, moderation, decision and NPU text-generation
examples remain in the [README](../README.md#api-examples). For exact request
limits and upstream behavior, see the pinned [NPU guide](https://github.com/peonist-ai/halogen-flash-server/blob/v0.18.0/docs/NPU.md)
and [changelog](https://github.com/peonist-ai/halogen-flash-server/blob/v0.18.0/CHANGELOG.md).
