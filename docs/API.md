# Halogen endpoints through Halobridge

Halobridge 0.1.38 supports Halogen 0.18.18. Install the Halobridge update, then
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
| `POST /v1/audio/speech` **new in 0.18.14** | NPU `qwen3-tts-0.6b-customvoice` or `qwen3-tts-1.7b-customvoice` | Speak text with a preset voice; WAV or PCM output. |
| `POST /classify` | Custom sequence classifier | Label inputs with the labels trained into your classifier. |
| `GET /halogen/handles` | Sessions enabled on the active GPU profile | Inspect kept conversation states. |
| `DELETE /halogen/handles/<name>` | Sessions enabled on the active GPU profile | Release one kept state. |
| `GET /v1/models` | No generation | Discover GPU profile IDs and currently loaded NPU IDs, tasks and context limits. |
| `GET /health` | Running backend | Check engine health and priority admission counters. |
| `GET /metrics` | Running backend | Scrape Prometheus metrics, including priority-slot gauges. |
| `GET /cache` | Running backend | Inspect prompt-cache hits, dropped/replaced entries, evictions and pool usage. |

The first five NPU models run on 0.16.2. System One needs 0.16.3+, Flux needs
0.17.0+, the decision/embedding/reranker 4B variants need 0.17.4+, ASR needs 0.18.0+, and TTS needs 0.18.14+. Use 0.18.18 for all features. Enable a model
on every profile where it should remain available after a GPU model switch.

## Test in the dashboard

Open **API Playground** to run these requests without curl. Each endpoint has
an example, model selection and editable JSON. Upload fields prepare images and
WAV files; speech and image results include playback/previews and downloads.
The result shows HTTP status, total time, first-byte time, raw data and headers.
First-byte time includes routing, admission and model work; it is not necessarily
time to the first generated token. Streaming text appears live; streamed PCM is
playable after completion. Stop cancels the browser request.

The advanced section accepts `X-Halogen-*` and Anthropic version/beta headers.
Use the dedicated API-key field for backend authentication. Credentials and
request inputs are kept in memory only. Requests pass through the usual router
and telemetry. No inference runs on opening the tab or selecting an example.

## Model metadata and benchmark discovery

`GET /v1/models` preserves the active GPU backend's model metadata, including
its context limit (`meta.n_ctx`, `n_ctx`, `context_length` or `max_model_len`,
as reported by the engine). This also works with `models.auto_discover = false`
and an explicit `manual.service`; no Quadlet or deployment support is needed.
Inactive GPU profiles remain listed, but do not inherit the active model's limits.
An unavailable backend leaves the configured IDs discoverable without inventing
a context length.

To verify the active model's metadata and run the Terminal Bench doctor:

```bash
curl --fail-with-body -sS http://127.0.0.1:8831/v1/models | python3 -m json.tool
curl --fail-with-body -sS http://127.0.0.1:8731/v1/models | python3 -m json.tool
./terminal_bench.py doctor --endpoint http://127.0.0.1:8731/v1 --model halogen-qwen3.8-flash-next
```

Use your actual model ID and configured backend port. `--model` selects the
benchmark target when Halobridge advertises multiple GPU or NPU models.

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

Images are generated locally. On 0.18.2, large images and ASR take turns at NPU
memory: idle model parts are released and reloaded as needed. Alternating them
costs time; a request that still cannot fit returns 503 while the server stays
running. On 0.18.0/0.18.1, sizes above 512x512 may fail with ASR loaded; use
256x256 or 512x512 on those engines. Image bodies pass through Halobridge;
request history stores duration, status, model and response size, not the image.

## Speech: turn a WAV recording into text

Update the target GPU profiles to 0.18.18 and select **one** ASR model in
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

## Speech: turn text into audio (0.18.14)

Activate one `qwen3-tts-0.6b-customvoice` or `qwen3-tts-1.7b-customvoice` model
in NPU Setup. This is independent of the one-ASR-model limit.

```bash
curl --fail-with-body -sS localhost:8731/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts-1.7b-customvoice","input":"Hello from Halobridge.","voice":"ryan","language":"english","response_format":"wav"}' \
  -o speech.wav
```

The nine voices are `aiden`, `dylan`, `eric`, `ono_anna`, `ryan`, `serena`,
`sohee`, `uncle_fu` and `vivian`. Input is limited to 4096 characters and about
2.5 minutes of output. `language` is optional (`auto` by default). The 1.7B model
also accepts `instructions` for speaking style; 0.6B refuses that field.
`response_format` is `wav` or `pcm` (16-bit mono, 24 kHz); use `stream:true`
with PCM. Compressed formats and `speed` values other than 1 are unsupported.

The talker runs on the NPU with CPU audio decoding; Flash stays loaded.
From 0.18.15, `HALOGEN_NPU_SPLICE=1` is the default: short decisions,
embeddings, reranks and moderation run between steps, with streaming guard
checks first. `0` queues them behind the long job instead. On 0.18.14 this
option defaults to 0. These controls are in the profile editor.
Audio content and speech text are excluded from Halobridge telemetry buffers;
history records duration, model, status and response bytes without invented tokens.

0.18.14 also enables `HALOGEN_POOL_GROW=1`: a long answer can reclaim idle prompt
cache to reach its requested budget. `0` restores the old behavior. Existing
checkpoint, context and slot settings are preserved during the engine update.

## Added through Halogen 0.18.14

Enable the required models in NPU Setup first. Existing API authentication still
applies to every example. Flash checkpoints do not need replacing.

### Decision 3.0

`d3-lite` and `d3-nano` answer System One questions with up to 255 choices,
yes/no probabilities or scores. They accept up to 8192 text tokens and optionally
one inline image; the server does not fetch image URLs. Use the actual model ID
when several deciders are enabled. The `vllm-sr/` aliases also work.

```bash
curl --fail-with-body -sS localhost:8731/v1/systemone \
  -H 'Content-Type: application/json' -d '{
  "model":"d3-lite","state":"The customer was charged twice.",
  "questions":{"team":{"type":"choice","instructions":"Which team should handle this?",
    "criteria":{"billing":"Payment problems","technical":"Software problems","other":null}}}}'
```

### Multimodal embeddings

`embeddinggemma-2` embeds text, images and WAV audio (up to 30 seconds).
`wemm-embedding-4b` embeds text and images and shares device programs with
`decider-4b`. Clients must index and query with the same embedding model.

```bash
curl --fail-with-body -sS localhost:8731/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2","input":["How do I enable the NPU?"],"prompt_name":"query"}'
```

EmbeddingGemma image inputs use `image_url` parts with inline base64, and audio
inputs use `input_audio` parts with `data` and `format:"wav"`; see the
[pinned NPU guide](https://github.com/peonist-ai/halogen-flash-server/blob/v0.18.18/docs/NPU.md).

### Image edits and variations

Both multipart routes use `flux2-klein-4b` and preserve the active GPU model.
On 0.18.12+, edits accept one to four repeated `image` or `image[]` fields,
up to 25 MiB per picture. Refer to them as image 1 through image 4 in upload
order. A single-image edit can return 1024x1024 with `size=1024x1024`; edits
with multiple images return 512x512. Variations still accept exactly one image
and return 512x512. Check the response's `size` for the output dimensions.
Disable both routes with `HALOGEN_NPU_IMAGE_EDITS=0` if desired.

```bash
curl --fail-with-body -sS localhost:8731/v1/images/edits \
  -F model=flux2-klein-4b -F image=@source.png -F 'prompt=Make the sky orange' \
  -F size=1024x1024
curl --fail-with-body -sS localhost:8731/v1/images/edits \
  -F model=flux2-klein-4b -F image=@scene.png -F image=@object.png \
  -F 'prompt=Place the object from image 2 in the scene from image 1'
curl --fail-with-body -sS localhost:8731/v1/images/variations \
  -F model=flux2-klein-4b -F image=@source.png
```

### Streaming guard

With `qwen3guard-stream-0.6b` loaded, request `guard:true` on the GPU chat call.
The guard reports `halogen_guard` verdicts and a summary; it does not block or
rewrite the answer. It is not a standalone chat model. Stream count, probability
threshold, backlog and final-verdict wait are configurable in the profile editor.

```bash
curl --fail-with-body -N -sS localhost:8731/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash","messages":[{"role":"user","content":"Explain safe password storage."}],"max_tokens":128,"stream":true,"guard":true}'
```

### Other controls and compatibility routes

`HALOGEN_KV8=1` is an opt-in memory saving mode with small numerical differences;
it cannot be combined with `HALOGEN_COMPOSABLE_CONTEXT=1`. The profile editor
also exposes `HALOGEN_FORK_MAP=auto`, `HALOGEN_FORK_AUTO_KEEPS`, template selection,
GPU selection, cache-prefix reuse, drafting thresholds/grouping and access logs.
Existing context and slots are preserved. Empty values retain upstream defaults.

`/props`, `/slots`, `/tokenize` and `/detokenize` are proxied for llama.cpp clients.
`max_tokens:-1` asks for the engine's maximum available output budget.
`timings.speculation` reports draft acceptance; `timings.max_tokens_clamped_from`
and `max_tokens_clamped_to` explain budgets shortened to the available context.
The router preserves these fields. The speed, schema, NPU-memory and multi-GPU
fixes take effect through the engine update.

## Downloading NPU files before host setup

In Models, select the tasks and backend profiles, then **Preview downloads**.
**Download files only** downloads and verifies the pinned model files without
requiring the NPU driver, XRT or IOMMU to be ready. It uses the existing progress
and cancellation controls and retains completed files for another attempt.

After the host checks pass, **Download & activate** prepares any missing files,
reuses verified downloads, configures the selected Quadlets, drains requests and
restarts the affected active backend. The host prerequisites are shown next to
these actions; administrator setup and a reboot may be needed for drivers and
boot/IOMMU settings.

The guarded dashboard API also supports this preparation step:

```bash
curl --fail-with-body -sS localhost:8731/dashboard/api/deploy/npu/download \
  -H 'Content-Type: application/json' -H 'X-Halogen-Action: deploy' \
  -d '{"models":"decider-4b","profiles":["official"]}'
```

Add the Halobridge login token when authentication is enabled. The response is
202 with job kind `quick-npu-download`; `/dashboard/api/deploy/job` reports its
progress. This step leaves profiles and the running backend unchanged.

## Strict tools and cached NPU decisions on 0.18.2

A tool with `strict:true` constrains its generated arguments to a supported
parameter schema. The flag is inside `function` for Chat Completions and on the
tool itself for Responses and Messages. Your client still executes the tool:

```bash
curl --fail-with-body -sS localhost:8731/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
  "model":"qwen3.8-flash","enable_thinking":false,"max_tokens":256,
  "messages":[{"role":"user","content":"Look up the weather in Berlin in Celsius."}],
  "parallel_tool_calls":false,
  "tools":[{"type":"function","function":{
    "name":"lookup_weather","description":"Look up a city weather report","strict":true,
    "parameters":{"type":"object","properties":{
      "city":{"type":"string"},"units":{"type":"string","enum":["celsius","fahrenheit"]}},
      "required":["city","units"],"additionalProperties":false}}}]}'
```

`HALOGEN_STRICT_TOOLS` defaults to 1; 0 restores unenforced strict declarations.
Unsupported schema features and strict calls with images or logprobs may fall
back to ordinary tool calls and are logged by Halogen. `/health` reports the
supported strict-tool rules.

Both deciders now accept up to 8192 tokens including their questions and
template. For repeated System One questions about long text, enable
`HALOGEN_NPU_DEC_RESUME=1` in the profile's NPU section to reuse text within a
multi-question request. Mode 2 also reuses matching prefixes across requests
whose text grows. It is off by default. `HALOGEN_NPU_DEC_ENTRIES` defaults to 6;
kept texts consume host RAM. Request shapes remain unchanged.

The engine detects old GPU firmware and enables its workaround automatically.
Leave `HALOGEN_QSA_FORCE_SEL` empty for that policy. Client-written chat-template
markers are read as text; `HALOGEN_CLIENT_MARKERS=special` restores the previous
interpretation. Both controls are available in the profile editor.

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

On 0.18.1+, `HALOGEN_FORK_MAP=claude-code` with `HALOGEN_FORK=1` maps Claude
Code's thread identifiers to these handles automatically. The profile editor
validates this dependency. For disconnect retries, `HALOGEN_RETRY_GRACE_S`
defaults to 20 seconds (0 disables, maximum 120); the client must resend the
same request within the grace period. Halobridge forwards the thread headers
and request fields. `/v1/responses` still requires the full input history;
`previous_response_id` alone returns 400.

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
GPU checkpoints remain unchanged from 0.16.2 through 0.18.18. Prompt-cache fixes, Flux tokenizer corrections, NPU health listing, clean container stops and the automatic GPU firmware workaround take effect after the engine update.

### Tool compatibility and vision discovery (0.18.18)

Update profiles directly to 0.18.18 in **System → Engine updates**. The
0.18.17/0.18.18 fixes use the existing endpoints: kept conversations remain
isolated in a full KV pool, automatic keeps no longer wait on themselves, and
short prompts stay responsive beside long ones despite metrics reads or paused
replies. Flash checkpoints, context and slot settings are unchanged. The
0.18.18 NPU catalog matches 0.18.14/0.18.16; verified assets are reused.

On `/v1/messages`, Claude Code's `tool_reference` blocks and `defer_loading`
fields pass through unchanged. Newly loaded tools no longer invalidate the
conversation's prompt cache. The editor exposes `HALOGEN_TOOL_DEFER` (0.18.18+):
leave it empty for the engine default `1`, or set `0` to load tools up front.
The `ENABLE_TOOL_SEARCH=false` client workaround is no longer needed. Tool-call
arguments containing literal tool markup are preserved on Chat Completions,
Responses and Messages, streamed or not.

`GET /v1/models` now carries the active backend's vision capability in
OpenRouter's format:

```json
{
  "architecture": {
    "input_modalities": ["text", "image"],
    "output_modalities": ["text"]
  }
}
```

Without its vision tower the inputs are `["text"]`. Halobridge preserves the
advertised values, including each NPU model's own metadata, and never copies
the active model's capabilities to an inactive GPU profile. No new endpoint
is required. Existing custom GGUF profiles can use ggml-org's IQ4_NL checkpoint
with the updated engine; HGN installations need no new checkpoint.

### Reliability and capacity (0.18.16)

The profile editor exposes `HALOGEN_CACHE_BRANCHES` (Halogen 0.13.3+).
Empty inherits the engine default `2`: one conversation and one side turn.
`3` retains resume points for a parent and two children, avoiding another
prefill when the parent continues after both children. More branches increase
host memory use and the derived `HALOGEN_CACHE_ENTRIES` default. An explicit
`HALOGEN_CACHE_ENTRIES` overrides that automatic entry count. Branches are cache
resume points, not concurrent slots; changing them does not change `HALOGEN_KV_SLOTS`.

Use **System → Engine updates** to update installed profiles. 0.18.16 fixes
kept replies being mixed when conversations move in the KV pool and turns off
0.18.15's automatic reply-end keeps after reports of long stalls. Empty
`HALOGEN_FORK_MAP` is off again; an explicitly configured `auto` stays on.
The editor accepts `claude-code,auto` to enable both maps with
`HALOGEN_FORK=1`. Existing settings, checkpoints and shared files are retained.
The 0.18.16 NPU catalog is identical to 0.18.14, so verified files are reused.

0.18.15's cancellation, truncated tool-call and `/detokenize` fixes use existing
endpoints. No new endpoint or request format is needed. The profile editor
exposes these additional controls (0.18.15+, except the older EOS guard):

| Setting | Default | Purpose |
| --- | --- | --- |
| `HALOGEN_EOS_GUARD` | `1` | Keep quoted end-of-turn tokens in reasoning and tools from stopping a reply; available since 0.13.4. |
| `HALOGEN_ANSWER_GUARD` | `1` | Avoid ending on unfinished code fences, headings and labels; needs the EOS guard. It can change replies that would have ended there. |
| `HALOGEN_DV_GRAMMAR` | `1` | Accelerate structured-output checks; `0` restores the previous path. |
| `HALOGEN_GRAMMAR_JOINT` | `1` | Accelerate grammar decoding across concurrent requests; `0` restores the previous path. |
| `HALOGEN_OOM_SCORE_ADJ` | `500` | Prefer ending the container over the desktop when the host runs out of memory; `0` leaves its inherited score. Range 0–1000. |
| `HALOGEN_NPU_HOST_FLOOR_GIB` | `4` | Refuse new NPU memory allocations with 503 / `host_memory` below this host-memory reserve; `0` disables the floor. |
| `HALOGEN_NPU_SPLICE` | `1` | Interleave short NPU jobs and prioritize streaming guard checks; 0.18.14 defaulted to `0`. |

If the NPU returns `host_memory`, enable fewer NPU models or reduce the KV
pool. A speech request that cannot fit now releases idle NPU memory and retries
once, then fails only that request if needed; other NPU models stay available.

The capacity editor accepts a KV pool up to **2,097,152** positions on 0.18.15+.
For a 128 GB host running the default official checkpoint, the optional
upstream eight-conversation recipe is:

```text
HALOGEN_KV8=1
HALOGEN_KV_SLOTS=8
HALOGEN_MAX_TOK=16384
HALOGEN_KV_POOL_POSITIONS=2097152
```

It leaves little host memory: use this recipe **without NPU models**. KV8 can
change output and is incompatible with composable context. The pool is shared
capacity, not a larger per-request context. The engine may lower it to fit.
Halobridge does not enable this recipe automatically; new templates keep
262144 context and two slots, and upgrades preserve user settings.

### Streaming during long prompts and NPU recovery (0.18.13)

These improvements use the existing endpoints. Empty profile-editor fields
inherit these engine defaults:

| Setting | Default | Purpose |
| --- | --- | --- |
| `HALOGEN_ADMIT_SPLICE_MS` | `500` | Keep streaming replies moving while reading another long prompt; `0` disables it. |
| `HALOGEN_ADMIT_SPLICE_FOR` | `all` | Apply to every stream; `h` limits it to human turns. |
| `HALOGEN_ADMIT_NEST` | `1` | Start a short text prompt during another request's prefill; `0` disables it. |
| `HALOGEN_ADMIT_NEST_MAX` | `1024` | Maximum new, uncached tokens for nested admission; `0` disables it. |
| `HALOGEN_CACHE_STAGE` | `2` | Use a 64 MiB pinned staging buffer for cache copies; `0` restores legacy copies. |
| `HALOGEN_NPU_RESTARTS` | `3` | Restart a faulted NPU engine while Flash keeps serving; `0` disables restarts. |

Nested admission does not apply to images or composable context. It reserves
about 0.33 GiB of pinned host memory and 32 MiB of GPU memory. NPU restarts back
off for 5, 10 and 15 seconds by default; the retry budget resets after ten
minutes of stable operation. Clients should handle temporary 503 responses.

The 0.18.12 and 0.18.13 NPU catalogs contain refreshed assets. The engine updater
prepares and verifies the selected models in a new shared tree before stopping
the active backend, and retains the previous tree for rollback.

On supported hosts, `GET /metrics` includes
`halogen:gpu_queues_evicted_seconds_total`: cumulative time the kernel evicted
GPU queues while moving mapped memory. A rising counter during slow requests
points to compaction or reclaim stalls. The engine's request log also reports
`evicted_ms` for this delay. Query through the router with:

```bash
curl --fail-with-body -sS localhost:8731/metrics \
  | grep 'halogen:gpu_queues_evicted_seconds_total'
```

Existing embeddings, reranking, moderation, decision and NPU text-generation
examples remain in the [README](../README.md#api-examples). For exact request
limits and upstream behavior, see the pinned [NPU guide](https://github.com/peonist-ai/halogen-flash-server/blob/v0.18.18/docs/NPU.md)
and [changelog](https://github.com/peonist-ai/halogen-flash-server/blob/v0.18.18/CHANGELOG.md).
