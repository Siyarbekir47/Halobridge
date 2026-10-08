# Halobridge

**Halobridge** is a router and dashboard for local [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) deployments.

It gives you one OpenAI-compatible endpoint in front of your local Halogen models,
plus a lightweight operations dashboard for usage, model switching, and safe
one-click container updates.

> **Independent community project.** Halobridge is not affiliated with,
> sponsored, or endorsed by Peonist, LLC. “halogen” and “Peonist” are
> trademarks of Peonist, LLC. Halobridge does not contain or modify the
> halogen-flash-server engine; it talks to the official container image over
> HTTP and manages local Podman Quadlet files.


<p align="center">
  <img width="1907" height="905" alt="1_1" src="https://github.com/user-attachments/assets/245536b5-73af-47e0-89b0-1428cd18b000" />
</p>

<p align="center"><em>Overview — active model, request volume, token usage and cache efficiency at a glance.</em></p>

## What it does

- Routes OpenAI-compatible requests to the currently active local Halogen model.
- Switches between model services through systemd user units.
- Discovers models from Podman Quadlet files.
- **Installs and manages Halogen backend profiles from the dashboard**: create a
  profile from a template, edit parameters such as KV slots, KV pool, context
  size and cache size, preview the resulting Quadlet as a diff, then apply with
  automatic backup and one-click rollback.
- Shows request counts, input/output tokens, cache ratio, latency, and history.
- Uses only API-reported `usage` values for token accounting.
- Provides a safe update flow for the official Halogen container image.
- Runs five NPU tasks alongside the GPU model through the same API, with
  shared downloads, host diagnostics and versioned update recovery.
- Can be protected with a token gate for non-local access.

## What it does not do

- It is not an official Peonist product.
- It does not store prompts, responses, images, tool contents, or secrets.
- It does not invent token counts when the API does not report usage.
- It does not update to `latest`, release candidates, or non-stable tags.
- It does not delete user-managed model or runtime cache directories. After a
  successful Uncensored quick setup, it removes only the downloaded source
  GGUF shards and their local Hugging Face metadata; the converted `.hgn`
  remains in place.

## Requirements

Currently supported distributions: **Ubuntu** and **Fedora 44**. Other Linux
distributions have not been tested.

- A Linux host supported by
  [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server),
  with its recommended BIOS/UMA configuration
- Podman and systemd user services
- Python 3.11+ and `pipx`
- Enough local disk space for the selected model (the official download is
  about 118 GiB, plus cache and working space)

You do **not** need to install Halogen, create a container, download a model or
write a Quadlet before starting Halobridge. The dashboard can do that from an
empty host.

## Quick start

### 1. Install Halobridge

```bash
pipx install git+https://github.com/Siyarbekir47/Halobridge.git
hash -r
halobridge --version
```

No configuration file is required. On a clean host, `halobridge doctor` warns
that no model exists yet; that is expected until step 3.

### 2. Start the dashboard

For access only from the server itself:

```bash
halobridge router
```

For access from your LAN or VPN:

```bash
halobridge router --bind 0.0.0.0
```

Then open one of these addresses:

```text
http://127.0.0.1:8731/dashboard
http://SERVER-IP:8731/dashboard
```

Binding to `0.0.0.0` exposes the API and dashboard to networks that can reach
the host. Prefer a trusted LAN/VPN, and configure the token gate before using
an untrusted network.

#### Optional: start automatically with systemd

Stop a foreground Halobridge process with `Ctrl+C`, then create and start a
systemd user service. This block works with the `pipx` installation above and
does not require a cloned repository:

```bash
mkdir -p ~/.config/systemd/user
tee ~/.config/systemd/user/halobridge.service >/dev/null <<'EOF'
[Unit]
Description=Halobridge router and dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=%h/.local/bin/halobridge router --bind 0.0.0.0
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now halobridge.service
sudo loginctl enable-linger "$USER"
```

`enable-linger` lets the user service start during boot without an interactive
login. Verify the service and follow its logs with:

```bash
systemctl --user status halobridge.service --no-pager
journalctl --user -u halobridge.service -f
```

For server-local access only, remove `--bind 0.0.0.0` from `ExecStart`.

### 3. Install the official model

In the dashboard:

1. Open **Models**.
2. Leave **Official model** selected.
3. Click **Install official model** and confirm.
4. Keep Halobridge running while the job finishes.

Halobridge creates the Quadlet, pulls the pinned official container image,
downloads the model on first start and streams the progress into the dashboard.
The tokenizer, vision tower and separate v2 N-Gram table are prepared in the
shared directory before activation. The official checkpoint downloads on first
start and resumes if interrupted; the selected engine/checkpoint determines
its size. Image input is enabled by default. Later starts reuse files on disk.

The container engine and model format belong to
[peonist-ai/halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server).
Halobridge provides the installation workflow, Quadlet management, router and
dashboard around that official image.

### 4. Verify it

Check health and model discovery:

```bash
curl -sS http://127.0.0.1:8731/health
curl -sS http://127.0.0.1:8731/v1/models
```

Send a short OpenAI-compatible request:

```bash
curl -sS --max-time 300 \
  http://127.0.0.1:8731/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash",
    "messages": [{"role": "user", "content": "Reply with exactly: Halogen is running."}],
    "max_tokens": 64,
    "temperature": 0,
    "reasoning_effort": "low"
  }'
```

If you started Halobridge with `--bind 0.0.0.0`, replace `127.0.0.1` with the
server's LAN or VPN address when testing from another machine.

### Later starts

The downloaded model stays under `~/halogen`. Starting Halobridge again
discovers the generated Quadlet and starts or adopts the official backend:

```bash
halobridge router --bind 0.0.0.0
```

## Configuration

Default config path:

```text
~/.config/halobridge/halobridge.toml
```

You can also use:

```bash
halobridge --config /path/to/config.toml router
```

or:

```bash
export HALOBRIDGE_CONFIG=/path/to/config.toml
halobridge doctor
```

A missing config file is valid and uses safe localhost defaults.

See [`config.example.toml`](config.example.toml) for the full commented example.

## Model discovery

By default, Halobridge scans:

```text
~/.config/containers/systemd
```

for `*.container` Quadlet files whose image belongs to the configured Halogen
image repository.

For each model it reads:

- `Image`
- `ContainerName`
- `Environment=HALOGEN_MODEL_ID=...`
- `Volume=...:/cache`

You can disable discovery and configure models explicitly:

```toml
[models]
auto_discover = false

[models.explicit]
"qwen3.8-flash" = "halogen-official.service"
"qwen3.8-flash-uncensored" = "halogen-uncensored.service"
```

## Dashboard

The dashboard is organized into four workspaces: **Overview** for usage and
breakdowns, **Requests** for paginated history and request details, **Models**
for profiles, installation and model files, and **System** for host health,
updates and telemetry maintenance. The active model and connection status stay
visible across workspaces. Monitoring views share the selected reporting period.

English is the default. The language selector remembers English or German and
applies it to labels, validation errors, update status and deployment job messages.
Changing language preserves profile edits and the selected reporting period.

The **Appearance** selector defaults to **System**, following your device's light
or dark theme as it changes. Select **Light** or **Dark** to override it. Your
choice is remembered in the browser and shared across open dashboard tabs.

Close the incomplete token usage notice with its **×** button. It stays hidden
across refreshes, reloads and reporting periods until a new request with missing
token usage is recorded. The browser remembers which existing requests you
acknowledged; complete requests do not make the notice reappear.

The dashboard shows:

- active model and switch target
- running and queued requests
- completed inference requests
- input/output tokens from API `usage`
- cache and reasoning breakdowns
- request history
- engine telemetry where available
- update status and recovery state

Important accounting rules:

- `0` means the API reported zero.
- `–` means the API did not report the value.
- Cache tokens are part of input tokens.
- Reasoning tokens are part of output tokens.
- Engine logs are independent and are not joined to router requests by timestamp.


### Request history

The **Requests** workspace shows completed requests with client/model, reported
input and output tokens, duration and status. Individual rows can be expanded for
more detail without storing prompt or response contents.

<p align="center">
<img width="1907" height="905" alt="1_2" src="https://github.com/user-attachments/assets/aaa5e3cc-d4b6-4fb3-a8fb-ba112d23a89d" />
</p>

<p align="center"><em>Requests — per-request token usage, latency, client/model and completion status.</em></p>

## Profiles & deployment

If Halogen is not installed yet, the dashboard can install and manage it. If
Quadlets already exist, they are imported and edited in place — the files on
disk stay the source of truth.

### One-click deploy

The **Install a model** section in **Models → Profiles & installation** needs no
configuration. Halobridge serves one backend at a time, so installing a model
takes over the host: the currently active backend is drained and stopped, the
GPU is released, the new model is installed and started, and the router points
at it. You do not have to stop the running model by hand first.


<p align="center">
<img width="1907" height="905" alt="1_3" src="https://github.com/user-attachments/assets/a1f8c71e-9296-4045-abe6-dd27f919e183" />
</p>

<p align="center"><em>Models — installed profiles, active backend state and one-click model installation.</em></p>

- **Official model** — one click installs the official profile and starts it.
  The first start downloads the upstream checkpoint through
  `HALOGEN_DOWNLOAD`; verified shared files are prepared first. The vision tower is
  enabled by default, so the resulting backend accepts image inputs. If a
  profile already exists, its checkpoint and runtime settings are preserved;
  shared asset bindings are connected to the canonical directory.
- **Uncensored model** — paste a Hugging Face read token and click once.
  Before the first download, sign in to Hugging Face, open the
  [gated OrcaRouter repository](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF),
  accept its terms or request access, and wait for approval. Then create a
  READ token from the same account. A valid token alone is not enough if that
  account has not been granted repository access. Halobridge downloads the
  uncensored GGUF, the draft head and shared assets (while the current model keeps
  serving), then stops the active backend, converts to `.hgn` with the GPU to
  itself, applies the profile and starts uncensored — all as one streaming job
  with visible steps. After the new backend passes its health check, the source
  GGUF shards and their local download metadata are removed automatically.
  If the converted `.hgn` already exists, the
  download/convert steps are skipped and no token is required.
- **Swift 1.5** and **Swift 1.5 Abliterated** — select a variant, review its
  download/storage preview and click **Install Swift model**. Both support
  text and images, use the existing official engine, and require no HF token
  or GGUF conversion. Their pinned complete checkpoints include the MTP draft;
  neither downloads the official v2 checkpoint. Services are
  `halogen-swift15.service` and `halogen-swift15-abliterated.service`, with API
  model IDs `halogen-swift15` and `halogen-swift15-abliterated`.

The Official and Uncensored buttons ask for a confirmation click (labelled "Stop active model &
install?") and respect the same safety boundaries as the advanced editor
(allowed roots, same-origin + `X-Halogen-Action: deploy`). The switch drains
in-flight requests and waits for GPU memory release before touching the
hardware, so it never runs two backends on the GPU at once.

The install runs as a streaming job. Halobridge forwards the backend service
output from the systemd journal and adds a heartbeat every 10 s, so download
and startup progress stay visible. A prepared backend has a 5-minute health
deadline; the first official start gets up to 6 hours because it downloads
about 118 GiB. If it does not come up — or you press **Cancel** — Halobridge
stops the half-started model and rolls back to the previously active one, so a
failed takeover never leaves the host without a backend or frozen. A missing
tokenizer is re-downloaded even when the `.hgn` already exists, so a partial
earlier install is repaired automatically.

#### Shared model assets and Swift installation

N-Gram, all six tokenizer files and the vision tower live under
`<models_root>/shared/halogen-v2/<pinned-revision>/`. Official, Orca and both
Swift profiles use the same files through read-only `ro,z` mounts. Existing
files are located through actual Quadlet mounts, verified and imported using
hard links on the same filesystem. Across filesystems a verified copy is
made; source files are retained. Shared assets use about 48.5 GiB once.
Deleting a profile never deletes shared assets, checkpoints or caches.

The Swift checkpoint sizes are about 63.5 GiB (Swift) and 62.1 GiB
(Abliterated). The preview distinguishes cached verification, unverified
existing files and missing downloads, with required/free disk space per
filesystem. Preparation checks pinned digests, including Git blob hashes for
small tokenizer files. Verification is streamed and cached by path, device,
inode, size and modification time. Interrupted downloads retain HF resume
metadata; changed files invalidate the cache. Invalid files are repaired by
Quick Setup without replacing legacy files.

Swift starts with engine `0.15.1`, 262144 context/KV pool, two slots,
`HALOGEN_MAX_TOK=8192`, weights locking enabled, adaptive speculation disabled,
temperature `1.0`, Top-P `0.95` and Top-K `20`. Reinstallation preserves edited
runtime settings. Checkpoint, tokenizer, N-Gram and vision paths are explicit,
and `HALOGEN_DOWNLOAD` is absent: later model switches cannot trigger downloads.
Missing or invalid assets are reported before stopping the old backend.

Downloads and verification keep the current backend serving. Only activation
drains requests, changes the Quadlets, releases GPU memory and starts Swift.
The durable `swift-install.json` in the dashboard state directory stores the
previous affected Quadlets and active model. Failed or cancelled activation
restores those exact files, retains downloads and restarts the old backend.
Startup also recovers an interrupted activation. External Quadlet edits are
preserved and leave the router in maintenance for manual recovery.

Model discovery and `/v1/models` refresh without a Halobridge restart. The
official checkpoint upgrader remains restricted to the official model; engine
image updates preserve each profile's checkpoint and shared bindings.

Sources: [Swift model card](https://huggingface.co/Quat3rnion/halogen-swift1.5-qwen3.8-flash-next-v2),
[Abliterated model card](https://huggingface.co/Quat3rnion/halogen-swift1.5-qwen3.8-flash-next-v2-abliterated).

HTTP endpoints (existing dashboard authentication applies):

```text
GET  /dashboard/api/deploy/quick/swift/plan?variant=swift15
POST /dashboard/api/deploy/quick/swift
     {"variant": "swift15"}  # or swift15-abliterated
```

POST requires `Content-Type: application/json` and `X-Halogen-Action: deploy`.
Progress and cancellation use the existing deployment job endpoints.

After installing the four profiles, run the opt-in GPU check on the Linux host
with `python3 tests/smoke_swift_linux.py` from a checkout. It sends text and
image requests, exercises all model switches and restores the initial model;
it does not install weights. Use `HALOBRIDGE_TOKEN` for an authenticated router
and `--url` if the endpoint differs from `http://127.0.0.1:8731`.

#### Why the checkpoint field is empty after Quick Setup

This is intentional for the official model. Quick Setup sets
`HALOGEN_DOWNLOAD=peonist-ai/halogen-qwen3.8-flash-next` and leaves
`HALOGEN_CHECKPOINT` empty. The official container then downloads and selects
its standard checkpoint from `/models`, matching the upstream
[halogen-flash-server Quickstart](https://github.com/peonist-ai/halogen-flash-server#quickstart).
The empty dashboard field therefore means **automatic official checkpoint
selection**, not “no checkpoint loaded”. The startup log and `/health` show the
checkpoint that is actually active.

Halobridge does set `HALOGEN_VISION_TOWER` explicitly because upstream keeps
image support opt-in. This makes a fresh Official Quick Setup accept images by
default.

| Checkpoint mode | Advantages | Tradeoffs |
| --- | --- | --- |
| Empty / automatic (Official Quick Setup) | Simplest setup; follows the official downloaded layout; fewer paths to maintain | Not suitable when you need to choose between several custom checkpoints |
| Explicit `HALOGEN_CHECKPOINT` | Deterministically selects one `.hgn` or supported GGUF; useful for custom and converted models | The path must stay correct; a renamed, moved or missing file prevents startup |

For the standard official model, leave the checkpoint field empty. The dashboard
checkpoint selector sets it explicitly when choosing v2 or HT43. Custom or
converted checkpoints can still be configured manually.

Official checkpoint changes affect only the active official profile. Uncensored
profiles keep their converted OrcaRouter checkpoint; Swift profiles retain their
own weights. Upgrading the container image is separate from choosing a checkpoint.

#### Halogen 0.16.0: optional HT43 checkpoint

Halogen **0.16.0 keeps v2 as the default**. Its new `qwen38-flash-next-ht43.hgn`
checkpoint is optional: about **53.7 GiB** on disk and roughly **8 GiB less RAM**
than v2, with somewhat slower prompt processing and draft decoding.
See the [upstream changelog](https://github.com/peonist-ai/halogen-flash-server/blob/v0.16.0/CHANGELOG.md)
and [official model card](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next).

1. Update the engine to `0.16.0` or newer in the dashboard.
2. With Official active, select **HT43** under **System → model checkpoint**.
3. Review the download/shared-file plan and confirm preparation and restart.

Halobridge downloads only the selected checkpoint and missing shared files from
pinned revisions, checks their sizes and hashes, and reuses the same N-Gram,
tokenizer and vision files as v2, Orca and Swift. Existing unverified files are
checked first. Downloads resume after interruption. Free space is checked per
filesystem, including a 5 GiB reserve.

Preparation runs while the existing model serves requests. Activation drains
requests, saves only the official Quadlet, releases GPU memory and verifies the
model, API/engine version, image and selected checkpoint. Failure or cancellation
restores the previous official configuration; startup recovery uses the durable
update journal. Downloads and previous weights are retained for retries and
switching back to **v2** through the same selector. Missing or changed files can
be prepared again with **Verify / repair checkpoint**. Prepared profiles start
without `HALOGEN_DOWNLOAD`; model switches cannot silently download another
checkpoint. Custom official checkpoints are excluded from this selector.

The authenticated dashboard endpoint also accepts an explicit choice:
`POST /dashboard/api/updates/checkpoint` with `{"target":"ht43"}` or
`{"target":"v2"}`. The existing same-origin JSON/action-header checks apply.
Engine updates preserve the chosen checkpoint for every profile.

### Advanced editor

Open **Profile & Deployment** in the dashboard:

- **New profile from template** — two starting points:
  - *Official model*: preconfigured for the upstream weights repo. The first
    start downloads the weights (~118 GiB) into the models directory through
    `HALOGEN_DOWNLOAD`; later starts are offline. The vision tower is enabled
    by default.
  - *Custom model*: for your own GGUF or `.hgn` files you place in the
    models directory yourself.
- **Editable parameters** — every field is validated against a typed allowlist
  of upstream `HALOGEN_*` variables (KV slots, KV pool positions, context
  size, prefill chunk, host RAM reserve, disk cache size and pruning,
  reasoning effort, token defaults and caps, and more). Unknown variables and
  out-of-range values are rejected before anything is written.
- **Preview (dry-run)** — shows the exact Quadlet diff against the current
  file before anything changes.
- **Apply** — backs up the previous Quadlet, writes atomically, reloads
  systemd, and verifies the generated unit. If verification fails, the
  previous file is restored automatically.
- **Rollback** — restores the most recent backup with one click.
- **Start / delete** — a profile can only be started when no other backend is
  active (Halobridge serves one backend at a time), and only deleted while
  stopped. Model and cache directories are never deleted by the dashboard.

Safety boundaries:

- Host paths a profile may mount must live under `deploy.allowed_roots`
  (default: the user's home directory).
- Mutations require the dashboard session plus the `X-Halogen-Action: deploy`
  header and same-origin checks.
- Set `[deploy] enabled = false` to disable the whole deployment surface.

### System workspace

The **System** workspace summarizes host and engine state, including RAM, GPU
load, KV-pool usage, configured context/slots, generation and prefill throughput,
engine version, and update state. Engine measurements come from local completion
logs and are intentionally kept separate from router request accounting.

<p align="center">
<img width="1907" height="905" alt="1_4" src="https://github.com/user-attachments/assets/dabf1f93-1508-4095-ad4c-7c56670fd8b3" />
</p>

<p align="center"><em>System — host resources, engine telemetry, runtime configuration and update status.</em></p>

## Updates

Halobridge updates the local Quadlet files to the newest stable upstream tag.

Default update source:

```text
GitHub tags for peonist-ai/halogen-flash-server
```

Only stable versions like `0.13.1` are accepted. `latest`, `-rc1`, build
metadata, and malformed tags are ignored.

Update flow:

1. Pull the new image while inference continues.
2. Atomically close router admission.
3. Drain running requests.
4. Back up both Quadlet files.
5. Replace only the `Image=` line in both Quadlets.
6. Reload systemd and restart the active model service.
7. Verify API version, engine version, model, capability probe, and container image ID.
8. Roll back automatically if verification fails.

You can disable container updates:

```toml
[updates]
enabled = false
```

You can keep the dashboard but disable installation:

```toml
[security]
allow_install = false
```

## Halogen 0.17.2 support

Halobridge supports the new NPU image and System One routes, priority admission,
and the engine's new cache, schema and decode settings. Update Halobridge, then
use **System → Engine updates** to update the backend profiles to **0.17.2**.
New Official and Orca templates use 0.17.2; existing settings are preserved.
Swift's initial template keeps its model-card version until you update the engine.

The GPU checkpoints are unchanged from 0.16.2. The engine update brings faster
decode and long-prompt processing, speculative decoding for two conversations,
better disk-cache reuse, corrected streamed tool calls, JSON-schema/tool fixes,
nullable-string argument fixes, and support for late system/developer turns.
These changes work through the existing chat, Responses and Messages routes.
Leave `HALOGEN_MTP_DEPTH` empty for the engine's adaptive depth; an explicit
value continues to fix the depth.

See the [endpoint guide](docs/API.md) for use cases, prerequisites and runnable
examples, and the [upstream changelog](https://github.com/peonist-ai/halogen-flash-server/blob/v0.17.2/CHANGELOG.md)
for the engine changes. The upstream Compose recipe is an alternative deployment;
Halobridge's managed setup continues to use Podman Quadlets.

## NPU models with Halogen 0.17.2

The GPU model and the selected NPU models run together behind Halobridge's
existing port. Requests select an NPU model by its `model` ID; they do not
switch or unload the GPU model. `/v1/models` advertises only the NPU models
loaded by the currently active backend.

| Model ID | Task | Endpoint |
| --- | --- | --- |
| `decider-0.8b` | Schema-based decisions with option probabilities | `/v1/chat/completions` |
| `qwen3-embedding-0.6b` | Text embeddings for search and RAG | `/v1/embeddings` |
| `qwen3-reranker-0.6b` | Rank documents against a query | `/v1/rerank` |
| `qwen3guard-gen-0.6b` | Moderation with safety labels | `/v1/moderations` |
| `qwen3.5-2b` | Text generation and summaries, with streaming | `/v1/chat/completions` |
| `decider-0.8b` | Typed questions with probabilities and confidence | `/v1/systemone` |
| `flux2-klein-4b` | Text-to-image generation on the NPU | `/v1/images/generations` |

### Host preparation

1. Update the selected backend profiles to **Halogen 0.17.2** using
   the dashboard's engine updater. Your GPU checkpoint remains selected.
2. Follow [upstream NPU host instructions](https://github.com/peonist-ai/halogen-flash-server/blob/v0.17.2/docs/NPU.md#host)
   for the Linux `amdxdna` driver, firmware, XRT and its NPU plugin. The service
   user needs access to `/dev/accel/accel0`. Enable IOMMU; `amd_iommu=off` and
   `iommu=off` are incompatible, while `iommu=pt` is supported.
3. Run `halobridge npu-host-setup` and execute its printed commands once on the
   host. They download the unmodified upstream fabric-clock helper and unit
   from a pinned revision, verify SHA-256, and install and start the system
   service with `sudo`. Halobridge itself does not execute privileged commands.
4. Run `halobridge npu-check`. Resolve its reported prerequisites before
   enabling any NPU models. Rootless Podman needs the host fabric-clock service
   running: concurrent GPU/NPU work requires the GPU fabric clock to be held.
5. In **Models → NPU models alongside your GPU model**, select the tasks and
   target backend profiles, preview downloads, then apply the selection.

Ubuntu and Fedora 44 remain the supported distributions. NPU support also
depends on the host driver, firmware and XRT; actual device operation must be
checked on your machine. Halobridge finds AMD's `/opt/xilinx/xrt` installation
or resolves installed distribution library symlinks. XRT mounts are read-only
and never receive a SELinux relabel operation.

### Downloads, updates and recovery

NPU weights, tokenizers and device programs are checked against the selected
container image's pinned file record. They live under
`models_root/shared/halogen-npu/<manifest-sha256>` and are shared by Official,
Orca, Swift and custom GPU profiles. Containers mount this directory read-only.
Flux needs about **7.5 GiB of downloads and 8 GB of host memory** in addition to
the GPU and other loaded models. Its decoder and device programs are included
in the plan. Images require Halogen 0.17.0 or newer; System One requires 0.16.3
or newer. The five earlier NPU models continue to work on 0.16.2.

The embedder, reranker and guard share their device programs; unused model
weights are not downloaded. Repeated setup verifies and reuses existing files.

Preparation, disk checks and XRT compatibility checks happen while the current
backend serves requests. Activation drains Halobridge requests and restarts
only the affected active backend. Failures or cancellation restore its previous
Quadlet and backend; a persistent journal recovers interrupted activation after
a router restart. Downloaded files stay available for retry. External Quadlet
edits pause recovery instead of being overwritten.

Engine updates prepare a new versioned NPU directory when its file record
changes. Unchanged files are reused; previous device programs stay intact for
rollback. A later model start with missing or invalid NPU files fails with a
repair instruction instead of starting an implicit download. Enable the same
NPU selection on every backend profile if the NPU models should remain
available after GPU model switches. Uncheck all tasks and apply to disable NPU
models while retaining their downloads.

Send inference through Halobridge so its request drain includes NPU work.
Calls made directly to the backend bypass that admission gate. The upstream
health counters describe GPU work; Halobridge tracks routed NPU requests too.
All six NPU tasks and System One requests appear in request history. Image and
System One requests stay outside text-generation token coverage. Anthropic
Messages enters the usage charts with its full input, including cached tokens;
generated image bodies are streamed through without buffering them for telemetry.

### API examples

These examples use the default local router address. Add your normal bearer
token header when the token gate is enabled.

```bash
# Decision: two to ten options; logprobs returns their probabilities.
curl --fail-with-body -sS localhost:8731/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
  "model":"decider-0.8b",
  "messages":[{"role":"user","content":"Please reset my account password."}],
  "response_format":{"type":"json_schema","json_schema":{
    "name":"topic","description":"Which support team should handle this?",
    "schema":{"enum":["accounts","billing","shipping"]}}},
  "logprobs":true,"top_logprobs":3}'

# Embeddings: 32–1024 dimensions; input must be text, not token IDs.
curl --fail-with-body -sS localhost:8731/v1/embeddings \
  -H 'Content-Type: application/json' -d '{
  "model":"qwen3-embedding-0.6b","dimensions":256,
  "input":["Instruct: Retrieve useful documentation\nQuery:How do I enable the NPU?",
           "Install the NPU driver, firmware and matching XRT plugin."]}'

# Reranking: results include the original index and relevance score.
curl --fail-with-body -sS localhost:8731/v1/rerank \
  -H 'Content-Type: application/json' -d '{
  "model":"qwen3-reranker-0.6b","query":"How do I enable the NPU?",
  "documents":["Install amdxdna and XRT.","The dashboard has a dark theme."],
  "top_n":2,"return_documents":true}'

# Moderation: strict also flags the Controversial label.
curl --fail-with-body -sS localhost:8731/v1/moderations \
  -H 'Content-Type: application/json' -d '{
  "model":"qwen3guard-gen-0.6b","input":"How do I bake bread?","strict":true}'

# Small text model: streaming is optional; no thinking, vision or tool calls.
curl --fail-with-body -sSN localhost:8731/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
  "model":"qwen3.5-2b","stream":true,"max_tokens":128,
  "messages":[{"role":"user","content":"Explain what a local inference router does in three sentences."}]}'
```

The first four models accept up to 4096 input tokens. `qwen3.5-2b` accepts
16384 prompt tokens, with 18432 total positions. Decisions choose a schema
option without sampling. Moderation reports labels and probabilities rather
than meaningful OpenAI category scores. NPU and GPU work share the chip's
memory bandwidth and power budget. See the
[NPU API and limits](https://github.com/peonist-ai/halogen-flash-server/blob/v0.17.2/docs/NPU.md)
for supported request options.

### Own NPU fine-tunes and advanced flags

Fine-tunes of the first four models are supported. Put `config.json`,
`model.safetensors` and `tokenizer.json` in a directory accessible through
a profile's `/models` mount or an additional mount. Enter the container path
in NPU Setup, for example `/models/my-guard`. Use a distinct directory basename:
it becomes the API model ID. For all-profile setup, that path must be available
in every selected profile.

Setup probes the actual container engine before activation and installs the
required base device programs. Fine-tune sources receive separate read-only
mounts, so conversion writes temporary files inside the container and runs
again at each start. The built-in shared files remain unchanged. A generation
or image fine-tune is not supported by upstream 0.17.2.

The profile editor exposes `HALOGEN_NPU_MODELS`, `HALOGEN_NPU_QUEUE` (default
64), `HALOGEN_NPU_PORT` (internal, default 8740), `HALOGEN_NPU_EMB_BATCH`,
`HALOGEN_NPU_VERIFY`, `HALOGEN_CACHE_EVICT`, `HALOGEN_MTP` and
`HALOGEN_PREFILL_CANCEL`. Boolean selectors preserve explicit `0`; an empty
value uses the engine's default. Use NPU Setup to provision files and mounts;
editing the model list alone does not install models. Keep file verification
enabled unless diagnosing a specific upstream issue.

The System view shows cache hits, evictions and replaced entries (`dropped`).
`HALOGEN_CACHE_EVICT=0` selects the earlier eviction policy; leaving it empty
uses the improved upstream default. These engine flags do not change the
selected GPU checkpoint.

Under **Configure a profile manually → Engine scheduling & compatibility**,
`HALOGEN_ADMISSION_RESERVE=1` holds one of the GPU engine's slots for requests
that send `X-Halogen-Priority: 1`. Background requests wait for the other slots;
running requests are not interrupted. The engine keeps at least one background
slot, and the System view shows the reserve, active priority work and waiting
background requests. `/health` and `/metrics` expose the same counters.

The editor also exposes `HALOGEN_SCHEMA_ESCAPE`, `HALOGEN_CACHE_DISK_DEEPEN`,
`HALOGEN_PLE_PAR`, `HALOGEN_MTP_DEPTH`, `HALOGEN_ADMIT_TICKS`,
`HALOGEN_PREFILL_KEEP_TRUNK` and `HALOGEN_REPETITION_PENALTY`. Empty fields retain
upstream defaults; explicit `0` and `off` survive edits and updates. Keeping
the unpacked trunk costs about 5.5 GiB; a repetition penalty other than 1 needs
sampling (`temperature > 0`). `reasoning_effort=max` is accepted as an alias
of `xhigh`.

## Updating Halobridge itself

Under **System → Halobridge updates**, the dashboard checks GitHub every six
hours and offers an update only when the package version increases. Checks
follow the branch recorded by your installation; use `app_updates.branch` to
choose a different branch. Commit changes without a version bump do not trigger
an update.

For a GitHub installation managed by **pipx** and running as a Linux systemd
**user service**, click **Update now**. Halobridge waits for active requests,
reinstalls the package from its trusted GitHub source, verifies the new version,
and restarts its own service. A separate background service runs the update,
so it survives the restart. The dashboard reconnects and reloads automatically.
Failed installations restore the previous environment; status and errors remain
available after a reload.

If verification fails after restarting, the dashboard requires manual recovery.
**Update details** shows the saved environment's path; its `venv` subdirectory
contains the previous installation. Stop the Halobridge user service before
restoring it, then restart and verify `halobridge --version`. After confirming
the restored service works, remove `app-update.json` from your configured
dashboard state directory to clear the recovery status.
Restart the user service once more after clearing that journal.

```toml
[app_updates]
enabled = true
check_interval_h = 6
branch = ""                    # auto-detect the installed branch
service = "halobridge.service"  # systemd user service running this instance
```

Set `app_updates.enabled = false` to disable these checks. The existing
`security.allow_install = false` also disables Halobridge installation. Other
installation methods display a manual update command when available. You can
still update from the terminal:

**Installed with pipx:**

```bash
pipx upgrade halobridge
```

**Installed from a git clone (`pip install -e .`):**

```bash
cd Halobridge
git pull
```

Editable installs pick up the new code immediately — no reinstall needed.

**Installed with plain pip from git:**

```bash
pip install -U git+https://github.com/Siyarbekir47/Halobridge.git
```

Then restart and verify:

```bash
systemctl --user restart halobridge.service   # only if you run it as a service
halobridge --version
halobridge doctor
```

`halobridge --version` reads the installed package metadata, so it always
shows the version you actually have running.

> **For maintainers:** when publishing a new release, bump `version` in
> `pyproject.toml`, add short English bullets to `CHANGELOG.md`, and tag the
> commit `vX.Y.Z` on GitHub.

## Token gate

If you expose the dashboard beyond localhost, configure a token:

```toml
[security]
auth_token = "replace-me"
```

Then:

- `/dashboard` requires login.
- Dashboard APIs require the session cookie or a bearer token.
- Update actions still require same-origin JSON and the action header.

Recommended bearer usage:

```bash
curl -H "Authorization: Bearer replace-me" \
  http://127.0.0.1:8731/router/status
```

Generate a strong token:

```bash
openssl rand -hex 32
```

## systemd user service

A template unit is provided in:

```text
systemd/halobridge.service
```

Install it for your user:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/halobridge.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now halobridge.service
```

Make sure the `halobridge` command is available to systemd. If you use `pipx`,
it is usually in `~/.local/bin`.

## CLI

```bash
halobridge --help
halobridge --version
halobridge --check-config
halobridge doctor
halobridge router
halobridge dashboard
halobridge update-check
halobridge npu-check
halobridge npu-host-setup
```

## Development

Run tests:

```bash
python -B -m unittest discover -s tests -v
```

After updating to Halogen 0.17.2 and enabling the five text NPU models on a Linux host, run the opt-in hardware
smoke test from a checkout:

```bash
python tests/npu_smoke.py --url http://127.0.0.1:8731
# Include Flux image generation after enabling its model:
python tests/npu_smoke.py --images
# Optional GPU switches; the script restores the initially active model:
python tests/npu_smoke.py --gpu-switches qwen3.8-flash halogen-swift15-abliterated
```

Set `HALOBRIDGE_TOKEN` if authentication is enabled. This test sends small
inference requests, checks the text tasks, System One and streaming, and
verifies that NPU requests leave the active GPU model unchanged. It does not
install models or alter host settings. Hardware smoke tests must be run on the
target machine; the automated suite simulates XRT, Podman and systemd.

Optional real-browser dashboard check:

```bash
python tests/browser_check.py --browser "/path/to/chrome"
```

For manual UI checks without Podman or a model backend, run
`python tests/preview_dashboard.py` and open `http://127.0.0.1:8732/dashboard`.
This fixture uses synthetic, in-memory telemetry and supports profile previews;
it does not deploy models or modify system services.

Dashboard markup, styles and behavior live in `src/halobridge_data/dashboard.html`,
`dashboard.css` and `dashboard.js`. The small `theme.js` script applies the saved
appearance before the stylesheet loads. UI translations are in `locales/en.json` and
`locales/de.json`. Python messages use English source text; add their German
translations to `locales/server.de.json`, using matching `{p0}`, `{p1}` placeholders
for dynamic values. Response localization leaves profile data and shared job state
unchanged, so each browser can select its own language.

## Security notes

- Default bind is `127.0.0.1`.
- Do not expose update controls without a token and a trusted network.
- Use HTTPS via a reverse proxy if the dashboard is reachable over a network.
- Keep Quadlet directories owned by the user running Halobridge.
- Do not put secrets in query strings.

## Credits

- [Peonist (peonist-ai)](https://github.com/peonist-ai/halogen-flash-server)
  for the Halogen engine, official checkpoints, shared model assets and NPU
  models and host tools. Host tools are fetched unmodified from upstream and
  remain subject to its [license](https://github.com/peonist-ai/halogen-flash-server/blob/v0.17.2/LICENSE.md).
- [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
  for FLUX.2-klein-4B, used by the NPU image profile.
- [UkisAI](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next)
  for the original Swift 1.5 fine-tune.
- [Quat3rnion](https://huggingface.co/Quat3rnion/halogen-swift1.5-qwen3.8-flash-next-v2)
  for the Halogen builds of Swift 1.5 and Swift 1.5 Abliterated.
- **johnlockejrr** for providing diagnostics that helped bring Ubuntu support.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
