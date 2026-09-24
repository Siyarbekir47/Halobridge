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
- Can be protected with a token gate for non-local access.

## What it does not do

- It is not an official Peonist product.
- It does not store prompts, responses, images, tool contents, or secrets.
- It does not invent token counts when the API does not report usage.
- It does not update to `latest`, release candidates, or non-stable tags.
- It does not delete model or cache directories; deployment only writes,
  backs up, and removes Quadlet files.

## Requirements

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

### 3. Install the official model

In the dashboard:

1. Open **Models**.
2. Leave **Official model** selected.
3. Click **Install official model** and confirm.
4. Keep Halobridge running while the job finishes.

Halobridge creates the Quadlet, pulls the pinned official container image,
downloads the model on first start and streams the progress into the dashboard.
The download is about 118 GiB and resumes if interrupted. The tokenizer,
quality overlay and vision tower come from the official weights repository;
image input is enabled by default. Later starts reuse the files on disk.

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

- **Official model** — one click installs the official profile and starts it.
  The first start downloads the upstream weights (~118 GiB) through
  `HALOGEN_DOWNLOAD`; later starts are offline. The downloaded vision tower is
  enabled by default, so the resulting backend accepts image inputs. If a
  profile already exists, the existing one is started instead of being
  overwritten.
- **Uncensored model** — paste a Hugging Face read token and click once.
  Before the first download, sign in to Hugging Face, open the
  [gated OrcaRouter repository](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF),
  accept its terms or request access, and wait for approval. Then create a
  READ token from the same account. A valid token alone is not enough if that
  account has not been granted repository access. Halobridge downloads the
  uncensored GGUF, the draft head and tokenizer (while the current model keeps
  serving), then stops the active backend, converts to `.hgn` with the GPU to
  itself, applies the profile and starts uncensored — all as one streaming job
  with visible steps. If the converted `.hgn` already exists, the
  download/convert steps are skipped and no token is required.

Both buttons ask for a confirmation click (labelled "Stop active model &
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

For the standard official model, leave the checkpoint field empty. Set it only
when deliberately running a custom or converted checkpoint.

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

You can disable updates:

```toml
[updates]
enabled = false
```

You can keep the dashboard but disable installation:

```toml
[security]
allow_install = false
```

## Updating Halobridge itself

The dashboard's one-click update handles the **halogen container**. To update
the **Halobridge package itself**, use the command matching how you installed it:

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
> `pyproject.toml` and tag the commit `vX.Y.Z` on GitHub.

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
```

## Development

Run tests:

```bash
python -B -m unittest discover -s tests -v
```

Optional real-browser dashboard check:

```bash
python tests/browser_check.py --browser "/path/to/chrome"
```

For manual UI checks without Podman or a model backend, run
`python tests/preview_dashboard.py` and open `http://127.0.0.1:8732/dashboard`.
This fixture uses synthetic, in-memory telemetry and supports profile previews;
it does not deploy models or modify system services.

Dashboard markup, styles and behavior live in `src/halobridge_data/dashboard.html`,
`dashboard.css` and `dashboard.js`. UI translations are in `locales/en.json` and
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

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
