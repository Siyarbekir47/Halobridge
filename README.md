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
- Shows request counts, input/output tokens, cache ratio, latency, and history.
- Uses only API-reported `usage` values for token accounting.
- Provides a safe update flow for the official Halogen container image.
- Can be protected with a token gate for non-local access.

## What it does not do

- It is not an official Peonist product.
- It does not store prompts, responses, images, tool contents, or secrets.
- It does not invent token counts when the API does not report usage.
- It does not update to `latest`, release candidates, or non-stable tags.

## Requirements

- Linux host with Podman and systemd user services
- Python 3.11+
- One or more Halogen model services defined as Podman Quadlets
- A reachable Halogen backend on the configured backend URL

## Quick start

```bash
git clone https://github.com/Siyarbekir47/Halobridge.git
cd Halobridge
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create a config (optional — safe localhost defaults work without one):

```bash
mkdir -p ~/.config/halobridge
cp config.example.toml ~/.config/halobridge/halobridge.toml
$EDITOR ~/.config/halobridge/halobridge.toml
```

Check the setup:

```bash
halobridge doctor
```

Start the router and dashboard:

```bash
halobridge router
```

Open:

```text
http://127.0.0.1:8731/dashboard
```

### 60-second version

If everything is already installed and configured, this is all you need:

```bash
pipx install git+https://github.com/Siyarbekir47/Halobridge.git
halobridge doctor   # shows what is ready and what is missing
halobridge router   # starts API + dashboard on http://127.0.0.1:8731
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

## Security notes

- Default bind is `127.0.0.1`.
- Do not expose update controls without a token and a trusted network.
- Use HTTPS via a reverse proxy if the dashboard is reachable over a network.
- Keep Quadlet directories owned by the user running Halobridge.
- Do not put secrets in query strings.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
