# Changelog

## 0.1.26

- Add shared, verified Flux NPU image generation setup.
- Route System One questions without switching GPU models.
- Track image, decision and Anthropic Messages requests correctly.
- Expose priority slots, cache, schema and decode settings.
- Use Halogen 0.17.2 for new Official and Orca profiles.
- Document endpoints, examples and hardware requirements.

## 0.1.25

- Add NPU setup for decisions, embeddings, reranking, moderation and text generation.
- Share verified, versioned NPU files across backend profiles.
- Prepare NPU updates before downtime; recover interrupted activation.
- Support custom NPU fine-tunes and host diagnostics.
- Expose NPU and engine flags; show prompt-cache eviction counters.
- Use Halogen 0.16.2 for new Official and Orca profiles.
- Route NPU requests without switching GPU models; include all tasks in history.

## 0.1.24

- Add optional HT43 checkpoint selection for Halogen 0.16.0+.
- Keep v2 as default and preserve checkpoints during engine updates.
- Use Halogen 0.16.0 for new Official and Orca templates.
- Prepare pinned, verified checkpoints before switching; reuse shared assets.
- Add download plans, checkpoint repair and rollback to the previous profile.

## 0.1.23

- Add Swift 1.5 and Swift 1.5 Abliterated quick templates with vision.
- Share verified N-Gram, tokenizer and vision files across model profiles.
- Add download previews, resumable installs and activation recovery.
- Refresh API model discovery without restarting Halobridge.
- Document Ubuntu and Fedora 44 support.
- Credit model creators and Ubuntu diagnostics contributor johnlockejrr.

## 0.1.22

- Limit checkpoint v2 upgrades and recovery to the official model profile.
- Prevent checkpoint upgrades during model switches.

## 0.1.21

- Keep token usage notices dismissed until a new incomplete request is recorded.

## 0.1.20

- Add system-aware dark mode with manual theme selection.
- Make the incomplete token usage notice dismissible.
- Add Halobridge version checks and dashboard updates for pipx installs.

## 0.1.19

- Add official checkpoint v2 upgrades.
