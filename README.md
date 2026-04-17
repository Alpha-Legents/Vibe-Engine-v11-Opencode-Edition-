# ⚡ Vibe Engine v11 (Opencode Edition)

## What's this?

Vibe Engine v11 is a working local-first agent runtime that:

* Runs large language models via Ollama (Qwen, Mistral, etc.)
* Exposes an OpenAI-compatible API
* Enables tool calling (write, bash, edit, grep, etc.)
* Streams responses in real-time
* Auto-deploys via tunnel for remote access
* Handles unstable model outputs via multi-format parsing fallback

Basically: a self-hosted coding agent backend that actually works.
---
Try it out: [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/Alpha-Legents/Vibe-Engine-v11-Opencode-Edition-/blob/main/engine.ipynb)
---

## Notes

* Tunneled using localtunnel (yeah, I know Tailscale / Cloudflare exist — Kaggle hates them)
* This is the **last stable version** before major experimental changes (v12–v18)

---

## Experimental direction (v12–v18)

* Moving from Ollama → vLLM (performance + better tool handling)
* Testing sharper models (e.g. IQuest Coder v1)
* Tried shifting to Claude Code… (didn’t go well..at all)

---

## Contribute / Fork

If this looks interesting:

> Fork → break it → improve it → push

---

## Why I built this

Didn’t want to rely on API costs or external compute. So I built my own.
