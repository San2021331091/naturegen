---
title: NatureGen API
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
python_version: '3.12'
app_file: space.py
pinned: false
---

# NatureGen API

FastAPI backend for the NatureGen studio: Tiny-SD image generation and short nature videos on CPU.
The web interface is hosted separately and calls this Space over HTTPS.

Set these under **Settings → Variables and secrets**:

| Name | Kind | Value |
| --- | --- | --- |
| `FRONTEND_ORIGIN` | Variable | The frontend's URL, e.g. `https://your-app.vercel.app` (comma-separate several) |
| `OPENROUTER_API_KEY` | Secret | Optional, enables prompt ideas |

Health check: `/api/health`. API docs: `/docs`.

Storage is not persistent on the free tier: generated files and history are cleared when the Space restarts.
