# NatureGen AI

A modern and new nature image/video studio with a Next.js dashboard and FastAPI backend. Image generation uses the local, pretrained `segmind/tiny-sd` Stable Diffusion model at 512x512. Short video generation uses the same Tiny-SD model and photo LoRA, then uses a small SegFormer model to animate water, sky, and foliage regions while keeping the camera fixed. A separate from-scratch diffusion model remains available for training experiments; its checkpoint does not power the Create page.

## What is built

- Responsive overview, generation, training, dataset, history, and settings pages; typed Axios client and Zustand state. Polling runs only while jobs are active.
- SQLite metadata/history/job storage; bounded background workers; cancellation and restart recovery; one training/generation job at a time.
- Licensed pair import, small metadata collection batches, review workflows, lexical matching evidence, availability validation, transient image hashing, and duplicate-group train/validation/test splits.
- HTTPS image loader with exact host allowlists, public DNS validation, pinned TLS connections, redirect revalidation, byte/decode limits, bounded retry/backoff, rate control, and bounded RAM caching.
- Local Stable Diffusion inference through Hugging Face Diffusers, with a fixed 512x512 output, seeded sampling, CPU support, progress reporting, and cached model files.
- Videos from 3–6 freshly generated Tiny-SD images, joined with transitions at 256x256 and 8 FPS, with optional transition chimes.
- Optional Tiny-SD U-Net LoRA fine-tuning from permitted Pexels/Unsplash photo-caption streams; Create applies the saved adapter automatically.
- Optional from-scratch research path: custom BPE and Transformer, multiscale residual U-Net, manually implemented diffusion, joint training, EMA, validation, and resumable checkpoints.
- Fixed-prompt/seed evaluation-grid command and automated numerical, security, storage, API, and synthetic overfit tests.

**Current local experiment:** the optional from-scratch checkpoint received only two optimizer updates and produces unstructured 64×64 noise. It is not used by the Create page. Stable Diffusion downloads its model files on first generation (about 1.06 GB for Tiny-SD) and caches them under `backend/models/`. The decorative landscape in the overview is CSS artwork, not model output.

## Deploy to Hugging Face Spaces (free ZeroGPU)

The Space runs the FastAPI backend from `backend/space.py`; the Next.js studio is deployed separately. Tiny-SD image and video generation stays on CPU. Model weights download on first use. Space storage is ephemeral, so generated images, videos, history, and LoRA data are not durable across restarts.

Create a **Gradio Space** with the Blank template and select **ZeroGPU**. Keep it Private; the Vercel server proxy authenticates using a read token stored as a server-side secret. After creating it, open PowerShell in this project folder and run these commands, replacing the owner and Space name:

```powershell
git init
git add .
git commit -m "Deploy NatureGen Gradio Space"
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
git push -u space HEAD:main
```

Git may ask you to sign in to Hugging Face during the push. Then open Space **Settings → Variables and secrets** and add `OPENROUTER_API_KEY` as a secret for prompt suggestions. Never add credentials to Git. Model weights download on first use; CPU inference is slower than GPU, and generated files are removed when the Space restarts. See the [Spaces overview](https://huggingface.co/docs/hub/spaces-overview).

## Deploy the Next.js studio to Vercel

The Vercel app uses the existing Next.js project in `frontend/` and proxies selected public-demo endpoints to the private HF Space. The Hugging Face token stays server-side. The proxy allows image/video generation and status/media reads; dataset editing, training, and history deletion remain unavailable to public visitors.

1. Push this project to a Git provider Vercel supports, or install the Vercel CLI and deploy from `frontend/`.
2. Set the Vercel project Root Directory to `frontend` when importing the repository.
3. In Vercel **Settings → Environment Variables**, add `HF_TOKEN` with a Hugging Face access token that has **read** access to this private Space. Add `HF_SPACE_URL=https://sanloxai-springspace.hf.space` if the Space has a different URL.
4. Deploy. Production API calls use the same-origin Next.js proxy; no `NEXT_PUBLIC_` token is used.

For CLI deployment from PowerShell, run `npx vercel` inside `frontend`, follow the project linking prompts, add `HF_TOKEN` through the Vercel dashboard, then run `npx vercel --prod`. Do not paste the token into source files or commit it. Hugging Face documents read-token access for private Spaces in its [Spaces API guide](https://huggingface.co/docs/hub/spaces-api-endpoints).

## Run on Windows

Python 3.10+ and Node.js 20.9+ are required. The current workspace uses Python 3.13 and Node 24. Use `npm.cmd` if PowerShell blocks `npm.ps1`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
Copy-Item backend/.env.example backend/.env
Copy-Item frontend/.env.local.example frontend/.env.local
cd frontend
npm.cmd ci
```

Run `start-backend.cmd` and `start-frontend.cmd` in separate terminals. Open **http://localhost:3000**. API documentation: **http://localhost:8000/docs**. Backend health: **http://localhost:8000/api/health**.

For production-mode local frontend testing: `cd frontend`, `npm.cmd run build`, then `npm.cmd start`. On Linux/macOS use `.venv/bin/python`, `npm`, and `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000` from `backend`.

The backend is a single-user local tool. Bind it to loopback and run **one Uvicorn worker**. Public hosting requires authentication, shared job coordination, deployment secrets, and a real compute service; this repository does not publish the local ML backend to a static hosting service.

## Environment and troubleshooting

`backend/.env.example` lists all settings. API secrets remain server-side. Add exact approved image hostnames to `ALLOWED_IMAGE_HOSTS` and restart the backend. Wildcards, private addresses, plain HTTP, credentials in URLs, and nonstandard ports are refused. Do not add untrusted hosts merely to get an import through.

Prompt ideas use the OpenRouter chat completions API and are restricted to nature scenes. Unrelated starting ideas are adapted into natural subjects. Set `OPENROUTER_API_KEY` in `backend/.env`; `OPENROUTER_MODEL` optionally selects a model and defaults to `openrouter/auto`. The frontend calls the local backend, so the API key is never sent to the browser. Prompt ideas and any text entered as a starting idea are sent to OpenRouter under the account's configured model/provider settings.

Run `.\.venv\Scripts\python.exe scripts/inspect_environment.py` for hardware diagnostics. Windows PyTorch DLL errors are exposed in the health response without preventing dataset/history API use. This machine's system PyTorch initially failed with WinError 1114, and the latest isolated build hit application-control WinError 4551. Use a compatible PyTorch installation approved by the machine's application-control policy; do not disable that policy. See `docs/verification.md` for the final tested environment.

The Create page uses `segmind/tiny-sd`, a distilled 512×512 Stable Diffusion checkpoint. It runs in float32 on CPU, or float16 when CUDA is selected and available, and defaults to 20 denoising steps. CPU generation may take a few minutes depending on the processor. The first generation needs internet access to fetch about 1.06 GB of model files; later generations use the local cache. Set `NATUREGEN_SD_MODEL` in `backend/.env` to select a compatible Diffusers text-to-image model and `DEVICE=cpu` to force CPU use. CPU support means it runs without a GPU, not that inference is instantaneous.

## Generate with Stable Diffusion

Create works without collecting data or training a checkpoint. It uses [`segmind/tiny-sd`](https://huggingface.co/segmind/tiny-sd), a distilled Stable Diffusion model that produces native 512x512 images under its [CreativeML Open RAIL-M license](https://huggingface.co/segmind/tiny-sd). The first generation downloads about 1.06 GB of model files from Hugging Face into `backend/models/`; later generations use that local cache. The backend runs inference in float32 on CPU or float16 on CUDA and defaults to 20 denoising steps. On CPU, generation can take a few minutes depending on the processor. Set `NATUREGEN_SD_MODEL` in `backend/.env` to choose another compatible Diffusers text-to-image model.

The Training page has two distinct workflows. **Fine-tune Tiny-SD** streams permission-cleared 512x512 Pexels/Unsplash photos and provider captions into a small U-Net LoRA adapter. Create applies the saved adapter automatically. Source pixels remain in bounded RAM; creator, attribution URL, image URL, provider, ID, and caption are recorded alongside the adapter in `backend/checkpoints/tiny-sd-lora/`. CPU fine-tuning is much slower than inference and needs several GB of free memory. This requires provider credentials and documented ML permission (`*_ML_PERMISSION=true`).

The separate custom experiment remains 64x64 and trains the from-scratch model. Its checkpoints are not loaded by the Create page.

## Local video generation

The Create page's **Video** mode uses Tiny-SD and the saved photo LoRA to generate multiple 512x512 images, then exports their transitions as a 256x256 MP4 at 8 FPS.

Video mode generates 3–6 fresh images from the prompt, varying the composition and seed for each image. It joins them with crossfade or slide transitions and an optional short chime at each transition. Each image occupies four seconds of the timeline. No nature ambience is mixed into these videos. More images require more generation time.

## Dataset workflow

1. Obtain licensed, genuinely paired image-caption metadata. Record the source, creator, attribution, and applicable ML permission. Captions from an unrelated dataset must never be paired with search results as though they were ground truth.
2. Configure the exact image host allowlist. Import up to 100 metadata records per request via the Dataset page or `POST /api/dataset/import`.
3. Select **Validate URLs & split**. Source images are fetched into RAM, decoded, hashed and released or retained in the bounded RAM cache. No source images are written to disk. Generated outputs are stored separately.
4. Select **Analyze matching signals** for lexical evidence. Coverage is not a confidence probability and does not approve a pair.
5. Inspect the real image and caption together, add review notes, then approve or reject. Approval requires successful image validation.
6. Review split counts. At least two approved training pairs and one approved validation pair are needed even for a small real-data run. Near-duplicates and shared image identities are grouped before deterministic 80/10/10 assignment. Small datasets may not contain all splits; add more distinct reviewed images.

Import schema (shape only; supply your own real values):

```typescript
{
  records: Array<{
    caption: string; image_url: string;
    source: string; source_id: string; license_note: string;
    creator?: string; attribution_url?: string;
    category?: string; tags?: string[]; description?: string;
  }>
}
```

Collection is bounded to 30 metadata records per call (20 in the UI). Pexels and Unsplash adapters require API credentials **and documented explicit permission for ML use**, acknowledged by the corresponding `*_ML_PERMISSION=true` backend setting. This is not a way to grant yourself permission. Hugging Face collection reads bounded Conceptual Captions metadata pages, filters nature words, and rejects URLs outside the allowlist; it may legitimately import zero records. Use the API `offset` parameter to inspect another page. See `docs/dataset-research.md`.

Validation is explicit before approval; training rechecks URL safety on uncached fetches. Revalidate older records before a new experiment. Failed training fetches quarantine the pair; they never substitute another image. Deduplication uses a lightweight difference hash and review; it is a practical heuristic, not proof that every near-duplicate has been detected.

## Train, stop, resume, and generate

Use the Training page to start a deliberately small experiment. Set epochs, batch size, channel width, and learning rate. The API exposes text dimensions/layers, accumulation, timestep count, CFG dropout, EMA decay, and seed. Native training is initially 64×64 only; 128/256 training and resolution transfer are deferred research stages.

Training and generation return a job ID; inspect `/api/jobs/{id}`. Stop training to save at an optimizer boundary. Resume uses the saved configuration, ordering, next batch, and RNG state. Changing the dataset manifest requires a new experiment. An interrupted process leaves its last valid checkpoint. If `training.lock` remains after a hard crash, verify that no training process exists before manually removing that one lock file.

Checkpoints contain raw and EMA weights, optimizer/scheduler/scaler states, RNG states, tokenizer/checksum, model configuration, dataset manifest hash, run ID, step/epoch/cursor, and software metadata. They are loaded strictly with `weights_only=True`. Keep checkpoint files local and trusted. `latest.previous.pt` retains the previous save. Starting another experiment eventually replaces `latest.pt`; copy an important checkpoint to an archive before doing so.

The Create page supports prompt, seed, guidance, and 4-50 denoising steps, defaulting to 20 for CPU use. The model's configured scheduler is used. Generated PNGs and settings are persisted, downloadable, reusable, and deletable from History.

## Streaming training from Pexels and Unsplash

**Fine-tune Tiny-SD** (`POST /api/training/tiny-sd-lora`) streams full-resolution provider images and original alt captions, trains rank-4 LoRA weights on the Tiny-SD U-Net, and saves adapter weights plus source attribution metadata. Stop saves the latest complete optimizer step. The Create pipeline loads the adapter automatically on generation and after backend restart. Source image pixels are never written to disk. `buffer_size` sets how many distinct photo-caption pairs are held in the training pool (default 128, up to 1024); each pair is encoded once and kept as latents, about 0.4 MB of RAM per pair, and one new photo is swapped in every four steps. Raise `steps` along with it so each pair is visited several times. First use also requires the Tiny-SD base model download. Pexels/Unsplash credentials and documented explicit ML permission are required.

**Stream photos to the custom experiment** (`POST /api/training/stream`) still trains the from-scratch model without the review workflow. It resizes streamed photos to 64x64 and keeps a bounded rolling pool in memory. Its checkpoint format is separate from Tiny-SD and never affects Create.
## Evaluation

```powershell
cd backend
..\.venv\Scripts\python.exe -m app.evaluate --steps 50
```

This creates `results/evaluation-grid.png` and a matching JSON assessment sheet using four fixed prompts and three fixed seeds. It uses EMA weights and records native resolution explicitly; display enlargement is not super-resolution. Fill out ratings using `docs/evaluation.md`. No pretrained CLIP, Inception, FID, or other external feature models are used.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
.\.venv\Scripts\python.exe scripts/benchmark.py
cd frontend
npm.cmd run build
```

The synthetic overfit test verifies gradients and decreasing loss on generated test tensors only. It does not produce a deployable model or claim useful image quality.

## API map

| Purpose | Endpoints |
| --- | --- |
| Health/settings | `GET /api/health`, `/api/settings` |
| Dataset | `GET /api/dataset/stats`, `/api/dataset/records`; `POST /api/dataset/import`, `/collect`, `/validate`, `/match`, `/records/{id}/review` |
| Training | `POST /api/training/start`, `/stop`, `/resume`; `GET /api/training/status`, `/history` |
| Generation | `POST /api/prompts/generate`, `/api/generate`, `/api/generate-video`; `GET /api/generated/{filename}`, `/api/videos/{filename}` |
| Jobs | `GET /api/jobs`, `/api/jobs/{id}`; `POST /api/jobs/{id}/cancel` |
| History | `GET /api/history`, `/api/video-history`; `DELETE /api/history/{id}`, `/api/video-history/{id}` |


[Open Link](https://naturegen-gules.vercel.app/)
