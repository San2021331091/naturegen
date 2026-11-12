"""Hugging Face Space entry point: the NatureGen API under /api plus a small Gradio studio at /.

Local development does not use this file; run uvicorn on app.main:app as before.
"""
import os
import time
from contextlib import asynccontextmanager

os.environ.setdefault('DEVICE', 'cpu')
os.environ.setdefault('TORCH_NUM_THREADS', '2')

# ZeroGPU hardware refuses to start unless one function is decorated with @spaces.GPU.
# Generation stays on CPU; this hook only satisfies that startup check.
try:
    import spaces
except ImportError:  # Not on ZeroGPU hardware (local runs, CPU Spaces).
    spaces = None

if spaces is not None:
    @spaces.GPU(duration=1)
    def _zerogpu_compatibility_hook():
        return None

import gradio as gr
from gradio import Server
from pydantic import ValidationError

from app import main as api
from app import storage
from app.config import RESULTS
from app.schemas import GenerationRequest, PromptIdeasRequest, VideoGenerationRequest

api_app = api.app


@asynccontextmanager
async def lifespan(_server):
    async with api_app.router.lifespan_context(api_app):
        yield


app = Server(
    title=api_app.title,
    version=api_app.version,
    description='NatureGen image and video generation API.',
)

# Reuse the existing API endpoints, security middleware, and error handlers.
app.include_router(api_app.router)
for middleware in reversed(api_app.user_middleware):
    app.add_middleware(middleware.cls, *middleware.args, **middleware.kwargs)
for exception, handler in api_app.exception_handlers.items():
    app.add_exception_handler(exception, handler)


def _run(start, request_type, fields, progress):
    """Queue a job through the same path the API uses, then wait for it and report progress."""
    try:
        job = start(request_type(**fields))
    except ValidationError as error:
        raise gr.Error(error.errors()[0]['msg']) from error
    except Exception as error:
        raise gr.Error(str(getattr(error, 'detail', None) or error)) from error
    while True:
        job = storage.get('job', job['id']) or job
        state = job.get('state')
        if state == 'completed':
            return job['result']
        if state in ('failed', 'cancelled', 'stopped'):
            raise gr.Error(job.get('error') or f'Generation {state}.')
        total = job.get('total') or 0
        phase = str(job.get('phase') or state or 'working').replace('-', ' ').capitalize()
        progress(min(1.0, job.get('current', 0) / total) if total else 0, desc=phase)
        time.sleep(1)


def generate_image(prompt, seed, steps, guidance, progress=gr.Progress()):
    fields = {'prompt': (prompt or '').strip(), 'seed': int(seed), 'steps': int(steps), 'guidance': float(guidance)}
    result = _run(api.generation, GenerationRequest, fields, progress)
    return str(RESULTS / f"{result['id']}.png"), f"Seed {result['seed']} · {result['steps']} steps · CPU"


def generate_video(prompt, seed, scene_count, transition, progress=gr.Progress()):
    fields = {'prompt': (prompt or '').strip(), 'seed': int(seed), 'scene_count': int(scene_count), 'transition': transition}
    result = _run(api.video_generation, VideoGenerationRequest, fields, progress)
    return str(RESULTS / f"{result['id']}.mp4"), f"{result['frames'] / result['fps']:.0f}s clip · seed {result['seed']} · CPU"


def suggest_prompt(mode):
    try:
        return api.generate_prompt_ideas(PromptIdeasRequest(mode=mode, count=1))['prompts'][0]
    except Exception as error:
        raise gr.Error(str(getattr(error, 'detail', None) or error)) from error


with gr.Blocks(title='NatureGen AI') as demo:
    gr.Markdown(
        '# NatureGen AI\n'
        'Create nature images and short clips. Generation runs on CPU, so an image takes a few minutes '
        'and the first one after a restart also downloads the model.'
    )
    with gr.Tabs():
        with gr.Tab('Image'):
            image_prompt = gr.Textbox(label='Nature prompt', lines=3, placeholder='Describe a landscape, plant, water, weather, or geological scene.')
            image_idea = gr.Button('Suggest a prompt')
            with gr.Row():
                image_seed = gr.Number(label='Seed', value=42, precision=0)
                image_steps = gr.Slider(label='Steps', minimum=4, maximum=50, step=1, value=20)
                image_guidance = gr.Slider(label='Guidance', minimum=1, maximum=12, step=0.5, value=7.5)
            image_button = gr.Button('Generate image', variant='primary')
            image_output = gr.Image(label='Generated image', type='filepath')
            image_info = gr.Markdown()
            image_idea.click(lambda: suggest_prompt('image'), outputs=image_prompt)
            image_button.click(generate_image, [image_prompt, image_seed, image_steps, image_guidance], [image_output, image_info])
        with gr.Tab('Video'):
            video_prompt = gr.Textbox(label='Nature prompt', lines=3, placeholder='Describe a nature scene with visible details that can move.')
            video_idea = gr.Button('Suggest a prompt')
            with gr.Row():
                video_seed = gr.Number(label='Seed', value=42, precision=0)
                video_scenes = gr.Slider(label='Images in clip', minimum=3, maximum=6, step=1, value=3)
                video_transition = gr.Dropdown(label='Transition', choices=['crossfade', 'slide'], value='crossfade')
            video_button = gr.Button('Generate video clip', variant='primary')
            video_output = gr.Video(label='Generated clip')
            video_info = gr.Markdown()
            video_idea.click(lambda: suggest_prompt('video'), outputs=video_prompt)
            video_button.click(generate_video, [video_prompt, video_seed, video_scenes, video_transition], [video_output, video_info])
    gr.Markdown('API health: `/api/health` · API docs: `/docs`')

demo.queue(default_concurrency_limit=1)


if __name__ == '__main__':
    # Gradio serves the page at / on the same app that already holds the /api routes.
    # No Node front-end: Python answers on 7860 directly.
    demo.launch(
        _app=app,
        app_kwargs={'lifespan': lifespan},
        server_name='0.0.0.0',
        server_port=7860,
        ssr_mode=False,
        allowed_paths=[str(RESULTS)],
    )
