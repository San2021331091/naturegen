"""Free Hugging Face Gradio Space entry point for NatureGen."""
import os

# Keep the requested CPU-friendly inference path even when ZeroGPU hardware is selected.
os.environ.setdefault('DEVICE', 'cpu')
os.environ.setdefault('TORCH_NUM_THREADS', '2')

# ZeroGPU requires a decorated entry point. NatureGen inference intentionally stays on CPU.
try:
    import spaces
except ImportError:  # Lets the same UI run locally outside Spaces.
    class spaces:
        @staticmethod
        def GPU(*_args, **_kwargs):
            return lambda function: function

@spaces.GPU(duration=1)
def _zerogpu_compatibility_hook():
    """ZeroGPU compatibility hook; all model inference remains on CPU."""
    return None

import gradio as gr
from PIL import Image

from backend.app import sd_generation, sd_video
from backend.app.config import RESULTS, SD_MODEL_ID
from backend.app.schemas import PromptIdeasRequest


def _progress(progress, current, total, phase):
    if total:
        progress(min(1.0, current / total), desc=phase.replace('-', ' ').capitalize())


def generate_image(prompt, seed, steps, guidance, progress=gr.Progress()):
    if not prompt or len(prompt.strip()) < 3:
        raise gr.Error('Enter a nature prompt with at least 3 characters.')
    seed = int(seed)
    request = {
        'prompt': prompt.strip(), 'seed': seed, 'steps': int(steps),
        'guidance': float(guidance), 'resolution': 512,
    }
    result = sd_generation.generate(
        request,
        lambda **state: _progress(progress, state.get('current', 0), state.get('total', 0), state.get('phase', 'generating')),
        lambda: False,
    )
    image_path = RESULTS / f"{result['id']}.png"
    with Image.open(image_path) as image:
        output = image.copy()
    return output, f"Generated locally on {sd_generation.device()} · {SD_MODEL_ID} · seed {seed}"


def generate_video(prompt, seed, scene_count, transition, progress=gr.Progress()):
    if not prompt or len(prompt.strip()) < 3:
        raise gr.Error('Enter a nature prompt with at least 3 characters.')
    seed = int(seed)
    result = sd_video.generate(
        {
            'prompt': prompt.strip(), 'seed': seed, 'scene_count': int(scene_count),
            'transition': transition, 'transition_sound': True,
        },
        lambda **state: _progress(progress, state.get('current', 0), state.get('total', 0), state.get('phase', 'generating')),
        lambda: False,
    )
    path = RESULTS / f"{result['id']}.mp4"
    duration = result['frames'] / result['fps']
    return str(path), f"{duration:.0f}s · {result['motion_type']} · Sound: {result['audio']['description']} · CPU"


def suggest_prompt(mode):
    from backend.app.main import generate_prompt_ideas
    try:
        result = generate_prompt_ideas(PromptIdeasRequest(mode=mode, count=1))
    except Exception as error:
        detail = getattr(error, 'detail', None)
        raise gr.Error(detail or str(error)) from error
    return result['prompts'][0]


with gr.Blocks(title='NatureGen AI') as demo:
    gr.Markdown('# NatureGen AI\nCreate nature images and short clips locally with Tiny-SD. Video motion is selected from each image using SegFormer B0. Inference runs on CPU.')
    with gr.Tabs():
        with gr.Tab('Image'):
            image_prompt = gr.Textbox(label='Nature prompt', lines=3, placeholder='Describe a landscape, plant, water, weather, or geological scene.')
            image_ideas = gr.Button('Suggest a fresh nature prompt')
            image_ideas.click(lambda: suggest_prompt('image'), outputs=image_prompt)
            with gr.Row():
                image_seed = gr.Number(label='Seed', value=42, precision=0)
                image_steps = gr.Slider(label='Steps', minimum=4, maximum=30, step=1, value=20)
                image_guidance = gr.Slider(label='Guidance', minimum=1, maximum=12, step=0.5, value=7.5)
            image_button = gr.Button('Generate image', variant='primary')
            image_output = gr.Image(label='Generated image', type='pil')
            image_info = gr.Markdown()
            image_button.click(generate_image, [image_prompt, image_seed, image_steps, image_guidance], [image_output, image_info])
        with gr.Tab('Video'):
            video_prompt = gr.Textbox(label='Nature prompt', lines=3, placeholder='Describe a nature scene with visible details that can move.')
            video_ideas = gr.Button('Suggest a fresh nature prompt')
            video_ideas.click(lambda: suggest_prompt('video'), outputs=video_prompt)
            with gr.Row():
                video_seed = gr.Number(label='Seed', value=42, precision=0)
                video_scenes = gr.Slider(label='Images in clip', minimum=3, maximum=6, step=1, value=3)
                video_transition = gr.Dropdown(label='Transition', choices=['crossfade', 'slide'], value='crossfade')
            gr.Markdown('Each source image gets localized motion based on detected regions, then the images are joined with transitions. CPU generation can take several minutes.')
            video_button = gr.Button('Generate video clip', variant='primary')
            video_output = gr.Video(label='Generated clip')
            video_info = gr.Markdown()
            video_button.click(generate_video, [video_prompt, video_seed, video_scenes, video_transition], [video_output, video_info])

demo.queue(default_concurrency_limit=1).launch(
    server_name='0.0.0.0', server_port=7860,
    allowed_paths=[str(RESULTS)],
)
