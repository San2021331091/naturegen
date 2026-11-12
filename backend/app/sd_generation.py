"""Local Stable Diffusion inference used by the Create page."""
import json
import gc
import os
import threading
import time
import uuid

import torch
from PIL import Image

from . import storage
from .config import MODEL_CACHE, RESULTS, SD_LORA_DIR, SD_MODEL_ID


_pipeline = None
_pipeline_lock = threading.Lock()
_generation_lock = threading.Lock()
_lora_loaded = False
_lora_mtime = None
LORA_NAME = 'naturegen'
_requested_device = os.getenv('DEVICE', 'auto')
_device = ('cuda' if torch.cuda.is_available() else 'cpu') if _requested_device == 'auto' else _requested_device


def device():
    return _device


def is_loaded():
    return _pipeline is not None


def unload_pipeline():
    """Release the image pipeline before loading the larger video pipeline."""
    global _pipeline, _lora_loaded, _lora_mtime
    with _pipeline_lock:
        pipeline = _pipeline
        _pipeline = None
        _lora_loaded = False
        _lora_mtime = None
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def lora_status():
    weights_path = SD_LORA_DIR / 'pytorch_lora_weights.safetensors'
    metadata_path = SD_LORA_DIR / 'metadata.json'
    if not weights_path.is_file():
        return {'available': False, 'steps': 0, 'images_seen': 0, 'providers': [], 'trained_at': None}
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        metadata = {}
    return {'available': True, 'steps': metadata.get('steps', 0), 'images_seen': metadata.get('images_seen', 0),
            'providers': metadata.get('providers', []), 'trained_at': metadata.get('trained_at')}


def _load_lora(pipeline):
    global _lora_loaded, _lora_mtime
    weights_path = SD_LORA_DIR / 'pytorch_lora_weights.safetensors'
    modified = weights_path.stat().st_mtime_ns if weights_path.is_file() else None
    if modified == _lora_mtime and _lora_loaded:
        return
    if _lora_loaded:
        pipeline.unload_lora_weights()
        _lora_loaded = False
    if modified is None:
        _lora_mtime = None
        return
    try:
        pipeline.load_lora_weights(str(SD_LORA_DIR), weight_name=weights_path.name, adapter_name=LORA_NAME)
    except (OSError, ValueError, RuntimeError) as error:
        raise ValueError('The saved Tiny-SD LoRA adapter could not be loaded. Fine-tune it again from Training.') from error
    _lora_loaded = True
    _lora_mtime = modified


def prepare_lora_training(pipeline):
    """Detach the active inference adapter before starting a fresh fine-tune."""
    global _lora_loaded, _lora_mtime
    if _lora_loaded:
        pipeline.unload_lora_weights()
    _lora_loaded = False
    _lora_mtime = None
    return pipeline


def mark_lora_loaded():
    global _lora_loaded, _lora_mtime
    weights_path = SD_LORA_DIR / 'pytorch_lora_weights.safetensors'
    _lora_loaded = weights_path.is_file()
    _lora_mtime = weights_path.stat().st_mtime_ns if _lora_loaded else None


def restore_lora(pipeline):
    """Restore the last committed adapter if fine-tuning exits before saving."""
    global _lora_loaded, _lora_mtime
    if _lora_loaded or LORA_NAME in getattr(pipeline.unet, 'peft_config', {}):
        try:
            pipeline.unload_lora_weights()
        except (AttributeError, RuntimeError, ValueError):
            pass
    _lora_loaded = False
    _lora_mtime = None
    _load_lora(pipeline)


def _load_pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        try:
            from diffusers import DiffusionPipeline
        except ImportError as error:
            raise ValueError(
                'Stable Diffusion packages are missing. Install backend/requirements.txt and restart the backend.'
            ) from error

        if _device == 'cpu':
            thread_count = max(1, min(os.cpu_count() or 2, int(os.getenv('TORCH_NUM_THREADS', '4'))))
            torch.set_num_threads(thread_count)
        try:
            pipeline = DiffusionPipeline.from_pretrained(
                SD_MODEL_ID,
                torch_dtype=torch.float16 if _device == 'cuda' else torch.float32,
                use_safetensors=False,
                cache_dir=str(MODEL_CACHE),
            )
        except OSError as error:
            raise ValueError('Could not download Tiny-SD model files from Hugging Face. Check the connection and retry; the first download is about 1 GB.') from error
        pipeline.to(_device)
        if hasattr(pipeline, 'set_progress_bar_config'):
            pipeline.set_progress_bar_config(disable=True)
        _pipeline = pipeline
        _load_lora(_pipeline)
        return _pipeline


def generate(request, update, cancelled):
    steps = request['steps']
    update(current=0, total=steps, phase='loading-model')
    with _generation_lock:
        if cancelled():
            raise InterruptedError('Generation cancelled.')
        pipeline = _load_pipeline()
        _load_lora(pipeline)
        if cancelled():
            raise InterruptedError('Generation cancelled.')

        generator = torch.Generator(device=_device).manual_seed(request['seed'])

        def report_progress(_pipeline, step_index, _timestep, callback_kwargs):
            update(current=step_index + 1, total=steps, phase='denoising')
            if cancelled():
                raise InterruptedError('Generation cancelled.')
            return callback_kwargs

        output = pipeline(
            prompt=request['prompt'],
            negative_prompt='blurry, low resolution, low quality, oversaturated, distorted, deformed',
            height=512,
            width=512,
            num_inference_steps=steps,
            guidance_scale=request['guidance'],
            generator=generator,
            callback_on_step_end=report_progress,
        )

    if cancelled():
        raise InterruptedError('Generation cancelled.')

    key = uuid.uuid4().hex
    image = output.images[0].convert('RGB')
    image.save(RESULTS / f'{key}.png')
    result = {
        **request,
        'sampler': 'default',
        'id': key,
        'image_url': f'/api/generated/{key}.png',
        'created_at': time.time(),
        'model_id': SD_MODEL_ID,
        'device': _device,
        'experimental': False,
    }
    storage.put('generation', result)
    return result
