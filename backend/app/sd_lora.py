"""LoRA fine-tuning of Tiny-SD from permitted, streamed photo-caption pairs."""
import json
import os
import queue
import random
import shutil
import threading
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from . import sd_generation
from .config import CHECKPOINTS, SD_LORA_DIR, SD_MODEL_ID
from .dataset import loader
from .stream import PER_PAGE, QUERIES, ImageStream
from .training import locked


WEIGHTS_NAME = 'pytorch_lora_weights.safetensors'
ADAPTER_NAME = sd_generation.LORA_NAME
LORA_TARGETS = ['to_q', 'to_k', 'to_v', 'to_out.0']


def train(request, update, cancelled):
    return locked(lambda: _train(request, update, cancelled))


def _save_adapter(unet, metadata):
    temporary = CHECKPOINTS / f'.tiny-sd-lora-{uuid.uuid4().hex}'
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(temporary),
            unet_lora_layers=get_peft_model_state_dict(unet, adapter_name=ADAPTER_NAME),
            safe_serialization=True,
        )
        weights = temporary / WEIGHTS_NAME
        if not weights.is_file():
            raise ValueError('Diffusers did not write the Tiny-SD LoRA weights.')
        SD_LORA_DIR.mkdir(parents=True, exist_ok=True)
        metadata_tmp = temporary / 'metadata.json'
        metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(weights, SD_LORA_DIR / WEIGHTS_NAME)
        os.replace(metadata_tmp, SD_LORA_DIR / 'metadata.json')
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _metadata(steps, source_photos, request, target, losses, loss_window):
    return {
        'base_model': SD_MODEL_ID,
        'method': 'unet-lora',
        'rank': 4,
        'steps': steps,
        'images_seen': len(source_photos),
        'providers': sorted({item['source'] for item in source_photos.values()}),
        'source_photos': list(source_photos.values()),
        'learning_rate': request['learning_rate'],
        'resolution': request.get('resolution', 512),
        'mean_loss': sum(losses[-loss_window:]) / min(loss_window, len(losses)),
        'device': str(target),
        'trained_at': time.time(),
    }


def _encode(pipeline, caption, image, target):
    """Encode a pair once; the pool keeps small latent statistics and text states instead of 512x512 pixels."""
    vae, tokenizer = pipeline.vae, pipeline.tokenizer
    pixels = torch.from_numpy(image).unsqueeze(0).to(device=target, dtype=next(vae.parameters()).dtype)
    tokens = tokenizer([caption], padding='max_length', truncation=True,
                       max_length=tokenizer.model_max_length, return_tensors='pt').input_ids.to(target)
    with torch.no_grad():
        distribution = vae.encode(pixels).latent_dist
        hidden = pipeline.text_encoder(tokens)[0]
    return distribution.mean.cpu(), distribution.std.cpu(), hidden.cpu()


def _saved_run():
    """Metadata of the committed adapter, for continuing it."""
    try:
        metadata = json.loads((SD_LORA_DIR / 'metadata.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        raise ValueError('There is no saved Tiny-SD adapter to continue. Start a new training run instead.') from None
    if not (SD_LORA_DIR / WEIGHTS_NAME).is_file() or not metadata.get('source_photos'):
        raise ValueError('There is no saved Tiny-SD adapter to continue. Start a new training run instead.')
    if metadata.get('base_model') != SD_MODEL_ID:
        raise ValueError('The saved adapter was trained on a different base model and cannot be continued.')
    return metadata


def _reload(photos, resolution):
    """Yield (photo, image) for saved pairs, downloading just ahead of the encoder; image is None when unavailable."""
    items, stopped = queue.Queue(maxsize=4), threading.Event()

    def offer(item):
        while not stopped.is_set():
            try:
                items.put(item, timeout=.5)
                return
            except queue.Full:
                pass

    def produce():
        try:
            for photo in photos:
                if stopped.is_set():
                    return
                try:
                    offer((photo, loader.load(photo['image_url'], resolution)))
                except ValueError:
                    offer((photo, None))
        finally:
            offer(None)

    threading.Thread(target=produce, daemon=True, name='naturegen-lora-reload').start()
    try:
        while True:
            item = items.get()
            if item is None:
                return
            yield item
    finally:
        stopped.set()


def _train(request, update, cancelled):
    target = torch.device(sd_generation.device())
    resume = request.get('resume', False)
    saved = _saved_run() if resume else None
    start_step = saved['steps'] if saved else 0
    if saved and start_step >= request['steps']:
        raise ValueError(f"The saved adapter already reached {start_step} steps. Raise total training steps to continue it.")
    # Offset the seed so a continued run does not replay the pair order of its first steps.
    seed = request['seed'] + start_step
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if target.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    source_photos = {f"{photo['source']}:{photo['source_id']}": photo for photo in saved['source_photos']} if saved else {}
    global_step = start_step
    saved_step = 0
    stream = None
    pipeline = None
    with sd_generation._generation_lock:
        try:
            update(phase='loading-model', current=0, total=request['steps'])
            pipeline = sd_generation._load_pipeline()
            unet = pipeline.unet
            if resume:
                # Continue from the committed adapter weights rather than a fresh initialization.
                sd_generation.restore_lora(pipeline)
                unet.requires_grad_(False)
                for name, parameter in unet.named_parameters():
                    if 'lora_' in name and f'.{ADAPTER_NAME}.' in name:
                        parameter.requires_grad_(True)
            else:
                sd_generation.prepare_lora_training(pipeline)
                unet.requires_grad_(False)
                unet.add_adapter(LoraConfig(
                    r=4,
                    lora_alpha=4,
                    init_lora_weights='gaussian',
                    target_modules=LORA_TARGETS,
                ), adapter_name=ADAPTER_NAME)
            trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
            if not trainable:
                raise ValueError('Tiny-SD did not expose trainable LoRA layers for its U-Net.')
            # Measured on CPU: gradient checkpointing recomputes activations and costs ~25% per step, and
            # channels-last convolutions are ~15% faster. One 512x512 sample fits in RAM without checkpointing.
            if target.type == 'cpu':
                if hasattr(unet, 'disable_gradient_checkpointing'):
                    unet.disable_gradient_checkpointing()
            elif hasattr(unet, 'enable_gradient_checkpointing'):
                unet.enable_gradient_checkpointing()
            unet.to(memory_format=torch.channels_last)
            pipeline.vae.to(memory_format=torch.channels_last)
            optimizer = torch.optim.AdamW(trainable, lr=request['learning_rate'], weight_decay=0.01)
            text_encoder, vae = pipeline.text_encoder, pipeline.vae
            scheduler = pipeline.scheduler
            unet.train()
            text_encoder.eval().requires_grad_(False)
            vae.eval().requires_grad_(False)

            pool = []
            seen_ids = set()
            if resume:
                # Re-encode the pairs the adapter was already training on; pixels and latents are never kept on disk.
                photos = saved['source_photos'][-request['buffer_size']:]
                update(phase='filling image pool', current=0, total=len(photos), images_seen=len(source_photos))
                reloaded = _reload(photos, request.get('resolution', 512))
                try:
                    for index, (photo, image) in enumerate(reloaded):
                        if cancelled():
                            raise InterruptedError('Tiny-SD fine-tuning cancelled while loading photos.')
                        if image is not None:
                            pool.append(_encode(pipeline, photo['caption'], image, target))
                        update(phase='filling image pool', current=index + 1, total=len(photos), images_seen=len(source_photos))
                finally:
                    reloaded.close()
                if len(pool) < 2:
                    raise ValueError('The saved training photos could not be downloaded again. Check the network connection.')
            else:
                # Larger pools use larger search pages so provider request quotas are not spent four photos at a time.
                stream = ImageStream(request.get('resolution', 512), capacity=4, include_metadata=True,
                                     page_size=max(4, min(PER_PAGE, request['buffer_size'] // (2 * len(QUERIES)))))
                update(phase='filling image pool', current=0, total=request['buffer_size'])
            while stream is not None and len(pool) < request['buffer_size']:
                if cancelled():
                    raise InterruptedError('Tiny-SD fine-tuning cancelled while loading photos.')
                caption, url, image, source = stream.take()
                key = (source['source'], source['source_id'])
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                source_photos[f'{key[0]}:{key[1]}'] = {**source, 'image_url': url}
                pool.append(_encode(pipeline, caption, image, target))
                update(phase='filling image pool', current=len(pool), total=request['buffer_size'], images_seen=len(source_photos))

            update(phase='training-lora', current=global_step, total=request['steps'], global_step=global_step,
                   images_seen=len(source_photos), device=str(target))
            scheduler_config = scheduler.config
            vae_scale = vae.config.scaling_factor
            loss_window = min(25, request['steps'])
            losses = []
            while global_step < request['steps']:
                if cancelled():
                    break
                if stream is not None and global_step and global_step % 4 == 0:
                    try:
                        fresh = stream.poll()
                    except ValueError:
                        # A dropped connection ends top-ups only; training carries on with the pairs already loaded.
                        stream.close()
                        stream = fresh = None
                    if fresh is not None:
                        caption, url, image, source = fresh
                        key = (source['source'], source['source_id'])
                        if key not in seen_ids:
                            seen_ids.add(key)
                            source_photos[f'{key[0]}:{key[1]}'] = {**source, 'image_url': url}
                            pool.append(_encode(pipeline, caption, image, target))
                            if len(pool) > request['buffer_size']:
                                pool.pop(0)

                mean, std, hidden = random.choice(pool)
                latents = (mean + std * torch.randn_like(mean)).to(target) * vae_scale
                hidden = hidden.to(target)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler_config.num_train_timesteps, (latents.shape[0],), device=target).long()
                noisy = scheduler.add_noise(latents, noise, timesteps)
                prediction_type = getattr(scheduler_config, 'prediction_type', 'epsilon')
                if prediction_type == 'epsilon':
                    expected = noise
                elif prediction_type == 'v_prediction':
                    expected = scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f'Unsupported Tiny-SD scheduler prediction type: {prediction_type}.')

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=target.type, enabled=target.type == 'cuda', dtype=torch.float16):
                    predicted = unet(noisy, timesteps, encoder_hidden_states=hidden).sample
                    loss = F.mse_loss(predicted.float(), expected.float(), reduction='mean')
                if not torch.isfinite(loss):
                    raise ValueError('Tiny-SD LoRA loss became nonfinite; the previous adapter was kept.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                global_step += 1
                losses.append(float(loss.detach()))
                update(phase='training-lora', current=global_step, total=request['steps'], global_step=global_step,
                       loss=losses[-1], images_seen=len(source_photos),
                       providers=sorted({item['source'] for item in source_photos.values()}),
                       learning_rate=optimizer.param_groups[0]['lr'])
                if global_step % loss_window == 0 or global_step == request['steps']:
                    _save_adapter(unet, _metadata(global_step, source_photos, request, target, losses, loss_window))
                    sd_generation.mark_lora_loaded()
                    saved_step = global_step
            if global_step > max(saved_step, start_step):
                _save_adapter(unet, _metadata(global_step, source_photos, request, target, losses, loss_window))
                sd_generation.mark_lora_loaded()
                saved_step = global_step
            if cancelled():
                return {'model': SD_MODEL_ID, 'adapter': str(SD_LORA_DIR), 'steps': global_step,
                        'images_seen': len(source_photos), 'stopped': True}
            return {'model': SD_MODEL_ID, 'adapter': str(SD_LORA_DIR), 'steps': global_step,
                    'images_seen': len(source_photos), 'providers': sorted({item['source'] for item in source_photos.values()})}
        finally:
            if stream is not None:
                stream.close()
            if pipeline is not None:
                if saved_step == 0 or global_step != saved_step:
                    sd_generation.restore_lora(pipeline)
                pipeline.unet.eval()
