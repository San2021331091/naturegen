"""Create a short, fixed-camera clip with semantic local motion masks."""
import gc
import re
import threading
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

from . import sd_generation, storage
from .config import MODEL_CACHE, RESULTS, SD_MODEL_ID


VIDEO_WIDTH = 256
VIDEO_HEIGHT = 256
VIDEO_FRAMES = 480
VIDEO_FPS = 8
VIDEO_STEPS = 20
SEQUENCE_SECONDS_PER_IMAGE = 4
TRANSITION_FRAMES = 6
SEGMENTER_ID = 'nvidia/segformer-b0-finetuned-ade-512-512'
WATER_LABELS = (21, 26, 60, 104, 109, 113, 128)
WATERFALL_LABEL = 113
TREE_LABELS = (4, 72)
PLANT_LABELS = (9, 17, 29, 66, 68)
FOLIAGE_LABELS = TREE_LABELS + PLANT_LABELS
SKY_LABEL = 2
FLAG_LABEL = 149
FOUNTAIN_LABEL = 104
_video_lock = threading.Lock()
_segmenter_lock = threading.Lock()
_segmenter_processor = None
_segmenter_model = None


def is_loaded():
    return sd_generation.is_loaded() or _segmenter_model is not None


def _load_segmenter():
    global _segmenter_processor, _segmenter_model
    with _segmenter_lock:
        if _segmenter_model is not None:
            return _segmenter_processor, _segmenter_model
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError as error:
            raise ValueError('The small motion-segmentation model needs Transformers. Install backend/requirements.txt and restart the backend.') from error
        _segmenter_processor = SegformerImageProcessor.from_pretrained(
            SEGMENTER_ID,
            cache_dir=str(MODEL_CACHE),
        )
        _segmenter_model = SegformerForSemanticSegmentation.from_pretrained(
            SEGMENTER_ID,
            cache_dir=str(MODEL_CACHE),
        ).to('cpu').eval()
        return _segmenter_processor, _segmenter_model


def _segment_motion_regions(image):
    processor, model = _load_segmenter()
    inputs = processor(images=image, return_tensors='pt')
    with torch.inference_mode():
        logits = model(**inputs).logits
        labels = F.interpolate(
            logits,
            size=(image.height, image.width),
            mode='bilinear',
            align_corners=False,
        ).argmax(dim=1)[0].cpu().numpy()
    water = np.isin(labels, WATER_LABELS)
    waterfall = labels == WATERFALL_LABEL
    trees = np.isin(labels, TREE_LABELS)
    plants = np.isin(labels, PLANT_LABELS)
    foliage = trees | plants
    sky = labels == SKY_LABEL
    flag = labels == FLAG_LABEL
    fountain = labels == FOUNTAIN_LABEL
    del inputs, logits
    gc.collect()
    return {'water': water, 'waterfall': waterfall, 'trees': trees, 'plants': plants,
            'foliage': foliage, 'sky': sky, 'flag': flag, 'fountain': fountain}


def _soft_mask(mask):
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(mask_image.filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32)[..., None] / 255.0


def _sample(image, x_map, y_map):
    height, width = image.shape[:2]
    x_map = np.clip(x_map, 0, width - 1)
    y_map = np.clip(y_map, 0, height - 1)
    x0 = np.floor(x_map).astype(np.int32)
    y0 = np.floor(y_map).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = (x_map - x0)[..., None]
    dy = (y_map - y0)[..., None]
    top = image[y0, x0] * (1 - dx) + image[y0, x1] * dx
    bottom = image[y1, x0] * (1 - dx) + image[y1, x1] * dx
    return top * (1 - dy) + bottom * dy


def _animate_regions(image, cancelled, update, frame_count=16):
    image = image.convert('RGB')
    masks = _segment_motion_regions(image)
    weights = {name: _soft_mask(mask) for name, mask in masks.items()}
    base = np.asarray(image, dtype=np.float32)
    height, width = base.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    frames = []
    for index in range(frame_count):
        if cancelled():
            raise InterruptedError('Video generation cancelled.')

        phase = 2.0 * np.pi * index / frame_count
        frame = base.copy()

        if np.any(masks['sky']):
            sky_shift = 1.5 * np.sin(phase)
            sky_sample = _sample(base, xx - sky_shift, yy)
            frame = frame * (1 - weights['sky']) + sky_sample * weights['sky']

        if np.any(masks['water']):
            water_x = xx + 1.8 * np.sin(yy * 0.055 + phase)
            water_y = yy + 1.5 * np.sin(xx * 0.065 - phase)
            flow_x = xx + 1.0 * np.sin(yy * 0.035 - phase)
            flow_y = yy - 3.0 * np.sin(yy * 0.045 - phase)
            water_sample = _sample(base, water_x, water_y)
            frame = frame * (1 - weights['water']) + water_sample * weights['water']

        if np.any(masks['waterfall']):
            waterfall_sample = _sample(base, xx + np.sin(yy * 0.035 - phase),
                                       yy - 3.0 * np.sin(yy * 0.045 - phase))
            frame = frame * (1 - weights['waterfall']) + waterfall_sample * weights['waterfall']

        for name, amplitude in (('trees', 1.8), ('plants', 3.2)):
            if np.any(masks[name]):
                swing = amplitude * np.clip(yy / max(1, height), 0, 1)
                sample = _sample(base, xx + swing * np.sin(phase + yy * 0.012),
                                 yy + 0.6 * np.sin(phase + xx * 0.018))
                frame = frame * (1 - weights[name]) + sample * weights[name]

        if np.any(masks['fountain']):
            spray = _sample(base, xx + 0.7 * np.sin(phase + yy * 0.08),
                            yy - 1.2 * np.sin(phase + xx * 0.06))
            frame = frame * (1 - weights['fountain']) + spray * weights['fountain']
        if np.any(masks['flag']):
            flutter = _sample(base, xx + 1.7 * np.sin(phase + yy * 0.1), yy)
            frame = frame * (1 - weights['flag']) + flutter * weights['flag']

        result = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
        frames.append(result.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS))
        update(current=index + 1, total=VIDEO_FRAMES, phase='animating-regions')
    scene = {name: float(mask.mean()) for name, mask in masks.items()}
    return frames, scene


def _image_sequence_frames(image_ids, transition, cancelled, update, images=None):
    images = list(images or [])
    for image_id in image_ids:
        if not re.fullmatch(r'[a-f0-9]{32}', image_id):
            raise ValueError('One or more selected image IDs are invalid.')
        record = storage.get('generation', image_id)
        path = RESULTS / f'{image_id}.png'
        if not record or record.get('image_url') != f'/api/generated/{image_id}.png' or not path.is_file():
            raise ValueError('A selected image is no longer available. Refresh image history and try again.')
        with Image.open(path) as source:
            images.append(ImageOps.fit(source.convert('RGB'), (VIDEO_WIDTH, VIDEO_HEIGHT), method=Image.Resampling.LANCZOS))

    images = [ImageOps.fit(image, (VIDEO_WIDTH, VIDEO_HEIGHT), method=Image.Resampling.LANCZOS) for image in images]
    frames_per_image = VIDEO_FPS * SEQUENCE_SECONDS_PER_IMAGE
    total_frames = frames_per_image * len(images)
    motion_frames = []
    scene = {'water': 0.0, 'waterfall': 0.0, 'foliage': 0.0, 'sky': 0.0,
             'trees': 0.0, 'plants': 0.0, 'flag': 0.0, 'fountain': 0.0}
    for image_index, image in enumerate(images):
        animated, detected = _animate_regions(
            image, cancelled,
            lambda current, total, phase: update(
                current=image_index * frames_per_image + current,
                total=total_frames, phase=phase),
            frame_count=frames_per_image,
        )
        motion_frames.append(animated)
        for name, coverage in detected.items():
            scene[name] = max(scene[name], coverage)
    transition_times = [index * SEQUENCE_SECONDS_PER_IMAGE for index in range(1, len(images))]
    frames = []
    for index in range(total_frames):
        if cancelled():
            raise InterruptedError('Video generation cancelled.')
        image_index, local_frame = divmod(index, frames_per_image)
        current = motion_frames[image_index][local_frame]
        if image_index > 0 and local_frame < TRANSITION_FRAMES:
            previous = motion_frames[image_index - 1][-1]
            progress = (local_frame + 1) / TRANSITION_FRAMES
            if transition == 'slide':
                shift = round(progress * VIDEO_WIDTH)
                frame = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT))
                frame.paste(previous.crop((shift, 0, VIDEO_WIDTH, VIDEO_HEIGHT)), (0, 0))
                frame.paste(current.crop((0, 0, shift, VIDEO_HEIGHT)), (VIDEO_WIDTH - shift, 0))
            else:
                frame = Image.blend(previous, current, progress)
        else:
            frame = current.copy()
        frames.append(frame)
        update(current=index + 1, total=total_frames, phase='transitioning-images')
    duration = total_frames / VIDEO_FPS
    return frames, scene, duration, transition_times


def generate(request, update, cancelled):
    with _video_lock:
        if cancelled():
            raise InterruptedError('Video generation cancelled.')

        transition = request.get('transition', 'crossfade')
        transition_times = []
        if request.get('image_ids'):
            frames, scene, duration, transition_times = _image_sequence_frames(
                request['image_ids'], transition, cancelled, update,
            )
            model_id = 'Generated image sequence'
            motion_type = 'semantic motion from detected image regions'
        else:
            with sd_generation._generation_lock:
                update(current=0, total=VIDEO_STEPS, phase='loading-model')
                pipeline = sd_generation._load_pipeline()
                sd_generation._load_lora(pipeline)

                images = []
                count = request.get('scene_count', 3)
                compositions = ['wide establishing view', 'closer detailed view', 'alternate viewpoint', 'low angle view', 'elevated view', 'intimate foreground composition']
                for scene_index in range(count):
                    if cancelled():
                        raise InterruptedError('Video generation cancelled.')
                    def report_scene(_pipeline, step_index, _timestep, callback_kwargs):
                        update(current=scene_index * VIDEO_STEPS + step_index + 1,
                               total=count * VIDEO_STEPS, phase='generating-keyframe')
                        if cancelled():
                            raise InterruptedError('Video generation cancelled.')
                        return callback_kwargs
                    output = pipeline(
                        prompt=f"{request['prompt']}, {compositions[scene_index]}",
                        negative_prompt='blurry, low resolution, distorted, text, watermark',
                        height=512, width=512, num_inference_steps=VIDEO_STEPS,
                        guidance_scale=7.5,
                        generator=torch.Generator(device=sd_generation.device()).manual_seed((request['seed'] + scene_index) % 2**32),
                        callback_on_step_end=report_scene,
                    )
                    images.append(output.images[0].convert('RGB'))
                    del output
            frames, scene, duration, transition_times = _image_sequence_frames(
                [], transition, cancelled, update, images=images,
            )
            model_id = f'{SD_MODEL_ID} multi-image video'
            motion_type = 'semantic motion from detected image regions'

        key = uuid.uuid4().hex
        filename = f'{key}.mp4'
        try:
            from diffusers.utils import export_to_video
            export_to_video(frames, str(RESULTS / filename), fps=VIDEO_FPS)
        except ImportError as error:
            raise ValueError('Video export packages are missing. Install backend/requirements.txt and restart the backend.') from error

        from .ambient_audio import add_ambient_audio
        total_frames = len(frames)
        update(current=total_frames, total=total_frames, phase='adding-audio')
        if cancelled():
            raise InterruptedError('Video generation cancelled.')
        audio = add_ambient_audio(
            RESULTS / filename, request['prompt'], duration, request['seed'], scene=scene,
            audio_style='quiet', transition_times=transition_times,
            transition_sound=bool(transition_times) and request.get('transition_sound', True),
        )

    detected_motion = []
    if scene:
        if scene.get('water', 0) >= 0.04:
            detected_motion.append('water flow')
        if scene.get('foliage', 0) >= 0.08:
            detected_motion.append('tree and plant sway')
        if scene.get('fountain', 0) >= 0.015:
            detected_motion.append('fountain spray')
        if scene.get('flag', 0) >= 0.01:
            detected_motion.append('flag flutter')
        if scene.get('sky', 0) >= 0.08:
            detected_motion.append('sky drift')
    if detected_motion:
        motion_type = ', '.join(detected_motion) + f' with {transition} transitions'
    elif scene is not None:
        motion_type = f'subtle localized motion with {transition} transitions'

    result = {
        'id': key,
        'prompt': request['prompt'],
        'seed': request['seed'],
        'video_url': f'/api/videos/{filename}',
        'width': VIDEO_WIDTH,
        'height': VIDEO_HEIGHT,
        'frames': len(frames),
        'fps': VIDEO_FPS,
        'steps': VIDEO_STEPS if not request.get('image_ids') else 0,
        'created_at': time.time(),
        'model_id': model_id,
        'device': sd_generation.device() if not request.get('image_ids') else 'local image sequence',
        'style_lora': sd_generation.lora_status() if not request.get('image_ids') else None,
        'motion_type': motion_type,
        'audio': audio,
        'transition': transition,
        'source_image_count': len(request.get('image_ids', [])) or request.get('scene_count', 3),
    }
    storage.put('video', result)
    return result
