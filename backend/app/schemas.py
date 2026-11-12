import re
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class GenerationRequest(StrictModel):
    prompt: str = Field(min_length=3, max_length=500)
    resolution: Literal[512] = 512
    steps: int = Field(default=20, ge=4, le=50)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    guidance: float = Field(default=7.5, ge=0, le=12)
    # Retain legacy values so saved UI settings can be replayed; Diffusers uses
    # the scheduler configured by the selected model.
    sampler: Literal['default', 'ddim', 'ddpm'] = 'default'


class VideoGenerationRequest(StrictModel):
    prompt: str = Field(min_length=3, max_length=500)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    scene_count: int = Field(default=3, ge=3, le=6)
    image_ids: list[str] = Field(default_factory=list, max_length=8)
    audio_style: Literal['auto', 'quiet', 'wind', 'stream', 'waterfall', 'surf', 'birds'] = 'auto'
    transition: Literal['crossfade', 'slide'] = 'crossfade'
    transition_sound: bool = True

    @field_validator('image_ids')
    @classmethod
    def valid_image_ids(cls, value):
        if len(value) == 1:
            raise ValueError('Select at least two images for a transition video.')
        if len(value) != len(set(value)):
            raise ValueError('Choose each image only once.')
        if any(not re.fullmatch(r'[a-f0-9]{32}', image_id) for image_id in value):
            raise ValueError('One or more selected image IDs are invalid.')
        return value


class PromptIdeasRequest(StrictModel):
    avoid: list[str] = Field(default_factory=list, max_length=20)
    mode: Literal['image', 'video'] = 'image'
    idea: str = Field(default='', max_length=500)
    style: str = Field(default='', max_length=80)
    count: int = Field(default=4, ge=1, le=4)


class TrainingRequest(StrictModel):
    spatial_self_attention: bool = True
    epochs: int = Field(default=5, ge=1, le=1000)
    batch_size: int = Field(default=2, ge=1, le=32)
    learning_rate: float = Field(default=.0001, ge=.000001, le=.01)
    text_learning_rate: float = Field(default=.0001, ge=.000001, le=.01)
    accumulation: int = Field(default=1, ge=1, le=32)
    base_channels: Literal[16, 32, 64] = 32
    text_dim: Literal[64, 128, 256] = 128
    text_layers: int = Field(default=2, ge=1, le=4)
    timesteps: int = Field(default=1000, ge=10, le=1000)
    cfg_dropout: float = Field(default=.1, ge=.05, le=.5)
    ema_decay: float = Field(default=.999, ge=.9, le=.9999)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class PairInput(StrictModel):
    caption: str = Field(min_length=3, max_length=500)
    image_url: str = Field(min_length=8, max_length=2048)
    source: str = Field(min_length=1, max_length=80)
    source_id: str = Field(min_length=1, max_length=200)
    creator: str = Field(default='', max_length=200)
    attribution_url: str = Field(default='', max_length=2048)
    license_note: str = Field(min_length=5, max_length=1000)
    category: str = Field(default='nature', min_length=1, max_length=80)
    tags: list[str] = Field(default_factory=list, max_length=30)
    description: str = Field(default='', max_length=1000)

    @field_validator('attribution_url')
    @classmethod
    def safe_link(cls, value):
        if value and not value.startswith('https://'):
            raise ValueError('Attribution link must use HTTPS.')
        return value


class ImportRequest(StrictModel):
    records: list[PairInput] = Field(min_length=1, max_length=100)


class ReviewRequest(StrictModel):
    review_status: Literal['approved', 'rejected', 'pending']
    note: str = Field(min_length=3, max_length=1000)


class CollectionRequest(StrictModel):
    provider: Literal['auto', 'huggingface', 'pexels', 'unsplash']
    query: str = Field(default='forest waterfall', min_length=3, max_length=120)
    limit: int = Field(default=20, ge=1, le=30)
    offset: int = Field(default=0, ge=0, le=3000000)
    license_note: str = Field(min_length=5, max_length=1000)


class StreamTrainingRequest(StrictModel):
    spatial_self_attention: bool = True
    steps: int = Field(default=5000, ge=10, le=500000)
    batch_size: int = Field(default=4, ge=1, le=32)
    buffer_size: int = Field(default=256, ge=16, le=2048)
    fresh_per_step: int = Field(default=1, ge=0, le=8)
    validate_every: int = Field(default=250, ge=10, le=10000)
    validation_images: int = Field(default=64, ge=8, le=256)
    min_replays: int = Field(default=8, ge=1, le=1000)
    preview_every: int = Field(default=1000, ge=0, le=10000)
    learning_rate: float = Field(default=.0002, ge=.000001, le=.01)
    text_learning_rate: float = Field(default=.0002, ge=.000001, le=.01)
    base_channels: Literal[16, 32, 64] = 32
    text_dim: Literal[64, 128, 256] = 128
    text_layers: int = Field(default=2, ge=1, le=4)
    timesteps: int = Field(default=1000, ge=10, le=1000)
    cfg_dropout: float = Field(default=.1, ge=.05, le=.5)
    ema_decay: float = Field(default=.999, ge=.9, le=.9999)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    resume: bool = False
    # Training image size. 384 is roughly twice as fast on CPU; Create still generates at 512.
    resolution: Literal[384, 512] = 512


class TinySDLoraRequest(StrictModel):
    steps: int = Field(default=100, ge=1, le=10000)
    buffer_size: int = Field(default=128, ge=4, le=1024)
    learning_rate: float = Field(default=0.00001, ge=0.000001, le=0.001)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    # Continue the saved adapter on its saved pairs up to `steps` total steps, instead of starting fresh.
    resume: bool = False
    # Training image size. 384 is roughly twice as fast on CPU; Create still generates at 512.
    resolution: Literal[384, 512] = 512
