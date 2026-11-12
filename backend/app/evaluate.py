"""Fixed prompt/seed grids using EMA weights; no pretrained evaluation networks."""
import argparse
import json
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from .config import CHECKPOINTS, RESULTS
from .diffusion import Diffusion
from .training import load_checkpoint, device

PROMPTS = [
    'A green forest in soft morning mist',
    'A blue ocean beneath a golden sunset',
    'A snow covered mountain under a clear sky',
    'A waterfall flowing through a tropical forest',
]


@torch.no_grad()
def preview(model, tokenizer, config, output, step, prompts=None, steps=50, guidance=2., seed=42):
    """Small fixed-seed grid during training; pixels are model output, never sharpened."""
    prompts = PROMPTS if prompts is None else prompts
    target = next(model.parameters()).device
    diffusion = Diffusion(config.timesteps, target)
    grid = Image.new('RGB', (len(prompts) * 180, 220), '#f7f8f4')
    draw = ImageDraw.Draw(grid)
    was_training = model.training
    try:
        for index, prompt in enumerate(prompts):
            ids = torch.tensor([tokenizer.encode(prompt)], device=target)
            null = torch.tensor([tokenizer.encode('', null=True)], device=target)
            tensor = diffusion.sample(model, ids, null, config.resolution, min(steps, config.timesteps), guidance, seed)
            pixels = ((tensor[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            grid.paste(Image.fromarray(pixels).resize((160, 160), Image.Resampling.NEAREST), (index * 180, 0))
            draw.multiline_text((index * 180, 165), '\n'.join(prompt[i:i + 25] for i in range(0, min(100, len(prompt)), 25)), fill='#253b30')
        output.parent.mkdir(parents=True, exist_ok=True)
        grid.save(output)
        output.with_suffix('.json').write_text(json.dumps({'checkpoint_step': step, 'prompts': prompts,
            'seed': seed, 'steps': min(steps, config.timesteps), 'guidance': guidance,
            'native_resolution': config.resolution, 'display_resize': 'nearest, 160px',
            'sampler': 'ddim'}, indent=2), encoding='utf-8')
    finally:
        model.train(was_training)
    return output


def evaluate(checkpoint, output, seeds=(42, 123, 2026), steps=50, guidance=4.):
    target = device()
    state, tokenizer, config, model = load_checkpoint(checkpoint, target)
    model.load_state_dict(state['ema'], strict=True)
    model.eval()
    diffusion = Diffusion(config.timesteps, target)
    cell = max(128, config.resolution)
    grid = Image.new('RGB', (len(seeds) * cell + 250, len(PROMPTS) * (cell + 24)), '#f7f8f4')
    draw = ImageDraw.Draw(grid)
    samples = []
    for row, prompt in enumerate(PROMPTS):
        draw.multiline_text((10, row * (cell + 24) + 20), '\n'.join([prompt[:30], prompt[30:]]), fill='#253b30')
        ids = torch.tensor([tokenizer.encode(prompt)], device=target)
        null = torch.tensor([tokenizer.encode('', null=True)], device=target)
        for col, seed in enumerate(seeds):
            tensor = diffusion.sample(model, ids, null, config.resolution, min(steps, config.timesteps), guidance, seed)
            image = Image.fromarray(((tensor[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8))
            # Display enlargement only; native resolution remains recorded explicitly.
            grid.paste(image.resize((cell, cell), Image.Resampling.NEAREST), (250 + col * cell, row * (cell + 24)))
            draw.text((255 + col * cell, row * (cell + 24) + cell + 5), f'Seed {seed}', fill='#253b30')
            samples.append({'prompt': prompt, 'seed': seed, 'quality_rating': None, 'adherence_rating': None, 'memorization_notes': ''})
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)
    output.with_suffix('.json').write_text(json.dumps({'checkpoint_step': state['global_step'], 'native_resolution': config.resolution,
        'manifest': state['manifest'], 'tokenizer_checksum': state['tokenizer_checksum'], 'guidance': guidance,
        'steps': min(steps, config.timesteps), 'created_at': time.time(), 'samples': samples}, indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=str, default=str(CHECKPOINTS / 'latest.pt'))
    parser.add_argument('--output', type=str, default=str(RESULTS / 'evaluation-grid.png'))
    parser.add_argument('--steps', type=int, default=50)
    args = parser.parse_args()
    from pathlib import Path
    print(evaluate(Path(args.checkpoint), Path(args.output), steps=args.steps))
