"""Measure one small forward/backward step; never trains or creates a checkpoint."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import torch
from app.model import ModelConfig, NatureModel
from app.training import device

target = device()
config = ModelConfig()
model = NatureModel(12000, config).to(target)
images = torch.randn(2, 3, 64, 64, device=target)
tokens = torch.randint(1, 12000, (2, 64), device=target)
started = time.perf_counter()
for _ in range(3):
    model.zero_grad(set_to_none=True)
    model(images, torch.tensor([50, 500], device=target), tokens).square().mean().backward()
if target.type == 'cuda': torch.cuda.synchronize()
seconds = (time.perf_counter() - started) / 3
print(json.dumps({'parameters': sum(p.numel() for p in model.parameters()), 'batch_size': 2, 'resolution': 64,
                  'device': str(target), 'seconds_per_forward_backward': seconds,
                  'gpu_peak_bytes': torch.cuda.max_memory_allocated() if target.type == 'cuda' else None,
                  'note': 'Synthetic mechanics benchmark; excludes network, optimizer, validation, and checkpoint costs.'}, indent=2))
