import time
import pytest
import torch
from app.diffusion import Diffusion, condition_dropout
from app.model import ModelConfig, NatureModel, TextEncoder
from app.tokenizer import BPETokenizer
from app.training import EMA, atomic_save, load_checkpoint, rng_state, restore_rng

torch.set_num_threads(2)


def tiny():
    config = ModelConfig(resolution=8, base_channels=8, multipliers=(1, 2), blocks=1, text_dim=16, text_layers=1, heads=2, max_length=16, timesteps=10)
    tokenizer = BPETokenizer(max_length=16).train(['green forest', 'blue ocean'])
    model = NatureModel(len(tokenizer.vocab), config)
    ids = torch.tensor([tokenizer.encode('green forest'), tokenizer.encode('blue ocean')])
    return config, tokenizer, model, ids


def test_schedule_and_velocity_roundtrip():
    diffusion = Diffusion(100)
    assert torch.isfinite(diffusion.beta).all() and (diffusion.beta > 0).all() and (diffusion.beta < 1).all()
    assert (diffusion.bar[1:] < diffusion.bar[:-1]).all()
    clean, noise = torch.randn(3, 3, 8, 8), torch.randn(3, 3, 8, 8)
    t = torch.tensor([0, 50, 99])
    noisy, velocity = diffusion.forward(clean, noise, t)
    restored, restored_noise = diffusion.convert(noisy, velocity, t)
    torch.testing.assert_close(restored, clean)
    torch.testing.assert_close(restored_noise, noise)


def test_padding_mask_and_conditioning_gradients():
    config, tok, model, ids = tiny()
    model.eval()
    original = model.text(ids)[0].detach()
    with torch.no_grad(): model.text.embedding.weight[0].fill_(10000)
    changed = model.text(ids)[0].detach()
    torch.testing.assert_close(original[ids != 0], changed[ids != 0])
    x = torch.randn(2, 3, 8, 8)
    prediction = model(x, torch.tensor([2, 6]), ids)
    assert prediction.shape == x.shape
    prediction.square().mean().backward()
    assert sum(p.grad.abs().sum() for p in model.text.parameters() if p.grad is not None) > 0
    assert sum(p.grad.abs().sum() for p in model.unet.parameters() if p.grad is not None) > 0
    unconditional = model(x, torch.tensor([2, 6]), torch.tensor([tok.encode('', null=True)] * 2))
    assert torch.isfinite(unconditional).all()


def test_cfg_dropout_rate():
    torch.manual_seed(42)
    ids = torch.ones(10000, 5, dtype=torch.long)
    null = torch.full_like(ids, 4)
    result = condition_dropout(ids, null, .1)
    assert .08 < (result[:, 0] == 4).float().mean() < .12


def test_samplers_repeatability_and_cancel():
    config, tok, model, ids = tiny()
    diffusion = Diffusion(10)
    null = torch.tensor([tok.encode('', null=True)] * 2)
    first = diffusion.sample(model, ids, null, 8, steps=4)
    second = diffusion.sample(model, ids, null, 8, steps=4)
    torch.testing.assert_close(first, second)
    assert first.shape == (2, 3, 8, 8) and first.min() >= -1 and first.max() <= 1
    assert torch.isfinite(diffusion.sample(model, ids, null, 8, steps=10, sampler='ddpm')).all()
    with pytest.raises(ValueError): diffusion.sample(model, ids, null, 8, steps=4, sampler='ddpm')
    with pytest.raises(InterruptedError): diffusion.sample(model, ids, null, 8, steps=4, cancelled=lambda: True)


def test_ema_checkpoint_rng(tmp_path):
    config, tok, model, ids = tiny()
    ema = EMA(model, .5)
    before = next(ema.model.parameters()).clone()
    with torch.no_grad(): next(model.parameters()).add_(2)
    ema.update(model)
    torch.testing.assert_close(next(ema.model.parameters()), before + 1)
    warm = EMA(model, .999)
    with torch.no_grad(): next(model.parameters()).add_(10)
    warm.update(model, 0)
    torch.testing.assert_close(next(warm.model.parameters()), next(model.parameters()) - 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.randn(2, 3, 8, 8), torch.ones(2, dtype=torch.long), ids).sum().backward(); optimizer.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    saved_rng = rng_state()
    expected = torch.rand(4); restore_rng(saved_rng); torch.testing.assert_close(torch.rand(4), expected)
    state = {'version': 1, 'model_config': config.dict(), 'model': model.state_dict(), 'ema': ema.model.state_dict(),
             'tokenizer': tok.state(), 'tokenizer_checksum': tok.checksum(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'rng': saved_rng}
    path = tmp_path / 'model.pt'; atomic_save(state, path); atomic_save(state, path)
    loaded, _, _, restored = load_checkpoint(path)
    restored_optimizer = torch.optim.AdamW(restored.parameters()); restored_optimizer.load_state_dict(loaded['optimizer'])
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, 1); restored_scheduler.load_state_dict(loaded['scheduler'])
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert len(restored_optimizer.state) == len(optimizer.state)
    for a, b in zip(model.parameters(), restored.parameters()): torch.testing.assert_close(a, b)
    assert path.with_suffix('.previous.pt').exists()


def test_tiny_synthetic_overfit():
    """Code mechanics only: synthetic tensors are not a nature training dataset."""
    torch.manual_seed(42)
    config, tok, model, ids = tiny()
    model.eval()  # eliminate dropout while testing fixed-batch overfit mechanics
    diffusion = Diffusion(10)
    clean = torch.zeros(2, 3, 8, 8); clean[0, 1] = 1; clean[1, 2] = 1
    noisy, expected = diffusion.forward(clean, torch.randn_like(clean), torch.tensor([3, 6]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    values = []
    for _ in range(35):
        optimizer.zero_grad()
        loss = (model(noisy, torch.tensor([3, 6]), ids) - expected).square().mean()
        loss.backward(); optimizer.step(); values.append(float(loss.detach()))
    assert values[-1] < values[0] * .35, values


def test_stream_training_saves_resumable_checkpoint(isolated_store, tmp_path, monkeypatch):
    import numpy as np
    from app import training
    def pixels(index):
        return np.random.default_rng(index).uniform(-1, 1, (3, 64, 64)).astype(np.float32)
    class FakeStream:
        def __init__(self, resolution, state=None):
            self.state = state or {'round': 0}; self.count = 0
        def take(self):
            self.count += 1
            return (f'green forest {self.count}', f'https://images.pexels.com/{self.count}.jpg', pixels(self.count))
        def poll(self):
            return self.take()
        def close(self):
            pass
    monkeypatch.setattr(training, 'ImageStream', FakeStream)
    monkeypatch.setattr(training, 'CHECKPOINTS', tmp_path)
    monkeypatch.setattr(training.loader, 'load', lambda url, resolution: pixels(int(url.rsplit('/', 1)[1].split('.')[0])))
    request = {'steps': 4, 'batch_size': 2, 'buffer_size': 16, 'fresh_per_step': 1, 'validate_every': 2, 'learning_rate': 1e-4,
               'text_learning_rate': 1e-4, 'base_channels': 16, 'text_dim': 64, 'text_layers': 1, 'timesteps': 10,
               'cfg_dropout': .1, 'ema_decay': .999, 'seed': 1, 'resume': False,
               'validation_images': 8, 'min_replays': 8, 'preview_every': 0}
    result = training.stream_train(request, lambda **k: None, lambda: False)
    assert result['global_step'] == 4 and result['images_seen'] == 16
    state, *_ = load_checkpoint(tmp_path / 'latest.pt')
    assert state['mode'] == 'stream' and len(state['validation']) == 8 and len(state['history']) == 2
    assert 'rng' in state and len(state['replay']['items']) == 16
    assert (tmp_path / f"{state['run_id']}-best.pt").is_file()
    resumed = training.stream_train({**request, 'steps': 6, 'resume': True}, lambda **k: None, lambda: False)
    assert resumed['global_step'] == 6 and resumed['run_id'] == state['run_id']
    after, *_ = load_checkpoint(tmp_path / 'latest.pt')
    assert [r['url'] for r in after['replay']['items']] == [r['url'] for r in state['replay']['items']]


def test_image_self_attention_and_legacy_config():
    from app.model import ImageSelfAttention
    attention = ImageSelfAttention(8, 2)
    x = torch.randn(2, 8, 4, 4, requires_grad=True)
    torch.testing.assert_close(attention(x), x)
    attention(x).square().mean().backward()
    assert attention.attn.out.weight.grad.abs().sum() > 0
    config, tok, model, ids = tiny()
    legacy_config = config.dict()
    legacy_config.pop('spatial_self_attention')
    restored = NatureModel(len(tok.vocab), ModelConfig(**legacy_config))
    restored.load_state_dict(model.state_dict(), strict=True)


def test_extended_schedule_does_not_take_zero_lr_step():
    from app.training import stream_scheduler
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([{'params': [parameter], 'lr': 0.}])
    request = {'steps': 50000, 'learning_rate': .0002, 'text_learning_rate': .0002}
    state = {'global_step': 20000, 'training_config': {'steps': 20000}}
    scheduler, spec = stream_scheduler(optimizer, request, state)
    assert 0 < optimizer.param_groups[0]['lr'] < .0002
    assert spec['start'] == 20000 and spec['total'] == 50000
    parameter.sum().backward(); optimizer.step(); scheduler.step()
    assert scheduler.last_epoch == 20001
    expected = optimizer.param_groups[0]['lr']
    again = {'global_step': 20001, 'lr_schedule': spec}
    stream_scheduler(optimizer, request, again)
    assert optimizer.param_groups[0]['lr'] == expected
