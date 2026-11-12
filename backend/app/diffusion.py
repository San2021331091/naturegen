import math
import torch


class Diffusion:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        curve = torch.cos(((t / timesteps + .008) / 1.008) * math.pi / 2).square()
        curve = curve / curve[0]
        self.beta = (1 - curve[1:] / curve[:-1]).clamp(1e-8, .999).float().to(device)
        self.alpha = 1 - self.beta
        self.bar = self.alpha.cumprod(0)

    def coefficients(self, t):
        return self.bar[t].sqrt()[:, None, None, None], (1 - self.bar[t]).sqrt()[:, None, None, None]

    def forward(self, clean, noise, t):
        a, s = self.coefficients(t)
        return a * clean + s * noise, a * noise - s * clean

    def convert(self, x, velocity, t):
        a, s = self.coefficients(t)
        return a * x - s * velocity, s * x + a * velocity

    @torch.no_grad()
    def sample(self, model, ids, null_ids, resolution, steps=50, guidance=4., seed=42, sampler='ddim', progress=None, cancelled=None):
        device = self.bar.device
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn((len(ids), 3, resolution, resolution), device=device, generator=generator)
        if sampler not in ('ddim', 'ddpm'):
            raise ValueError('Unsupported sampler.')
        if sampler == 'ddpm' and steps != self.timesteps:
            raise ValueError('Reference DDPM requires all training timesteps.')
        if not 2 <= steps <= self.timesteps:
            raise ValueError('Sampling steps must be between 2 and training timesteps.')
        schedule = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
        model.eval()
        for index, current in enumerate(schedule):
            if cancelled and cancelled():
                raise InterruptedError('Generation cancelled.')
            t = torch.full((len(ids),), current, dtype=torch.long, device=device)
            conditional = model(x, t, ids)
            unconditional = model(x, t, null_ids)
            velocity = unconditional + guidance * (conditional - unconditional)
            clean, noise = self.convert(x, velocity, t)
            clean = clean.clamp(-1, 1)
            # Re-derive epsilon after clipping x0 for a consistent DDIM update.
            a, s = self.coefficients(t)
            noise = (x - a * clean) / s.clamp_min(1e-8)
            previous = schedule[index + 1] if index + 1 < len(schedule) else -1
            prev_bar = self.bar[previous] if previous >= 0 else torch.ones((), device=device)
            if sampler == 'ddim':
                x = prev_bar.sqrt() * clean + (1 - prev_bar).sqrt() * noise
            else:
                bar, beta, alpha = self.bar[current], self.beta[current], self.alpha[current]
                mean = (prev_bar.sqrt() * beta / (1 - bar)) * clean + (alpha.sqrt() * (1 - prev_bar) / (1 - bar)) * x
                variance = beta * (1 - prev_bar) / (1 - bar)
                x = mean + variance.sqrt() * torch.randn(x.shape, device=device, generator=generator) if current else mean
            if not torch.isfinite(x).all():
                raise ValueError('Sampler produced nonfinite values.')
            if progress:
                progress(index + 1, len(schedule))
        return x.clamp(-1, 1)


def condition_dropout(ids, null_ids, probability):
    mask = torch.rand(len(ids), device=ids.device) < probability
    return torch.where(mask[:, None], null_ids, ids)
