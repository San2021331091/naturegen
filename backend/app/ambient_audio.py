"""Lightweight synthesized nature ambience; no additional model downloads."""
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def select_ambience(prompt, scene, audio_style='auto', prompt_fallback=False):
    """Use visible region coverage to gate sounds; text only refines their type."""
    text = prompt.lower()
    # Remove explicitly excluded subjects before interpreting descriptive words.
    text = re.sub(r'\b(?:no|without|absence of)\b[^,.;]*(?=[,.;]|$)', '', text)
    words = set(re.findall(r'[a-z]+', text))
    if audio_style != 'auto':
        choices = {
            'quiet': ('quiet', False),
            'wind': ('soft wind', False),
            'stream': ('stream', False),
            'waterfall': ('waterfall', False),
            'surf': ('surf', False),
            'birds': ('soft wind', True),
        }
        return choices[audio_style]
    has_scene_coverage = any(scene.values())
    kind = 'quiet'
    if scene.get('waterfall', 0) >= 0.015 or (prompt_fallback and not has_scene_coverage and words & {'waterfall', 'falls'}):
        kind = 'waterfall'
    elif scene.get('water', 0) >= 0.04 or (prompt_fallback and not has_scene_coverage and words & {'water', 'river', 'stream', 'brook', 'rapids', 'creek', 'ocean', 'sea', 'surf', 'waves', 'coast', 'beach'}):
        if words & {'ocean', 'sea', 'surf', 'waves', 'coast', 'beach'}:
            kind = 'surf'
        elif words & {'river', 'stream', 'brook', 'rapids', 'creek'}:
            kind = 'stream'
        else:
            kind = 'gentle water'
    elif scene.get('foliage', 0) >= 0.08 or (prompt_fallback and not has_scene_coverage and words & {'forest', 'trees', 'leaves', 'foliage', 'jungle', 'woodland'}):
        kind = 'soft foliage'
    # A forest alone does not imply audible birds. Generic chirps also cannot
    # represent specific species, so require an explicit birdsong request.
    birds = bool(words & {'birdsong', 'chirping', 'chirps'}) and scene.get('foliage', 0) >= 0.08
    return kind, birds


def add_ambient_audio(path, prompt, duration, seed=42, scene=None, audio_style='auto',
                      transition_times=None, transition_sound=False):
    path = Path(path)
    prompt_fallback = scene is None
    scene = scene or {}
    kind, birds = select_ambience(prompt, scene, audio_style, prompt_fallback)
    rate = 22050
    rng = np.random.default_rng(seed)
    t = np.arange(round(duration * rate), dtype=np.float32) / rate
    noise = rng.normal(0, 1, len(t)).astype(np.float32)
    smooth = np.convolve(noise, np.ones(32) / 32, mode='same')
    if kind == 'waterfall':
        signal = (0.055 * noise + 0.3 * smooth) * (0.95 + 0.05 * np.sin(t))
    elif kind == 'surf':
        signal = (0.04 * noise + 0.28 * smooth) * (0.15 + 0.85 * np.sin(np.pi * t / 7) ** 4)
    elif kind == 'stream':
        signal = (0.018 * noise + 0.16 * smooth) * (0.8 + 0.2 * np.sin(t * 3.1) ** 2)
    elif kind == 'gentle water':
        signal = 0.045 * smooth * (0.6 + 0.4 * np.sin(t * 1.3) ** 2)
    elif kind in ('soft foliage', 'soft wind'):
        signal = (0.003 * noise + 0.06 * smooth) * (0.4 + 0.6 * np.sin(np.pi * t / 9) ** 2)
    else:
        signal = np.zeros_like(t)
    if birds:
        kind += ' and bird chirps'
        for start in np.arange(1.0, duration - 0.4, 3.7):
            start += rng.uniform(-0.3, 0.3)
            a = int(start * rate)
            n = int(0.28 * rate)
            u = np.arange(n) / rate
            chirp = np.sin(2 * np.pi * (2400 * u + 2200 * u * u))
            signal[a:a+n] += 0.09 * chirp * np.sin(np.pi * u / 0.28) ** 2
    transition_times = transition_times or []
    if transition_sound:
        for start in transition_times:
            a = max(0, round(start * rate))
            n = min(round(0.65 * rate), len(signal) - a)
            if n <= 0:
                continue
            u = np.arange(n, dtype=np.float32) / rate
            phase = 2 * np.pi * 880 * u
            envelope = np.sin(np.pi * np.arange(n, dtype=np.float32) / max(1, n)) ** 2
            burst = (0.075 * np.sin(phase) + 0.035 * np.sin(phase * 1.5)) * np.exp(-5 * u)
            signal[a:a+n] += burst * envelope
    signal *= np.minimum(1, t / 0.5) * np.minimum(1, (duration - t) / 0.8)
    pcm = (np.clip(signal, -1, 1) * 32767).astype('<i2')
    with tempfile.TemporaryDirectory(prefix='ambient-', dir=path.parent) as temporary:
        audio = Path(temporary) / 'ambient.wav'
        output = Path(temporary) / 'with-audio.mp4'
        with wave.open(str(audio), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(pcm.tobytes())
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-v', 'error',
            '-i', str(path), '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '96k', '-t', str(duration),
            '-movflags', '+faststart', str(output),
        ], check=True, capture_output=True, timeout=120)
        output.replace(path)
    return {'type': 'synthesized ambience', 'description': 'transition chime' if transition_sound and transition_times else kind, 'duration': duration,
            'selection': 'scene matched' if audio_style == 'auto' else 'user selected',
            'audio_style': audio_style, 'transition_sound': bool(transition_sound),
            'transition_effect_count': len(transition_times) if transition_sound else 0,
            'scene_coverage': scene}
