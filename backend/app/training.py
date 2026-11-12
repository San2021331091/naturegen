"""Joint text/U-Net training, EMA, deterministic validation, atomic resume."""
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
import uuid
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from . import storage
from .config import CHECKPOINTS, RESULTS
from .dataset import approved_training_records, loader, training_splits
from .diffusion import Diffusion, condition_dropout
from .model import ModelConfig, NatureModel
from .stream import ImageStream
from .replay import ReplayPool, image_hash, split_validation
from .tokenizer import BPETokenizer

torch.set_num_threads(int(os.getenv('TORCH_NUM_THREADS', '4')))


def device():
    requested = os.getenv('DEVICE', 'auto')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable.')
    return torch.device('cuda' if (requested == 'auto' and torch.cuda.is_available()) else 'cpu' if requested == 'auto' else requested)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']; np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


class EMA:
    def __init__(self, model, decay=.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, source, step=None):
        # Warm up the decay so short runs are not dominated by the random initialisation.
        decay = self.decay if step is None else min(self.decay, (1 + step) / (10 + step))
        for target, value in zip(self.model.parameters(), source.parameters()):
            target.lerp_(value, 1 - decay)
        for target, value in zip(self.model.buffers(), source.buffers()):
            target.copy_(value)


def atomic_save(payload, path):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    if path.exists():
        backup = path.with_suffix('.previous.pt')
        # Keep the previous valid checkpoint until replacement is complete.
        import shutil
        shutil.copy2(path, backup)
    # Windows refuses the replace while a reader (e.g. a generation job) has the file open.
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.25)


def load_checkpoint(path, target='cpu'):
    if not path.is_file():
        raise ValueError('No trained checkpoint exists. Train a model on approved pairs first.')
    state = torch.load(path, map_location=target, weights_only=True)
    if state.get('version') != 1 or state['model_config'].get('prediction_type') != 'v' or state['model_config'].get('schedule') != 'cosine':
        raise ValueError('Incompatible checkpoint format, schedule, or prediction target.')
    tokenizer = BPETokenizer.restore(state['tokenizer'])
    if tokenizer.checksum() != state['tokenizer_checksum']:
        raise ValueError('Checkpoint tokenizer checksum mismatch.')
    config = ModelConfig(**state['model_config'])
    model = NatureModel(len(tokenizer.vocab), config).to(target)
    model.load_state_dict(state['model'], strict=True)
    return state, tokenizer, config, model


def manifest(records):
    values = [{key: row.get(key) for key in ('id', 'caption', 'image_url', 'split', 'phash', 'review_status', 'license_note')} for row in sorted(records, key=lambda r: r['id'])]
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def batch(records, tokenizer, resolution, target):
    images, captions = [], []
    for row in records:
        try:
            images.append(loader.load(row['image_url'], resolution)); captions.append(row['caption'])
        except ValueError as error:
            storage.put('pair', {**row, 'status': 'unavailable', 'error': str(error)})
            loader.stats['quarantined'] += 1
    if not images:
        return None
    return torch.from_numpy(np.stack(images)).to(target), torch.tensor([tokenizer.encode(c) for c in captions], device=target)


@torch.no_grad()
def validate(model, diffusion, rows, tokenizer, config, batch_size, target):
    # Validation consumes no training RNG state and never applies augmentation.
    before = rng_state(); seed_all(771)
    model.eval(); losses = []
    try:
        for offset in range(0, len(rows), batch_size):
            loaded = batch(rows[offset:offset + batch_size], tokenizer, config.resolution, target)
            if loaded is None:
                continue
            images, ids = loaded
            t = torch.randint(config.timesteps, (len(images),), device=target)
            noisy, expected = diffusion.forward(images, torch.randn_like(images), t)
            loss = (model(noisy, t, ids) - expected).square().mean()
            losses.append(float(loss))
    finally:
        restore_rng(before)
    if not losses:
        raise ValueError('No validation images could be loaded.')
    return sum(losses) / len(losses)


def train(request, update, cancelled, resume=False):
    return locked(lambda: _train(request, update, cancelled, resume))


def stream_train(request, update, cancelled):
    return locked(lambda: _stream_train(request, update, cancelled))


def locked(task):
    # Exclusive file ownership also protects against a second backend process.
    lock_path = CHECKPOINTS / 'training.lock'
    try:
        handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError('Checkpoint directory is locked. If the previous process crashed, verify it is stopped before removing training.lock.') from None
    try:
        os.write(handle, str(os.getpid()).encode())
        keep_awake(True)
        return task()
    finally:
        keep_awake(False)
        os.close(handle); lock_path.unlink(missing_ok=True)


def keep_awake(on):
    # Idle sleep silently suspends a long CPU run; hold Windows awake only while training owns the lock.
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def _train(request, update, cancelled, resume):
    target = device(); seed_all(request['seed'])
    records = approved_training_records()
    train_rows, val_rows = training_splits(records)
    train_rows = sorted(train_rows, key=lambda r: r['id'])
    val_rows = sorted(val_rows, key=lambda r: r['id'])
    if len(train_rows) < 2 or not val_rows:
        raise ValueError('Training needs at least 2 approved training pairs and 1 approved validation pair. Import, validate, and review more records.')
    version = manifest(records)
    if resume:
        state, tokenizer, config, model = load_checkpoint(CHECKPOINTS / 'latest.pt', target)
        if state['manifest'] != version:
            raise ValueError('Dataset manifest changed. Start a new run instead of resuming.')
        previous = state['training_config']
        # Preserve optimizer schedule and architecture; epochs is the total run target.
        for field in ('batch_size', 'learning_rate', 'text_learning_rate', 'accumulation', 'seed', 'epochs', 'base_channels', 'text_dim', 'text_layers', 'timesteps'):
            if previous[field] != request[field]:
                raise ValueError(f'Resume requires the original {field} setting.')
    else:
        config = ModelConfig(base_channels=request['base_channels'], text_dim=request['text_dim'], text_layers=request['text_layers'], timesteps=request['timesteps'], spatial_self_attention=request.get('spatial_self_attention', True))
        tokenizer = BPETokenizer().train([r['caption'] for r in train_rows])
        model = NatureModel(len(tokenizer.vocab), config).to(target)
        state = None
    optimizer = torch.optim.AdamW([{'params': model.unet.parameters(), 'lr': request['learning_rate']},
                                   {'params': model.text.parameters(), 'lr': request['text_learning_rate']}], weight_decay=1e-4)
    count = math.ceil(len(train_rows) / request['batch_size'])
    updates = request['epochs'] * math.ceil(count / request['accumulation'])
    warmup = max(1, min(100, updates // 10))
    def schedule(step):
        return (step + 1) / warmup if step < warmup else .5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, updates - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=target.type == 'cuda')
    ema = EMA(model, request['ema_decay'])
    diffusion = Diffusion(config.timesteps, target)
    epoch, cursor, global_step = 0, 0, 0
    order, history = list(range(len(train_rows))), []
    if state:
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']); ema.model.load_state_dict(state['ema'])
        epoch, cursor, global_step = state['epoch'], state['cursor'], state['global_step']
        order, history = state['order'], state['history']; restore_rng(state['rng'])
    started = time.time()
    run_id = state['run_id'] if state else uuid.uuid4().hex
    def save(ep, position):
        payload = {'version': 1, 'run_id': run_id, 'model': model.state_dict(), 'ema': ema.model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'model_config': config.dict(), 'training_config': request, 'tokenizer': tokenizer.state(), 'tokenizer_checksum': tokenizer.checksum(),
            'manifest': version, 'epoch': ep, 'cursor': position, 'order': order, 'global_step': global_step, 'rng': rng_state(),
            'history': history, 'software': {'python': platform.python_version(), 'torch': str(torch.__version__)},
            'device': str(target), 'saved_at': time.time(), 'elapsed_seconds': time.time() - started,
            'experimental': True, 'trained_on_real_pairs': True}
        atomic_save(payload, CHECKPOINTS / 'latest.pt')
    update(parameters=sum(p.numel() for p in model.parameters()), device=str(target), manifest=version, run_id=run_id)
    for ep in range(epoch, request['epochs']):
        if cursor == 0:
            random.shuffle(order)
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses, accumulated = [], 0
        for index in range(cursor, count):
            if cancelled() and accumulated == 0:
                save(ep, index); return {'checkpoint': 'latest.pt', 'global_step': global_step}
            selected = [train_rows[i] for i in order[index * request['batch_size']:(index + 1) * request['batch_size']]]
            loaded = batch(selected, tokenizer, config.resolution, target)
            if loaded is not None:
                images, ids = loaded
                null_ids = torch.tensor([tokenizer.encode('', null=True)] * len(ids), device=target)
                ids = condition_dropout(ids, null_ids, request['cfg_dropout'])
                t = torch.randint(config.timesteps, (len(images),), device=target)
                noisy, expected = diffusion.forward(images, torch.randn_like(images), t)
                with torch.autocast(device_type=target.type, enabled=target.type == 'cuda', dtype=torch.float16):
                    loss = (model(noisy, t, ids) - expected).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError('Training loss became nonfinite; previous checkpoint retained.')
                scaler.scale(loss).backward(); accumulated += 1
                losses.append(float(loss.detach()))
            if accumulated and (accumulated >= request['accumulation'] or index == count - 1 or cancelled()):
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                scheduler.step(); ema.update(model, global_step); global_step += 1; accumulated = 0
                if global_step % 25 == 0 or cancelled():
                    save(ep, index + 1)
            update(epoch=ep + 1, current=index + 1, total=count, global_step=global_step,
                   loss=losses[-1] if losses else None, learning_rate=optimizer.param_groups[0]['lr'],
                   gpu_memory=torch.cuda.memory_allocated() if target.type == 'cuda' else 0)
        if cancelled():
            save(ep, count); return {'checkpoint': 'latest.pt', 'global_step': global_step}
        if not losses and cursor == 0:
            raise ValueError('No training images could be loaded.')
        validation = validate(ema.model, diffusion, val_rows, tokenizer, config, request['batch_size'], target)
        metric = {'id': f'{run_id}-{ep}', 'run_id': run_id, 'epoch': ep + 1, 'loss': sum(losses) / len(losses) if losses else None,
                  'validation_loss': validation, 'global_step': global_step, 'created_at': time.time()}
        history.append(metric); storage.put('metric', metric)
        update(validation_loss=validation)
        save(ep + 1, 0); cursor = 0
    return {'checkpoint': 'latest.pt', 'global_step': global_step, 'run_id': run_id}


VALIDATION_IMAGES = 64


def stream_scheduler(optimizer, request, state=None):
    """Rebuild the saved schedule; extending a run gets a short, explicit warmup."""
    step = state['global_step'] if state else 0
    spec = state.get('lr_schedule') if state else None
    if spec is None:
        total = state['training_config']['steps'] if state else request['steps']
        spec = {'start': 0, 'total': total, 'warmup': max(1, min(500, total // 20))}
    if spec['total'] != request['steps']:
        spec = {'start': step, 'total': request['steps'],
                'warmup': max(1, min(500, (request['steps'] - step) // 20))}
    def factor(current):
        elapsed = current - spec['start']
        if elapsed < spec['warmup']:
            return (elapsed + 1) / spec['warmup']
        progress = (elapsed - spec['warmup']) / max(1, spec['total'] - spec['start'] - spec['warmup'])
        return .5 * (1 + math.cos(math.pi * min(1, progress)))
    for group, rate in zip(optimizer.param_groups, (request['learning_rate'], request['text_learning_rate'])):
        group['lr'] = group['initial_lr'] = rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor, last_epoch=step - 1)
    return scheduler, spec


def _stream_train(request, update, cancelled):
    """Train on a deduplicated, rehearsed pool. Save metadata, never source pixels."""
    request = {'validation_images': VALIDATION_IMAGES, 'min_replays': 8, 'preview_every': 1000, **request}
    target = device(); seed_all(request['seed'])
    if request['resume'] and not (CHECKPOINTS / 'latest.pt').is_file():
        raise ValueError('No checkpoint exists to resume.')
    resuming = request['resume'] and (CHECKPOINTS / 'latest.pt').is_file()
    state = None
    if resuming:
        state, tokenizer, config, model = load_checkpoint(CHECKPOINTS / 'latest.pt', target)
        if state.get('mode') != 'stream':
            raise ValueError('The latest checkpoint is not a streaming run. Start a new streaming run instead.')
        if state['global_step'] >= request['steps']:
            raise ValueError(f"The checkpoint already reached {state['global_step']} steps. Raise total training steps to continue it.")
        for field in ('base_channels', 'text_dim', 'text_layers', 'timesteps'):
            if request[field] != getattr(config, field):
                raise ValueError(f'Resume cannot change {field}; use the saved architecture.')
    stream = ImageStream(64, state['stream'] if state else None)
    try:
        update(phase='filling image pool', current=0, total=request['buffer_size'])
        if state:
            held_out = [(c, url, loader.load(url, 64)) for c, url in state['validation']]
            if state.get('validation_hashes') and [image_hash(im) for _, _, im in held_out] != state['validation_hashes']:
                raise ValueError('Validation images changed at their URLs.')
        else:
            held_out = []
        pool = ReplayPool(request['buffer_size'], request['min_replays'], held_out, (state or {}).get('replay'))
        if state and state.get('replay'):
            pool.restore(state['replay'], loader.load, config.resolution)
        else:
            initial = ReplayPool(request['buffer_size'] + (0 if state else request['validation_images']), validation=held_out)
            attempts = 0
            while len(initial.items) < initial.capacity:
                if cancelled():
                    raise InterruptedError('Streaming cancelled while filling the image pool.')
                attempts += 1
                if attempts > initial.capacity * 30:
                    raise ValueError('Not enough distinct images to fill the pool. Reduce buffer_size or review the source.')
                initial.offer(stream.take()); update(current=len(initial.items), total=initial.capacity)
            if not state:
                held_out, training_items = split_validation(initial.items, request['validation_images'])
                pool = ReplayPool(request['buffer_size'], request['min_replays'], held_out)
            else:
                training_items = initial.items
            for item in training_items:
                pool.offer(item)
        validation = [(c, image) for c, _, image in held_out]
        if not state:
            state = {'validation': [(c, url) for c, url, _ in held_out]}
            config = ModelConfig(base_channels=request['base_channels'], text_dim=request['text_dim'], text_layers=request['text_layers'], timesteps=request['timesteps'], spatial_self_attention=request.get('spatial_self_attention', True))
            # Character BPE still encodes words first seen later, so the vocabulary can be fixed up front.
            tokenizer = BPETokenizer().train([c for c, _, _ in pool.items], vocab_size=4000)
            model = NatureModel(len(tokenizer.vocab), config).to(target)
        optimizer = torch.optim.AdamW([{'params': model.unet.parameters(), 'lr': request['learning_rate']},
                                       {'params': model.text.parameters(), 'lr': request['text_learning_rate']}], weight_decay=1e-4)
        ema = EMA(model, request['ema_decay'])
        global_step, seen, history = 0, len(pool.items), []
        run_id = uuid.uuid4().hex
        if 'model' in state:
            optimizer.load_state_dict(state['optimizer'])
            ema.model.load_state_dict(state['ema'])
            global_step, seen, history, run_id = state['global_step'], state['images_seen'], state['history'], state['run_id']
            if 'rng' in state:
                restore_rng(state['rng'])
        scheduler, lr_schedule = stream_scheduler(optimizer, request, state if 'model' in state else None)
        diffusion = Diffusion(config.timesteps, target)
        null = tokenizer.encode('', null=True)
        started = time.time()
        best_validation = state.get('best_validation', float('inf'))
        def save(path=None):
            atomic_save({'version': 1, 'mode': 'stream', 'run_id': run_id, 'model': model.state_dict(), 'ema': ema.model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model_config': config.dict(),
                'training_config': {**request, 'resume': False}, 'tokenizer': tokenizer.state(), 'tokenizer_checksum': tokenizer.checksum(),
                'manifest': 'stream', 'global_step': global_step, 'images_seen': seen, 'history': history,
                'stream': stream.state, 'validation': state['validation'],
                'validation_hashes': [image_hash(im) for _, _, im in held_out],
                'replay': pool.state(), 'rng': rng_state(), 'lr_schedule': lr_schedule,
                'best_validation': best_validation, 'validation_protocol': 'four-noise-levels-v1',
                'software': {'python': platform.python_version(), 'torch': str(torch.__version__)}, 'device': str(target),
                'saved_at': time.time(), 'elapsed_seconds': time.time() - started, 'experimental': True, 'trained_on_real_pairs': True}, path or CHECKPOINTS / 'latest.pt')
        update(phase='training', parameters=sum(p.numel() for p in model.parameters()), device=str(target), run_id=run_id,
               current=global_step, total=request['steps'])
        losses = []
        model.train()
        while global_step < request['steps']:
            if cancelled():
                save(); return {'checkpoint': 'latest.pt', 'global_step': global_step, 'images_seen': seen}
            picks = pool.sample(request['batch_size'])
            # Preserve caption directionality; a caption may explicitly say left/right.
            images = torch.from_numpy(np.stack([image for _, _, image in picks])).to(target)
            ids = torch.tensor([tokenizer.encode(c) for c, _, _ in picks], device=target)
            ids = condition_dropout(ids, torch.tensor([null] * len(ids), device=target), request['cfg_dropout'])
            t = torch.randint(config.timesteps, (len(images),), device=target)
            noisy, expected = diffusion.forward(images, torch.randn_like(images), t)
            loss = (model(noisy, t, ids) - expected).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Training loss became nonfinite; previous checkpoint retained.')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); ema.update(model, global_step); global_step += 1
            losses.append(float(loss.detach()))
            # Swap in photos that finished downloading; slow downloads never stall training.
            for _ in range(request['fresh_per_step']):
                if not pool.can_refresh():
                    break
                item = stream.poll()
                if item is None:
                    break
                if pool.offer(item):
                    seen += 1
            update(current=global_step, global_step=global_step, loss=losses[-1], images_seen=seen,
                   unique_images=pool.admitted, duplicate_images=pool.duplicates,
                   learning_rate=optimizer.param_groups[0]['lr'])
            if global_step % request['validate_every'] == 0 or global_step == request['steps']:
                validation_loss = _validate_arrays(ema.model, diffusion, validation, tokenizer, config, target)
                metric = {'id': f'{run_id}-{global_step}', 'run_id': run_id, 'epoch': global_step // request['validate_every'],
                          'loss': sum(losses) / len(losses), 'validation_loss': validation_loss, 'global_step': global_step,
                          'images_seen': seen, 'created_at': time.time()}
                history.append(metric); storage.put('metric', metric); losses = []
                if validation_loss < best_validation:
                    best_validation = validation_loss
                    save(CHECKPOINTS / f'{run_id}-best.pt')
                update(validation_loss=validation_loss); save(); model.train()
            elif global_step % 100 == 0:
                save()
            if request['preview_every'] and (global_step % request['preview_every'] == 0 or global_step == request['steps']):
                from .evaluate import preview
                path = RESULTS / run_id / f'step-{global_step}.png'
                before = rng_state()
                try:
                    preview(ema.model, tokenizer, config, path, global_step)
                finally:
                    restore_rng(before)
                update(preview_path=str(path.relative_to(RESULTS)))
        return {'checkpoint': 'latest.pt', 'global_step': global_step, 'images_seen': seen, 'run_id': run_id}
    finally:
        stream.close()


@torch.no_grad()
def _validate_arrays(model, diffusion, items, tokenizer, config, target):
    before = rng_state(); seed_all(771)
    model.eval()
    try:
        total, count = 0., 0
        for offset in range(0, len(items), 8):
            chunk = items[offset:offset + 8]
            images = torch.from_numpy(np.stack([image for _, image in chunk])).to(target)
            ids = torch.tensor([tokenizer.encode(c) for c, _ in chunk], device=target)
            for fraction in (.1, .35, .65, .9):
                t = torch.full((len(images),), min(config.timesteps - 1, int(config.timesteps * fraction)), device=target, dtype=torch.long)
                noisy, expected = diffusion.forward(images, torch.randn_like(images), t)
                total += float((model(noisy, t, ids) - expected).square().mean()) * len(images)
                count += len(images)
        return total / count
    finally:
        restore_rng(before)


def generate(request, update, cancelled):
    target = device()
    state, tokenizer, config, model = load_checkpoint(CHECKPOINTS / 'latest.pt', target)
    if not state.get('trained_on_real_pairs') or state['global_step'] < 1:
        raise ValueError('Checkpoint has not been trained on approved real image-caption pairs.')
    if request['resolution'] != config.resolution:
        raise ValueError(f'Loaded checkpoint supports {config.resolution}x{config.resolution} only.')
    model.load_state_dict(state['ema'], strict=True)
    ids = torch.tensor([tokenizer.encode(request['prompt'])], device=target)
    null = torch.tensor([tokenizer.encode('', null=True)], device=target)
    diffusion = Diffusion(config.timesteps, target)
    tensor = diffusion.sample(model, ids, null, config.resolution, request['steps'], request['guidance'], request['seed'], request['sampler'],
                              lambda current, total: update(current=current, total=total), cancelled)
    key = uuid.uuid4().hex
    image = Image.fromarray(((tensor[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype('uint8'))
    image.save(RESULTS / f'{key}.png')
    result = {**request, 'id': key, 'image_url': f'/api/generated/{key}.png', 'created_at': time.time(),
              'checkpoint_step': state['global_step'], 'run_id': state['run_id'], 'experimental': True}
    storage.put('generation', result)
    return result
