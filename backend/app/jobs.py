import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from . import storage

executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='naturegen')
lock = threading.Lock()
events = {}
stop_requests = set()


def start(kind, task, config=None):
    with lock:
        active = [j for j in storage.all_records('job') if j.get('state') in ('queued', 'running', 'stopping')]
        if any((j.get('kind') == kind) or (j.get('kind') in ('training', 'generation') and kind in ('training', 'generation'))
               or (str(j.get('kind', '')).startswith('dataset') and kind.startswith('dataset')) for j in active):
            raise ValueError('A conflicting job is already active. Wait or cancel it first.')
        if len(active) >= 2:
            raise ValueError('Worker capacity reached. Wait for an active job to finish.')
        key = uuid.uuid4().hex
        event = threading.Event()
        events[key] = event
        job = storage.put('job', {'id': key, 'kind': kind, 'state': 'queued', 'current': 0, 'total': 0,
                                'created_at': time.time(), 'updated_at': time.time(), 'config': config or {}, 'error': None})

    def update(**fields):
        with lock:
            current = storage.get('job', key)
            return storage.put('job', {**current, **fields, 'updated_at': time.time()})

    def run():
        try:
            update(state='running')
            result = task(update, event.is_set)
            with lock:
                terminal = 'stopped' if key in stop_requests else 'cancelled' if event.is_set() else 'completed'
            update(state=terminal, result=result)
        except InterruptedError as error:
            with lock:
                terminal = 'stopped' if key in stop_requests else 'cancelled'
            update(state=terminal, error=None if terminal == 'stopped' else str(error))
        except Exception as error:
            # Avoid leaking signed URLs/API credentials through third-party exceptions.
            message = str(error) if isinstance(error, ValueError) else f'{type(error).__name__}: job failed. Check configuration and connectivity.'
            update(state='failed', error=message)
        finally:
            with lock:
                events.pop(key, None)
                stop_requests.discard(key)
    executor.submit(run)
    return job


def cancel(key):
    with lock:
        event = events.get(key)
        if not event:
            raise ValueError('Job is not active.')
        event.set()
        job = storage.get('job', key)
        return storage.put('job', {**job, 'state': 'stopping', 'updated_at': time.time()})


def stop(key):
    with lock:
        event = events.get(key)
        if not event:
            raise ValueError('Job is not active.')
        stop_requests.add(key)
        event.set()
        job = storage.get('job', key)
        return storage.put('job', {**job, 'state': 'stopping', 'updated_at': time.time()})
