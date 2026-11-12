"""Metadata collection and bounded, DNS-pinned HTTPS image retrieval."""
import hashlib
import http.client
import io
import ipaddress
import os
import re
import socket
import ssl
import threading
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit
import httpx
import numpy as np
from PIL import Image, ImageOps
from . import storage
from .config import HOSTS

Image.MAX_IMAGE_PIXELS = 24_000_000
NATURE = set('forest rainforest waterfall mountain mountains ocean beach river lake desert snow wildlife tree trees flowers sunset sunrise valley meadow glacier coast woodland landscape nature'.split())


def concepts(text):
    return set(re.findall(r'[a-z]+', text.lower()))


def validate_url(url: str, hosts=HOSTS):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None, 443) or parsed.fragment:
        raise ValueError('Only HTTPS image URLs without credentials, fragments, or custom ports are accepted.')
    if not parsed.hostname or parsed.hostname.lower() not in hosts:
        raise ValueError('Image host is not in ALLOWED_IMAGE_HOSTS.')
    addresses = {entry[4][0] for entry in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)}
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise ValueError('Image host resolves to a non-public address.')
    return parsed, sorted(addresses)[0]


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, address, timeout):
        super().__init__(host, timeout=timeout, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        sock = socket.create_connection((self.address, 443), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class ImageLoader:
    def __init__(self, hosts=HOSTS, max_bytes=12 * 1024 * 1024, cache_mb=None):
        self.hosts, self.max_bytes = hosts, max_bytes
        self.capacity = int(cache_mb if cache_mb is not None else os.getenv('MAX_IMAGE_CACHE_MB', '64')) * 1024**2
        self.cache, self.used = OrderedDict(), 0
        self.lock = threading.Lock()
        self.stats = Counter()
        self.last_request = {}

    def _fetch(self, url):
        for attempt in range(5):
            parsed, address = validate_url(url, self.hosts)
            host = parsed.hostname.lower()
            interval = 1.0 if host.endswith(('wikimedia.org', 'wikipedia.org')) else .2
            delay = interval - (time.monotonic() - self.last_request.get(host, 0.))
            if delay > 0:
                time.sleep(delay)
            self.last_request[host] = time.monotonic()
            conn = PinnedHTTPS(parsed.hostname, address, float(os.getenv('IMAGE_REQUEST_TIMEOUT', '15')))
            try:
                conn.request('GET', parsed.path + ('?' + parsed.query if parsed.query else ''), headers={
                    'User-Agent': 'NatureGen/1.0 (local research image validation)', 'Accept': 'image/*'})
                response = conn.getresponse()
                if response.status in (301, 302, 303, 307, 308):
                    url = urljoin(url, response.getheader('Location', ''))
                    continue
                if response.status == 429:
                    if attempt == 4:
                        raise ValueError('Image source is rate-limiting requests; retry validation later.')
                    retry_after = response.getheader('Retry-After', '')
                    try:
                        retry_seconds = max(0., float(retry_after))
                    except ValueError:
                        try:
                            retry_at = parsedate_to_datetime(retry_after)
                            if retry_at.tzinfo is None:
                                retry_at = retry_at.replace(tzinfo=timezone.utc)
                            retry_seconds = max(0., (retry_at - datetime.now(timezone.utc)).total_seconds())
                        except (TypeError, ValueError, OverflowError):
                            retry_seconds = min(2 ** attempt, 8.)
                    if retry_seconds > 30:
                        raise ValueError('Image source requested a long rate-limit wait; retry validation later.')
                    time.sleep(retry_seconds)
                    continue
                if response.status != 200:
                    raise ValueError(f'Image request returned HTTP {response.status}.')
                if response.getheader('Content-Type', '').split(';')[0].lower() not in ('image/jpeg', 'image/png', 'image/webp'):
                    raise ValueError('Unsupported image content type.')
                size = response.getheader('Content-Length')
                if size and int(size) > self.max_bytes:
                    raise ValueError('Image response exceeds size limit.')
                chunks, total = [], 0
                while True:
                    chunk = response.read(min(65536, self.max_bytes + 1 - total))
                    if not chunk:
                        break
                    chunks.append(chunk); total += len(chunk)
                    if total > self.max_bytes:
                        raise ValueError('Image response exceeds size limit.')
                return b''.join(chunks)
            finally:
                conn.close()
        raise ValueError('Too many image redirects.')

    def load(self, url, resolution=64):
        # Serial network access bounds concurrency and prevents duplicate fetches.
        with self.lock:
            key = (url, resolution)
            if key in self.cache:
                self.stats['cached'] += 1
                self.cache.move_to_end(key)
                return self.cache[key].copy()
            for attempt in range(3):
                try:
                    payload = self._fetch(url)
                    with Image.open(io.BytesIO(payload)) as source:
                        if min(source.size) < 32:
                            raise ValueError('Image is smaller than 32 pixels.')
                        source.load()
                        image = ImageOps.fit(ImageOps.exif_transpose(source).convert('RGB'), (resolution, resolution), method=Image.Resampling.LANCZOS)
                        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
                    self.stats['fetched'] += 1
                    while self.cache and self.used + array.nbytes > self.capacity:
                        _, old = self.cache.popitem(last=False); self.used -= old.nbytes
                    if array.nbytes <= self.capacity:
                        self.cache[key] = array; self.used += array.nbytes
                    return array.copy()
                except (OSError, http.client.HTTPException):
                    if attempt == 2:
                        self.stats['failed'] += 1
                        raise ValueError('Image transport failed after bounded retries.') from None
                    time.sleep(.5 * 2**attempt)
                except (ValueError, Image.DecompressionBombError):
                    self.stats['failed'] += 1
                    raise


loader = ImageLoader()


def fingerprint(array):
    image = Image.fromarray(((array.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype('uint8'))
    gray = np.asarray(image.convert('L').resize((9, 8)))
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return f'{sum(int(bit) << i for i, bit in enumerate(bits)):016x}'


def assign_splits():
    records = sorted(storage.all_records('pair'), key=lambda r: r['id'])
    groups = []
    for record in records:
        if not record.get('phash'):
            continue
        matching = [g for g in groups if any(
            record['image_url'] == other['image_url'] or
            (record['source'], record['source_id']) == (other['source'], other['source_id']) or
            (int(record['phash'], 16) ^ int(other['phash'], 16)).bit_count() <= 5 for other in g)]
        group = [record]
        for old in matching:
            group.extend(old); groups.remove(old)
        groups.append(group)
    for group in groups:
        identity = min(r['id'] for r in group)
        bucket = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 100
        split = 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'
        for record in group:
            storage.put('pair', {**record, 'split': split, 'duplicate_group': identity})


def import_pair(item):
    # A URL is checked at import and again on every network request.
    validate_url(item['image_url'])
    identity = hashlib.sha256(f"{item['source']}:{item['source_id']}".encode()).hexdigest()[:24]
    existing = storage.get('pair', identity)
    if existing:
        return existing
    return storage.put('pair', {**item, 'id': identity, 'review_status': 'pending', 'status': 'unchecked', 'split': None,
                                'created_at': time.time(), 'match_confidence': None, 'match_signals': {}})


def validate_records(update, cancelled):
    records = storage.all_records('pair')
    for index, record in enumerate(records):
        if cancelled():
            raise InterruptedError('Validation cancelled.')
        try:
            array = loader.load(record['image_url'])
            record.update(status='available', phash=fingerprint(array), last_checked=time.time(), error=None)
        except ValueError as error:
            record.update(status='unavailable', error=str(error), last_checked=time.time())
        storage.put('pair', record)
        update(current=index + 1, total=len(records))
    assign_splits()


def match_records(update, cancelled):
    records = storage.all_records('pair')
    for index, record in enumerate(records):
        if cancelled():
            raise InterruptedError('Matching cancelled.')
        caption = concepts(record['caption'])
        metadata = concepts(' '.join(record.get('tags', [])) + ' ' + record.get('description', ''))
        overlap = sorted(caption & metadata)
        record['match_signals'] = {'caption_concepts': sorted(caption & NATURE), 'metadata_overlap': overlap,
                                   'coverage': len(overlap) / max(len(caption), 1)}
        # Lexical overlap is evidence, not a calibrated probability or approval.
        storage.put('pair', record)
        update(current=index + 1, total=len(records))


def provider_unavailable(provider):
    """Why a stock provider cannot be used, or None when it is configured and permitted."""
    if provider == 'huggingface':
        return None
    if os.getenv(f'{provider.upper()}_ML_PERMISSION', '').lower() != 'true':
        return f'{provider.title()} collection requires explicit ML permission; configure the backend only after obtaining it.'
    if not os.getenv('PEXELS_API_KEY' if provider == 'pexels' else 'UNSPLASH_ACCESS_KEY', ''):
        return f'Missing {provider} API key.'
    return None


def collect(request, update, cancelled):
    if request['provider'] != 'auto':
        reason = provider_unavailable(request['provider'])
        if reason:
            raise ValueError(reason)
        return {'imported': {request['provider']: collect_from(request['provider'], request, request['limit'], update, cancelled)}}
    # Hugging Face first; Pexels then Unsplash fill any shortfall, but only when configured and permitted.
    imported, skipped = {}, {}
    for provider in ('huggingface', 'pexels', 'unsplash'):
        remaining = request['limit'] - sum(imported.values())
        if remaining <= 0:
            break
        reason = provider_unavailable(provider)
        if reason:
            skipped[provider] = reason
            continue
        try:
            imported[provider] = collect_from(provider, request, remaining, update, cancelled)
        except httpx.HTTPError as error:
            skipped[provider] = f'{provider.title()} request failed: {error}'
    return {'imported': imported, 'skipped': skipped}


def collect_from(provider, request, limit, update, cancelled):
    before = len(storage.all_records('pair'))
    if provider == 'huggingface':
        collect_huggingface(request, limit, update, cancelled)
    else:
        collect_stock(provider, request, limit, update, cancelled)
    return len(storage.all_records('pair')) - before


def collect_stock(provider, request, limit, update, cancelled):
    query = request['query']
    key = os.getenv('PEXELS_API_KEY' if provider == 'pexels' else 'UNSPLASH_ACCESS_KEY', '')
    with httpx.Client(timeout=20, follow_redirects=False) as client:
        if provider == 'pexels':
            response = client.get('https://api.pexels.com/v1/search', params={'query': query, 'per_page': limit}, headers={'Authorization': key})
        else:
            response = client.get('https://api.unsplash.com/search/photos', params={'query': query, 'per_page': limit}, headers={'Authorization': f'Client-ID {key}'})
        response.raise_for_status()
        rows = response.json().get('photos' if provider == 'pexels' else 'results', [])
    for i, row in enumerate(rows):
        if cancelled():
            raise InterruptedError('Collection cancelled.')
        caption = row.get('alt') if provider == 'pexels' else row.get('alt_description') or row.get('description')
        if not caption:
            continue
        import_pair({'source': provider, 'source_id': str(row['id']), 'caption': caption,
            'image_url': row['src']['large'] if provider == 'pexels' else row['urls']['regular'],
            'creator': row.get('photographer', '') if provider == 'pexels' else row['user']['name'],
            'attribution_url': row['url'] if provider == 'pexels' else row['links']['html'],
            'license_note': request['license_note'], 'category': next(iter(sorted(concepts(caption) & NATURE)), 'nature'),
            'description': caption, 'tags': [], 'width': row['width'], 'height': row['height']})
        update(current=i + 1, total=len(rows))


def collect_huggingface(request, limit, update, cancelled):
    # HF viewer returns bounded metadata rows; never downloads dataset image shards.
    with httpx.Client(timeout=30) as client:
        response = client.get('https://datasets-server.huggingface.co/rows', params={
            'dataset': 'google-research-datasets/conceptual_captions', 'config': 'unlabeled',
            'split': 'train', 'offset': request.get('offset', 0), 'length': limit})
        response.raise_for_status()
        rows = response.json()['rows']
    skipped = 0
    for i, entry in enumerate(rows):
        if cancelled():
            raise InterruptedError('Collection cancelled.')
        row = entry['row']
        if concepts(row['caption']) & NATURE:
            try:
                import_pair({'source': 'conceptual_captions', 'source_id': str(entry['row_idx']),
                    'caption': row['caption'], 'image_url': row['image_url'], 'creator': '', 'attribution_url': '',
                    'license_note': request['license_note'], 'tags': [], 'description': row['caption'],
                    'category': sorted(concepts(row['caption']) & NATURE)[0]})
            except ValueError:
                skipped += 1
        else:
            skipped += 1
        update(current=i + 1, total=len(rows), skipped=skipped)


def stats():
    records = storage.all_records('pair')
    available = [r for r in records if r.get('status') == 'available']
    invalid = [r for r in records if r.get('status') == 'unavailable']
    approved = [r for r in records if r.get('review_status') == 'approved' and r.get('status') == 'available']
    train_rows, validation_rows = training_splits()
    categories = Counter(r.get('category') for r in records if r.get('category'))
    sources = Counter(r.get('source') for r in records if r.get('source'))
    splits = Counter(r.get('split') for r in records if r.get('split'))
    return {'total': len(records), 'available': len(available), 'invalid': len(invalid),
            'approved': len(approved), 'categories': dict(categories), 'sources': dict(sources),
            'splits': dict(splits), 'loader': dict(loader.stats),
            'training_readiness': training_readiness(train_rows, validation_rows)}


def approved_training_records():
    return [r for r in storage.all_records('pair')
            if r.get('review_status') == 'approved' and r.get('status') == 'available' and r.get('phash')]


def training_splits(records=None):
    approved = approved_training_records() if records is None else records
    return ([r for r in approved if r.get('split') == 'train'],
            [r for r in approved if r.get('split') == 'validation'])


def training_readiness(train_rows=None, validation_rows=None):
    if train_rows is None or validation_rows is None:
        train_rows, validation_rows = training_splits()
    ready = len(train_rows) >= 2 and len(validation_rows) >= 1
    return {'ready': ready, 'train_pairs': len(train_rows), 'validation_pairs': len(validation_rows),
            'message': None if ready else
                'Training needs at least 2 approved, available training pairs and 1 approved, available validation pair.'}
