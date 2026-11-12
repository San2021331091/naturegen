"""Streamed Pexels/Unsplash training images: searched page by page and held in memory only, never written to disk."""
import os
import queue
import threading
import time
import httpx
from .dataset import loader, provider_unavailable

QUERIES = ['forest', 'waterfall', 'mountain lake', 'ocean waves', 'sunset beach', 'river valley', 'snowy mountains',
           'desert dunes', 'misty forest', 'flower meadow', 'canyon', 'autumn forest', 'tropical jungle', 'coastal cliffs',
           'night sky stars', 'green hills', 'pine trees', 'lake reflection', 'rocky coast', 'countryside field']
PER_PAGE = 30


def usable_providers():
    usable = [p for p in ('pexels', 'unsplash') if provider_unavailable(p) is None]
    if not usable:
        raise ValueError('Streaming needs Pexels or Unsplash configured with an API key and ML permission in backend/.env.')
    return usable


def search(client, provider, query, page, include_metadata=False, page_size=PER_PAGE):
    """Search a bounded provider page, optionally retaining attribution metadata."""
    if provider == 'pexels':
        response = client.get('https://api.pexels.com/v1/search', params={'query': query, 'page': page, 'per_page': page_size},
                              headers={'Authorization': os.getenv('PEXELS_API_KEY', '')})
        response.raise_for_status()
        rows = response.json().get('photos', [])
        if include_metadata:
            return [{'caption': row.get('alt') or '', 'image_url': row['src']['large'], 'source': 'pexels',
                     'source_id': str(row['id']), 'creator': row.get('photographer', ''),
                     'attribution_url': row.get('url', '')} for row in rows]
        return [(row.get('alt') or '', row['src']['medium']) for row in rows]
    response = client.get('https://api.unsplash.com/search/photos', params={'query': query, 'page': page, 'per_page': page_size},
                          headers={'Authorization': f"Client-ID {os.getenv('UNSPLASH_ACCESS_KEY', '')}"})
    response.raise_for_status()
    rows = response.json().get('results', [])
    if include_metadata:
        return [{'caption': row.get('alt_description') or row.get('description') or '', 'image_url': row['urls']['regular'],
                 'source': 'unsplash', 'source_id': str(row['id']), 'creator': row.get('user', {}).get('name', ''),
                 'attribution_url': row.get('links', {}).get('html', '')} for row in rows]
    return [(row.get('alt_description') or row.get('description') or '', row['urls']['small']) for row in rows]


class ImageStream:
    """Background producer of (caption, url, image array) items; `state` records search positions for resume."""

    def __init__(self, resolution, state=None, capacity=64, include_metadata=False, page_size=None):
        self.resolution, self.providers = resolution, usable_providers()
        self.include_metadata = include_metadata
        self.page_size = page_size or (4 if include_metadata else PER_PAGE)
        self.state = state or {'round': 0, 'pages': {}, 'exhausted': []}
        self.cooling = {}
        self.items, self.stopped, self.error = queue.Queue(maxsize=capacity), threading.Event(), None
        self.thread = threading.Thread(target=self._run, daemon=True, name='naturegen-stream')
        self.thread.start()

    def _run(self):
        try:
            with httpx.Client(timeout=20, follow_redirects=False) as client:
                empty = 0
                while not self.stopped.is_set():
                    rows = self._next_page(client)
                    if rows is None:
                        continue
                    empty = 0 if rows else empty + 1
                    if empty > len(QUERIES) * 2:
                        raise ValueError('Image search stopped returning results.')
                    for row in rows:
                        if self.include_metadata:
                            caption, url = row['caption'], row['image_url']
                        else:
                            caption, url = row
                        if self.stopped.is_set():
                            return
                        if len(caption.split()) < 2:
                            continue
                        try:
                            image = loader.load(url, self.resolution)
                        except ValueError:
                            continue
                        while not self.stopped.is_set():
                            try:
                                item = (caption, url, image, row) if self.include_metadata else (caption, url, image)
                                self.items.put(item, timeout=.5)
                                break
                            except queue.Full:
                                pass
        except Exception as error:
            if isinstance(error, ValueError):
                self.error = error
            elif isinstance(error, httpx.HTTPStatusError):
                self.error = ValueError(f'Image streaming failed with HTTP {error.response.status_code}.')
            else:
                self.error = ValueError(f'Image streaming failed ({type(error).__name__}).')

    def _next_page(self, client):
        if not self.providers:
            raise ValueError('Pexels and Unsplash both rejected the configured API keys or permissions.')
        ready = [p for p in self.providers if self.cooling.get(p, 0) <= time.time()]
        if not ready:
            # Every provider is over its hourly quota; training carries on with the current pool meanwhile.
            self.stopped.wait(30)
            return None
        round_ = self.state['round']; self.state['round'] += 1
        provider = ready[round_ % len(ready)]
        query = QUERIES[(round_ // len(ready)) % len(QUERIES)]
        key = f'{provider}:{query}'
        if key in self.state['exhausted']:
            return []
        page = self.state['pages'].get(key, 0) + 1
        self.state['pages'][key] = page
        try:
            rows = search(client, provider, query, page, self.include_metadata, self.page_size)
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status == 429:
                # Hourly quota spent: retry this page after a pause instead of dropping the provider for the whole run.
                self.cooling[provider] = time.time() + 600
                self.state['pages'][key] = page - 1
                return None
            if status in (401, 403):
                # Credentials or permissions failed for this provider; retain the other source.
                self.providers.remove(provider)
                return []
            # A bad page or temporary provider failure must not terminate a long training job.
            # The page cursor has already advanced, so skip this query and try another topic.
            self.state['exhausted'].append(key)
            return []
        if not rows:
            self.state['exhausted'].append(key)
        return rows

    def take(self, timeout=90):
        for _ in range(timeout):
            try:
                return self.items.get(timeout=1)
            except queue.Empty:
                if self.error:
                    raise self.error from None
        raise ValueError('No streamed images arrived. Check the network connection and API keys.')

    def poll(self):
        """A fresh item if one is ready, without waiting; raises once the producer has failed and drained."""
        try:
            return self.items.get_nowait()
        except queue.Empty:
            if self.error:
                raise self.error
            return None

    def close(self):
        self.stopped.set()
