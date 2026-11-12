"""Bounded image replay with exact decoded-image deduplication and caption groups.

Only URL/caption/hash metadata is serialized; source image bytes stay in RAM.
Caption groups are a sampling heuristic, not verified semantic labels.
"""
import hashlib
import random
import re
from collections import Counter, defaultdict
from urllib.parse import urlsplit

import numpy as np


def image_hash(image):
    pixels = ((image + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def source_key(url):
    parts = urlsplit(url)
    # Provider resizing/crop query parameters do not identify a different photo.
    return parts.netloc.lower() + parts.path


def caption_group(caption):
    words = set(re.findall(r'[a-z]+', caption.lower()))
    for group, terms in (
        ('waterfall', {'waterfall', 'waterfalls'}),
        ('mountain', {'mountain', 'mountains', 'peak', 'peaks', 'alpine'}),
        ('coast', {'ocean', 'sea', 'beach', 'coast', 'coastal', 'waves'}),
        ('forest', {'forest', 'trees', 'tree', 'woods', 'jungle', 'canopy'}),
        ('desert', {'desert', 'dunes', 'canyon'}),
        ('freshwater', {'lake', 'river', 'pond', 'stream'}),
        ('field', {'field', 'meadow', 'flowers', 'hills', 'grass'}),
        ('sky', {'sky', 'stars', 'clouds', 'sunset'}),
    ):
        if words & terms:
            return group
    return 'other'


def split_validation(items, count):
    """Spread held-out examples across the caption groups present in the pool."""
    groups = defaultdict(list)
    for item in items:
        groups[caption_group(item[0])].append(item)
    for rows in groups.values():
        random.shuffle(rows)
    held_out = []
    while len(held_out) < count:
        for group in sorted(groups):
            if groups[group] and len(held_out) < count:
                held_out.append(groups[group].pop())
        if not any(groups.values()):
            break
    return held_out, [item for rows in groups.values() for item in rows]


class ReplayPool:
    def __init__(self, capacity, min_replays=8, validation=(), state=None):
        self.capacity, self.min_replays = capacity, min_replays
        self.items, self.uses = [], []
        self.blocked_urls = {source_key(url) for _, url, _ in validation}
        self.blocked_hashes = {image_hash(im) for _, _, im in validation}
        self.seen_urls = set((state or {}).get('seen_urls', []))
        self.seen_hashes = set((state or {}).get('seen_hashes', []))
        self.admitted = (state or {}).get('admitted', 0)
        self.duplicates = (state or {}).get('duplicates', 0)

    def offer(self, item):
        caption, url, image = item
        key, digest = source_key(url), image_hash(image)
        if (key in self.blocked_urls or digest in self.blocked_hashes or
                key in self.seen_urls or digest in self.seen_hashes):
            self.duplicates += 1
            return False
        if len(self.items) == self.capacity:
            eligible = [i for i, uses in enumerate(self.uses) if uses >= self.min_replays]
            if not eligible:
                return False
            counts = Counter(caption_group(row[0]) for row in self.items)
            # Evict a well-rehearsed image from an overrepresented caption group.
            index = max(eligible, key=lambda i: (counts[caption_group(self.items[i][0])], self.uses[i]))
            self.items[index], self.uses[index] = item, 0
        else:
            self.items.append(item)
            self.uses.append(0)
        self.seen_urls.add(key); self.seen_hashes.add(digest)
        self.admitted += 1
        return True

    def can_refresh(self):
        return len(self.items) < self.capacity or any(n >= self.min_replays for n in self.uses)

    def sample(self, batch_size):
        groups = defaultdict(list)
        for i, item in enumerate(self.items):
            groups[caption_group(item[0])].append(i)
        chosen = []
        while len(chosen) < min(batch_size, len(self.items)):
            names = list(groups)
            random.shuffle(names)
            for name in names:
                if not groups[name] or len(chosen) >= batch_size:
                    continue
                least_used = min(self.uses[i] for i in groups[name])
                index = random.choice([i for i in groups[name] if self.uses[i] == least_used])
                groups[name].remove(index)
                chosen.append(index)
        for index in chosen:
            self.uses[index] += 1
        return [self.items[i] for i in chosen]

    def state(self):
        return {'items': [{'caption': c, 'url': u, 'hash': image_hash(im), 'uses': n}
                          for (c, u, im), n in zip(self.items, self.uses)],
                'seen_urls': sorted(self.seen_urls), 'seen_hashes': sorted(self.seen_hashes),
                'admitted': self.admitted, 'duplicates': self.duplicates}

    def restore(self, state, load, resolution):
        if len(state['items']) != self.capacity:
            raise ValueError('Resume requires the saved buffer_size.')
        for row in state['items']:
            image = load(row['url'], resolution)
            if image_hash(image) != row['hash']:
                raise ValueError('A replay image changed at its URL; refusing an inconsistent resume.')
            if source_key(row['url']) in self.blocked_urls or row['hash'] in self.blocked_hashes:
                raise ValueError('Replay image overlaps the validation set.')
            self.items.append((row['caption'], row['url'], image))
            self.uses.append(row['uses'])
