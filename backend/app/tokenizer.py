"""Deterministic character BPE, trained only on this project's training captions."""
import hashlib
import json
import unicodedata
from collections import Counter


class BPETokenizer:
    special = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<NULL>']

    def __init__(self, vocab=None, merges=None, max_length=64):
        self.vocab = vocab or {s: i for i, s in enumerate(self.special)}
        self.merges = [tuple(p) for p in (merges or [])]
        self.max_length = max_length

    @staticmethod
    def normalize(text):
        return ' '.join(unicodedata.normalize('NFKC', text).lower().split())

    @staticmethod
    def merge(tokens, pair):
        result, i = [], 0
        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                result.append(tokens[i] + tokens[i + 1]); i += 2
            else:
                result.append(tokens[i]); i += 1
        return result

    def train(self, captions, vocab_size=12000):
        sequences = [list(self.normalize(t)) for t in captions if self.normalize(t)]
        if not sequences:
            raise ValueError('Tokenizer needs nonempty training captions.')
        self.vocab = {s: i for i, s in enumerate(self.special)}
        for char in sorted(set(c for seq in sequences for c in seq)):
            self.vocab[char] = len(self.vocab)
        self.merges = []
        while len(self.vocab) < vocab_size:
            counts = Counter(pair for seq in sequences for pair in zip(seq, seq[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            if counts[pair] < 2:
                break
            self.merges.append(pair)
            joined = ''.join(pair)
            if joined not in self.vocab:
                self.vocab[joined] = len(self.vocab)
            sequences = [self.merge(seq, pair) for seq in sequences]
        return self

    def encode(self, text, null=False):
        tokens = list(self.normalize(text))
        for pair in self.merges:
            tokens = self.merge(tokens, pair)
        ids = [2, 4, 3] if null else [2] + [self.vocab.get(t, 1) for t in tokens][:self.max_length - 2] + [3]
        return ids + [0] * (self.max_length - len(ids))

    def decode(self, ids):
        inverse = {i: s for s, i in self.vocab.items()}
        return ''.join(inverse.get(i, '<UNK>') for i in ids if i not in (0, 2, 3, 4))

    def state(self):
        return {'vocab': self.vocab, 'merges': self.merges, 'max_length': self.max_length, 'normalization': 'NFKC-lower-whitespace-v1'}

    @classmethod
    def restore(cls, state):
        if state.get('normalization') != 'NFKC-lower-whitespace-v1':
            raise ValueError('Unsupported tokenizer normalization.')
        return cls(state['vocab'], state['merges'], state['max_length'])

    def checksum(self):
        return hashlib.sha256(json.dumps(self.state(), sort_keys=True).encode()).hexdigest()
