import io
import socket
import time
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app import dataset, jobs, storage
from app.main import app
from app.tokenizer import BPETokenizer


def public_dns(*args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))]


def test_tokenizer_roundtrip_determinism():
    captions = ['A green forest at sunrise.', 'A green forest in rain.', 'A mountain at sunrise.']
    first = BPETokenizer().train(captions)
    second = BPETokenizer().train(captions)
    assert first.state() == second.state()
    assert first.decode(first.encode(captions[0])) == captions[0].lower()
    assert first.encode('', null=True)[:3] == [2, 4, 3]
    assert first.checksum() == BPETokenizer.restore(first.state()).checksum()
    assert first.encode('🙂')[1] == 1
    assert first.encode('a ' * 200)[-1] == 3


@pytest.mark.parametrize('url', ['http://images.test/a.png', 'https://127.0.0.1/a.png', 'https://images.test:444/a.png', 'https://user:secret@images.test/a.png', 'https://evil.test/a.png'])
def test_unsafe_url_rejected(url, monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', public_dns)
    with pytest.raises(ValueError):
        dataset.validate_url(url, {'images.test'})


def test_private_dns_rejected(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [(2, 1, 6, '', ('127.0.0.1', 443))])
    with pytest.raises(ValueError, match='non-public'):
        dataset.validate_url('https://images.test/a.png', {'images.test'})


class Response:
    def __init__(self, status=200, headers=None, body=b''):
        self.status = status; self.headers = headers or {}; self.body = io.BytesIO(body)
    def getheader(self, name, default=None):
        return self.headers.get(name, default)
    def read(self, size):
        return self.body.read(size)


def network(monkeypatch, responses):
    class Connection:
        def __init__(self, host, address, timeout):
            assert address == '93.184.216.34'  # validated IP is passed to the pinned transport
        def request(self, *args, **kwargs): pass
        def getresponse(self): return responses.pop(0)
        def close(self): pass
    monkeypatch.setattr(socket, 'getaddrinfo', public_dns)
    monkeypatch.setattr(dataset, 'PinnedHTTPS', Connection)


def test_redirect_checked_before_connect(monkeypatch):
    network(monkeypatch, [Response(302, {'Location': 'https://localhost/private'})])
    with pytest.raises(ValueError, match='host'):
        dataset.ImageLoader(hosts={'images.test'})._fetch('https://images.test/a.png')


@pytest.mark.parametrize('headers,body', [({'Content-Type': 'text/html'}, b'error'), ({'Content-Type': 'image/png', 'Content-Length': '999'}, b'a'), ({'Content-Type': 'image/png'}, b'a' * 101)])
def test_content_and_size_limits(monkeypatch, headers, body):
    network(monkeypatch, [Response(headers=headers, body=body)])
    with pytest.raises(ValueError):
        dataset.ImageLoader(hosts={'images.test'}, max_bytes=100)._fetch('https://images.test/a.png')


def test_decode_cache_bounded(monkeypatch):
    data = io.BytesIO(); Image.new('RGB', (64, 64), 'green').save(data, format='PNG')
    loader = dataset.ImageLoader(cache_mb=1)
    monkeypatch.setattr(loader, '_fetch', lambda url: data.getvalue())
    first = loader.load('https://images.test/1', 64)
    assert first.shape == (3, 64, 64) and first.min() >= -1 and first.max() <= 1
    first[:] = 0
    assert not np.array_equal(first, loader.load('https://images.test/1', 64))
    for i in range(30): loader.load(f'https://images.test/{i}', 64)
    assert loader.used <= loader.capacity
    assert len(loader.cache) < 30


def test_duplicate_groups_do_not_leak(isolated_store):
    for i, phash in enumerate(['0000000000000000', '0000000000000001', 'ffffffffffffffff']):
        storage.put('pair', {'id': str(i), 'phash': phash, 'source': 'test', 'source_id': str(i), 'image_url': f'https://images.test/{i}'})
    dataset.assign_splits()
    assert storage.get('pair', '0')['split'] == storage.get('pair', '1')['split']
    assert storage.get('pair', '0')['duplicate_group'] == storage.get('pair', '1')['duplicate_group']


def test_api_empty_state_and_validation(isolated_store, monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, 'engine', lambda: (_ for _ in ()).throw(main.HTTPException(503, 'Runtime unavailable')))
    with TestClient(app) as client:
        assert client.get('/api/health').json()['runtime_error'] == 'Runtime unavailable'
        assert client.get('/api/dataset/stats').json()['total'] == 0
        assert client.get('/api/history').json() == []
        assert client.post('/api/generate', json={'prompt': ''}).status_code == 422
        assert client.post('/api/generate', json={'prompt': 'forest'}).status_code == 503
        assert client.get('/api/generated/not-an-image').status_code == 404
        assert client.post('/api/dataset/import', json={'records': []}).status_code == 422
        assert client.post('/api/training/stop', headers={'Origin': 'https://evil.test'}).status_code == 403
        assert client.get('/api/jobs/missing').status_code == 404


def test_review_requires_validation(isolated_store):
    storage.put('pair', {'id': 'record', 'status': 'unchecked', 'review_status': 'pending'})
    with TestClient(app) as client:
        result = client.post('/api/dataset/records/record/review', json={'review_status': 'approved', 'note': 'Reviewed scene'})
        assert result.status_code == 400


def test_dataset_stats_ignores_partial_rows(isolated_store):
    storage.put('pair', {'id': 'partial', 'category': 'forest', 'source': 'manual'})
    stats = dataset.stats()
    assert stats['total'] >= 1
    assert stats['available'] == 0
    assert stats['invalid'] == 0
    assert stats['approved'] == 0
    assert stats['categories']['forest'] >= 1
    assert stats['sources']['manual'] >= 1


def test_job_completion_failure_and_recovery(isolated_store):
    job = jobs.start('test', lambda update, cancelled: {'ok': True})
    for _ in range(100):
        if storage.get('job', job['id'])['state'] == 'completed': break
        time.sleep(.01)
    assert storage.get('job', job['id'])['result'] == {'ok': True}
    storage.put('job', {'id': 'stale', 'state': 'running'})
    storage.recover_jobs()
    assert storage.get('job', 'stale')['state'] == 'interrupted'


def test_provider_requires_permission(monkeypatch):
    monkeypatch.delenv('PEXELS_ML_PERMISSION', raising=False)
    with pytest.raises(ValueError, match='permission'):
        dataset.collect({'provider': 'pexels', 'limit': 1, 'query': 'forest'}, lambda **kw: None, lambda: False)


def test_auto_collect_falls_back_to_permitted_providers(isolated_store, monkeypatch):
    monkeypatch.setenv('PEXELS_ML_PERMISSION', 'true'); monkeypatch.setenv('PEXELS_API_KEY', 'k')
    monkeypatch.delenv('UNSPLASH_ML_PERMISSION', raising=False)
    calls = []
    def fake(provider, request, limit, update, cancelled):
        calls.append((provider, limit))
        return {'huggingface': 1, 'pexels': 2}[provider]
    monkeypatch.setattr(dataset, 'collect_from', fake)
    request = {'provider': 'auto', 'query': 'forest', 'limit': 5, 'offset': 0, 'license_note': 'checked terms'}
    result = dataset.collect(request, lambda **k: None, lambda: False)
    assert calls == [('huggingface', 5), ('pexels', 4)]
    assert result['imported'] == {'huggingface': 1, 'pexels': 2}
    assert 'unsplash' in result['skipped']
    with pytest.raises(ValueError):
        dataset.collect({**request, 'provider': 'unsplash'}, lambda **k: None, lambda: False)
