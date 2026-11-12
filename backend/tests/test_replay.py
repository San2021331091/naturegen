import random
import numpy as np
import pytest
from app.replay import ReplayPool, caption_group, split_validation


def item(index, caption='green forest'):
    return caption, f'https://images.pexels.com/{index}.jpg', np.full((3, 8, 8), index / 100, np.float32)


def test_duplicates_and_validation_never_enter_training():
    held_out = item(1)
    pool = ReplayPool(2, validation=[held_out])
    assert not pool.offer(held_out)
    assert not pool.offer(('new caption', 'https://images.pexels.com/alias.jpg', held_out[2]))
    assert pool.offer(item(2))
    assert not pool.offer((item(2)[0], item(2)[1] + '?width=256', item(3)[2]))
    assert not pool.offer(item(2))
    assert pool.admitted == 1 and pool.duplicates == 4


def test_rehearsal_and_caption_balancing():
    random.seed(7)
    pool = ReplayPool(4, min_replays=2)
    for index in range(3):
        assert pool.offer(item(index, 'green forest'))
    assert pool.offer(item(4, 'blue ocean'))
    assert not pool.offer(item(5, 'high mountains'))
    first = pool.sample(2)
    assert {caption_group(row[0]) for row in first} == {'forest', 'coast'}
    pool.sample(4); pool.sample(4)
    assert pool.can_refresh() and pool.offer(item(5, 'high mountains'))
    assert any(caption_group(row[0]) == 'coast' for row in pool.items)
    assert len(pool.items) == 4


def test_metadata_resume_and_changed_image_rejection():
    pool = ReplayPool(2)
    pool.offer(item(1)); pool.offer(item(2)); pool.sample(1)
    state = pool.state()
    assert all('image' not in row for row in state['items'])
    images = {item(i)[1]: item(i)[2] for i in (1, 2)}
    restored = ReplayPool(2, state=state)
    restored.restore(state, lambda url, resolution: images[url], 8)
    assert restored.state() == state
    assert not restored.offer(item(1))
    with pytest.raises(ValueError, match='changed'):
        ReplayPool(2, state=state).restore(state, lambda url, resolution: item(9)[2], 8)


def test_validation_covers_available_caption_groups():
    rows = [item(i, 'green forest') for i in range(10)] + [item(11, 'blue ocean'), item(12, 'snowy mountains')]
    validation, training = split_validation(rows, 3)
    assert {caption_group(row[0]) for row in validation} == {'forest', 'coast', 'mountain'}
    assert len(training) == 9
    assert not {row[1] for row in validation} & {row[1] for row in training}
