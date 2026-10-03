"""Guards against the bug that broke the first version of this project.

test_duplicate_paths_are_rejected is the one that matters. The original code
built `paths = X_data + X_data` and then split on indices, which put the same
image on both sides of the split. These tests make that impossible to
reintroduce without something going red.
"""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.splits import split_by_image


def make_data(n=400):
    paths = [f"img_{i}.jpg" for i in range(n)]
    labels = list(np.repeat([0, 1, 2, 3], n // 4))
    return paths, labels


def test_no_image_appears_in_two_splits():
    paths, labels = make_data()
    train, val, test = split_by_image(paths, labels, seed=42)

    train_files = {paths[i] for i in train}
    val_files = {paths[i] for i in val}
    test_files = {paths[i] for i in test}

    assert not train_files & test_files
    assert not train_files & val_files
    assert not val_files & test_files


def test_splits_cover_every_image_exactly_once():
    paths, labels = make_data()
    train, val, test = split_by_image(paths, labels, seed=0)
    assert sorted(train + val + test) == list(range(len(paths)))


def test_duplicate_paths_are_rejected():
    """The original bug, now an error instead of a silent 4-point gain."""
    paths, labels = make_data()
    with pytest.raises(ValueError, match="duplicate"):
        split_by_image(paths * 2, labels * 2, seed=42)


def test_original_approach_did_leak():
    """Documents the size of the old failure so it stays concrete.

    Splitting a duplicated path list on indices puts most of the test set into
    training. This reproduces that, and asserts it was severe.
    """
    paths, labels = make_data()
    dup_paths = paths * 2
    dup_labels = labels * 2

    idx = list(range(len(dup_paths)))
    train, test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=dup_labels
    )
    train, _ = train_test_split(
        train, test_size=0.25, random_state=42,
        stratify=[dup_labels[i] for i in train]
    )

    train_files = {dup_paths[i] for i in train}
    leaked = sum(dup_paths[i] in train_files for i in test)
    assert leaked / len(test) > 0.5


@pytest.mark.parametrize("seed", [0, 42, 123, 456])
def test_split_is_deterministic_per_seed(seed):
    paths, labels = make_data()
    assert split_by_image(paths, labels, seed) == split_by_image(paths, labels, seed)


def test_class_balance_is_preserved():
    paths, labels = make_data(800)
    train, val, test = split_by_image(paths, labels, seed=7)
    for split in (train, val, test):
        counts = np.bincount([labels[i] for i in split], minlength=4)
        assert counts.min() > 0
        assert counts.max() - counts.min() <= 1
