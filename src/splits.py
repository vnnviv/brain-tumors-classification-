"""Train / val / test splitting.

Kept separate from data.py so it imports nothing heavy. The rule this module
enforces is the whole reason it exists: split unique images first, augment
afterwards. Getting that backwards is what broke the first version of this
project, and it is cheap to guard against once it lives in one place.
"""

from sklearn.model_selection import train_test_split

from .config import TEST_FRAC, VAL_FRAC


def split_by_image(paths, labels, seed):
    """Split unique images into train / val / test index lists.

    Raises if the same path appears twice, because duplicating before the
    split is exactly the bug this guards against. Duplicate afterwards, on the
    training indices only.
    """
    if len(set(paths)) != len(paths):
        raise ValueError(
            "duplicate paths passed to split_by_image; duplicate after "
            "splitting, never before"
        )

    idx = list(range(len(paths)))
    train, test = train_test_split(
        idx, test_size=TEST_FRAC, random_state=seed, stratify=labels
    )
    train, val = train_test_split(
        train,
        test_size=VAL_FRAC / (1 - TEST_FRAC),
        random_state=seed,
        stratify=[labels[i] for i in train],
    )
    return train, val, test
