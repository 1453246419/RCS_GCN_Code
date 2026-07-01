import numpy as np

from rcsgcn.splits import make_splits


def test_stratified_splits_are_reproducible_and_disjoint():
    labels = np.repeat(np.arange(4), 10)
    first = make_splits(labels, n_splits=5, seed=42)
    second = make_splits(labels, n_splits=5, seed=42)
    for (train_a, test_a), (train_b, test_b) in zip(first, second):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
        assert not set(train_a).intersection(test_a)
        assert set(labels[test_a]) == {0, 1, 2, 3}

