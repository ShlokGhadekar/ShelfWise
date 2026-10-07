"""The fold splitter decides what the model is allowed to see - a bug here silently leaks the future."""

from __future__ import annotations

import pytest

from shelfwise.forecasting.folds import make_folds


def test_last_fold_ends_on_last_day_and_folds_are_ordered():
    folds = make_folds(last_day=1941, horizon=28, n_folds=3, step=28, train_window=700, first_available_day=1100)
    assert [f.k for f in folds] == [1, 2, 3]
    assert folds[-1].test_end == 1941
    assert [f.origin for f in folds] == [1857, 1885, 1913]
    assert all(a.origin < b.origin for a, b in zip(folds, folds[1:], strict=False))


@pytest.mark.parametrize("n_folds,step", [(1, 28), (3, 28), (5, 35), (4, 56)])
def test_train_strictly_before_test_and_windows_are_horizon_long(n_folds, step):
    folds = make_folds(last_day=1941, horizon=28, n_folds=n_folds, step=step, train_window=700)
    for f in folds:
        assert f.train_end == f.origin
        assert f.train_end < f.test_start                       # no overlap train/test
        assert f.test_start == f.origin + 1                     # no gap either
        assert f.test_end - f.test_start + 1 == 28
        assert f.train_start <= f.val_start <= f.train_end      # inner validation is inside training...
        assert f.train_end - f.val_start + 1 == 28              # ...and is its most recent 28 days
        assert f.fit_end == f.val_start - 1


def test_test_windows_never_overlap():
    folds = make_folds(last_day=1941, horizon=28, n_folds=6, step=28, train_window=400)
    days = [d for f in folds for d in range(f.test_start, f.test_end + 1)]
    assert len(days) == len(set(days))


def test_train_window_respects_first_available_day():
    folds = make_folds(last_day=1941, horizon=28, n_folds=3, step=28, train_window=10_000, first_available_day=1100)
    assert all(f.train_start == 1100 for f in folds)


def test_not_enough_history_raises():
    with pytest.raises(ValueError):
        make_folds(last_day=200, horizon=28, n_folds=5, step=28, train_window=700, first_available_day=150)


def test_overlapping_step_raises():
    with pytest.raises(ValueError):
        make_folds(last_day=1941, horizon=28, n_folds=3, step=7, train_window=700)
