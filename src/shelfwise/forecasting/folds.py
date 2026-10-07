"""Walk-forward (rolling-origin) backtest folds.

Why not a random train/test split? Because rows are ordered in time. A random split
puts days from *after* a test day into the training set; the model then "remembers"
the future (e.g. a price change or a trend that has not happened yet at forecast time)
and the test score is optimistic. It is the most common mistake in forecasting projects.

Instead we simulate how the model would actually be used:

    fold 1:  train [ ........ origin1 ]  test (origin1+1 .. origin1+28)
    fold 2:  train [ ............. origin2 ]  test (origin2+1 .. origin2+28)
    fold 3:  train [ .................. origin3 ]  test (origin3+1 .. origin3+28)

The last fold's test window ends on the last observed day. Several folds show whether a
result is stable across periods rather than a lucky month.

Inside each fold, the last `horizon` days of the training period are held out as an
*inner* validation window for LightGBM early stopping - again a time-based split.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Fold:
    k: int                 # 1-based fold number (1 = oldest)
    origin: int            # last day of training data ("today" for this fold)
    train_start: int       # first training target day
    test_start: int
    test_end: int
    val_start: int         # inner validation window for early stopping (inside training)

    @property
    def train_end(self) -> int:
        return self.origin

    @property
    def fit_end(self) -> int:
        """Last day used to fit before early stopping (the day before the inner validation window)."""
        return self.val_start - 1

    def as_dict(self) -> dict:
        return dataclasses.asdict(self) | {"train_end": self.train_end, "fit_end": self.fit_end}


def make_folds(last_day: int, horizon: int, n_folds: int, step: int, train_window: int,
               first_available_day: int = 1) -> list[Fold]:
    """Build `n_folds` rolling-origin folds whose last test window ends on `last_day`.

    Raises ValueError if there is not enough history for the requested folds.
    """
    if horizon < 1 or n_folds < 1 or step < 1 or train_window < 2 * horizon:
        raise ValueError("horizon, n_folds, step must be >= 1 and train_window >= 2 * horizon")
    if step < horizon:
        # Overlapping test windows would count the same days twice in averaged metrics.
        raise ValueError(f"step ({step}) must be >= horizon ({horizon}) so test windows don't overlap")

    folds = []
    for i in range(n_folds):
        origin = last_day - horizon - (n_folds - 1 - i) * step
        train_start = max(origin - train_window + 1, first_available_day)
        if origin - train_start + 1 < 2 * horizon:
            raise ValueError(f"fold {i + 1}: only {origin - train_start + 1} training days before origin {origin}")
        folds.append(Fold(k=i + 1, origin=origin, train_start=train_start,
                          test_start=origin + 1, test_end=origin + horizon,
                          val_start=origin - horizon + 1))
    return folds
