"""Binary classification metrics that work on scalars and on numpy arrays of confusion counts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Confusion:
    tp: np.ndarray | int
    fp: np.ndarray | int
    tn: np.ndarray | int
    fn: np.ndarray | int

    @classmethod
    def from_predictions(
        cls, labels: np.ndarray, predictions: np.ndarray, axis: int = 0
    ) -> Confusion:
        """Count along ``axis``. ``predictions`` may have more dimensions than ``labels`` (grids)."""
        y = np.asarray(labels, dtype=bool)
        p = np.asarray(predictions, dtype=bool)
        y = y.reshape(y.shape + (1,) * (p.ndim - y.ndim))
        return cls(
            tp=np.sum(y & p, axis=axis),
            fp=np.sum(~y & p, axis=axis),
            tn=np.sum(~y & ~p, axis=axis),
            fn=np.sum(y & ~p, axis=axis),
        )

    def __add__(self, other: Confusion) -> Confusion:
        return Confusion(
            self.tp + other.tp, self.fp + other.fp, self.tn + other.tn, self.fn + other.fn
        )

    @property
    def total(self):
        return self.tp + self.fp + self.tn + self.fn

    def as_dict(self) -> dict[str, int]:
        return {k: int(getattr(self, k)) for k in ("tp", "fp", "tn", "fn")}


def _divide(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def precision(c: Confusion):
    return _divide(c.tp, c.tp + c.fp)


def recall(c: Confusion):
    return _divide(c.tp, c.tp + c.fn)


def specificity(c: Confusion):
    return _divide(c.tn, c.tn + c.fp)


def f1(c: Confusion):
    """F1 score. ``nan`` if there is neither a positive label nor a positive prediction."""
    return _divide(2 * c.tp, 2 * c.tp + c.fp + c.fn)


def mcc(c: Confusion):
    """Matthews correlation coefficient, 0 when undefined (a constant prediction or label)."""
    tp, fp, tn, fn = (np.asarray(v, dtype=np.float64) for v in (c.tp, c.fp, c.tn, c.fn))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            denominator > 0, (tp * tn - fp * fn) / np.where(denominator > 0, denominator, 1), 0.0
        )


def balanced_accuracy(c: Confusion):
    return (recall(c) + specificity(c)) / 2


def accuracy(c: Confusion):
    return _divide(c.tp + c.tn, c.total)


METRICS = {"f1": f1, "mcc": mcc}


def summary(c: Confusion) -> dict[str, float]:
    return {
        **c.as_dict(),
        "precision": float(precision(c)),
        "recall": float(recall(c)),
        "f1": float(f1(c)),
        "mcc": float(mcc(c)),
        "balanced_accuracy": float(balanced_accuracy(c)),
        "accuracy": float(accuracy(c)),
    }
