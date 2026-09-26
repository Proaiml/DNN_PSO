"""
Cost functions for PSO training (lower is better).

PSO only needs a number per candidate network, so the cost does not have to be
differentiable: accuracy, F1, profit or a simulation result work as well as the
original mean squared error.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Objective = Callable[[np.ndarray, np.ndarray], float]


def mse(y_true: np.ndarray, y_out: np.ndarray) -> float:
    """The original cost of prodnnv10: 100 * mean squared error."""
    return float(100.0 * np.mean((np.asarray(y_true, dtype=float) - y_out) ** 2))


def _labels(y_out: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    y_out = np.asarray(y_out)
    return y_out.argmax(axis=1) if y_out.ndim == 2 and y_out.shape[1] > 1 else (y_out.ravel() >= threshold).astype(int)


def _true_labels(y_true: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true)
    return y_true.argmax(axis=1) if y_true.ndim == 2 and y_true.shape[1] > 1 else y_true.ravel().astype(int)


def error_rate(y_true: np.ndarray, y_out: np.ndarray) -> float:
    """100 * (1 - accuracy). Piecewise constant: impossible for gradient descent, fine for PSO."""
    return float(100.0 * np.mean(_labels(y_out) != _true_labels(y_true)))


def f1_loss(y_true: np.ndarray, y_out: np.ndarray) -> float:
    """100 * (1 - F1) for binary targets; useful for imbalanced data."""
    t, p = _true_labels(y_true), _labels(y_out)
    tp = float(np.sum((t == 1) & (p == 1)))
    fp = float(np.sum((t == 0) & (p == 1)))
    fn = float(np.sum((t == 1) & (p == 0)))
    f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return 100.0 * (1.0 - f1)


def mse_plus_error(y_true: np.ndarray, y_out: np.ndarray) -> float:
    """MSE gives a direction, the error rate the goal; together they avoid long plateaus."""
    return mse(y_true, y_out) + error_rate(y_true, y_out)


OBJECTIVES: dict[str, Objective | None] = {
    "mse": None,                     # None = the fast vectorised path of the engine
    "accuracy": error_rate,
    "f1": f1_loss,
    "mse+accuracy": mse_plus_error,
}


def resolve(objective: str | Objective | None) -> Objective | None:
    if objective is None or callable(objective):
        return objective
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be a function or one of {sorted(OBJECTIVES)}")
    return OBJECTIVES[objective]
