"""
Vectorised engine for the prodnnv10 network (class_prodnn.py).

The original class is the reference and is not modified. This module computes
the SAME model with NumPy matrices so that a whole swarm can be evaluated in
one call:

* the same architecture rule (``transition_per``: every hidden layer is
  ``int(previous * transition_per)`` neurons while that is larger than the
  output layer),
* the same weight order (``w{layer}_{neuron}_{input}``, layer by layer, the
  order PSO positions are mapped in),
* the same neuron: ``sigmoid(inputs · weights)``, no bias,
* the same cost: ``100 * mean((y - first_output) ** 2)``.

tests/test_equivalence.py checks, for several architectures and random
weights, that outputs and costs are identical to ``prodnnv10``.

Extensions (all off by default, so the default behaviour is the original one):
``neuron_bias`` (a bias per neuron), other activations (``step`` for binary
neurons, ``tanh``, ``relu``) and ``all_outputs`` (use every output neuron,
the original uses only the first one).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0))),
    "step": lambda z: (z > 0).astype(float),          # binary neurons: no gradient, PSO still works
    "tanh": np.tanh,
    "relu": lambda z: np.maximum(z, 0.0),
}


def pyramid_layers(inputs: int, outputs: int, transition_per: float) -> list[int]:
    """The architecture rule of prodnnv10, with the inputs checked first.

    The original loops forever for ``transition_per >= 1``; here it is an error.
    """
    if int(inputs) < 1 or int(outputs) < 1:
        raise ValueError("inputs and outputs must be >= 1")
    if not 0 < float(transition_per) < 1:
        raise ValueError(f"transition_per must be between 0 and 1 (got {transition_per}); "
                         "the original class never stops for values >= 1")
    layers = [int(inputs)]
    current = int(inputs)
    while True:
        nxt = int(current * transition_per)
        if nxt > outputs:
            layers.append(nxt)
            current = nxt
        else:
            break
    layers.append(int(outputs))
    return layers


class FastProdnn:
    """The prodnnv10 network as matrices. ``layers`` e.g. [3, 2, 1]."""

    def __init__(self, layers: Sequence[int], activation: str = "sigmoid", neuron_bias: bool = False,
                 all_outputs: bool = False) -> None:
        if len(layers) < 2:
            raise ValueError("a network needs at least an input and an output layer")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
        self.layers = [int(n) for n in layers]
        self.activation = activation
        self.neuron_bias = bool(neuron_bias)
        self.all_outputs = bool(all_outputs)
        self._act = ACTIVATIONS[activation]
        # (rows, cols) of every weight matrix in PSO order; a bias column is appended last
        self.shapes = [(n_out, n_in + (1 if self.neuron_bias else 0))
                       for n_in, n_out in zip(self.layers[:-1], self.layers[1:])]
        self.dimensions = int(sum(r * c for r, c in self.shapes))
        self.weights: list[np.ndarray] = [np.random.random(s) for s in self.shapes]

    @classmethod
    def pyramid(cls, inputs: int, outputs: int, transition_per: float, **kwargs) -> "FastProdnn":
        return cls(pyramid_layers(inputs, outputs, transition_per), **kwargs)

    # ------------------------------------------------------------ weights <-> PSO positions
    def unflatten(self, positions: np.ndarray) -> list[np.ndarray]:
        """(particles, D) -> one array (particles, rows, cols) per layer."""
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        if positions.shape[1] != self.dimensions:
            raise ValueError(f"expected {self.dimensions} weights, got {positions.shape[1]}")
        out, start = [], 0
        for rows, cols in self.shapes:
            size = rows * cols
            out.append(positions[:, start:start + size].reshape(-1, rows, cols))
            start += size
        return out

    def flatten(self) -> np.ndarray:
        return np.concatenate([w.ravel() for w in self.weights])

    def set_flat(self, vector: np.ndarray) -> None:
        self.weights = [w[0].copy() for w in self.unflatten(np.asarray(vector, dtype=float)[None, :])]

    # ------------------------------------------------------------ forward pass
    def forward_batch(self, X: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Outputs for every particle: (particles, samples, outputs)."""
        a = np.asarray(X, dtype=float)[None, :, :]                       # (1, N, in)
        a = np.broadcast_to(a, (np.atleast_2d(positions).shape[0],) + a.shape[1:])
        for W in self.unflatten(positions):
            if self.neuron_bias:
                a = np.concatenate([a, np.ones(a.shape[:2] + (1,))], axis=2)
            a = self._act(np.einsum("pni,poi->pno", a, W))
        return a

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Outputs with the current weights: (samples, outputs)."""
        return self.forward_batch(X, self.flatten()[None, :])[0]

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """What prodnnv10 reports: the first output neuron (or all of them with all_outputs)."""
        out = self.forward(X)
        return out if self.all_outputs else out[:, 0]

    # ------------------------------------------------------------ cost (identical to cost_function)
    @staticmethod
    def original_cost(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """100 * mean squared error; y_pred may carry a leading particle axis."""
        return 100.0 * np.mean((np.asarray(y, dtype=float) - y_pred) ** 2,
                               axis=tuple(range(1, np.ndim(y_pred))) if np.ndim(y_pred) > 1 else None)

    def swarm_cost(self, X: np.ndarray, y: np.ndarray, positions: np.ndarray,
                   objective: Callable[[np.ndarray, np.ndarray], float] | None = None) -> np.ndarray:
        """Cost of every particle, the function handed to pyswarms."""
        out = self.forward_batch(X, positions)                          # (P, N, O)
        pred = out if self.all_outputs else out[:, :, 0]
        if objective is None:
            y_arr = np.asarray(y, dtype=float)
            if self.all_outputs and y_arr.ndim == 1:
                y_arr = y_arr[:, None]
            diff = (y_arr[None, ...] - pred) ** 2
            return 100.0 * diff.reshape(diff.shape[0], -1).mean(axis=1)
        return np.array([float(objective(np.asarray(y), p)) for p in pred])

    # ------------------------------------------------------------ interoperability with prodnnv10
    def to_prodnn_weights(self) -> dict[str, float]:
        """Weights as prodnnv10.all_Weights (only without neuron_bias)."""
        if self.neuron_bias:
            raise ValueError("prodnnv10 has no biases; this network uses neuron_bias")
        result = {}
        for index, W in enumerate(self.weights, start=1):
            for neuron in range(W.shape[0]):
                for inp in range(W.shape[1]):
                    result[f"w{index}_{neuron}_{inp}"] = float(W[neuron, inp])
        return result

    @classmethod
    def from_prodnn(cls, model) -> "FastProdnn":
        """Same network and weights as a prodnnv10 instance (its all_Weights order = PSO order)."""
        net = cls(model.all_layers)
        net.set_flat(np.array(list(model.all_Weights.values()), dtype=float))
        return net
