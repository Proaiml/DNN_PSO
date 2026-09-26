"""
Black-box training: the network is judged by ANY function, not by labelled data.

This is where training by PSO is genuinely different from backpropagation:
the cost may come from a simulation (a controller driving a plant), a game,
a trading back-test (profit, Sharpe ratio) or a hardware measurement. Nothing
has to be differentiable; PSO only needs one number per candidate network.

    from dnnpso.blackbox import fit_blackbox

    def evaluate(policy):            # policy(inputs) -> network outputs
        ...run a simulation with policy...
        return cost                  # lower is better

    net, cost, history = fit_blackbox(inputs=2, outputs=1, evaluate=evaluate)

With ``batch=True`` the whole swarm is evaluated in one call: ``policy(x)``
then takes one input row PER PARTICLE, shape (particles, inputs), returns
(particles, outputs), and ``evaluate`` returns one cost per particle. A
simulation written with NumPy arrays then steps all candidate networks at
once, which is typically 10-100x faster than evaluating them one by one.
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np

from .engine import FastProdnn, pyramid_layers

Policy = Callable[[np.ndarray], np.ndarray]


def fit_blackbox(evaluate: Callable[[Policy], float], inputs: int | None = None, outputs: int = 1,
                 transition_per: float = 2 / 3, layers: Sequence[int] | None = None, activation: str = "sigmoid",
                 neuron_bias: bool = True, particles: int = 30, iterations: int = 100, restarts: int = 2,
                 options: dict | None = None, weight_bounds: tuple[float, float] | None = None,
                 batch: bool = False, seed: int | None = None,
                 verbose: bool = False) -> tuple[FastProdnn, float, list[float]]:
    """Searches network weights that minimise ``evaluate(policy)``.

    ``policy(x)`` takes one input vector (or a batch, shape (n, inputs)) and returns the network output(s).
    The architecture is the prodnnv10 pyramid (``inputs`` -> ``outputs`` with ``transition_per``) unless
    ``layers`` is given explicitly. Returns (network with the best weights, best cost, cost per iteration).
    """
    import pyswarms as ps

    if layers is None:
        if inputs is None:
            raise ValueError("give inputs (and outputs) or an explicit layers list")
        layers = pyramid_layers(inputs, outputs, transition_per)
    net = FastProdnn(layers, activation=activation, neuron_bias=neuron_bias, all_outputs=True)
    if seed is not None:
        np.random.seed(seed)

    def policy_for(position: np.ndarray) -> Policy:
        weights = net.unflatten(position[None, :])

        def policy(x: np.ndarray) -> np.ndarray:
            a = np.atleast_2d(np.asarray(x, dtype=float))
            for W in weights:
                if net.neuron_bias:
                    a = np.hstack([a, np.ones((a.shape[0], 1))])
                a = net._act(a @ W[0].T)
            return a[0] if np.ndim(x) == 1 else a
        return policy

    def batch_policy_for(P: np.ndarray) -> Policy:
        weights = net.unflatten(P)                                   # (particles, rows, cols) per layer

        def policy(x: np.ndarray) -> np.ndarray:                     # x: (particles, inputs)
            a = np.asarray(x, dtype=float)
            for W in weights:
                if net.neuron_bias:
                    a = np.hstack([a, np.ones((a.shape[0], 1))])
                a = net._act(np.einsum("pi,poi->po", a, W))
            return a
        return policy

    def swarm_cost(P: np.ndarray) -> np.ndarray:
        if batch:
            return np.asarray(evaluate(batch_policy_for(P)), dtype=float).reshape(-1)
        return np.array([float(evaluate(policy_for(p))) for p in P])

    bounds = None
    if weight_bounds is not None:
        bounds = (np.full(net.dimensions, float(weight_bounds[0])), np.full(net.dimensions, float(weight_bounds[1])))
    logger = logging.getLogger("pyswarms")
    level = logger.level
    if not verbose:
        logger.setLevel(logging.ERROR)
    best_cost, best_pos, history = np.inf, None, []
    try:
        for _ in range(max(1, int(restarts))):
            optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=net.dimensions,
                                                options=dict(options or {"c1": 1.0, "c2": 0.8, "w": 0.9}),
                                                bounds=bounds)
            cost, pos = optimizer.optimize(swarm_cost, iters=iterations, verbose=verbose)
            history.extend(float(c) for c in optimizer.cost_history)
            if cost < best_cost:
                best_cost, best_pos = float(cost), np.asarray(pos, dtype=float)
    finally:
        logger.setLevel(level)
    net.set_flat(best_pos)
    return net, best_cost, history
