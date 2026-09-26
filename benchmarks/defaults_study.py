"""
Which PSONetwork defaults work best? Six small problems x five settings x three seeds.

    python benchmarks/defaults_study.py

Settings:
  A original  : bias none, no scaling, unbounded weights (exactly prodnnv10)
  B input     : bias column, scaling, unbounded weights      <- PSONetwork default
  C input±10  : bias column, scaling, weights in [-10, 10]
  D input±5   : bias column, scaling, weights in [-5, 5]
  E neuron±10 : a bias per neuron, scaling, weights in [-10, 10]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnnpso import PSONetwork, train_test_split  # noqa: E402


def datasets():
    rng = np.random.default_rng(42)
    bits = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=float)
    yield "3-bit (repo data)", bits, np.array([1, 0, 0, 0, 0, 1, 0, 0]), False
    yield "XOR", bits[:4, 1:], np.array([0, 1, 1, 0]), False
    parity = np.array([[(i >> k) & 1 for k in range(4)] for i in range(16)], dtype=float)
    yield "4-bit parity", parity, parity.sum(axis=1).astype(int) % 2, False
    t = rng.uniform(0, np.pi, 150)
    moons = np.vstack([np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)]])
    moons += rng.normal(0, 0.15, moons.shape)
    yield "two moons (300)", moons, np.r_[np.zeros(150), np.ones(150)].astype(int), True
    r = np.r_[rng.uniform(0, 0.5, 150), rng.uniform(0.8, 1.2, 150)]
    a = rng.uniform(0, 2 * np.pi, 300)
    yield "circles (300)", np.c_[r * np.cos(a), r * np.sin(a)], np.r_[np.zeros(150), np.ones(150)].astype(int), True
    x = rng.uniform(0, 1, 200)
    yield "sine regression", x[:, None], np.sin(2 * np.pi * x) + rng.normal(0, 0.05, 200), True


SETTINGS = {
    "A original": dict(bias="none", scale=None, weight_bounds=None),
    "B input": dict(bias="input", weight_bounds=None),
    "C input±10": dict(bias="input", weight_bounds=(-10, 10)),
    "D input±5": dict(bias="input", weight_bounds=(-5, 5)),
    "E neuron±10": dict(bias="neuron", weight_bounds=(-10, 10)),
}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"{'problem':<20}" + "".join(f"{k:>14}" for k in SETTINGS))
    total = {k: [] for k in SETTINGS}
    started = time.perf_counter()
    for name, X, y, split in datasets():
        row = f"{name:<20}"
        for key, kw in SETTINGS.items():
            scores = []
            for seed in range(3):
                if split:
                    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, seed)
                else:
                    Xtr = Xte = X
                    ytr = yte = y
                task = "regression" if name.startswith("sine") else "auto"
                net = PSONetwork(transition_per=2 / 3, particles=30, iterations=200, restarts=3,
                                 seed=seed, task=task, **kw).fit(Xtr, ytr)
                scores.append(net.score(Xte, yte))
            total[key].append(np.mean(scores))
            row += f"{np.mean(scores):>14.3f}"
        print(row, flush=True)
    print(f"{'mean':<20}" + "".join(f"{np.mean(v):>14.3f}" for v in total.values()))
    print(f"(test accuracy / R² on held-out data where split, training data otherwise; "
          f"{time.perf_counter() - started:.0f} s)")
