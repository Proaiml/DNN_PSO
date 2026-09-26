"""
Original class vs fast engine: same data, same PSO budget.

    python benchmarks/speed_memory.py

Both runs train the same network ([9, 6, 4, 2, 1] = 8 features + bias column)
on 1000 samples with 20 particles x 20 iterations, and report wall time and
peak Python memory (tracemalloc). The original run takes about 6 minutes;
``--skip-original`` runs only the fast engine.

Measured on a desktop PC (Python 3.11, NumPy 2.3, pyswarms 1.3):
  original prodnnv10 : 362.78 s, peak memory 357.1 MB, accuracy 0.566
  fast engine        :   0.19 s, peak memory   3.5 MB, accuracy 0.566  (same seed, same result)
  fast engine, default budget (30 particles x 200 iterations x 3 restarts): see the last line
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
__import__("pyswarms")      # loaded up front so its import cost is not measured

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dnnpso import PSONetwork  # noqa: E402


def run(engine: str, X, y):
    tracemalloc.start()
    started = time.perf_counter()
    net = PSONetwork(particles=20, iterations=20, restarts=1, engine=engine, seed=0).fit(X, y)
    seconds = time.perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return seconds, peak, net


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.random((1000, 8))
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0.8).astype(int)
    t_fast, m_fast, n_fast = run("fast", X, y)
    print(f"network {n_fast.layers_}, 1000 samples, 20 particles x 20 iterations")
    if "--skip-original" not in sys.argv:
        t_orig, m_orig, n_orig = run("original", X, y)
        print(f"original prodnnv10 : {t_orig:7.2f} s, peak memory {m_orig:7.1f} MB, "
              f"accuracy {n_orig.score(X, y):.3f}")
    print(f"fast engine        : {t_fast:7.2f} s, peak memory {m_fast:7.1f} MB, accuracy {n_fast.score(X, y):.3f}")
    if "--skip-original" not in sys.argv:
        print(f"-> {t_orig / t_fast:.0f}x faster, {m_orig / max(m_fast, 0.1):.0f}x less memory")
    started = time.perf_counter()
    full = PSONetwork(seed=0).fit(X, y)
    print(f"fast engine, default budget (30 x 200 x 3): {time.perf_counter() - started:.1f} s, "
          f"accuracy {full.score(X, y):.3f}")
