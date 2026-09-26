"""
Runs the ORIGINAL prodnnv10 class (class_prodnn.py) unchanged.

prodnnv10 reads its data from JSON files and prints every PSO step; this
adapter writes temporary files, keeps the console quiet and hands the trained
weights to the fast engine for predictions on new data. Use it when the result
must come from the original code itself.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from .engine import FastProdnn

REPO = Path(__file__).resolve().parent.parent


def train_original(X: np.ndarray, y: np.ndarray, transition_per: float, particles: int, iterations: int,
                   restarts: int, options: dict | None = None, quiet: bool = True):
    """Trains prodnnv10 and returns (FastProdnn with the best weights, best cost, per-evaluation costs)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from class_prodnn import prodnnv10          # the original, imported as is

    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("the original class uses only the first output neuron: give a 1-D target")
    with tempfile.TemporaryDirectory(prefix="dnnpso_") as tmp:
        xp, yp = os.path.join(tmp, "x.json"), os.path.join(tmp, "y.json")
        with open(xp, "w") as f:
            json.dump(np.asarray(X, dtype=float).tolist(), f)
        with open(yp, "w") as f:
            json.dump(y.tolist(), f)
        model = prodnnv10(int(np.shape(X)[1]), 1, transition_per, xp, yp, iterations, particles, restarts)
        if options:
            model.options = dict(options)
        sink = io.StringIO()
        logger = logging.getLogger("pyswarms")
        level = logger.level
        try:
            if quiet:
                logger.setLevel(logging.ERROR)
            with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext(), \
                    contextlib.redirect_stderr(sink) if quiet else contextlib.nullcontext():
                best = model.trainer()
                model.bestweight_upload()
        finally:
            logger.setLevel(level)
    net = FastProdnn.from_prodnn(model)
    return net, float(min(best)), list(map(float, model.costs))
