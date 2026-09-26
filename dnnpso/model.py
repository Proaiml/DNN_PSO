"""
PSONetwork: fit / predict / save / load around the prodnnv10 idea.

    from dnnpso import PSONetwork
    net = PSONetwork(transition_per=2/3, particles=30, iterations=200, restarts=3, seed=1)
    net.fit(X, y)
    net.predict(X_new)
    net.save("model.json")

What it adds to the original class (class_prodnn.py is not changed):
* arrays, lists or CSV instead of JSON file paths; predictions on NEW data
* whole-swarm evaluation with NumPy (hundreds of times faster), no unbounded
  memory growth during training
* input scaling to [0, 1] (the sigmoid saturates on large inputs)
* ``bias="input"``: a constant 1 input column. The original network has no
  bias, so a sample of zeros always gives 0.5 in the first layer and 2-input
  problems such as XOR get no hidden layer at all; the extra column fixes both
  without touching the original code
* multi-class targets (one output neuron per class) and regression targets
* non-differentiable objectives (accuracy, F1, any function)
* optional weight bounds, early stopping, seeds (benchmarks/defaults_study.py:
  on six test problems the bias column raised the mean score from 0.62 to 0.86,
  while weight bounds made it worse, so they are off by default)
* ``engine="original"`` trains with the original class itself
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import objectives as obj
from .engine import FastProdnn, pyramid_layers

FORMAT = "dnnpso-model-1"
DEFAULT_OPTIONS = {"c1": 1.0, "c2": 0.8, "w": 0.9}        # the values of prodnnv10


class PSONetwork:
    def __init__(self, transition_per: float = 2 / 3, hidden_layers: list[int] | None = None,
                 particles: int = 30, iterations: int = 200,
                 restarts: int = 3, options: dict | None = None, bias: str = "input",
                 activation: str = "sigmoid", scale: str | None = "minmax",
                 weight_bounds: tuple[float, float] | None = None,
                 objective: str | Callable | None = "mse", tol: float | None = None, tol_iter: int = 20,
                 engine: str = "fast", task: str = "auto", seed: int | None = None,
                 verbose: bool = False) -> None:
        if bias not in ("input", "none", "neuron"):
            raise ValueError("bias must be 'input', 'none' or 'neuron'")
        if engine not in ("fast", "original"):
            raise ValueError("engine must be 'fast' or 'original'")
        if scale not in (None, "minmax"):
            raise ValueError("scale must be 'minmax' or None")
        if task not in ("auto", "binary", "multiclass", "regression"):
            raise ValueError("task must be 'auto', 'binary', 'multiclass' or 'regression'")
        self.task = task
        self.transition_per = float(transition_per)
        # None = the prodnnv10 pyramid rule; a list (e.g. [8, 4]) sets the hidden layers explicitly
        self.hidden_layers = [int(n) for n in hidden_layers] if hidden_layers else None
        self.particles = int(particles)
        self.iterations = int(iterations)
        self.restarts = max(1, int(restarts))
        self.options = dict(options or DEFAULT_OPTIONS)
        self.bias = bias
        self.activation = activation
        self.scale = scale
        self.weight_bounds = tuple(weight_bounds) if weight_bounds is not None else None
        self.objective = objective
        self.tol = tol
        self.tol_iter = int(tol_iter)
        self.engine = engine
        self.seed = seed
        self.verbose = verbose
        # set by fit()
        self.net_: FastProdnn | None = None
        self.task_: str | None = None
        self.classes_: list | None = None
        self.x_min_ = self.x_max_ = None
        self.y_min_ = self.y_max_ = None
        self.best_cost_: float | None = None
        self.cost_history_: list[float] = []
        self.fit_seconds_: float | None = None
        self.n_features_: int | None = None
        self.feature_names_: list[str] | None = None             # set by the CLI for CSV data

    @classmethod
    def original(cls, **kwargs) -> "PSONetwork":
        """The exact behaviour of prodnnv10: no bias, no scaling, unbounded weights, MSE."""
        base = dict(bias="none", scale=None, weight_bounds=None, objective="mse")
        base.update(kwargs)
        return cls(**base)

    # ------------------------------------------------------------ data preparation
    def _prepare_x(self, X, fitting: bool) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if fitting:
            self.n_features_ = X.shape[1]
        elif self.n_features_ is not None and X.shape[1] != self.n_features_:
            raise ValueError(f"the model was trained on {self.n_features_} input columns, got {X.shape[1]}")
        if self.scale == "minmax":
            if fitting:
                self.x_min_, self.x_max_ = X.min(axis=0), X.max(axis=0)
            span = np.where(self.x_max_ - self.x_min_ > 0, self.x_max_ - self.x_min_, 1.0)
            X = (X - self.x_min_) / span
        if self.bias == "input":
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        return X

    def _prepare_y(self, y) -> tuple[np.ndarray, int]:
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] > 1:                          # already one-hot / multi-output
            self.task_, self.classes_ = "multioutput", None
            return y.astype(float), y.shape[1]
        y = y.ravel()
        if y.dtype == bool:
            y = y.astype(int)
        values = np.unique(y)
        is_number = np.issubdtype(y.dtype, np.number)
        task = self.task
        if task == "auto":
            if is_number and set(values.tolist()) <= {0, 1}:
                task = "binary"
            elif not is_number or (np.all(np.equal(np.mod(values, 1), 0)) and len(values) <= 20):
                task = "multiclass"
            else:
                task = "regression"
        if task == "binary":
            if not set(values.tolist()) <= {0, 1}:
                raise ValueError("task='binary' needs 0/1 targets")
            self.task_, self.classes_ = "binary", [0, 1]
            return y.astype(float), 1
        if task == "multiclass":
            self.task_, self.classes_ = "multiclass", values.tolist()
            onehot = (y[:, None] == values[None, :]).astype(float)
            return onehot, len(values)
        self.task_, self.classes_ = "regression", None
        self.y_min_, self.y_max_ = float(y.min()), float(y.max())
        span = (self.y_max_ - self.y_min_) or 1.0
        return (y.astype(float) - self.y_min_) / span, 1

    # ------------------------------------------------------------ training
    def fit(self, X, y) -> "PSONetwork":
        import pyswarms as ps

        started = time.perf_counter()
        if self.seed is not None:
            np.random.seed(self.seed)
        Xp = self._prepare_x(X, fitting=True)
        yp, outputs = self._prepare_y(y)
        layers = ([Xp.shape[1], *self.hidden_layers, outputs] if self.hidden_layers
                  else pyramid_layers(Xp.shape[1], outputs, self.transition_per))
        if self.engine == "original":
            if outputs != 1 or self.bias == "neuron" or self.activation != "sigmoid" or self.hidden_layers:
                raise ValueError("engine='original' supports one output, sigmoid, the pyramid rule "
                                 "and bias 'none'/'input' only")
            from .original import train_original
            self.net_, self.best_cost_, self.cost_history_ = train_original(
                Xp, yp, self.transition_per, self.particles, self.iterations, self.restarts, self.options,
                quiet=not self.verbose)
            self.fit_seconds_ = time.perf_counter() - started
            return self

        net = FastProdnn(layers, activation=self.activation, neuron_bias=self.bias == "neuron",
                         all_outputs=outputs > 1)
        cost_fn = obj.resolve(self.objective)
        bounds = None
        if self.weight_bounds is not None:
            low, high = self.weight_bounds
            bounds = (np.full(net.dimensions, float(low)), np.full(net.dimensions, float(high)))
        logger = logging.getLogger("pyswarms")
        level = logger.level
        if not self.verbose:
            logger.setLevel(logging.ERROR)
        best_cost, best_pos, history = np.inf, None, []
        try:
            for _ in range(self.restarts):
                kwargs = {}
                if self.tol is not None:
                    kwargs.update(ftol=float(self.tol), ftol_iter=self.tol_iter)
                optimizer = ps.single.GlobalBestPSO(n_particles=self.particles, dimensions=net.dimensions,
                                                    options=self.options, bounds=bounds, **kwargs)
                cost, pos = optimizer.optimize(lambda P: net.swarm_cost(Xp, yp, P, cost_fn),
                                               iters=self.iterations, verbose=self.verbose)
                history.extend(float(c) for c in optimizer.cost_history)
                if cost < best_cost:
                    best_cost, best_pos = float(cost), np.asarray(pos, dtype=float)
        finally:
            logger.setLevel(level)
        net.set_flat(best_pos)
        self.net_, self.best_cost_, self.cost_history_ = net, best_cost, history
        self.fit_seconds_ = time.perf_counter() - started
        return self

    # ------------------------------------------------------------ prediction
    def _check(self) -> FastProdnn:
        if self.net_ is None:
            raise RuntimeError("call fit() or load() first")
        return self.net_

    def predict_raw(self, X) -> np.ndarray:
        """Network outputs in (0, 1): shape (n,) for one output, (n, outputs) otherwise."""
        net = self._check()
        out = net.forward(self._prepare_x(X, fitting=False))
        return out[:, 0] if out.shape[1] == 1 else out

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.task_ == "binary":
            return (raw >= threshold).astype(int)
        if self.task_ == "multiclass":
            return np.asarray(self.classes_)[raw.argmax(axis=1)]
        if self.task_ == "regression":
            return raw * ((self.y_max_ - self.y_min_) or 1.0) + self.y_min_
        return raw

    def score(self, X, y) -> float:
        """Accuracy for classification, R² for regression."""
        pred = self.predict(X)
        y = np.asarray(y)
        if self.task_ in ("binary", "multiclass"):
            return float(np.mean(pred == y.ravel()))
        ss_res = float(np.sum((y.ravel() - pred) ** 2))
        ss_tot = float(np.sum((y.ravel() - y.mean()) ** 2)) or 1.0
        return 1.0 - ss_res / ss_tot

    @property
    def layers_(self) -> list[int]:
        return self._check().layers

    def summary(self) -> str:
        net = self._check()
        extra = {"input": " (+1 bias input)", "neuron": " (+bias per neuron)", "none": ""}[self.bias]
        return (f"layers {net.layers}{extra}, {net.dimensions} weights, activation {net.activation}, "
                f"task {self.task_}, best cost {self.best_cost_:.4f}, "
                f"trained in {self.fit_seconds_:.2f} s" if self.fit_seconds_ is not None else "")

    def plot_history(self, show: bool = True, path: str | None = None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.cost_history_)
        ax.set_xlabel("PSO iteration (all restarts)" if self.engine == "fast" else "evaluation")
        ax.set_ylabel("best cost")
        ax.set_title("DNN_PSO training")
        ax.grid(alpha=0.3)
        if path:
            fig.savefig(path, dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        net = self._check()
        return {
            "format": FORMAT, "layers": net.layers, "activation": net.activation,
            "neuron_bias": net.neuron_bias, "all_outputs": net.all_outputs,
            "weights": [w.tolist() for w in net.weights],
            "params": {"transition_per": self.transition_per, "hidden_layers": self.hidden_layers,
                       "particles": self.particles,
                       "iterations": self.iterations, "restarts": self.restarts, "options": self.options,
                       "bias": self.bias, "scale": self.scale,
                       "weight_bounds": list(self.weight_bounds) if self.weight_bounds else None,
                       "objective": self.objective if isinstance(self.objective, str) else "custom",
                       "engine": self.engine, "task": self.task, "seed": self.seed},
            "task": self.task_, "classes": self.classes_,
            "x_min": None if self.x_min_ is None else np.asarray(self.x_min_).tolist(),
            "x_max": None if self.x_max_ is None else np.asarray(self.x_max_).tolist(),
            "y_min": self.y_min_, "y_max": self.y_max_,
            "best_cost": self.best_cost_, "fit_seconds": self.fit_seconds_,
            "n_features": self.n_features_, "feature_names": self.feature_names_,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PSONetwork":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} file")
        params = dict(data["params"])
        if params.get("objective") == "custom":
            params["objective"] = "mse"
        params["weight_bounds"] = tuple(params["weight_bounds"]) if params.get("weight_bounds") else None
        model = cls(**params)
        net = FastProdnn(data["layers"], activation=data["activation"], neuron_bias=data["neuron_bias"],
                         all_outputs=data["all_outputs"])
        net.weights = [np.asarray(w, dtype=float) for w in data["weights"]]
        model.net_ = net
        model.task_, model.classes_ = data["task"], data["classes"]
        model.x_min_ = None if data["x_min"] is None else np.asarray(data["x_min"])
        model.x_max_ = None if data["x_max"] is None else np.asarray(data["x_max"])
        model.y_min_, model.y_max_ = data["y_min"], data["y_max"]
        model.best_cost_, model.fit_seconds_ = data["best_cost"], data["fit_seconds"]
        model.n_features_ = data.get("n_features")
        model.feature_names_ = data.get("feature_names")
        return model

    def to_prodnn_weights(self) -> dict[str, float]:
        """Weights in prodnnv10.all_Weights format (bias 'none' or 'input')."""
        return self._check().to_prodnn_weights()


def train_test_split(X, y, test_size: float = 0.25, seed: int | None = 0):
    """Small helper so the examples need nothing but NumPy."""
    X, y = np.asarray(X), np.asarray(y)
    idx = np.random.default_rng(seed).permutation(len(X))
    cut = int(round(len(X) * (1 - test_size)))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]
