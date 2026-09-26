"""
Tests for the dnnpso layer.

    python -m unittest discover -s tests -v

The most important ones prove that the fast engine computes exactly the same
network, outputs and costs as the original prodnnv10 class, and that the
original files are never modified.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.getLogger("pyswarms").setLevel(logging.ERROR)

from class_prodnn import prodnnv10  # noqa: E402  (the original, unchanged)
from dnnpso import FastProdnn, PSONetwork, fit_blackbox, objectives, pyramid_layers  # noqa: E402
from dnnpso.cli import main as cli_main  # noqa: E402

# SHA-256 of the original files (line endings normalised to \n), as written by İlhan Koçaslan
ORIGINALS = {
    "class_prodnn.py": "e5d1aea98e5fe875b64a6a74110bbafc49160072f3931bfcf3b0fe9b21d19111",
    "dnn+pso.py": "d8a42c877b497303cf28868f3e2a373c5c379bf00827c8f32d6b62edb734240b",
}
ARCHITECTURES = [(3, 1, 2 / 3), (8, 1, 2 / 3), (16, 1, 0.5), (10, 3, 0.8), (2, 2, 0.5), (5, 1, 0.9)]


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


class OriginalModel:
    """Builds a prodnnv10 instance from arrays (it only reads JSON files)."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()

    def build(self, inputs, outputs, t, X, y, iters=1, particles=2, loops=1):
        xp, yp = os.path.join(self.tmp.name, "x.json"), os.path.join(self.tmp.name, "y.json")
        Path(xp).write_text(json.dumps(np.asarray(X).tolist()))
        Path(yp).write_text(json.dumps(np.asarray(y).tolist()))
        return prodnnv10(inputs, outputs, t, xp, yp, iters, particles, loops)

    def close(self):
        self.tmp.cleanup()


class OriginalUntouchedTest(unittest.TestCase):
    def test_original_files_are_unchanged(self):
        for name, digest in ORIGINALS.items():
            data = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(digest, hashlib.sha256(data).hexdigest(), f"{name} was modified")


class EquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.orig = OriginalModel()
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        self.orig.close()

    def test_architecture_rule_is_the_same(self):
        for inputs, outputs, t in ARCHITECTURES:
            m = self.orig.build(inputs, outputs, t, [[0] * inputs], [0])
            self.assertEqual(m.all_layers, pyramid_layers(inputs, outputs, t))
            self.assertEqual(m.dimensions, FastProdnn(m.all_layers).dimensions)

    def test_invalid_transition_is_an_error_instead_of_an_endless_loop(self):
        for t in (1.0, 1.5, 0.0, -0.5):
            with self.assertRaises(ValueError):
                pyramid_layers(3, 1, t)

    def test_outputs_and_cost_are_identical(self):
        for inputs, outputs, t in ARCHITECTURES:
            X = self.rng.random((40, inputs)).round(3)
            y = self.rng.integers(0, 2, 40)
            m = self.orig.build(inputs, outputs, t, X, y)
            for key in m.all_Weights:
                m.all_Weights[key] = self.rng.normal(0, 3)
            cost = quiet(m.propagate_forward)
            net = FastProdnn.from_prodnn(m)
            np.testing.assert_allclose(net.predict_raw(X), m.full_outputs_lastmend, rtol=0, atol=1e-12)
            self.assertAlmostEqual(cost, net.original_cost(y, net.predict_raw(X)), places=10)

    def test_swarm_cost_equals_the_original_objective_f(self):
        X = self.rng.random((20, 8))
        y = self.rng.integers(0, 2, 20)
        m = self.orig.build(8, 1, 2 / 3, X, y)
        positions = self.rng.normal(0, 2, (7, m.dimensions))
        original = quiet(m.f, positions.copy())
        fast = FastProdnn(m.all_layers).swarm_cost(X, y, positions)
        np.testing.assert_allclose(fast, original, rtol=1e-12)

    def test_same_seed_same_training_result_as_the_original_class(self):
        X = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=float)
        y = np.array([1, 0, 0, 0, 0, 1, 0, 0])
        budget = dict(particles=10, iterations=20, restarts=2, seed=5)
        a = PSONetwork.original(engine="original", **budget).fit(X, y)
        b = PSONetwork.original(engine="fast", **budget).fit(X, y)
        self.assertAlmostEqual(a.best_cost_, b.best_cost_, places=9)
        np.testing.assert_allclose(a.predict_raw(X), b.predict_raw(X), atol=1e-12)

    def test_weights_round_trip_into_the_original_class(self):
        X = self.rng.random((10, 3))
        y = self.rng.integers(0, 2, 10)
        net = PSONetwork.original(iterations=10, seed=1).fit(X, y)
        m = self.orig.build(3, 1, 2 / 3, X, y)
        m.all_Weights.update(net.to_prodnn_weights())
        quiet(m.propagate_forward)
        np.testing.assert_allclose(m.full_outputs_lastmend, net.predict_raw(X), atol=1e-12)


class ModelTest(unittest.TestCase):
    XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    XOR_Y = np.array([0, 1, 1, 0])

    def test_xor_needs_the_bias_column(self):
        original = PSONetwork.original(seed=0).fit(self.XOR_X, self.XOR_Y)
        self.assertEqual([2, 1], original.layers_)
        np.testing.assert_allclose(original.predict_raw(self.XOR_X), 0.5, atol=1e-3)   # stuck at 0.5
        self.assertLess(original.score(self.XOR_X, self.XOR_Y), 1.0)
        solved = sum(PSONetwork(seed=s).fit(self.XOR_X, self.XOR_Y).score(self.XOR_X, self.XOR_Y) == 1.0
                     for s in range(5))
        self.assertGreaterEqual(solved, 3)

    def test_multiclass_and_regression(self):
        rng = np.random.default_rng(1)
        centers = np.array([[0, 0], [3, 0], [0, 3]])
        X = np.vstack([c + rng.normal(0, 0.4, (40, 2)) for c in centers])
        y = np.repeat(["a", "b", "c"], 40)
        clf = PSONetwork(hidden_layers=[4], seed=0).fit(X, y)
        self.assertEqual("multiclass", clf.task_)
        self.assertGreater(clf.score(X, y), 0.9)
        self.assertEqual({"a", "b", "c"}, set(clf.predict(X).tolist()))
        x = rng.uniform(0, 1, 80)
        reg = PSONetwork(hidden_layers=[4], seed=0, task="regression").fit(x[:, None], 3 * x + 10)
        self.assertGreater(reg.score(x[:, None], 3 * x + 10), 0.95)

    def test_save_and_load_give_the_same_predictions(self):
        rng = np.random.default_rng(2)
        X = rng.random((30, 4)) * 50
        y = (X[:, 0] > 25).astype(int)
        net = PSONetwork(seed=0, iterations=50).fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            loaded = PSONetwork.load(net.save(Path(tmp) / "m.json"))
        np.testing.assert_allclose(net.predict_raw(X), loaded.predict_raw(X))
        self.assertEqual(net.layers_, loaded.layers_)

    def test_non_differentiable_objective_and_step_neurons(self):
        rng = np.random.default_rng(3)
        X = rng.random((120, 2))
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        net = PSONetwork(hidden_layers=[3], activation="step", objective="accuracy", seed=0).fit(X, y)
        self.assertGreater(net.score(X, y), 0.9)

    def test_objectives(self):
        y = np.array([1, 0, 1, 1])
        out = np.array([0.9, 0.2, 0.4, 0.8])
        self.assertAlmostEqual(25.0, objectives.error_rate(y, out))
        self.assertAlmostEqual(100 * (1 - 0.8), objectives.f1_loss(y, out))
        self.assertAlmostEqual(objectives.mse(y, out), 100 * np.mean((y - out) ** 2))

    def test_original_engine_rejects_what_it_cannot_do(self):
        with self.assertRaises(ValueError):
            PSONetwork(engine="original", hidden_layers=[3]).fit(self.XOR_X, self.XOR_Y)


class BlackBoxTest(unittest.TestCase):
    def test_finds_a_policy_from_a_cost_function_only(self):
        target = np.array([0.2, 0.8])
        xs = np.array([[0.0, 1.0], [1.0, 0.0]])

        def evaluate(policy):
            return float(np.sum((policy(xs)[:, 0] - target) ** 2))

        net, cost, history = fit_blackbox(evaluate, layers=[2, 2, 1], iterations=80, seed=0)
        self.assertLess(cost, 1e-3)
        self.assertEqual(len(history), 160)

    def test_batch_mode_gives_the_same_costs(self):
        xs = np.array([[0.1, 0.9, 0.5]])
        costs = {}
        for batch in (False, True):
            def evaluate(policy, batch=batch):
                if batch:
                    return (policy(np.repeat(xs, 30, axis=0))[:, 0] - 0.7) ** 2
                return float((policy(xs[0])[0] - 0.7) ** 2)
            _, costs[batch], _ = fit_blackbox(evaluate, layers=[3, 2, 1], iterations=15, restarts=1,
                                              batch=batch, seed=9)
        self.assertAlmostEqual(costs[False], costs[True], places=12)


class CliTest(unittest.TestCase):
    def test_train_predict_info_with_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = os.path.join(tmp, "m.json")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(0, cli_main(["train", "--x", str(ROOT / "data" / "xor_x.json"), "--y",
                                              str(ROOT / "data" / "xor_y.json"), "--seed", "1", "--out", model]))
                self.assertEqual(0, cli_main(["predict", "--model", model, "--x", str(ROOT / "data" / "xor_x.json")]))
                self.assertEqual(0, cli_main(["info", "--model", model]))
            self.assertIn("[0, 1, 1, 0]", out.getvalue())
            csv_path = os.path.join(tmp, "d.csv")
            rows = ["a,b,label"] + [f"{a},{b},{int(a) ^ int(b)}" for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
            Path(csv_path).write_text("\n".join(rows))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, cli_main(["train", "--csv", csv_path, "--target", "label", "--seed", "1",
                                              "--out", model]))
                self.assertEqual(0, cli_main(["predict", "--model", model, "--csv", csv_path,
                                              "--out", os.path.join(tmp, "p.csv")]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
