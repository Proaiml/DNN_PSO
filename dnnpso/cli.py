"""
Command line:

    python -m dnnpso train   --x data/data_x.json --y data/data_y.json --out model.json
    python -m dnnpso train   --csv data.csv --target label --test-size 0.25 --out model.json --plot history.png
    python -m dnnpso predict --model model.json --x new_x.json [--out predictions.json]
    python -m dnnpso predict --model model.json --csv new.csv [--out predictions.csv]
    python -m dnnpso info    --model model.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .model import PSONetwork, train_test_split


def load_json(path: str) -> np.ndarray:
    return np.asarray(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def load_csv(path: str, target: str | None) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Returns (X, y, feature names). ``target``: column name or index; without it every column is a feature."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r]
    header, body = rows[0], rows[1:]
    try:
        [float(v) for v in header]
        header, body = [f"x{i}" for i in range(len(rows[0]))], rows             # no header row
    except ValueError:
        pass
    if target is None:
        return np.asarray(body, dtype=float), None, header
    t = header.index(target) if target in header else int(target)
    X = np.asarray([[v for i, v in enumerate(r) if i != t] for r in body], dtype=float)
    raw = [r[t] for r in body]
    try:
        y = np.asarray(raw, dtype=float)
        if np.all(np.equal(np.mod(y, 1), 0)):
            y = y.astype(int)
    except ValueError:
        y = np.asarray(raw)
    return X, y, [h for i, h in enumerate(header) if i != t]


def _data(args, features: list[str] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    if args.csv:
        X, y, names = load_csv(args.csv, getattr(args, "target", None))
        if features and all(f in names for f in features):
            X = X[:, [names.index(f) for f in features]]          # the model's columns, in its order
        args.feature_names = names
        return X, y
    if not args.x:
        raise SystemExit("give --x (JSON) or --csv")
    return load_json(args.x), (load_json(args.y) if getattr(args, "y", None) else None)


def cmd_train(args) -> int:
    X, y = _data(args)
    if y is None:
        raise SystemExit("training needs targets: --y (JSON) or --csv with --target")
    Xtr, Xte, ytr, yte = (train_test_split(X, y, args.test_size, args.seed) if args.test_size > 0
                          else (X, X, y, y))
    hidden = [int(n) for n in args.hidden.split(",")] if args.hidden else None
    net = PSONetwork(transition_per=args.transition, hidden_layers=hidden, particles=args.particles,
                     iterations=args.iterations,
                     restarts=args.restarts, bias=args.bias, activation=args.activation, objective=args.objective,
                     task=args.task, engine=args.engine, seed=args.seed, verbose=args.verbose)
    net.fit(Xtr, ytr)
    net.feature_names_ = getattr(args, "feature_names", None)
    print(net.summary())
    label = "R²" if net.task_ == "regression" else "accuracy"
    print(f"training {label}: {net.score(Xtr, ytr):.4f}")
    if args.test_size > 0:
        print(f"test {label} ({len(Xte)} held-out samples): {net.score(Xte, yte):.4f}")
    if args.out:
        print(f"model saved: {net.save(args.out)}")
    if args.plot:
        net.plot_history(show=False, path=args.plot)
        print(f"cost curve: {args.plot}")
    return 0


def cmd_predict(args) -> int:
    net = PSONetwork.load(args.model)
    X, _ = _data(args, net.feature_names_)
    pred = net.predict(X)
    values = [v.item() if hasattr(v, "item") else v for v in pred]
    if args.out and args.out.lower().endswith(".csv"):
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["prediction"])
            w.writerows([[v] for v in values])
    elif args.out:
        Path(args.out).write_text(json.dumps(values), encoding="utf-8")
    else:
        print(json.dumps(values))
    if args.out:
        print(f"{len(values)} predictions written to {args.out}")
    return 0


def cmd_info(args) -> int:
    net = PSONetwork.load(args.model)
    data = net.to_dict()
    print(net.summary())
    print(json.dumps({k: data[k] for k in ("task", "classes", "params")}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser(prog="python -m dnnpso", description="Neural networks trained by PSO")
    sub = parser.add_subparsers(dest="command", required=True)

    def data_args(p, targets: bool):
        p.add_argument("--x", help="inputs as JSON (list of rows)")
        p.add_argument("--csv", help="CSV file (header row optional)")
        if targets:
            p.add_argument("--y", help="targets as JSON (list)")
            p.add_argument("--target", help="CSV target column (name or index)")

    t = sub.add_parser("train", help="train and save a model")
    data_args(t, True)
    t.add_argument("--out", help="model file (.json)")
    t.add_argument("--transition", type=float, default=2 / 3, help="layer shrink ratio (0-1), default 2/3")
    t.add_argument("--hidden", help="explicit hidden layers, e.g. 8,4 (default: the pyramid rule)")
    t.add_argument("--particles", type=int, default=30)
    t.add_argument("--iterations", type=int, default=200)
    t.add_argument("--restarts", type=int, default=3)
    t.add_argument("--bias", choices=["input", "none", "neuron"], default="input")
    t.add_argument("--activation", choices=["sigmoid", "step", "tanh", "relu"], default="sigmoid")
    t.add_argument("--objective", choices=["mse", "accuracy", "f1", "mse+accuracy"], default="mse")
    t.add_argument("--task", choices=["auto", "binary", "multiclass", "regression"], default="auto")
    t.add_argument("--engine", choices=["fast", "original"], default="fast")
    t.add_argument("--test-size", type=float, default=0.0, help="share of samples held out for testing")
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--plot", help="save the cost curve as an image")
    t.add_argument("--verbose", action="store_true")
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="predict with a saved model")
    data_args(p, False)
    p.add_argument("--model", required=True)
    p.add_argument("--out", help="predictions file (.json or .csv); printed when omitted")
    p.set_defaults(func=cmd_predict)

    i = sub.add_parser("info", help="show a saved model")
    i.add_argument("--model", required=True)
    i.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)
