"""
dnnpso: ready-to-use layer around DNN_PSO (neural networks trained by particle swarm optimisation).

The original code (class_prodnn.py, dnn+pso.py) is left untouched; this package
uses the same model and adds a fast engine, a fit/predict API, persistence and
a command line. See README.md.
"""
from .engine import ACTIVATIONS, FastProdnn, pyramid_layers
from .model import PSONetwork, train_test_split
from .blackbox import fit_blackbox
from . import objectives

__version__ = "1.0.0"
__all__ = ["PSONetwork", "fit_blackbox", "FastProdnn", "pyramid_layers", "train_test_split", "objectives", "ACTIVATIONS"]
