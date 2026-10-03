"""Standard backpropagation (SGD / Adam) without any plasticity intervention."""
from .base import Method


class Baseline(Method):
    name = "baseline"
