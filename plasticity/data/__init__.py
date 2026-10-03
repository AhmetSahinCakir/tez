from .mnist import load_mnist
from .cifar100 import load_cifar100
from .streams import (
    PermutedMNISTStream,
    CIFAR100BinaryStream,
    StationaryDataset,
    Task,
    build_stream,
)

__all__ = [
    "load_mnist",
    "load_cifar100",
    "PermutedMNISTStream",
    "CIFAR100BinaryStream",
    "StationaryDataset",
    "Task",
    "build_stream",
]
