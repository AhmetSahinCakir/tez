#!/usr/bin/env bash
# Downloads MNIST (PyTorch/ossci S3 mirror) and CIFAR-100 (fast.ai S3 mirror, PNG archive) into data/raw
# and converts them once into data/processed/{mnist,cifar100}.npz.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  [ -f "$RAW/mnist_$f" ] || curl -sS -L --retry 4 -o "$RAW/mnist_$f" "https://ossci-datasets.s3.amazonaws.com/mnist/$f"
done
if [ ! -f "$RAW/cifar-100-python.tar.gz" ] && [ ! -f "$RAW/cifar100.tgz" ]; then
  curl -sS -L --retry 4 -o "$RAW/cifar100.tgz" "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz" \
    || curl -sS -L --retry 4 -o "$RAW/cifar-100-python.tar.gz" "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
fi
cd "$ROOT" && python3 -c "from plasticity.data import load_mnist, load_cifar100; m=load_mnist(); c=load_cifar100(); print('MNIST', m['x_train'].shape, 'CIFAR-100', c['x_train'].shape)"
