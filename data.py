"""Fashion-MNIST loading and the fixed PCA-8 representation for Round 2."""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class FashionPCA:
    pca: PCA
    scale: np.ndarray

    def transform(self, images: np.ndarray) -> np.ndarray:
        z = self.pca.transform(images.reshape(len(images), -1)) / self.scale
        return z.astype(np.float64, copy=False)

    def inverse_transform(self, latents: np.ndarray) -> np.ndarray:
        flat = self.pca.inverse_transform(np.asarray(latents) * self.scale)
        return flat.reshape(-1, 28, 28).clip(-1.0, 1.0)


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, = struct.unpack(">I", handle.read(4))
        ndim = magic & 0xFF
        if magic >> 8 != 0x000008:
            raise ValueError(f"{path}: expected unsigned-byte IDX, got magic {magic:#x}")
        shape = struct.unpack(">" + "I" * ndim, handle.read(4 * ndim))
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: truncated IDX payload")
    return data.reshape(shape)


def load_fashion_mnist(root: str | Path = "data/fashion-mnist/raw"):
    """Return official train/test images in [-1,1] and integer labels."""
    root = Path(root)
    train_x = _read_idx(root / "train-images-idx3-ubyte.gz").astype(np.float64)
    train_y = _read_idx(root / "train-labels-idx1-ubyte.gz").astype(np.int64)
    test_x = _read_idx(root / "t10k-images-idx3-ubyte.gz").astype(np.float64)
    test_y = _read_idx(root / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
    return train_x / 127.5 - 1.0, train_y, test_x / 127.5 - 1.0, test_y


def load_breast_mnist(path: str | Path = "data/breastmnist/breastmnist.npz"):
    """Return the official BreastMNIST train/validation/test splits in [-1, 1]."""
    with np.load(Path(path)) as payload:
        splits = []
        for split in ("train", "val", "test"):
            images = payload[f"{split}_images"].astype(np.float32) / 127.5 - 1.0
            labels = payload[f"{split}_labels"].astype(np.int64).reshape(-1)
            if images.ndim != 3 or images.shape[1:] != (28, 28):
                raise ValueError(
                    f"BreastMNIST {split}: expected N x 28 x 28, got {images.shape}"
                )
            splits.append((images, labels))
    return tuple(splits)


def fixed_split(root: str | Path = "data/fashion-mnist/raw", seed: int = 20260729):
    """Fixed stratified 50k/10k train/validation split plus official test."""
    train_x, train_y, test_x, test_y = load_fashion_mnist(root)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for label in range(10):
        idx = np.flatnonzero(train_y == label)
        rng.shuffle(idx)
        val_idx.extend(idx[:1000])
        train_idx.extend(idx[1000:])
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return (train_x[train_idx], train_y[train_idx]), (train_x[val_idx], train_y[val_idx]), (test_x, test_y)


def fit_pca(train_images: np.ndarray, n_components: int, seed: int = 20260729) -> FashionPCA:
    """Fit whitening on training data only, then scale each PC by its train max.

    Whitening fixes every component to unit training variance.  The additional
    deterministic max-absolute scaling is required by the bounded quadrature
    encoder; it uses training data only and is retained for exact inversion.
    """
    flat = train_images.reshape(len(train_images), -1)
    pca = PCA(n_components=n_components, whiten=True, svd_solver="randomized", random_state=seed)
    whitened = pca.fit_transform(flat)
    scale = np.maximum(np.max(np.abs(whitened), axis=0), 1e-12)
    return FashionPCA(pca=pca, scale=scale)


def fit_pca8(train_images: np.ndarray, seed: int = 20260729) -> FashionPCA:
    return fit_pca(train_images, 8, seed)
