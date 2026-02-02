import csv
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset


def _read_csv(path: str, feature_cols: Tuple[str, ...], target_cols: Tuple[str, ...]):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = []
        for r in reader:
            row = {}
            for k, v in r.items():
                if v is None:
                    row[k] = None
                else:
                    row[k] = v.replace(",", ".")
            rows.append(row)

    features = []
    targets = []
    for r in rows:
        try:
            x = [float(r[c]) for c in feature_cols]
            y = [float(r[c]) for c in target_cols]
        except (TypeError, ValueError, KeyError):
            continue
        features.append(x)
        targets.append(y)

    return torch.tensor(features, dtype=torch.float32), torch.tensor(
        targets, dtype=torch.float32
    )


def _train_val_test_split(n: int, test_split: float, val_split: float, seed: int):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx, val_idx, test_idx


def load_data(
    path: str,
    feature_cols: Tuple[str, ...],
    target_cols: Tuple[str, ...],
    test_split: float,
    val_split: float,
    seed: int,
):
    x, y = _read_csv(path, feature_cols, target_cols)
    train_idx, val_idx, test_idx = _train_val_test_split(
        x.shape[0], test_split, val_split, seed
    )

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)
    std[std == 0] = 1.0

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True)
    y_std[y_std == 0] = 1.0

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), mean, std, y_mean, y_std


class SerpaDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def augment_repeat(
    x: torch.Tensor, y: torch.Tensor, factor: int, noise_std: float, seed: int
):
    if factor <= 1:
        return x, y
    torch.manual_seed(seed)
    xs = [x]
    ys = [y]
    for _ in range(factor - 1):
        noise = torch.randn_like(x) * noise_std if noise_std > 0 else 0.0
        xs.append(x + noise)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)
