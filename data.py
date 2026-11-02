import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

SEED = 42
N_TRAIN = 20000
N_VAL = 5000
N_TEST = 5000

DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_FILE = PROCESSED_DIR / "mnist_split.npz"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mnist_dataset(data_dir: Path = DATA_DIR):
    """Carica MNIST ufficiale, appiattisce le immagini e normalizza i pixel in [0,1]."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)

    x_train = train_set.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_set.targets.numpy().astype(np.int64)

    x_test = test_set.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test_set.targets.numpy().astype(np.int64)

    return x_train, y_train, x_test, y_test


def build_split(seed: int = SEED, output_dir: Path = PROCESSED_DIR, save: bool = True):
    """Crea uno split train/validation/test mantenendo la distribuzione delle classi e salva gli indici."""
    x_train_full, y_train_full, x_test_full, y_test_full = load_mnist_dataset(DATA_DIR)

    train_idx, _ = train_test_split(
        np.arange(len(x_train_full)),
        train_size=N_TRAIN / len(x_train_full),
        random_state=seed,
        stratify=y_train_full,
    )

    remaining_train_idx = np.setdiff1d(np.arange(len(x_train_full)), train_idx)
    val_idx, _ = train_test_split(
        remaining_train_idx,
        train_size=N_VAL / len(remaining_train_idx),
        random_state=seed + 1,
        stratify=y_train_full[remaining_train_idx],
    )

    test_idx, _ = train_test_split(
        np.arange(len(x_test_full)),
        train_size=N_TEST / len(x_test_full),
        random_state=seed + 2,
        stratify=y_test_full,
    )

    X_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    X_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]
    X_test = x_test_full[test_idx]
    y_test = y_test_full[test_idx]

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
    }

    if save:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_dir / "mnist_split.npz",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )

        summary = {
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "input_dim": int(X_train.shape[1]),
            "n_classes": int(len(np.unique(y_train))),
            "train_class_counts": {str(k): int((y_train == k).sum()) for k in range(10)},
            "val_class_counts": {str(k): int((y_val == k).sum()) for k in range(10)},
            "test_class_counts": {str(k): int((y_test == k).sum()) for k in range(10)},
        }

        with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return splits


def load_splits(device=None, return_indices=False):
    """Carica lo split salvato e lo restituisce come dizionario dei tensori PyTorch."""
    if not SPLIT_FILE.exists():
        build_split()

    data = np.load(SPLIT_FILE)
    result = {}

    for name in ("train", "val", "test"):
        X = torch.tensor(data[f"X_{name}"], dtype=torch.float32, device=device)
        y = torch.tensor(data[f"y_{name}"], dtype=torch.long, device=device)
        result[name] = (X, y)

    if return_indices:
        result["indices"] = {
            "train": torch.tensor(data["train_idx"], dtype=torch.long, device=device),
            "val": torch.tensor(data["val_idx"], dtype=torch.long, device=device),
            "test": torch.tensor(data["test_idx"], dtype=torch.long, device=device),
        }

    return result


def make_stratified_split(x_train, y_train, train_size=N_TRAIN, val_size=N_VAL, test_size=N_TEST, seed=SEED):
    """Compatibilità con la vecchia API: crea lo split in modo stratificato sulla parte di training."""
    if train_size + val_size + test_size > len(x_train):
        raise ValueError("Requested split sizes exceed the available training samples.")

    x_subset = x_train[: train_size + val_size + test_size]
    y_subset = y_train[: train_size + val_size + test_size]

    x_train_split, x_rest, y_train_split, y_rest = train_test_split(
        x_subset,
        y_subset,
        train_size=train_size / (train_size + val_size + test_size),
        random_state=seed,
        stratify=y_subset,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_rest,
        y_rest,
        train_size=val_size / (val_size + test_size),
        random_state=seed + 1,
        stratify=y_rest,
    )

    return x_train_split, y_train_split, x_val, y_val, x_test, y_test


def save_split(x_train, y_train, x_val, y_val, x_test, y_test, output_dir: Path = PROCESSED_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "mnist_split.npz",
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        X_test=x_test,
        y_test=y_test,
    )

    summary = {
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "input_dim": int(x_train.shape[1]),
        "n_classes": int(len(np.unique(y_train))),
        "train_class_counts": {str(k): int((y_train == k).sum()) for k in range(10)},
        "val_class_counts": {str(k): int((y_val == k).sum()) for k in range(10)},
        "test_class_counts": {str(k): int((y_test == k).sum()) for k in range(10)},
    }

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    set_seed(SEED)
    splits = build_split(seed=SEED, output_dir=PROCESSED_DIR, save=True)

    for name in ("train", "val", "test"):
        X, y = splits[name]
        print(f"{name}: X={X.shape}, y={y.shape}")

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")


if __name__ == "__main__":
    main()
