import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

# Seed fissato per rendere il progetto riproducibile
SEED = 42
# Cartella dove verranno scaricati i dati MNIST
DATA_DIR = Path(__file__).resolve().parent / "data"
# Cartella dove salviamo i dati preprocessati e i file di summary
PROCESSED_DIR = DATA_DIR / "processed"


def set_seed(seed: int = SEED) -> None:
    # Imposta il seed per Python, NumPy e PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mnist_dataset(data_dir: Path = DATA_DIR):
    """Load the MNIST dataset with PyTorch and return flattened normalized inputs and labels."""
    # Garantisce che la cartella di dati esista
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Trasformazione standard: le immagini vengono convertite in tensori PyTorch
    # e i pixel vengono portati in scala [0, 1]
    transform = transforms.ToTensor()

    # Dataset di training MNIST
    train_set = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )

    # Dataset di test MNIST
    test_set = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    # Trasformiamo le immagini 28x28 in vettori di dimensione 784
    # e normalizziamo i pixel da [0, 255] a [0, 1]
    x_train = train_set.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_set.targets.numpy().astype(np.int64)

    x_test = test_set.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test_set.targets.numpy().astype(np.int64)

    return x_train, y_train, x_test, y_test


def make_stratified_split(x_train, y_train, train_size=20000, val_size=5000, test_size=5000, seed=SEED):
    """Create a stratified train/validation/test split from the MNIST training set."""
    # Controllo che le dimensioni richieste non superino il dataset disponibile
    if train_size + val_size + test_size > len(x_train):
        raise ValueError("Requested split sizes exceed the available training samples.")

    # Seleziona solo i campioni necessari per costruire i tre insiemi
    x_subset = x_train[: train_size + val_size + test_size]
    y_subset = y_train[: train_size + val_size + test_size]

    # Prima split: training vs resto
    # train_size/(train+val+test) = 20000/30000
    x_train_split, x_rest, y_train_split, y_rest = train_test_split(
        x_subset,
        y_subset,
        train_size=train_size / (train_size + val_size + test_size),
        random_state=seed,
        stratify=y_subset,
    )

    # Seconda split: resto -> validation + test
    # val_size/(val+test) = 5000/10000
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest,
        y_rest,
        train_size=val_size / (val_size + test_size),
        random_state=seed + 1,
        stratify=y_rest,
    )

    return x_train_split, y_train_split, x_val, y_val, x_test, y_test


def save_split(x_train, y_train, x_val, y_val, x_test, y_test, output_dir: Path = PROCESSED_DIR):
    # Assicura che la cartella di output esista
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salva tutti i dati in un unico file .npz
    np.savez(
        output_dir / "mnist_split.npz",
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        X_test=x_test,
        y_test=y_test,
    )

    # Costruisce un riassunto leggibile delle dimensioni e delle classi
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

    # Salva il riassunto in JSON per controllo manuale
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Dataset saved to:", output_dir)
    print(json.dumps(summary, indent=2))


def main():
    # Fisso il seed per avere risultati riproducibili
    set_seed(SEED)

    # Carico MNIST completo
    x_train_full, y_train_full, x_test_full, y_test_full = load_mnist_dataset(DATA_DIR)

    # Costruisco la split richiesta: training, validation, test
    x_train, y_train, x_val, y_val, x_test, y_test = make_stratified_split(
        x_train_full,
        y_train_full,
        train_size=20000,
        val_size=5000,
        test_size=5000,
        seed=SEED,
    )

    # Salvo i dati preprocessati in una cartella dedicata
    save_split(x_train, y_train, x_val, y_val, x_test, y_test, PROCESSED_DIR)

    # Stampo le dimensioni finali dei dati
    print("\nFinal split shapes:")
    print(f"X_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {x_val.shape}, y_val: {y_val.shape}")
    print(f"X_test:  {x_test.shape}, y_test: {y_test.shape}")


if __name__ == "__main__":
    # Esegue la pipeline quando il file viene lanciato come script
    main()
