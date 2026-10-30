"""
data.py — Caricamento di MNIST e creazione dello split train / validation / test.

COME SI USA
-----------
1) Da terminale, una volta sola (o quando si cambia seed/dimensioni):

       python data.py

   Scarica MNIST (solo la prima volta), crea lo split e lo salva in:
       data/processed/mnist_split.npz       -> dati + indici delle immagini scelte
       data/processed/dataset_summary.json  -> riassunto leggibile (conteggi per classe)

2) Dagli altri script del progetto (autoencoder, classificatore, ...):

       from data import load_splits, get_device
       s = load_splits(device=get_device())
       X_train, y_train = s["train"]   # X: (20000, 784) float32 in [0,1], y: (20000,) int64
       X_val,   y_val   = s["val"]     # X: (5000, 784),  y: (5000,)
       X_test,  y_test  = s["test"]    # X: (5000, 784),  y: (5000,)

SCELTE DEL PROTOCOLLO
---------------------
- Training (20.000) e validation (5.000) sono estratti a caso dal training set
  ufficiale MNIST (60.000 immagini).
- Il test (5.000) è estratto dal test set ufficiale MNIST (10.000 immagini):
  è disgiunto per costruzione e scritto da persone diverse rispetto a
  training e validation, quindi misura meglio la generalizzazione.
- Tutti e tre gli insiemi sono stratificati: ogni cifra 0-9 compare con la
  stessa proporzione che ha nel dataset originale.
- Lo split è riproducibile: con lo stesso SEED si ottengono sempre le stesse immagini.
- Si restituiscono tensori interi e non DataLoader, perché RProp lavora in
  modalità batch: a ogni epoca il gradiente è calcolato su tutto il training set.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets

# =========================================================================== #
# CONFIGURAZIONE
# Tutti i parametri del dataset sono qui: per cambiare lo split basta
# modificare queste costanti e rilanciare `python data.py`.
# =========================================================================== #

# Seed fissato per rendere il progetto riproducibile
SEED = 42

# Numero di immagini per ciascun insieme
N_TRAIN = 20_000   # usate per aggiornare i pesi
N_VAL = 5_000      # usate per early stopping e scelta del checkpoint
N_TEST = 5_000     # usate solo per la valutazione finale

# Cartella dove torchvision scarica i file originali di MNIST
DATA_DIR = Path(__file__).resolve().parent / "data"
# Cartella dove salviamo lo split preprocessato e il file di riassunto
PROCESSED_DIR = DATA_DIR / "processed"
# File con lo split pronto all'uso (letto da load_splits)
SPLIT_FILE = PROCESSED_DIR / "mnist_split.npz"


# =========================================================================== #
# UTILITÀ
# =========================================================================== #

def set_seed(seed: int = SEED) -> None:
    """Fissa il seed di Python, NumPy e PyTorch (CPU e GPU).

    Va chiamata all'inizio di ogni esperimento, così inizializzazione dei pesi
    e qualsiasi altra operazione casuale sono identiche a ogni esecuzione.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Restituisce la GPU ("cuda") se disponibile, altrimenti la CPU.

    Si passa a load_splits() e ai modelli con .to(device), così dati e
    pesi stanno sullo stesso dispositivo.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================== #
# CARICAMENTO E SPLIT
# =========================================================================== #

def load_mnist_dataset(data_dir: Path = DATA_DIR):
    """Carica il dataset MNIST completo (training e test ufficiali).

    Se i file non sono presenti in data_dir li scarica con torchvision.

    Restituisce:
        x_train: float32 (60000, 784) — immagini appiattite, pixel in [0, 1]
        y_train: int64   (60000,)     — label 0-9
        x_test:  float32 (10000, 784)
        y_test:  int64   (10000,)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # download=True scarica i file solo se mancano
    train_set = datasets.MNIST(root=str(data_dir), train=True, download=True)
    test_set = datasets.MNIST(root=str(data_dir), train=False, download=True)

    # .data contiene le immagini come interi 0-255 di forma (N, 28, 28):
    # le trasformiamo in vettori da 784 valori e portiamo i pixel in [0, 1]
    x_train = train_set.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_set.targets.numpy().astype(np.int64)
    x_test = test_set.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test_set.targets.numpy().astype(np.int64)

    return x_train, y_train, x_test, y_test


def make_stratified_split(y_train_full, y_test_full, train_size=N_TRAIN,
                          val_size=N_VAL, test_size=N_TEST, seed=SEED):
    """Sceglie quali immagini vanno in training, validation e test.

    Lavora solo sulle label: restituisce gli INDICI delle immagini scelte
    nei set ufficiali, non le immagini. Così lo split è leggero da salvare
    e si può sempre verificare da dove viene ogni immagine.

    - train e validation: dal training set ufficiale, senza sovrapposizioni
    - test: dal test set ufficiale
    - stratify=... mantiene la stessa proporzione delle 10 cifre in ogni insieme

    Restituisce un dizionario {"train": indici, "val": indici, "test": indici}.
    """
    # Controllo che le dimensioni richieste siano compatibili con MNIST
    if train_size + val_size > len(y_train_full):
        raise ValueError("Train + validation superano il training set ufficiale.")
    if test_size > len(y_test_full):
        raise ValueError("Il test richiesto supera il test set ufficiale.")

    # Train e validation: un'unica estrazione stratificata dal training ufficiale.
    # Con train_size e test_size interi si ottengono esattamente 20.000 e 5.000
    # indici, disgiunti tra loro; le immagini restanti non vengono usate.
    idx_train, idx_val = train_test_split(
        np.arange(len(y_train_full)),
        train_size=train_size,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_full,
    )

    # Test: campione stratificato del test set ufficiale.
    # (Se si chiedono tutte le 10.000 immagini, le si prende direttamente.)
    if test_size == len(y_test_full):
        idx_test = np.arange(len(y_test_full))
    else:
        idx_test, _ = train_test_split(
            np.arange(len(y_test_full)),
            train_size=test_size,
            random_state=seed + 1,   # seed diverso: estrazione indipendente
            stratify=y_test_full,
        )

    return {"train": idx_train, "val": idx_val, "test": idx_test}


def build_split(seed: int = SEED, data_dir: Path = DATA_DIR):
    """Carica MNIST e costruisce lo split completo.

    Restituisce due dizionari:
        data: {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"} (array NumPy)
        idx:  {"train", "val", "test"} con gli indici nei set ufficiali
    """
    x_tr, y_tr, x_te, y_te = load_mnist_dataset(data_dir)
    idx = make_stratified_split(y_tr, y_te, seed=seed)

    # Selezione delle immagini tramite gli indici:
    # train/val dal training ufficiale, test dal test ufficiale
    data = {
        "X_train": x_tr[idx["train"]], "y_train": y_tr[idx["train"]],
        "X_val": x_tr[idx["val"]], "y_val": y_tr[idx["val"]],
        "X_test": x_te[idx["test"]], "y_test": y_te[idx["test"]],
    }
    return data, idx


def save_split(data: dict, idx: dict, seed: int = SEED, output_dir: Path = PROCESSED_DIR):
    """Salva lo split su disco.

    - mnist_split.npz: dati, label, indici e seed (compresso)
    - dataset_summary.json: riassunto leggibile con numero di campioni
      e conteggio delle immagini per ogni cifra in ogni insieme
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File binario con tutto quello che serve agli altri script.
    # Gli indici servono a check_data.py per verificare la disgiunzione.
    np.savez_compressed(
        output_dir / SPLIT_FILE.name,
        **data,
        idx_train=idx["train"], idx_val=idx["val"], idx_test=idx["test"],
        seed=np.array(seed),
    )

    # Riassunto in JSON, da aprire a mano per un controllo veloce
    summary = {
        "seed": seed,
        "source": {"train": "MNIST train ufficiale", "val": "MNIST train ufficiale",
                   "test": "MNIST test ufficiale"},
        "train_samples": int(len(data["y_train"])),
        "val_samples": int(len(data["y_val"])),
        "test_samples": int(len(data["y_test"])),
        "input_dim": int(data["X_train"].shape[1]),
        "n_classes": int(len(np.unique(data["y_train"]))),
        "train_class_counts": {str(k): int((data["y_train"] == k).sum()) for k in range(10)},
        "val_class_counts": {str(k): int((data["y_val"] == k).sum()) for k in range(10)},
        "test_class_counts": {str(k): int((data["y_test"] == k).sum()) for k in range(10)},
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Dataset salvato in:", output_dir)
    print(json.dumps(summary, indent=2))


# =========================================================================== #
# FUNZIONE DA USARE NEGLI ALTRI SCRIPT
# =========================================================================== #

def load_splits(device: torch.device | str = "cpu", split_file: Path = SPLIT_FILE,
                return_indices: bool = False) -> dict:
    """Carica lo split salvato come tensori PyTorch, pronti per l'addestramento.

    Se il file non esiste ancora, lo crea automaticamente (come `python data.py`).

    Argomenti:
        device: dove mettere i tensori, di solito get_device() (GPU se c'è)
        split_file: percorso del file .npz con lo split
        return_indices: se True aggiunge anche gli indici nei set ufficiali

    Restituisce:
        {"train": (X, y), "val": (X, y), "test": (X, y)}
        X: float32 (N, 784) in [0, 1], y: int64 (N,)
        (+ "indices": {"train", "val", "test"} se return_indices=True)
    """
    split_file = Path(split_file)

    # Primo avvio: lo split non esiste ancora, lo generiamo
    if not split_file.exists():
        data, idx = build_split()
        save_split(data, idx)

    # Lettura del file e conversione NumPy -> tensori PyTorch sul device scelto
    with np.load(split_file) as f:
        out = {
            name: (torch.from_numpy(f[f"X_{name}"]).to(device),
                   torch.from_numpy(f[f"y_{name}"]).to(device))
            for name in ("train", "val", "test")
        }
        if return_indices:
            out["indices"] = {k: f[f"idx_{k}"] for k in ("train", "val", "test")}
    return out


# =========================================================================== #
# ESECUZIONE DIRETTA: python data.py
# =========================================================================== #

def main():
    """Genera lo split e lo salva in data/processed/, poi stampa le dimensioni."""
    set_seed(SEED)
    data, idx = build_split(SEED)
    save_split(data, idx, SEED)

    print("\nDimensioni finali:")
    for name in ("train", "val", "test"):
        print(f"X_{name}: {data[f'X_{name}'].shape}, y_{name}: {data[f'y_{name}'].shape}")


if __name__ == "__main__":
    main()
