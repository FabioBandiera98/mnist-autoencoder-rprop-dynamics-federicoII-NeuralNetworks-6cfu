"""Autoencoder MNIST con training batch, validazione ed early stopping."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn

from data import SEED, get_device, load_splits, set_seed

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"
HIDDEN_SIZES = (25, 50, 75, 100, 125)
DEFAULT_HIDDEN_SIZE = HIDDEN_SIZES[0]
DEFAULT_MAX_EPOCHS = 300
DEFAULT_PATIENCE = 10
DEFAULT_MIN_DELTA = 1e-6
RPROP_LR = 0.01


class Autoencoder(nn.Module):
    """Comprime un'immagine MNIST e prova a ricostruirla."""

    def __init__(self, input_size: int = 784, hidden_size: int = 25) -> None:
        super().__init__()

        # Il sigmoid mantiene la rappresentazione latente in un intervallo limitato.
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
        )

        # Il sigmoid finale produce pixel ricostruiti nell'intervallo [0, 1].
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Converte ogni immagine nel vettore latente della dimensione scelta."""
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Restituisce la ricostruzione dell'immagine in ingresso."""
        latent = self.encode(inputs)
        return self.decoder(latent)


def check_model(hidden_sizes: tuple[int, ...] = HIDDEN_SIZES) -> None:
    """Verifica tutte le forme e il range senza avviare il training."""
    example_batch = torch.rand(4, 784)

    # Controlla che ogni dimensione latente comprima e ricostruisca la forma attesa.
    for hidden_size in hidden_sizes:
        model = Autoencoder(hidden_size=hidden_size)
        reconstructed_batch = model(example_batch)
        latent_shape = tuple(model.encode(example_batch).shape)
        output_shape = tuple(reconstructed_batch.shape)
        if latent_shape != (4, hidden_size) or output_shape != (4, 784):
            raise RuntimeError(f"Shape non valida per hidden_size={hidden_size}.")
        if reconstructed_batch.min() < 0 or reconstructed_batch.max() > 1:
            raise RuntimeError(f"Output fuori da [0,1] per hidden_size={hidden_size}.")

        print(
            f"m={hidden_size:3d} | input={tuple(example_batch.shape)} | "
            f"latent={latent_shape} | reconstruction={output_shape} | "
            f"output=[{reconstructed_batch.min():.3f}, {reconstructed_batch.max():.3f}]"
        )


def evaluate_reconstruction(model: Autoencoder, inputs: torch.Tensor) -> float:
    """Calcola la MSE media per pixel senza aggiornare i pesi."""
    model.eval()
    with torch.no_grad():
        # La media include tutti i campioni e tutti i pixel dello split.
        loss = nn.functional.mse_loss(model(inputs), inputs, reduction="mean")
    return float(loss.item())


def train_autoencoder(
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
) -> dict:
    """Allena l'autoencoder con RProp batch e seleziona il checkpoint via validation MSE."""
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs e patience devono essere almeno 1.")

    # Fissa le casualità e seleziona GPU, se disponibile.
    set_seed(SEED)
    device = get_device()

    # Carica solo train e validation: il test non partecipa al training né alla selezione.
    splits = load_splits(device=device, names=("train", "val"))
    X_train, _ = splits["train"]
    X_val, _ = splits["val"]

    # Crea il modello 784 -> hidden_size -> 784 con RProp full-batch.
    model = Autoencoder(input_size=X_train.shape[1], hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Rprop(model.parameters(), lr=RPROP_LR)

    # Prepara le cartelle e i nomi dei file di output.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"autoencoder_m{hidden_size}_best.pt"
    history_path = RESULTS_DIR / f"autoencoder_m{hidden_size}_history.csv"
    summary_path = RESULTS_DIR / f"autoencoder_m{hidden_size}_summary.json"

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stop_reason = "max_epochs"
    epoch_records = []
    start_time = time.perf_counter()

    # Ogni epoca calcola un unico gradiente sull'intero training set.
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        columns = ["epoch", "train_mse", "val_mse", "epoch_seconds"]
        writer = csv.DictWriter(history_file, fieldnames=columns)
        writer.writeheader()

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            optimizer.zero_grad()
            train_reconstruction = model(X_train)
            train_loss = nn.functional.mse_loss(train_reconstruction, X_train)
            train_loss.backward()
            optimizer.step()

            # La validation misura la ricostruzione senza partecipare agli aggiornamenti.
            train_mse = evaluate_reconstruction(model, X_train)
            val_mse = evaluate_reconstruction(model, X_val)
            epoch_seconds = time.perf_counter() - epoch_start
            record = {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "epoch_seconds": epoch_seconds,
            }
            epoch_records.append(record)
            writer.writerow(record)
            history_file.flush()

            # Conserva i pesi migliori secondo la validation MSE.
            if val_mse < best_val_loss - min_delta:
                best_val_loss = val_mse
                best_epoch = epoch
                epochs_without_improvement = 0
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "input_size": X_train.shape[1],
                    "hidden_size": hidden_size,
                    "best_epoch": best_epoch,
                    "best_val_mse": best_val_loss,
                    "seed": SEED,
                    "optimizer": "RProp",
                    "optimizer_lr": RPROP_LR,
                    "batch_mode": "full_training_set",
                    "loss": "MSE",
                }
                torch.save(checkpoint, checkpoint_path)
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch:03d}/{max_epochs} | "
                f"train MSE: {train_mse:.6f} | val MSE: {val_mse:.6f} | "
                f"best val MSE: {best_val_loss:.6f} (epoch {best_epoch})"
            )

            # Ferma dopo patience epoche senza un miglioramento significativo.
            if epochs_without_improvement >= patience:
                stop_reason = "validation_patience"
                break

    total_seconds = time.perf_counter() - start_time
    stopped_early = stop_reason == "validation_patience"

    # Ripristina il checkpoint scelto usando solo i risultati di validation.
    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    summary = {
        "hidden_size": hidden_size,
        "optimizer": "RProp",
        "optimizer_lr": RPROP_LR,
        "batch_mode": "full_training_set",
        "loss": "MSE",
        "max_epochs": max_epochs,
        "epochs_completed": len(epoch_records),
        "best_epoch": best_epoch,
        "best_train_mse": epoch_records[best_epoch - 1]["train_mse"],
        "best_val_mse": best_val_loss,
        "patience": patience,
        "min_delta": min_delta,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "training_seconds": total_seconds,
        "device": str(device),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR)),
        "history": str(history_path.relative_to(PROJECT_DIR)),
    }
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(f"\nBest checkpoint: {checkpoint_path}")
    print(f"Per-epoch log:   {history_path}")
    print(f"Summary:         {summary_path}")
    print(f"Stopped because: {summary['stop_reason']}")
    return summary


def parse_args() -> argparse.Namespace:
    """Legge la dimensione latente e le opzioni del training dalla riga di comando."""
    parser = argparse.ArgumentParser(description="Allena gli autoencoder MNIST con hidden size 25, 50, 75, 100, 125.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--hidden-size",
        type=int,
        choices=HIDDEN_SIZES,
        default=DEFAULT_HIDDEN_SIZE,
        help="Allena un solo autoencoder (default: 25).",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Allena in sequenza tutti e cinque gli autoencoder.",
    )
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--check-model", action="store_true", help="Verifica tutte le architetture, senza training.")
    return parser.parse_args()


if __name__ == "__main__":
    # Il flag --check-model esegue il controllo architetturale; altrimenti parte il training.
    args = parse_args()
    if args.check_model:
        check_model()
    elif args.all:
        # Esegue gli stessi criteri e parametri per ciascuna dimensione latente.
        for hidden_size in HIDDEN_SIZES:
            print(f"\n=== Autoencoder con hidden size {hidden_size} ===")
            train_autoencoder(
                max_epochs=args.max_epochs,
                patience=args.patience,
                min_delta=args.min_delta,
                hidden_size=hidden_size,
            )
    else:
        train_autoencoder(
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            hidden_size=args.hidden_size,
        )
