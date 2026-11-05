"""Estrae e salva le rappresentazioni latenti dei cinque autoencoder MNIST."""

import json
from pathlib import Path

import numpy as np
import torch

from autoencoder import Autoencoder, CHECKPOINT_DIR, HIDDEN_SIZES, PROJECT_DIR
from data import get_device, load_splits

FEATURES_DIR = PROJECT_DIR / "data" / "features"
BATCH_SIZE = 1024


def encode_in_batches(model: Autoencoder, inputs: torch.Tensor, device: torch.device) -> np.ndarray:
    """Codifica un intero split a piccoli batch e restituisce una matrice NumPy."""
    model.eval()
    latent_batches = []

    # La dimensione batch limita la memoria richiesta, soprattutto quando si usa una GPU.
    with torch.no_grad():
        for start in range(0, len(inputs), BATCH_SIZE):
            input_batch = inputs[start : start + BATCH_SIZE].to(device)
            latent_batch = model.encode(input_batch)
            latent_batches.append(latent_batch.cpu())

    # Concatena i batch e garantisce che le feature salvate siano float32.
    return torch.cat(latent_batches, dim=0).numpy().astype(np.float32, copy=False)


def extract_for_hidden_size(hidden_size: int, splits: dict, device: torch.device) -> dict:
    """Estrae le feature dei tre split usando il checkpoint selezionato in Day 7."""
    checkpoint_path = CHECKPOINT_DIR / f"autoencoder_m{hidden_size}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint mancante per m={hidden_size}: {checkpoint_path}")

    # Carica i metadati e i pesi del modello selezionato con validation.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["hidden_size"] != hidden_size or checkpoint["input_size"] != 784:
        raise ValueError(f"Architettura checkpoint non coerente per m={hidden_size}.")

    # Ricostruisce la stessa architettura e applica i pesi salvati.
    model = Autoencoder(input_size=checkpoint["input_size"], hidden_size=hidden_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    feature_splits = {}
    for split_name in ("train", "val", "test"):
        # Le immagini vengono codificate; le etichette restano associate alle feature.
        inputs, labels = splits[split_name]
        features = encode_in_batches(model, inputs, device)
        labels_array = labels.numpy().astype(np.int64, copy=False)

        # Interrompe il processo se la matrice non ha forma o valori attesi.
        if features.shape != (len(labels_array), hidden_size):
            raise ValueError(f"Dimensione feature non valida per {split_name}, m={hidden_size}: {features.shape}")
        if not np.isfinite(features).all() or features.min() < 0.0 or features.max() > 1.0:
            raise ValueError(f"Feature non finite o fuori da [0,1] per {split_name}, m={hidden_size}.")

        feature_splits[f"X_{split_name}"] = features
        feature_splits[f"y_{split_name}"] = labels_array
        print(
            f"m={hidden_size:3d} | {split_name:5s} | "
            f"X={features.shape}, y={labels_array.shape} | "
            f"feature range=[{features.min():.4f}, {features.max():.4f}]"
        )

    # Salva le tre matrici e le label abbinate in un file per autoencoder.
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FEATURES_DIR / f"mnist_features_m{hidden_size}.npz"
    np.savez_compressed(output_path, **feature_splits)

    return {
        "hidden_size": hidden_size,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR)),
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_best_val_mse": float(checkpoint["best_val_mse"]),
        "output_file": str(output_path.relative_to(PROJECT_DIR)),
        "feature_shapes": {
            split_name: list(feature_splits[f"X_{split_name}"].shape)
            for split_name in ("train", "val", "test")
        },
    }


def main() -> None:
    """Estrae le feature per tutti i cinque modelli e crea un manifest di controllo."""
    device = get_device()
    print(f"Device: {device}")

    # Carica una sola volta immagini e label di tutti gli split.
    splits = load_splits(device="cpu", names=("train", "val", "test"))
    summaries = []

    # Usa in sequenza il checkpoint selezionato di ciascuna dimensione latente.
    for hidden_size in HIDDEN_SIZES:
        print(f"\n=== Estrazione feature con m={hidden_size} ===")
        summaries.append(extract_for_hidden_size(hidden_size, splits, device))

    # Registra provenienza, checkpoint e forme delle matrici generate.
    manifest_path = FEATURES_DIR / "feature_summary.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(
            {
                "feature_count": len(HIDDEN_SIZES) * 3,
                "splits": ["train", "val", "test"],
                "batch_size": BATCH_SIZE,
                "device": str(device),
                "models": summaries,
            },
            manifest_file,
            indent=2,
        )

    print(f"\nGenerate {len(HIDDEN_SIZES) * 3} matrici di feature.")
    print(f"File riepilogativo: {manifest_path}")


if __name__ == "__main__":
    # Avvia l'estrazione quando il file viene eseguito direttamente.
    main()
