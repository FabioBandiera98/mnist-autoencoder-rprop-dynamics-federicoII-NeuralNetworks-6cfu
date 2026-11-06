"""Estrae e salva le rappresentazioni latenti dei cinque autoencoder MNIST."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from autoencoder import Autoencoder, HIDDEN_SIZES, PROJECT_DIR, format_duration, load_autoencoder, output_paths
from data import SEED, get_device, load_splits

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


def seed_suffix(seed: int) -> str:
    """Suffisso dei file di output: vuoto per il seed di default, '_s<seed>' altrimenti."""
    return "" if seed == SEED else f"_s{seed}"


def sha256_file(path: Path) -> str:
    """Calcola l'impronta del checkpoint per identificare la rete che ha prodotto le feature."""
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_for_hidden_size(hidden_size: int, seed: int, splits: dict, device: torch.device) -> dict:
    """Estrae le feature dei tre split usando il checkpoint selezionato in Day 7."""
    # Usa gli stessi percorsi di autoencoder.py, così checkpoint e riepilogo corrispondono al seed scelto.
    checkpoint_path, _, ae_summary_path = output_paths(hidden_size, seed)
    if not checkpoint_path.exists() or not ae_summary_path.exists():
        raise FileNotFoundError(
            f"Checkpoint o riepilogo mancante per m={hidden_size}, seed={seed}: lancia prima autoencoder.py."
        )

    # Ricarica il modello selezionato con validation, con la stessa funzione usata dagli altri script.
    model = load_autoencoder(checkpoint_path, device)

    # Verifica che l'architettura del checkpoint sia quella attesa: 784 -> m.
    encoder_layer = model.encoder[0]
    if encoder_layer.in_features != 784 or encoder_layer.out_features != hidden_size:
        raise ValueError(f"Architettura checkpoint non coerente per m={hidden_size}.")

    # Legge dal riepilogo dell'autoencoder come si è concluso il suo training.
    with ae_summary_path.open(encoding="utf-8") as summary_file:
        ae_summary = json.load(summary_file)
    if not ae_summary["stopped_early"]:
        print(f"ATTENZIONE: l'autoencoder m={hidden_size} si è fermato al limite di epoche, non è converso.")

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
    output_path = FEATURES_DIR / f"mnist_features_m{hidden_size}{seed_suffix(seed)}.npz"
    np.savez_compressed(output_path, **feature_splits)

    # Il manifest registra anche se l'autoencoder era converso: serve a riconoscere feature "vecchie".
    return {
        "hidden_size": hidden_size,
        "seed": seed,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_best_epoch": int(ae_summary["best_epoch"]),
        "checkpoint_best_val_mse": float(ae_summary["best_val_mse"]),
        "autoencoder_converged": bool(ae_summary["stopped_early"]),
        "autoencoder_stop_reason": ae_summary["stop_reason"],
        "autoencoder_epochs_completed": int(ae_summary["epochs_completed"]),
        # Il riepilogo prodotto con la versione precedente di autoencoder.py non ha la test MSE.
        "autoencoder_test_mse": ae_summary.get("test_mse"),
        "output_file": str(output_path.relative_to(PROJECT_DIR)),
        "feature_shapes": {
            split_name: list(feature_splits[f"X_{split_name}"].shape)
            for split_name in ("train", "val", "test")
        },
    }


def parse_args() -> argparse.Namespace:
    """Legge da riga di comando il seed degli autoencoder da cui estrarre le feature."""
    parser = argparse.ArgumentParser(description="Estrae le feature latenti dei cinque autoencoder.")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed degli autoencoder (default: 42).")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help="Dispositivo per estrarre le feature; CPU aiuta a renderle confrontabili fra PC.",
    )
    return parser.parse_args()


def main() -> None:
    """Estrae le feature per tutti i cinque modelli e crea un manifest di controllo."""
    # Avvia il timer del programma: misura tutto, dal lancio alla fine.
    program_start = time.perf_counter()

    args = parse_args()
    # La CPU è opzionale per mantenere l'estrazione uniforme tra computer diversi.
    device = torch.device("cpu") if args.device == "cpu" else get_device()
    print(f"Device: {device} | seed autoencoder: {args.seed}")

    # Carica una sola volta immagini e label di tutti gli split.
    splits = load_splits(device="cpu", names=("train", "val", "test"))
    summaries = []

    # Usa in sequenza il checkpoint selezionato di ciascuna dimensione latente.
    for hidden_size in HIDDEN_SIZES:
        print(f"\n=== Estrazione feature con m={hidden_size} ===")
        summaries.append(extract_for_hidden_size(hidden_size, args.seed, splits, device))

    # Registra provenienza, checkpoint, convergenza e forme delle matrici generate.
    manifest_path = FEATURES_DIR / f"feature_summary{seed_suffix(args.seed)}.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(
            {
                "feature_count": len(HIDDEN_SIZES) * 3,
                "splits": ["train", "val", "test"],
                "batch_size": BATCH_SIZE,
                "device": str(device),
                "seed": args.seed,
                "all_autoencoders_converged": all(s["autoencoder_converged"] for s in summaries),
                "models": summaries,
            },
            manifest_file,
            indent=2,
        )

    print(f"\nGenerate {len(HIDDEN_SIZES) * 3} matrici di feature.")
    print(f"File riepilogativo: {manifest_path}")

    # Tempo effettivo di tutto il programma.
    print(f"\nTempo totale del programma: {format_duration(time.perf_counter() - program_start)}")


if __name__ == "__main__":
    # Avvia l'estrazione quando il file viene eseguito direttamente.
    main()
