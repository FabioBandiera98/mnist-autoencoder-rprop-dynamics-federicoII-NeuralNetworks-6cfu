"""Esegue e verifica la campagna completa dei classificatori (giorni 11-14)."""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import classifier
from autoencoder import HIDDEN_SIZES, output_paths as autoencoder_paths
from classifier import CLASSIFIER_HIDDEN_SIZE, CLASSIFIER_SEEDS, NUM_CLASSES, REPRESENTATIONS
from data import N_TEST, N_TRAIN, N_VAL, SEED, get_device, load_splits
from extract_features import FEATURES_DIR, sha256_file

PROJECT_DIR = Path(__file__).resolve().parent
CAMPAIGNS_DIR = PROJECT_DIR / "results" / "campaigns"


def write_json(path: Path, contents: dict) -> None:
    """Scrive JSON in modo atomico, così un'interruzione non lascia un file troncato."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(contents, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def update_array_hash(digest, array: np.ndarray) -> None:
    """Aggiunge a un hash dtype, forma e contenuto numerico di un array."""
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.view(np.uint8))


def validate_autoencoder_runs() -> tuple[dict, bool]:
    """Controlla che esistano i cinque checkpoint seed-42 del protocollo concordato."""
    run_metadata = {}
    features_are_current = True
    feature_manifest_path = FEATURES_DIR / "feature_summary.json"
    feature_manifest = None

    if feature_manifest_path.exists():
        feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))

    manifest_models = {
        model["hidden_size"]: model
        for model in (feature_manifest or {}).get("models", [])
    }

    for hidden_size in HIDDEN_SIZES:
        checkpoint_path, _, summary_path = autoencoder_paths(hidden_size, SEED)
        if not checkpoint_path.exists() or not summary_path.exists():
            raise FileNotFoundError(
                f"Checkpoint/summary AE mancante per m={hidden_size}. "
                "Completa prima: python autoencoder.py --all --seed 42"
            )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("seed") != SEED
            or summary.get("max_epochs") != 10000
            or summary.get("patience") != 100
            or summary.get("min_rel_delta") != 1e-3
        ):
            raise ValueError(
                f"Protocollo AE non aggiornato per m={hidden_size}; "
                "rigenera il modello con python autoencoder.py --all --seed 42"
            )

        checkpoint_hash = sha256_file(checkpoint_path)
        feature_entry = manifest_models.get(hidden_size, {})
        if feature_entry.get("checkpoint_sha256") != checkpoint_hash:
            features_are_current = False

        run_metadata[str(hidden_size)] = {
            "seed": SEED,
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR)),
            "checkpoint_sha256": checkpoint_hash,
            "best_epoch": summary["best_epoch"],
            "best_val_mse": summary["best_val_mse"],
            "converged": summary["stopped_early"],
            "stop_reason": summary["stop_reason"],
        }

    return run_metadata, features_are_current


def refresh_features_if_needed(force_refresh: bool) -> None:
    """Rigenera feature seed-42 se manca il manifest o non corrisponde ai checkpoint AE."""
    _, features_are_current = validate_autoencoder_runs()
    if features_are_current and not force_refresh:
        print("Feature esistenti: checkpoint AE e manifest coincidono.")
        return

    print("Estraggo/aggiorno le feature usando gli encoder seed 42 su CPU...")
    extractor_path = PROJECT_DIR / "extract_features.py"
    subprocess.run(
        [sys.executable, str(extractor_path), "--seed", str(SEED), "--device", "cpu"],
        cwd=PROJECT_DIR,
        check=True,
    )

    _, features_are_current = validate_autoencoder_runs()
    if not features_are_current:
        raise RuntimeError("Le feature generate non corrispondono ai checkpoint AE seed-42.")


def preflight() -> dict:
    """Verifica split, label, feature archive e calcola impronte per il confronto fra PC."""
    device = get_device()
    splits = load_splits(device="cpu", names=("train", "val", "test"))
    expected_sizes = {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}

    # Controlla forma, normalizzazione e label prima di iniziare run potenzialmente lunghi.
    split_digest = hashlib.sha256()
    for split_name, expected_count in expected_sizes.items():
        inputs, labels = splits[split_name]
        if inputs.shape != (expected_count, 784) or labels.shape != (expected_count,):
            raise ValueError(f"Shape non valida nello split {split_name}: {inputs.shape}, {labels.shape}.")
        if not torch.isfinite(inputs).all() or inputs.min() < 0 or inputs.max() > 1:
            raise ValueError(f"Pixel non finiti o fuori da [0,1] nello split {split_name}.")
        update_array_hash(split_digest, inputs.numpy())
        update_array_hash(split_digest, labels.numpy())

    # Verifica che le 15 matrici abbiano label identiche agli split MNIST originali.
    feature_hashes = {}
    feature_shapes = {}
    for hidden_size in HIDDEN_SIZES:
        feature_path = FEATURES_DIR / f"mnist_features_m{hidden_size}.npz"
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature mancanti: {feature_path}; controlla extract_features.py.")

        feature_digest = hashlib.sha256()
        with np.load(feature_path) as archive:
            for split_name, expected_count in expected_sizes.items():
                features = archive[f"X_{split_name}"]
                labels = archive[f"y_{split_name}"]
                original_labels = splits[split_name][1].numpy()
                if features.shape != (expected_count, hidden_size):
                    raise ValueError(f"Shape feature non valida per m={hidden_size}, {split_name}.")
                if not np.isfinite(features).all() or features.min() < 0 or features.max() > 1:
                    raise ValueError(f"Feature non finite o fuori da [0,1] per m={hidden_size}, {split_name}.")
                if not np.array_equal(labels, original_labels):
                    raise ValueError(f"Label delle feature diverse da MNIST per m={hidden_size}, {split_name}.")
                update_array_hash(feature_digest, features)
                update_array_hash(feature_digest, labels)
                feature_shapes[f"m{hidden_size}_{split_name}"] = list(features.shape)
        feature_hashes[str(hidden_size)] = feature_digest.hexdigest()

    # Registra gli split e gli encoder: il confronto fra PC può segnalare input differenti.
    autoencoders, features_are_current = validate_autoencoder_runs()
    if not features_are_current:
        raise RuntimeError("Le feature non corrispondono ai checkpoint AE correnti; rigenerazione non riuscita.")

    combined_digest = hashlib.sha256()
    combined_digest.update(split_digest.hexdigest().encode("ascii"))
    for hidden_size in HIDDEN_SIZES:
        combined_digest.update(feature_hashes[str(hidden_size)].encode("ascii"))

    return {
        "device": str(device),
        "dataset_sha256": split_digest.hexdigest(),
        "feature_sha256": feature_hashes,
        "input_sha256": combined_digest.hexdigest(),
        "feature_shapes": feature_shapes,
        "autoencoders": autoencoders,
        "all_autoencoders_converged": all(item["converged"] for item in autoencoders.values()),
    }


def environment_info(device: torch.device) -> dict:
    """Registra macchina, PyTorch e GPU usata per interpretare i tempi dei run."""
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def campaign_protocol(args: argparse.Namespace) -> dict:
    """Ritorna i parametri confrontabili fra computer e fissati dal protocollo."""
    return {
        "representations": list(REPRESENTATIONS),
        "classifier_seeds": list(CLASSIFIER_SEEDS),
        "autoencoder_feature_seed": SEED,
        "hidden_size": CLASSIFIER_HIDDEN_SIZE,
        "classes": NUM_CLASSES,
        "optimizer": "RProp",
        "batch_mode": "full_training_set",
        "loss": "CrossEntropyLoss",
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "min_rel_delta": args.min_rel_delta,
        "log_every": args.log_every,
    }


def validate_run_id(run_id: str) -> None:
    """Limita il nome della macchina a caratteri sicuri per nomi di cartelle."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", run_id):
        raise ValueError("run-id deve avere 1-40 caratteri: lettere, numeri, trattini o underscore.")


def load_completed_run(campaign_dir: Path, representation: str, seed: int, protocol: dict):
    """Riusa solo run già conclusi con stesso protocollo e artefatti presenti."""
    checkpoint_path, history_path, summary_path = classifier.output_paths(
        representation, seed, campaign_dir
    )
    if not (checkpoint_path.exists() and history_path.exists() and summary_path.exists()):
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "representation": representation,
        "seed": seed,
        "hidden_size": CLASSIFIER_HIDDEN_SIZE,
        "max_epochs": protocol["max_epochs"],
        "patience": protocol["patience"],
        "min_rel_delta": protocol["min_rel_delta"],
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Risultato esistente con protocollo diverso: {summary_path}.")
    return summary


def parse_args() -> argparse.Namespace:
    """Configura il nome del PC, il resume, il preflight e il protocollo di training."""
    parser = argparse.ArgumentParser(description="Campagna completa classificatori MNIST, giorni 11-14.")
    parser.add_argument("--run-id", required=True, help="Nome univoco del PC, ad esempio Fabio-3050 o Yegor-3080.")
    parser.add_argument("--max-epochs", type=int, default=classifier.DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=classifier.DEFAULT_PATIENCE)
    parser.add_argument("--min-rel-delta", type=float, default=classifier.DEFAULT_MIN_REL_DELTA)
    parser.add_argument("--log-every", type=int, default=classifier.DEFAULT_LOG_EVERY)
    parser.add_argument("--refresh-features", action="store_true", help="Rigenera le feature anche se aggiornate.")
    parser.add_argument("--dry-run", action="store_true", help="Preflight completo senza allenare classificatori.")
    parser.add_argument("--allow-cpu", action="store_true", help="Consente la campagna su CPU, molto più lenta della GPU.")
    return parser.parse_args()


def main() -> None:
    """Prepara gli input, esegue/resume i 30 run e conserva riepiloghi per PC."""
    args = parse_args()
    validate_run_id(args.run_id)
    if args.max_epochs < 1 or args.patience < 1 or args.log_every < 1:
        raise ValueError("max-epochs, patience e log-every devono essere almeno 1.")
    if not 0 <= args.min_rel_delta < 1:
        raise ValueError("min-rel-delta deve essere nell'intervallo [0,1).")

    campaign_dir = CAMPAIGNS_DIR / args.run_id
    manifest_path = campaign_dir / "run_manifest.json"
    existing_manifest = None
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.refresh_features:
            raise ValueError("Non rigenerare feature durante una campagna esistente; usa un nuovo --run-id.")

    # Una nuova campagna aggiorna le feature obsolete; un resume non le modifica mai.
    if existing_manifest:
        _, features_are_current = validate_autoencoder_runs()
        if not features_are_current:
            raise ValueError(
                "I checkpoint AE sono cambiati rispetto alla campagna esistente. "
                "I risultati sono preservati: usa un nuovo --run-id per i nuovi encoder."
            )
    else:
        try:
            refresh_features_if_needed(force_refresh=args.refresh_features)
            input_info = preflight()
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            print(f"Preflight non superato: {error}", file=sys.stderr)
            raise SystemExit(2) from None

    if existing_manifest:
        try:
            input_info = preflight()
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            print(f"Preflight non superato: {error}", file=sys.stderr)
            raise SystemExit(2) from None
    protocol = campaign_protocol(args)
    environment = environment_info(get_device())

    if args.dry_run:
        print(f"Preflight superato per {args.run_id}.")
        print(f"Device: {environment['device']} | GPU: {environment['gpu_name'] or 'nessuna'}")
        print("Run previsti: 6 rappresentazioni x 5 seed = 30.")
        print(f"Protocollo: {json.dumps(protocol, ensure_ascii=False)}")
        print(f"Input fingerprint: {input_info['input_sha256']}")
        if not input_info["all_autoencoders_converged"]:
            print("ATTENZIONE: almeno un encoder non ha raggiunto la patience; controlla i riepiloghi AE.")
        return

    # Evita accidentalmente una campagna di migliaia di epoche su CPU.
    if environment["device"] == "cpu" and not args.allow_cpu:
        raise RuntimeError("CUDA non disponibile. Installa PyTorch CUDA oppure usa --allow-cpu consapevolmente.")

    if existing_manifest:
        if existing_manifest["protocol"] != protocol:
            raise ValueError("Questa campagna esiste con parametri diversi; scegli un nuovo --run-id.")
        if existing_manifest["inputs"]["input_sha256"] != input_info["input_sha256"]:
            raise ValueError("Gli input sono cambiati dopo l'inizio della campagna; scegli un nuovo --run-id.")
        manifest = existing_manifest
    else:
        manifest = {
            "run_id": args.run_id,
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol,
            "environment": environment,
            "inputs": input_info,
            "completed_runs": [],
        }

    campaign_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    summaries = []
    total_runs = len(REPRESENTATIONS) * len(CLASSIFIER_SEEDS)

    # Esegue le 30 configurazioni in sequenza; i run già completi vengono saltati al resume.
    for representation in REPRESENTATIONS:
        for seed in CLASSIFIER_SEEDS:
            summary = load_completed_run(campaign_dir, representation, seed, protocol)
            if summary is None:
                print(f"\n=== {args.run_id}: {representation}, seed {seed} ===")
                summary = classifier.train_classifier(
                    representation=representation,
                    seed=seed,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    min_rel_delta=args.min_rel_delta,
                    log_every=args.log_every,
                    campaign_dir=campaign_dir,
                )
            else:
                print(f"Già completato, riprendo senza riaddestrare: {representation}, seed {seed}.")

            summaries.append(summary)
            manifest["completed_runs"] = [
                f"{item['representation']}_s{item['seed']}" for item in summaries
            ]
            manifest["status"] = "running"
            write_json(manifest_path, manifest)
            classifier.save_aggregate(summaries, results_dir=campaign_dir / "results")
            print(f"Avanzamento: {len(summaries)}/{total_runs} configurazioni completate.")

    manifest["status"] = "complete"
    manifest["completed_runs"] = [
        f"{item['representation']}_s{item['seed']}" for item in summaries
    ]
    write_json(manifest_path, manifest)

    # Conserverà anche quanti modelli sono arrivati alla regola di patience.
    summary_path = campaign_dir / "campaign_summary.json"
    write_json(
        summary_path,
        {
            "run_id": args.run_id,
            "status": "complete",
            "run_count": len(summaries),
            "converged_count": sum(item["converged"] for item in summaries),
            "not_converged": [
                f"{item['representation']}_s{item['seed']}"
                for item in summaries
                if not item["converged"]
            ],
            "comparison_csv": str((campaign_dir / "results" / "classifier_comparison.csv").relative_to(PROJECT_DIR)),
        },
    )
    print(f"\nCampagna completata: {len(summaries)} run.")
    print(f"Riepilogo: {summary_path}")
    print(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    # Avvia la campagna solo se lo script è eseguito direttamente.
    main()
