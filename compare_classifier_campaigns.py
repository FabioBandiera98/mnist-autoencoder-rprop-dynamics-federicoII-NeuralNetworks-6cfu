"""Confronta due campagne classifier eseguite su computer diversi."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from classifier import CLASSIFIER_SEEDS, REPRESENTATIONS
from data import SEED

CAMPAIGNS_DIR = Path(__file__).resolve().parent / "results" / "campaigns"
PROJECT_DIR = Path(__file__).resolve().parent
ROOT_RESULTS_DIR = PROJECT_DIR / "results"
ROOT_CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
ROOT_RESULTS_ID = "Fabio-3050"


def comparable_protocol(protocol: dict) -> dict:
    """Seleziona i parametri che influenzano i pesi e le metriche, non la frequenza di stampa."""
    keys = (
        "representations",
        "classifier_seeds",
        "autoencoder_feature_seed",
        "hidden_size",
        "classes",
        "optimizer",
        "batch_mode",
        "loss",
        "max_epochs",
        "patience",
        "min_rel_delta",
    )
    return {key: protocol[key] for key in keys}


def load_root_results_as_campaign(run_id: str) -> tuple[dict, list[dict]]:
    """Importa i run root-level di Fabio, creati prima che esistesse il campaign runner."""
    comparison_path = ROOT_RESULTS_DIR / "classifier_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Risultati classifier non trovati: {comparison_path}")

    summaries = []
    for representation in REPRESENTATIONS:
        for seed in CLASSIFIER_SEEDS:
            summary_path = ROOT_RESULTS_DIR / f"classifier_{representation}_s{seed}_summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Riepilogo mancante: {summary_path}")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            history_path = PROJECT_DIR / summary["history"]
            checkpoint_path = PROJECT_DIR / summary["checkpoint"]
            if not history_path.exists() or not checkpoint_path.exists():
                raise FileNotFoundError(f"Log o checkpoint mancante per {representation}/seed {seed}.")
            if summary.get("representation") != representation or summary.get("seed") != seed:
                raise ValueError(f"Identità del run non coerente: {summary_path}")
            if (
                summary.get("hidden_size") != 64
                or summary.get("num_classes") != 10
                or summary.get("max_epochs") != 10000
                or summary.get("patience") != 100
                or summary.get("min_rel_delta") != 1e-3
                or summary.get("optimizer") != "RProp"
                or summary.get("batch_mode") != "full_training_set"
                or summary.get("loss") != "CrossEntropyLoss"
            ):
                raise ValueError(f"Protocollo non conforme in {summary_path}")
            summaries.append(summary)

    # Non associare ai risultati vecchi feature generate dopo il loro training.
    feature_manifest = PROJECT_DIR / "data" / "features" / "feature_summary.json"
    if not feature_manifest.exists():
        raise FileNotFoundError(f"Manifest feature mancante: {feature_manifest}")
    feature_time = feature_manifest.stat().st_mtime
    stale_summaries = [
        path.name
        for path in ROOT_RESULTS_DIR.glob("classifier_*_summary.json")
        if path.stat().st_mtime < feature_time
    ]
    if stale_summaries:
        raise ValueError("I risultati classifier precedono le feature attuali; non è possibile provarne la provenienza.")

    # Riusa il preflight del runner per calcolare fingerprint su split, label e feature correnti.
    from run_classifier_campaign import preflight

    inputs = preflight()
    device_names = {summary["device"] for summary in summaries}
    if len(device_names) != 1:
        raise ValueError("I risultati root-level sono stati prodotti su dispositivi diversi.")

    reference = summaries[0]
    manifest = {
        "run_id": run_id,
        "status": "complete",
        "protocol": {
            "representations": list(REPRESENTATIONS),
            "classifier_seeds": list(CLASSIFIER_SEEDS),
            "autoencoder_feature_seed": SEED,
            "hidden_size": reference["hidden_size"],
            "classes": reference["num_classes"],
            "optimizer": reference["optimizer"],
            "batch_mode": reference["batch_mode"],
            "loss": reference["loss"],
            "max_epochs": reference["max_epochs"],
            "patience": reference["patience"],
            "min_rel_delta": reference["min_rel_delta"],
        },
        "environment": {
            "device": reference["device"],
            "gpu_name": None,
        },
        "inputs": inputs,
    }
    print(f"Importati 30 risultati root-level come {run_id} (nessun retraining).")
    return manifest, summaries


def load_campaign(run_id: str) -> tuple[dict, list[dict]]:
    """Carica una campagna o, per Fabio, importa i risultati root-level già completi."""
    campaign_dir = CAMPAIGNS_DIR / run_id
    manifest_path = campaign_dir / "run_manifest.json"
    summary_path = campaign_dir / "campaign_summary.json"
    if not manifest_path.exists() or not summary_path.exists():
        if run_id == ROOT_RESULTS_ID:
            return load_root_results_as_campaign(run_id)
        raise FileNotFoundError(f"Campagna incompleta o non trovata: {campaign_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    campaign_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or campaign_summary.get("run_count") != 30:
        raise ValueError(f"La campagna {run_id} non contiene 30 run completati.")

    summaries = []
    for representation in REPRESENTATIONS:
        for seed in CLASSIFIER_SEEDS:
            path = campaign_dir / "results" / f"classifier_{representation}_s{seed}_summary.json"
            history_path = campaign_dir / "results" / f"classifier_{representation}_s{seed}_history.csv"
            checkpoint_path = campaign_dir / "checkpoints" / f"classifier_{representation}_s{seed}_best.pt"
            if not path.exists():
                raise FileNotFoundError(f"Riepilogo mancante: {path}")
            if not history_path.exists() or not checkpoint_path.exists():
                raise FileNotFoundError(f"Log o checkpoint mancante per {representation}/seed {seed}.")

            summary = json.loads(path.read_text(encoding="utf-8"))
            expected = {
                "representation": representation,
                "seed": seed,
                "max_epochs": manifest["protocol"]["max_epochs"],
                "patience": manifest["protocol"]["patience"],
                "min_rel_delta": manifest["protocol"]["min_rel_delta"],
            }
            if any(summary.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Riepilogo con protocollo incoerente: {path}")
            summaries.append(summary)

    return manifest, summaries


def aggregate(run_id: str, manifest: dict, summaries: list[dict], inputs_match: bool) -> list[dict]:
    """Calcola media e deviazione standard dei cinque seed per ogni input."""
    device = manifest["environment"]["device"]
    gpu_name = manifest["environment"].get("gpu_name") or "CPU"
    rows = []

    for representation in REPRESENTATIONS:
        runs = [item for item in summaries if item["representation"] == representation]
        if len(runs) != len(CLASSIFIER_SEEDS):
            raise ValueError(f"Mancano seed classificatore per {run_id}/{representation}.")

        accuracies = [item["test_accuracy"] for item in runs]
        losses = [item["test_loss"] for item in runs]
        epochs = [item["best_epoch"] for item in runs]
        training_times = [item["training_seconds"] for item in runs]
        rows.append(
            {
                "run_id": run_id,
                "device": device,
                "gpu_name": gpu_name,
                "representation": representation,
                "input_size": runs[0]["input_size"],
                "runs": len(runs),
                "converged_runs": sum(item["converged"] for item in runs),
                "best_epoch_mean": statistics.mean(epochs),
                "best_epoch_std": statistics.stdev(epochs),
                "test_accuracy_mean": statistics.mean(accuracies),
                "test_accuracy_std": statistics.stdev(accuracies),
                "test_loss_mean": statistics.mean(losses),
                "test_loss_std": statistics.stdev(losses),
                "training_seconds_mean": statistics.mean(training_times),
                "training_seconds_std": statistics.stdev(training_times),
                "same_inputs_across_campaigns": inputs_match,
            }
        )

    return rows


def main() -> None:
    """Legge due campagne complete e salva un confronto distinto per computer."""
    parser = argparse.ArgumentParser(description="Confronta due campagne classifier complete.")
    parser.add_argument("run_id_a", help="Primo run-id, per esempio Fabio-3050.")
    parser.add_argument("run_id_b", help="Secondo run-id, per esempio Yegor-3080.")
    args = parser.parse_args()

    manifest_a, summaries_a = load_campaign(args.run_id_a)
    manifest_b, summaries_b = load_campaign(args.run_id_b)
    if comparable_protocol(manifest_a["protocol"]) != comparable_protocol(manifest_b["protocol"]):
        raise ValueError("Le campagne hanno protocolli diversi: non sono confrontabili direttamente.")

    # Hash uguali confermano stesso split e stessi valori delle feature, non solo nomi uguali.
    inputs_match = manifest_a["inputs"]["input_sha256"] == manifest_b["inputs"]["input_sha256"]
    if inputs_match:
        print("OK: split MNIST e matrici di feature sono identici tra i PC.")
    else:
        print("ATTENZIONE: split o feature differiscono; i risultati restano separati e non vanno aggregati tra PC.")

    rows = aggregate(args.run_id_a, manifest_a, summaries_a, inputs_match)
    rows.extend(aggregate(args.run_id_b, manifest_b, summaries_b, inputs_match))

    # Salva 12 righe: sei rappresentazioni per ciascuna macchina, senza fondere i seed fra PC.
    output_path = CAMPAIGNS_DIR / "combined_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as comparison_file:
        writer = csv.DictWriter(comparison_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Confronto salvato: {output_path.relative_to(PROJECT_DIR)}")
    for row in rows:
        print(
            f"{row['run_id']:>14} | {row['representation']:>4} | "
            f"accuracy {row['test_accuracy_mean']:.4f} ± {row['test_accuracy_std']:.4f} | "
            f"best epoch {row['best_epoch_mean']:.1f} ± {row['best_epoch_std']:.1f} | "
            f"training {row['training_seconds_mean']:.1f}s ± {row['training_seconds_std']:.1f}s | "
            f"converged {row['converged_runs']}/{row['runs']}"
        )


if __name__ == "__main__":
    # Avvia il confronto quando si esegue direttamente questo script.
    main()
