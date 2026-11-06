"""Classificatore MNIST con RProp full-batch e arresto basato sulla validation loss."""

import argparse
import copy
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from autoencoder import (
    HIDDEN_SIZES,
    PROJECT_DIR,
    RPROP_ETAS,
    RPROP_LR,
    RPROP_STEP_SIZES,
    format_duration,
)
from data import SEED, get_device, load_splits, set_seed

# Dimensioni dell'hidden layer e classi: identici per ogni rappresentazione.
CLASSIFIER_HIDDEN_SIZE = 64
NUM_CLASSES = 10

# Seed diversi per misurare la variabilità dell'inizializzazione del classificatore.
CLASSIFIER_SEEDS = (1, 2, 3, 4, 5)

# Regola di arresto fissata prima degli esperimenti.
DEFAULT_MAX_EPOCHS = 10000
DEFAULT_PATIENCE = 100
DEFAULT_MIN_REL_DELTA = 1e-3
DEFAULT_LOG_EVERY = 1

# Il classificatore usa sempre le feature dell'autoencoder addestrato con seed 42.
AUTOENCODER_FEATURE_SEED = SEED
FEATURES_DIR = PROJECT_DIR / "data" / "features"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"
REPRESENTATIONS = ("raw", *(f"m{size}" for size in HIDDEN_SIZES))


class Classifier(nn.Module):
    """Rete 1-hidden-layer: sigmoid nello strato nascosto, logits per 10 classi."""

    def __init__(self, input_size: int, hidden_size: int = CLASSIFIER_HIDDEN_SIZE) -> None:
        super().__init__()

        # Lo strato nascosto sigmoidale rispetta la traccia dell'esame.
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
        )

        # Restituisce logits grezzi: CrossEntropyLoss applica internamente log-softmax.
        self.output = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Produce dieci logits, uno per ciascuna cifra da 0 a 9."""
        return self.output(self.hidden(inputs))


def check_classifier() -> None:
    """Prova loss, gradienti e RProp per baseline raw e cinque input latenti."""
    set_seed(SEED)
    raw_inputs, raw_labels = load_splits(device="cpu", names=("train",))["train"]

    # Prova le dimensioni previste, usando veri campioni MNIST per l'input grezzo.
    for input_size in (784, *HIDDEN_SIZES):
        model = Classifier(input_size=input_size)
        optimizer = torch.optim.Rprop(
            model.parameters(), lr=RPROP_LR, etas=RPROP_ETAS, step_sizes=RPROP_STEP_SIZES
        )
        if input_size == 784:
            inputs, labels = raw_inputs[:8], raw_labels[:8]
        else:
            inputs = torch.rand(8, input_size)
            labels = torch.randint(0, NUM_CLASSES, (8,))

        # CrossEntropyLoss riceve logits e label intere: softmax non si applica a mano.
        loss = nn.functional.cross_entropy(model(inputs), labels)
        loss.backward()
        optimizer.step()

        # Verifica il contratto dell'uscita e che logits e loss siano numericamente validi.
        logits = model(inputs)
        if logits.shape != (8, NUM_CLASSES) or not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise RuntimeError(f"Smoke test non superato per input_size={input_size}.")
        print(
            f"Smoke test OK | input={tuple(inputs.shape)} | logits={tuple(logits.shape)} | "
            f"cross-entropy={loss.item():.4f} | RProp update eseguito"
        )


def load_representation(
    representation: str,
    device: torch.device,
    names: tuple[str, ...] = ("train", "val", "test"),
) -> dict:
    """Carica solo gli split richiesti, da raw MNIST o dalle feature archiviate."""
    if representation == "raw":
        # La baseline usa gli stessi split e le immagini originali appiattite a 784 pixel.
        return load_splits(device=device, names=names)

    # I classificatori confrontano le rappresentazioni estratte dall'autoencoder seed 42.
    feature_path = FEATURES_DIR / f"mnist_features_{representation}.npz"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature mancanti: {feature_path}. Prima eseguire "
            "python autoencoder.py --all --seed 42 e python extract_features.py --seed 42."
        )

    # Converte soltanto gli array e le label richiesti sul device scelto.
    with np.load(feature_path) as archive:
        splits = {
            name: (
                torch.tensor(archive[f"X_{name}"], dtype=torch.float32, device=device),
                torch.tensor(archive[f"y_{name}"], dtype=torch.long, device=device),
            )
            for name in names
        }

    return splits


def evaluate(model: Classifier, inputs: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Calcola cross-entropy e accuracy su uno split senza aggiornare i pesi."""
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
    return float(loss.item()), float(accuracy.item())


def output_paths(
    representation: str,
    seed: int,
    campaign_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Costruisce percorsi per checkpoint, log e riepilogo, isolati per campagna se richiesto."""
    tag = f"classifier_{representation}_s{seed}"
    checkpoint_dir = CHECKPOINT_DIR if campaign_dir is None else campaign_dir / "checkpoints"
    results_dir = RESULTS_DIR if campaign_dir is None else campaign_dir / "results"
    return (
        checkpoint_dir / f"{tag}_best.pt",
        results_dir / f"{tag}_history.csv",
        results_dir / f"{tag}_summary.json",
    )


def train_classifier(
    representation: str,
    seed: int,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    min_rel_delta: float = DEFAULT_MIN_REL_DELTA,
    log_every: int = DEFAULT_LOG_EVERY,
    campaign_dir: Path | None = None,
) -> dict:
    """Allena una configurazione e seleziona i pesi migliori secondo la validation loss."""
    if max_epochs < 1 or patience < 1 or log_every < 1:
        raise ValueError("max_epochs, patience e log_every devono essere almeno 1.")
    if not 0.0 <= min_rel_delta < 1.0:
        raise ValueError("min_rel_delta deve essere nell'intervallo [0, 1).")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Rappresentazione non supportata: {representation}.")

    # Ripete in modo deterministico l'inizializzazione del classificatore.
    set_seed(seed)
    device = get_device()

    # Durante la selezione carica solo training e validation: il test resta isolato.
    splits = load_representation(representation, device, names=("train", "val"))
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    # Costruisce la stessa rete a 64 neuroni nascosti per ogni dimensione di input.
    model = Classifier(input_size=X_train.shape[1]).to(device)
    optimizer = torch.optim.Rprop(
        model.parameters(), lr=RPROP_LR, etas=RPROP_ETAS, step_sizes=RPROP_STEP_SIZES
    )

    # Crea cartelle isolate per questa campagna o usa quelle storiche del classificatore.
    checkpoint_path, history_path, summary_path = output_paths(representation, seed, campaign_dir)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # Registra le prestazioni iniziali: l'epoca 0 non aggiorna i pesi.
    initial_train_loss, initial_train_accuracy = evaluate(model, X_train, y_train)
    initial_val_loss, initial_val_accuracy = evaluate(model, X_val, y_val)
    initial_record = {
        "epoch": 0,
        "train_loss": initial_train_loss,
        "train_accuracy": initial_train_accuracy,
        "val_loss": initial_val_loss,
        "val_accuracy": initial_val_accuracy,
        "epoch_seconds": 0.0,
    }

    # Conserva in memoria i pesi con migliore validation loss, anche prima del primo update.
    best_val_loss = initial_val_loss
    # Questa soglia separata controlla la patience, non la scelta del checkpoint.
    best_significant_val_loss = initial_val_loss
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    stop_reason = "max_epochs"
    start_time = time.perf_counter()

    print(
        f"representation={representation} | input={X_train.shape[1]} | seed={seed} | "
        f"RProp batch | max_epochs={max_epochs} | patience={patience} | "
        f"min_rel_delta={min_rel_delta:.4g} | device={device}"
    )

    # Ogni epoca usa un solo gradiente calcolato sui 20.000 esempi di training.
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        columns = list(initial_record.keys())
        writer = csv.DictWriter(history_file, fieldnames=columns)
        writer.writeheader()
        writer.writerow(initial_record)

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            optimizer.zero_grad()

            # Cross-entropy su tutto il training set: un backward e un update RProp.
            train_logits = model(X_train)
            train_loss = nn.functional.cross_entropy(train_logits, y_train)
            train_loss.backward()
            optimizer.step()

            # Registra loss e accuracy dopo l'aggiornamento; la validation non fa backward.
            train_loss_value, train_accuracy = evaluate(model, X_train, y_train)
            val_loss, val_accuracy = evaluate(model, X_val, y_val)
            record = {
                "epoch": epoch,
                "train_loss": train_loss_value,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            writer.writerow(record)
            history_file.flush()

            # Il checkpoint segue sempre la validation loss assoluta più bassa.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            # La patience si azzera solo quando il miglioramento relativo supera la soglia.
            if val_loss < best_significant_val_loss * (1.0 - min_rel_delta):
                best_significant_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Per default stampa ogni epoca, mantenendo il log CSV completo in ogni caso.
            if epoch % log_every == 0:
                print(
                    f"Epoch {epoch:5d} | train loss {train_loss_value:.5f} acc {train_accuracy:.4f} | "
                    f"val loss {val_loss:.5f} acc {val_accuracy:.4f} | "
                    f"best val loss {best_val_loss:.5f} (epoch {best_epoch})"
                )

            # L'early stopping scatta dopo patience epoche senza il miglioramento richiesto.
            if epochs_without_improvement >= patience:
                stop_reason = "validation_patience"
                break

    training_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)

    # Carica e usa il test una sola volta, dopo aver selezionato l'epoca migliore.
    test_split = load_representation(representation, device, names=("test",))
    X_test, y_test = test_split["test"]
    test_loss, test_accuracy = evaluate(model, X_test, y_test)
    epochs_completed = epoch if max_epochs else 0
    converged = stop_reason == "validation_patience"

    # Salva i pesi e i metadati che permettono di ricostruire esattamente il modello.
    torch.save(
        {
            "model_state_dict": best_state,
            "input_size": X_train.shape[1],
            "hidden_size": CLASSIFIER_HIDDEN_SIZE,
            "num_classes": NUM_CLASSES,
            "representation": representation,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "optimizer": "RProp",
            "optimizer_lr": RPROP_LR,
            "batch_mode": "full_training_set",
            "loss": "CrossEntropyLoss",
        },
        checkpoint_path,
    )

    # Salva metriche, stop reason, tempi e accuratezza di test nel riepilogo del run.
    summary = {
        "representation": representation,
        "input_size": int(X_train.shape[1]),
        "hidden_size": CLASSIFIER_HIDDEN_SIZE,
        "num_classes": NUM_CLASSES,
        "seed": seed,
        "optimizer": "RProp",
        "optimizer_lr": RPROP_LR,
        "rprop_etas": RPROP_ETAS,
        "rprop_step_sizes": RPROP_STEP_SIZES,
        "batch_mode": "full_training_set",
        "loss": "CrossEntropyLoss",
        "max_epochs": max_epochs,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_train_loss": initial_train_loss if best_epoch == 0 else None,
        "best_val_accuracy": initial_val_accuracy if best_epoch == 0 else None,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "patience": patience,
        "min_rel_delta": min_rel_delta,
        "converged": converged,
        "stop_reason": stop_reason,
        "training_seconds": training_seconds,
        "device": str(device),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR)),
        "history": str(history_path.relative_to(PROJECT_DIR)),
    }

    # Ricava train e validation metriche dall'epoca che ha generato il best checkpoint.
    if best_epoch > 0:
        summary["best_train_loss"] = float(record_at_epoch(history_path, best_epoch, "train_loss"))
        summary["best_val_accuracy"] = float(record_at_epoch(history_path, best_epoch, "val_accuracy"))

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(
        f"Fine {representation}/seed {seed}: best epoch={best_epoch}, "
        f"stop epoch={epochs_completed} ({stop_reason}), test loss={test_loss:.5f}, "
        f"test accuracy={test_accuracy:.4f}, training={format_duration(training_seconds)}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Log:        {history_path}")
    print(f"Summary:    {summary_path}")
    return summary


def record_at_epoch(history_path: Path, epoch: int, column: str) -> float:
    """Legge dal CSV il valore di train/validation associato all'epoca migliore."""
    with history_path.open(newline="", encoding="utf-8") as history_file:
        for row in csv.DictReader(history_file):
            if int(row["epoch"]) == epoch:
                return float(row[column])
    raise ValueError(f"Epoca {epoch} non trovata in {history_path}.")


def save_aggregate(summaries: list[dict], results_dir: Path | None = None) -> Path:
    """Salva media e deviazione standard dei run per ogni rappresentazione."""
    results_dir = RESULTS_DIR if results_dir is None else results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for summary in summaries:
        # Raggruppa le inizializzazioni del classificatore con la stessa rappresentazione.
        grouped.setdefault(summary["representation"], []).append(summary)

    rows = []
    for representation, runs in sorted(grouped.items()):
        test_accuracies = [run["test_accuracy"] for run in runs]
        test_losses = [run["test_loss"] for run in runs]
        best_epochs = [run["best_epoch"] for run in runs]
        training_times = [run["training_seconds"] for run in runs]

        # Con un solo seed la deviazione standard non è definita, quindi viene lasciata vuota.
        def sample_std(values: list[float]) -> float | str:
            return statistics.stdev(values) if len(values) > 1 else ""

        rows.append(
            {
                "representation": representation,
                "runs": len(runs),
                "seeds": ",".join(str(run["seed"]) for run in runs),
                "converged_runs": sum(run["converged"] for run in runs),
                "best_epoch_mean": statistics.mean(best_epochs),
                "best_epoch_std": sample_std(best_epochs),
                "test_accuracy_mean": statistics.mean(test_accuracies),
                "test_accuracy_std": sample_std(test_accuracies),
                "test_loss_mean": statistics.mean(test_losses),
                "test_loss_std": sample_std(test_losses),
                "training_seconds_mean": statistics.mean(training_times),
                "training_seconds_std": sample_std(training_times),
            }
        )

    # Scrive una riga per input: raw baseline e ciascuna dimensione latente.
    output_path = results_dir / "classifier_comparison.csv"
    with output_path.open("w", newline="", encoding="utf-8") as aggregate_file:
        writer = csv.DictWriter(aggregate_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Riepilogo media ± deviazione standard: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Legge configurazione, seed e opzioni di smoke test dalla riga di comando."""
    parser = argparse.ArgumentParser(description="Allena classificatori MNIST con RProp full-batch.")
    parser.add_argument(
        "--representation",
        choices=(*REPRESENTATIONS, "all"),
        default="m25",
        help="Feature latenti da usare, baseline raw o tutte le configurazioni.",
    )
    parser.add_argument("--seed", type=int, choices=CLASSIFIER_SEEDS, default=CLASSIFIER_SEEDS[0])
    parser.add_argument("--all-seeds", action="store_true", help="Ripete la rappresentazione scelta con i 5 seed.")
    parser.add_argument("--all", action="store_true", help="Esegue tutti gli input (raw e cinque m) con tutti i seed.")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-rel-delta", type=float, default=DEFAULT_MIN_REL_DELTA)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--smoke-test", action="store_true", help="Testa rete, cross-entropy e RProp su pochi campioni.")
    return parser.parse_args()


def main() -> None:
    """Lancia uno, più seed o tutte le configurazioni secondo le opzioni CLI."""
    args = parse_args()
    if args.smoke_test:
        check_classifier()
        return

    if args.all and args.representation != "m25":
        raise ValueError("Usare --all senza --representation, oppure scegliere una singola rappresentazione.")

    # Seleziona le rappresentazioni e i seed richiesti dall'utente.
    representations = REPRESENTATIONS if args.all else (args.representation,)
    seeds = CLASSIFIER_SEEDS if args.all or args.all_seeds else (args.seed,)
    summaries = []

    # Ogni combinazione rappresentazione/seed è un run indipendente e riproducibile.
    for representation in representations:
        for seed in seeds:
            summaries.append(
                train_classifier(
                    representation=representation,
                    seed=seed,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    min_rel_delta=args.min_rel_delta,
                    log_every=args.log_every,
                )
            )

    # Mostra test accuracy dei run effettuati senza usare il test per altre scelte.
    print("\n=== Riepilogo test ===")
    for summary in summaries:
        print(
            f"{summary['representation']:>4} | seed {summary['seed']} | "
            f"best epoch {summary['best_epoch']:>5} | converged={summary['converged']} | "
            f"test accuracy={summary['test_accuracy']:.4f}"
        )

    # Con più seed produce il confronto aggregato previsto dal protocollo.
    if len(summaries) > 1:
        save_aggregate(summaries)


if __name__ == "__main__":
    # Esegue il training o lo smoke test quando il modulo è lanciato direttamente.
    main()
