"""Autoencoder MNIST con training batch, validazione ed early stopping."""

import argparse
import copy
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
AUTOENCODER_SEEDS = (42, 1, 2)
DEFAULT_HIDDEN_SIZE = HIDDEN_SIZES[0]

# Early stopping ("convergenza pratica"): un'epoca conta come miglioramento solo se abbassa
# la miglior validation MSE di almeno lo 0,1% (soglia RELATIVA, quindi equa per tutti gli m:
# la MSE va da ~0.003 per m=125 a ~0.022 per m=25). Ci si ferma dopo 100 epoche consecutive
# senza miglioramento. 10000 epoche è solo un limite di sicurezza.
DEFAULT_MAX_EPOCHS = 10000
DEFAULT_PATIENCE = 100
DEFAULT_MIN_REL_DELTA = 1e-3

# Iperparametri RProp, scritti per esteso (sono i default di torch.optim.Rprop).
# Passo iniziale Δ0, uguale per ogni peso prima del primo aggiornamento.
RPROP_LR = 0.01
# (η-, η+): il passo si dimezza se il segno del gradiente cambia, cresce del 20% se resta uguale.
RPROP_ETAS = (0.5, 1.2)
# (Δmin, Δmax): limiti entro cui resta il passo di ogni peso.
RPROP_STEP_SIZES = (1e-6, 50.0)

# Ogni quante epoche stampare l'avanzamento a terminale.
DEFAULT_LOG_EVERY = 100


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


def mean_rprop_step(optimizer: torch.optim.Rprop) -> float:
    """Media dei passi Δ_ij di RProp su tutti i pesi: mostra come RProp adatta i passi."""
    steps = [
        optimizer.state[p]["step_size"].flatten()
        for group in optimizer.param_groups
        for p in group["params"]
        if "step_size" in optimizer.state[p]
    ]
    # Prima del primo aggiornamento lo stato è vuoto: ogni passo vale ancora Δ0.
    return torch.cat(steps).mean().item() if steps else RPROP_LR


def format_duration(seconds: float) -> str:
    """Formatta una durata in modo leggibile, per esempio '2 min 05.3 s' oppure '12.1 s'."""
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)} min {secs:04.1f} s" if minutes >= 1 else f"{secs:.1f} s"


def output_paths(hidden_size: int, seed: int) -> tuple[Path, Path, Path]:
    """Percorsi di checkpoint, log e riepilogo; il seed compare nel nome solo se non è quello di default."""
    tag = f"autoencoder_m{hidden_size}" + ("" if seed == SEED else f"_s{seed}")
    return (
        CHECKPOINT_DIR / f"{tag}_best.pt",
        RESULTS_DIR / f"{tag}_history.csv",
        RESULTS_DIR / f"{tag}_summary.json",
    )


def load_autoencoder(checkpoint_path: str | Path, device: torch.device | None = None) -> Autoencoder:
    """Ricarica un autoencoder salvato: dal giorno 7 si userà model.encode(x) come input del classificatore."""
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=True)
    model = Autoencoder(input_size=checkpoint["input_size"], hidden_size=checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device or "cpu").eval()


def train_autoencoder(
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    min_rel_delta: float = DEFAULT_MIN_REL_DELTA,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    seed: int = SEED,
    log_every: int = DEFAULT_LOG_EVERY,
) -> dict:
    """Allena l'autoencoder con RProp batch e seleziona il checkpoint via validation MSE."""
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs e patience devono essere almeno 1.")

    # Fissa le casualità (inizializzazione dei pesi) e seleziona GPU, se disponibile.
    set_seed(seed)
    device = get_device()

    # Carica solo train e validation: il test non partecipa al training né alla selezione.
    splits = load_splits(device=device, names=("train", "val"))
    X_train, _ = splits["train"]
    X_val, _ = splits["val"]

    # Crea il modello 784 -> hidden_size -> 784 con RProp full-batch.
    model = Autoencoder(input_size=X_train.shape[1], hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Rprop(
        model.parameters(), lr=RPROP_LR, etas=RPROP_ETAS, step_sizes=RPROP_STEP_SIZES
    )

    # Prepara le cartelle e i nomi dei file di output.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path, history_path, summary_path = output_paths(hidden_size, seed)

    # Epoca 0: errore dei pesi iniziali, così le curve partono dal modello non addestrato.
    initial_record = {
        "epoch": 0,
        "train_mse": evaluate_reconstruction(model, X_train),
        "val_mse": evaluate_reconstruction(model, X_val),
        "mean_step": RPROP_LR,
        "epoch_seconds": 0.0,
    }
    epoch_records = [initial_record]

    # Il miglior modello si tiene in memoria e si salva su disco una volta sola, alla fine.
    best_val_loss = initial_record["val_mse"]
    # La soglia relativa governa la patience, separatamente dal checkpoint migliore.
    best_significant_val_loss = initial_record["val_mse"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    stop_reason = "max_epochs"
    start_time = time.perf_counter()

    print("Loss = MSE di ricostruzione (media per pixel); più è bassa, migliore è la ricostruzione.")
    print(
        f"m={hidden_size} | seed={seed} | max_epochs={max_epochs} | patience={patience} | "
        f"min_rel_delta={min_rel_delta} | device={device}"
    )

    # Ogni epoca calcola un unico gradiente sull'intero training set.
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=list(initial_record.keys()))
        writer.writeheader()
        writer.writerow(initial_record)

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            optimizer.zero_grad()
            train_reconstruction = model(X_train)
            train_loss = nn.functional.mse_loss(train_reconstruction, X_train)
            train_loss.backward()
            optimizer.step()

            # Train e validation ricalcolate sui pesi appena aggiornati (stessi pesi per entrambe).
            train_mse = evaluate_reconstruction(model, X_train)
            val_mse = evaluate_reconstruction(model, X_val)
            epoch_seconds = time.perf_counter() - epoch_start
            record = {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "mean_step": mean_rprop_step(optimizer),
                "epoch_seconds": epoch_seconds,
            }
            epoch_records.append(record)
            writer.writerow(record)

            # Il checkpoint segue sempre la validation MSE assoluta più bassa.
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_epoch = epoch
                # Copia dei pesi: state_dict() da solo è un riferimento e cambierebbe insieme al modello.
                best_state = copy.deepcopy(model.state_dict())

            # La patience si azzera solo con un miglioramento relativo significativo.
            if val_mse < best_significant_val_loss * (1.0 - min_rel_delta):
                best_significant_val_loss = val_mse
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Stampa ogni log_every epoche: con migliaia di epoche una riga per epoca è troppo.
            if epoch % log_every == 0:
                # Il CSV resta leggibile anche se il training si interrompe.
                history_file.flush()
                print(
                    f"Epoch {epoch:5d} | train loss: {train_mse:.6f} | val loss: {val_mse:.6f} | "
                    f"best val loss: {best_val_loss:.6f} (epoch {best_epoch}) | mean step: {record['mean_step']:.2e}"
                )

            # Ferma dopo patience epoche senza un miglioramento significativo.
            if epochs_without_improvement >= patience:
                stop_reason = "validation_patience"
                break

    total_seconds = time.perf_counter() - start_time
    stopped_early = stop_reason == "validation_patience"

    # Ripristina i pesi di E_best, scelti usando solo la validation.
    model.load_state_dict(best_state)

    # Valutazione finale: il test viene caricato e usato una sola volta, sui pesi di E_best.
    X_test, _ = load_splits(device=device, names=("test",))["test"]
    test_mse = evaluate_reconstruction(model, X_test)

    summary = {
        "hidden_size": hidden_size,
        "seed": seed,
        "optimizer": "RProp (iRprop-, torch.optim.Rprop)",
        "optimizer_lr": RPROP_LR,
        "rprop_etas": RPROP_ETAS,
        "rprop_step_sizes": RPROP_STEP_SIZES,
        "batch_mode": "full_training_set",
        "loss": "MSE",
        "max_epochs": max_epochs,
        # Epoche di training effettive, esclusa l'epoca 0 (pesi iniziali).
        "epochs_completed": len(epoch_records) - 1,
        # Epoca con la miglior validation MSE: è il numero di epoche alla convergenza.
        "best_epoch": best_epoch,
        "best_train_mse": epoch_records[best_epoch]["train_mse"],
        "best_val_mse": best_val_loss,
        "test_mse": test_mse,
        "patience": patience,
        "min_rel_delta": min_rel_delta,
        # True = fermato per patience (converso); False = fermato al limite di epoche.
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "training_seconds": total_seconds,
        "device": str(device),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_DIR)),
        "history": str(history_path.relative_to(PROJECT_DIR)),
    }

    # Checkpoint dei pesi di E_best, con i metadati necessari a ricaricarlo.
    torch.save(
        {
            "model_state_dict": best_state,
            "input_size": X_train.shape[1],
            "hidden_size": hidden_size,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_loss,
            "seed": seed,
            "optimizer": "RProp",
            "optimizer_lr": RPROP_LR,
            "batch_mode": "full_training_set",
            "loss": "MSE",
        },
        checkpoint_path,
    )
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    # La loss è la MSE di ricostruzione: media su tutti i pixel di tutte le immagini.
    print(
        f"Fine: best epoch {best_epoch}, stop epoch {summary['epochs_completed']} ({stop_reason}) | "
        f"val loss {best_val_loss:.6f} | test loss {test_mse:.6f} | {total_seconds:.1f}s"
    )
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Per-epoch log:   {history_path}")
    print(f"Summary:         {summary_path}")
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
    selection.add_argument(
        "--all-seeds",
        action="store_true",
        help="Allena i cinque autoencoder con i seed 42, 1 e 2.",
    )
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-rel-delta", type=float, default=DEFAULT_MIN_REL_DELTA,
                        help="Miglioramento relativo minimo della validation MSE (default: 0.001 = 0,1%%).")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed per l'inizializzazione dei pesi.")
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--check-model", action="store_true", help="Verifica tutte le architetture, senza training.")
    return parser.parse_args()


if __name__ == "__main__":
    # Avvia il timer del programma: misura tutto, dal lancio alla fine.
    program_start = time.perf_counter()

    # Il flag --check-model esegue il controllo architetturale; altrimenti parte il training.
    args = parse_args()
    if args.check_model:
        check_model()
    else:
        # Sceglie le combinazioni dimensione/seed richieste, con identico protocollo.
        if args.all_seeds:
            configurations = [(m, seed) for seed in AUTOENCODER_SEEDS for m in HIDDEN_SIZES]
        else:
            hidden_sizes = HIDDEN_SIZES if args.all else (args.hidden_size,)
            configurations = [(m, args.seed) for m in hidden_sizes]
        timings = []
        for hidden_size, seed in configurations:
            print(f"\n=== Autoencoder m={hidden_size}, seed={seed} ===")
            # Tempo totale del modello: caricamento dati, training, test e salvataggi.
            model_start = time.perf_counter()
            summary = train_autoencoder(
                max_epochs=args.max_epochs,
                patience=args.patience,
                min_rel_delta=args.min_rel_delta,
                hidden_size=hidden_size,
                seed=seed,
                log_every=args.log_every,
            )
            timings.append((hidden_size, seed, summary["epochs_completed"],
                            summary["training_seconds"], time.perf_counter() - model_start))

        # Riepilogo dei tempi: solo ciclo di training e totale per modello.
        print("\n=== Tempi ===")
        print(f"{'m':>4} {'seed':>5} {'epoche':>7} {'training':>14} {'totale modello':>16}")
        for hidden_size, seed, epochs, train_seconds, model_seconds in timings:
            print(f"{hidden_size:>4} {seed:>5} {epochs:>7} {format_duration(train_seconds):>14} "
                  f"{format_duration(model_seconds):>16}")

    # Tempo effettivo di tutto il programma.
    print(f"\nTempo totale del programma: {format_duration(time.perf_counter() - program_start)}")
