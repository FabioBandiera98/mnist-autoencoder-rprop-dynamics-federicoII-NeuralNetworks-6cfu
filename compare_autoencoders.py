"""
compare_autoencoders.py — Giorno 6: confronto dei cinque autoencoder.

Non addestra niente: legge checkpoint, log e riepiloghi prodotti da
`python autoencoder.py --all`, quindi si può rilanciare quante volte si vuole.

Produce:
    results/autoencoder_comparison.csv              tabella riassuntiva (una riga per m)
    results/figures/ae_val_loss.png                 curve di validation MSE sovrapposte
    results/figures/ae_loss_m<m>.png                train e val MSE di ogni autoencoder
    results/figures/ae_mse_vs_m.png                 MSE finale (train/val/test) al variare di m
    results/figures/ae_epochs_vs_m.png              epoche alla convergenza al variare di m
    results/figures/ae_reconstructions_test.png     stesse 10 cifre di test ricostruite da ogni m

Uso:
    python compare_autoencoders.py
    python compare_autoencoders.py --seed 0
"""

import argparse
import csv
import json

import matplotlib

# Backend senza finestre: le figure vengono solo salvate su file.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from autoencoder import HIDDEN_SIZES, PROJECT_DIR, RESULTS_DIR, load_autoencoder, output_paths
from data import SEED, get_device, load_splits

FIG_DIR = RESULTS_DIR / "figures"

# Soglie per le statistiche dell'encoder: sono solo descrittive e non influenzano il training.
# Un'attivazione sigmoidale è considerata satura se è sotto 0.05 o sopra 0.95.
SATURATION_LOW, SATURATION_HIGH = 0.05, 0.95
# Un neurone è quasi costante se la sua deviazione standard sul training è sotto 0.01.
CONSTANT_STD = 0.01


def hidden_colors() -> dict:
    """Un colore per m da una scala ordinata: m piccolo = scuro, m grande = chiaro."""
    cmap = plt.get_cmap("viridis")
    n = len(HIDDEN_SIZES) - 1
    return {m: cmap(0.85 * i / n) for i, m in enumerate(HIDDEN_SIZES)}


def load_run(hidden_size: int, seed: int) -> dict:
    """Legge riepilogo JSON e log per epoca di un autoencoder; errore chiaro se mancano."""
    checkpoint_path, history_path, summary_path = output_paths(hidden_size, seed)
    if not (checkpoint_path.exists() and history_path.exists() and summary_path.exists()):
        raise FileNotFoundError(f"Mancano i risultati di m={hidden_size}: lancia 'python autoencoder.py --all'.")

    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    with history_path.open(encoding="utf-8") as f:
        # Tutte le colonne del log sono numeriche
        history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    if "test_mse" not in summary:
        raise ValueError(f"Il riepilogo di m={hidden_size} è della versione vecchia (senza test MSE): rifai il training.")
    return {"summary": summary, "history": history, "checkpoint": checkpoint_path}


@torch.no_grad()
def encoder_statistics(model, X: torch.Tensor) -> dict:
    """Statistiche di z = E(x) sul training set: è l'input che riceverà il classificatore."""
    # Codifica tutto il training set: matrice (N, m) con valori in (0, 1).
    z = model.encode(X)
    # Frazione di attivazioni sature, calcolata su tutti i neuroni e tutte le immagini.
    saturated = ((z < SATURATION_LOW) | (z > SATURATION_HIGH)).float().mean().item()
    # Variabilità di ogni neurone sulle immagini del training.
    unit_std = z.std(dim=0)
    return {
        # Attivazione media su tutti i neuroni e le immagini.
        "z_mean": round(z.mean().item(), 4),
        # Percentuale di attivazioni vicine a 0 o a 1.
        "z_saturated_pct": round(100 * saturated, 1),
        # Neuroni quasi costanti, che non portano informazione al classificatore.
        "constant_units": int((unit_std < CONSTANT_STD).sum().item()),
    }


def build_table(runs: dict, X_train: torch.Tensor, device: torch.device) -> list:
    """Una riga per autoencoder: metriche del training e statistiche dell'encoder."""
    rows = []
    for m, run in runs.items():
        s = run["summary"]
        model = load_autoencoder(run["checkpoint"], device)
        rows.append({
            "m": m,
            # Fattore di compressione: quante volte z è più piccolo dell'immagine.
            "compression": round(784 / m, 1),
            "params": sum(p.numel() for p in model.parameters()),
            "best_epoch": s["best_epoch"],
            "epochs_completed": s["epochs_completed"],
            "converged": s["stopped_early"],
            "train_mse": round(s["best_train_mse"], 6),
            "val_mse": round(s["best_val_mse"], 6),
            "test_mse": round(s["test_mse"], 6),
            "training_seconds": round(s["training_seconds"], 1),
            **encoder_statistics(model, X_train),
        })
    return rows


def save_table(rows: list) -> None:
    """Salva la tabella in CSV e la stampa a terminale."""
    path = RESULTS_DIR / "autoencoder_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    header = (f"{'m':>4} {'compr':>6} {'best':>6} {'stop':>6} {'conv':>5} {'train MSE':>10} "
              f"{'val MSE':>10} {'test MSE':>10} {'satur%':>7} {'cost.':>5} {'sec':>7}")
    print("\n" + header + "\n" + "-" * len(header))
    for r in rows:
        print(f"{r['m']:>4} {r['compression']:>6} {r['best_epoch']:>6} {r['epochs_completed']:>6} "
              f"{'sì' if r['converged'] else 'no':>5} {r['train_mse']:>10.6f} {r['val_mse']:>10.6f} "
              f"{r['test_mse']:>10.6f} {r['z_saturated_pct']:>7.1f} {r['constant_units']:>5} "
              f"{r['training_seconds']:>7.1f}")
    print(f"\nTabella: {path.relative_to(PROJECT_DIR)}")


def plot_single_curves(runs: dict) -> None:
    """Per ogni m: train e val MSE nel tempo, con la best epoch tratteggiata."""
    for m, run in runs.items():
        h, s = run["history"], run["summary"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["epoch"] for r in h], [r["train_mse"] for r in h], label="training")
        ax.plot([r["epoch"] for r in h], [r["val_mse"] for r in h], label="validation")
        ax.axvline(s["best_epoch"], color="gray", linestyle="--", linewidth=1,
                   label=f"best epoch = {s['best_epoch']}")
        # La MSE scende di ordini di grandezza.
        ax.set_yscale("log")
        ax.set_xlabel("epoca (= aggiornamento RProp)")
        ax.set_ylabel("MSE di ricostruzione")
        ax.set_title(f"Autoencoder m = {m}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"ae_loss_m{m}.png", dpi=150)
        plt.close(fig)


def plot_val_curves(runs: dict, colors: dict) -> None:
    """Validation MSE di tutti gli autoencoder nello stesso grafico; pallino = best epoch."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, run in runs.items():
        h, s = run["history"], run["summary"]
        ax.plot([r["epoch"] for r in h], [r["val_mse"] for r in h], color=colors[m], label=f"m = {m}")
        ax.plot(s["best_epoch"], s["best_val_mse"], "o", color=colors[m], markersize=5)
    ax.set_yscale("log")
    ax.set_xlabel("epoca (= aggiornamento RProp)")
    ax.set_ylabel("validation MSE")
    ax.set_title("Validation MSE degli autoencoder (pallino = best epoch)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ae_val_loss.png", dpi=150)
    plt.close(fig)


def plot_mse_vs_m(rows: list) -> None:
    """MSE finale (pesi della best epoch) al variare di m."""
    m = [r["m"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label, marker in (("train_mse", "training", "o"),
                               ("val_mse", "validation", "s"),
                               ("test_mse", "test", "^")):
        ax.plot(m, [r[key] for r in rows], marker=marker, label=label)
    ax.set_xticks(m)
    ax.set_xlabel("m (neuroni nascosti)")
    ax.set_ylabel("MSE di ricostruzione")
    ax.set_title("Errore di ricostruzione al variare di m")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ae_mse_vs_m.png", dpi=150)
    plt.close(fig)


def plot_epochs_vs_m(rows: list, colors: dict) -> None:
    """Best epoch (barre piene) ed epoca di arresto (barre vuote) al variare di m."""
    m = [r["m"] for r in rows]
    x = list(range(len(m)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, [r["epochs_completed"] for r in rows], color="none", edgecolor="gray", label="epoca di arresto")
    ax.bar(x, [r["best_epoch"] for r in rows], color=[colors[v] for v in m], label="best epoch")
    for i, r in enumerate(rows):
        # Scrive "limite" sopra i modelli fermati al massimo di epoche.
        if not r["converged"]:
            ax.annotate("limite", (i, r["epochs_completed"]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [str(v) for v in m])
    ax.set_xlabel("m (neuroni nascosti)")
    ax.set_ylabel("epoche")
    ax.set_title("Epoche alla convergenza degli autoencoder")
    # Lascia spazio in alto per le etichette "limite".
    ax.set_ylim(0, 1.12 * max(r["epochs_completed"] for r in rows))
    # Legenda sotto il grafico, così non copre le barre.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ae_epochs_vs_m.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_reconstructions(runs: dict, X_test: torch.Tensor, y_test: torch.Tensor, device: torch.device) -> None:
    """Prima riga: una cifra di test per classe. Righe successive: le ricostruzioni di ogni m."""
    # Prima immagine di ogni classe: le cifre mostrate sono sempre le stesse.
    idx = torch.stack([torch.nonzero(y_test == c)[0, 0] for c in range(10)])
    originals = X_test[idx]

    rows = [("originale", originals)]
    for m, run in runs.items():
        rows.append((f"m = {m}", load_autoencoder(run["checkpoint"], device)(originals)))

    fig, axes = plt.subplots(len(rows), 10, figsize=(11, 1.15 * len(rows) + 0.6))
    for r, (label, images) in enumerate(rows):
        for c in range(10):
            axes[r, c].imshow(images[c].view(28, 28).cpu(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("Ricostruzioni sul test set al variare di m (pesi della best epoch)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ae_reconstructions_test.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Costruisce la tabella e tutte le figure di confronto dei cinque autoencoder."""
    # Legge da riga di comando il seed dei modelli da confrontare.
    parser = argparse.ArgumentParser(description="Confronto dei cinque autoencoder (giorno 6).")
    parser.add_argument("--seed", type=int, default=SEED, help="seed degli autoencoder da confrontare")
    args = parser.parse_args()

    device = get_device()
    # Carica tutti gli split: il test serve solo per la figura delle ricostruzioni.
    splits = load_splits(device=device)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Legge i risultati dei cinque autoencoder, in ordine di m crescente.
    runs = {m: load_run(m, args.seed) for m in HIDDEN_SIZES}
    colors = hidden_colors()

    rows = build_table(runs, splits["train"][0], device)
    save_table(rows)
    plot_single_curves(runs)
    plot_val_curves(runs, colors)
    plot_mse_vs_m(rows)
    plot_epochs_vs_m(rows, colors)
    plot_reconstructions(runs, *splits["test"], device)
    print(f"Figure: {FIG_DIR.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
