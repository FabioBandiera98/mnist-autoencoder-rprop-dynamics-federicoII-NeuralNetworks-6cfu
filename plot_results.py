"""
plot_results.py — Giorni 15-16: tabella comparativa finale e grafici dei classificatori.

Non addestra niente e non usa PyTorch: legge solo i riepiloghi JSON e i log CSV prodotti da
`autoencoder.py` e `classifier.py`, quindi si può rilanciare quante volte si vuole.

Produce (in results/, oppure nella cartella della campagna se si usa --campaign):
    final_table.csv                         tabella finale, una riga per rappresentazione
    figures/clf_test_accuracy.png           test accuracy media ± dev. std. e singoli seed
    figures/clf_epochs.png                  epoche alla convergenza (best epoch) media ± dev. std.
    figures/clf_time.png                    tempo di training medio ± dev. std.
    figures/clf_loss_curves.png             curve di loss training/validation, un riquadro per input
    figures/clf_val_accuracy.png            validation accuracy nel tempo, tutti gli input insieme
    figures/clf_train_val_test.png          accuracy su training, validation e test (overfitting)
    figures/clf_accuracy_vs_ae_loss.png     test accuracy rispetto alla loss di ricostruzione dell'encoder

Uso:
    python plot_results.py
    python plot_results.py --campaign Yegor-3080
    python plot_results.py --curve-seed 3
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import matplotlib

# Backend senza finestre: le figure vengono solo salvate su file.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Costanti ripetute qui, invece di importarle da autoencoder.py, per non caricare PyTorch.
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
HIDDEN_SIZES = (25, 50, 75, 100, 125)
REPRESENTATIONS = ("raw", *(f"m{size}" for size in HIDDEN_SIZES))

# Seed dell'autoencoder da cui provengono le feature usate dai classificatori.
AUTOENCODER_FEATURE_SEED = 42
# Seed del classificatore mostrato nelle figure con le curve per epoca.
DEFAULT_CURVE_SEED = 1


def format_duration(seconds: float) -> str:
    """Formatta una durata in modo leggibile, per esempio '2 min 05.3 s' oppure '12.1 s'."""
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)} min {secs:04.1f} s" if minutes >= 1 else f"{secs:.1f} s"


def representation_label(representation: str) -> str:
    """Etichetta leggibile per gli assi: 'pixel (784)' per la baseline, 'm = 50' per le feature."""
    return "pixel (784)" if representation == "raw" else f"m = {representation[1:]}"


def representation_colors() -> dict:
    """Grigio per la baseline; per gli m la stessa scala di compare_autoencoders.py."""
    cmap = plt.get_cmap("viridis")
    n = len(HIDDEN_SIZES) - 1
    colors = {f"m{m}": cmap(0.85 * i / n) for i, m in enumerate(HIDDEN_SIZES)}
    colors["raw"] = "gray"
    return colors


def sample_std(values: list) -> float:
    """Deviazione standard campionaria; 0 se c'è un solo run."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_history(history_path: Path) -> list:
    """Legge il log per epoca di un run come lista di dizionari numerici."""
    with history_path.open(encoding="utf-8") as f:
        # Tutte le colonne del log sono numeriche.
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def load_classifier_runs(results_dir: Path) -> dict:
    """Legge riepilogo e log di tutti i classificatori trovati, raggruppati per rappresentazione."""
    runs = {}
    for representation in REPRESENTATIONS:
        runs[representation] = []
        # Un file di riepilogo per seed: classifier_<rappresentazione>_s<seed>_summary.json.
        for summary_path in sorted(results_dir.glob(f"classifier_{representation}_s*_summary.json")):
            with summary_path.open(encoding="utf-8") as f:
                summary = json.load(f)
            history_path = results_dir / f"classifier_{representation}_s{summary['seed']}_history.csv"
            if not history_path.exists():
                raise FileNotFoundError(f"Manca il log per epoca: {history_path}.")
            history = load_history(history_path)

            # La training accuracy della best epoch non è nel riepilogo: si prende dal log.
            summary["best_train_accuracy"] = history[summary["best_epoch"]]["train_accuracy"]
            runs[representation].append({"summary": summary, "history": history})

        if not runs[representation]:
            raise FileNotFoundError(
                f"Nessun risultato per '{representation}' in {results_dir}: lancia 'python classifier.py --all'."
            )
    return runs


def load_autoencoder_summaries() -> dict:
    """Legge i riepiloghi dei cinque autoencoder (seed 42) che hanno prodotto le feature."""
    summaries = {}
    for m in HIDDEN_SIZES:
        # Il seed 42 è quello di default e non compare nel nome del file.
        summary_path = RESULTS_DIR / f"autoencoder_m{m}_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Manca {summary_path}: lancia 'python autoencoder.py --all'.")
        with summary_path.open(encoding="utf-8") as f:
            summaries[f"m{m}"] = json.load(f)
        if summaries[f"m{m}"]["seed"] != AUTOENCODER_FEATURE_SEED:
            raise ValueError(f"Il riepilogo di m={m} non è del seed {AUTOENCODER_FEATURE_SEED}.")
    return summaries


def build_table(runs: dict, ae_summaries: dict) -> list:
    """Una riga per rappresentazione: dati dell'autoencoder e media ± dev. std. dei classificatori."""
    rows = []
    for representation in REPRESENTATIONS:
        summaries = [run["summary"] for run in runs[representation]]
        # La baseline sui pixel non ha un autoencoder: le sue colonne restano vuote.
        ae = ae_summaries.get(representation)

        best_epochs = [s["best_epoch"] for s in summaries]
        train_acc = [s["best_train_accuracy"] for s in summaries]
        val_acc = [s["best_val_accuracy"] for s in summaries]
        test_acc = [s["test_accuracy"] for s in summaries]
        seconds = [s["training_seconds"] for s in summaries]

        rows.append({
            "representation": representation,
            "input_size": summaries[0]["input_size"],
            "ae_best_epoch": ae["best_epoch"] if ae else "",
            "ae_val_mse": round(ae["best_val_mse"], 6) if ae else "",
            "ae_test_mse": round(ae["test_mse"], 6) if ae else "",
            "runs": len(summaries),
            "seeds": ",".join(str(s["seed"]) for s in summaries),
            # Run fermati per patience; gli altri hanno raggiunto il massimo di epoche.
            "converged_runs": sum(bool(s["converged"]) for s in summaries),
            "clf_best_epoch_mean": round(statistics.mean(best_epochs), 1),
            "clf_best_epoch_std": round(sample_std(best_epochs), 1),
            "train_accuracy_mean": round(statistics.mean(train_acc), 4),
            "val_accuracy_mean": round(statistics.mean(val_acc), 4),
            "val_accuracy_std": round(sample_std(val_acc), 4),
            "test_accuracy_mean": round(statistics.mean(test_acc), 4),
            "test_accuracy_std": round(sample_std(test_acc), 4),
            "training_seconds_mean": round(statistics.mean(seconds), 2),
            "training_seconds_std": round(sample_std(seconds), 2),
        })
    return rows


def save_table(rows: list, output_dir: Path) -> None:
    """Salva la tabella in CSV e la stampa a terminale."""
    path = output_dir / "final_table.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    header = (f"{'input':>12} {'ep. AE':>7} {'val loss AE':>12} {'ep. clf':>14} {'conv':>5} "
              f"{'train acc':>10} {'val acc':>16} {'test acc':>16} {'sec':>6}")
    print("\nAccuracy in %, media ± dev. std. sui seed del classificatore, sui pesi della best epoch.")
    print(header + "\n" + "-" * len(header))
    for r in rows:
        ae_epoch = r["ae_best_epoch"] if r["ae_best_epoch"] != "" else "-"
        ae_loss = f"{r['ae_val_mse']:.6f}" if r["ae_val_mse"] != "" else "-"
        epochs = f"{r['clf_best_epoch_mean']:.1f} ± {r['clf_best_epoch_std']:.1f}"
        val = f"{100 * r['val_accuracy_mean']:.2f} ± {100 * r['val_accuracy_std']:.2f}"
        test = f"{100 * r['test_accuracy_mean']:.2f} ± {100 * r['test_accuracy_std']:.2f}"
        print(f"{representation_label(r['representation']):>12} {ae_epoch:>7} {ae_loss:>12} {epochs:>14} "
              f"{r['converged_runs']}/{r['runs']:<3} {100 * r['train_accuracy_mean']:>10.2f} {val:>16} "
              f"{test:>16} {r['training_seconds_mean']:>6.2f}")
    print(f"\nTabella: {path.relative_to(PROJECT_DIR)}")


def plot_mean_std(runs: dict, rows: list, colors: dict, key: str, ylabel: str, title: str,
                  filename: str, fig_dir: Path, scale: float = 1.0) -> None:
    """Grafico generico: un punto per seed, più media e barra di errore (± dev. std.) per ogni input."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, representation in enumerate(REPRESENTATIONS):
        values = [scale * run["summary"][key] for run in runs[representation]]
        # Pallini chiari = singoli seed: mostrano quanto conta l'inizializzazione.
        ax.plot([i] * len(values), values, "o", color=colors[representation], alpha=0.35, markersize=5)
        ax.errorbar(i, statistics.mean(values), yerr=sample_std(values), fmt="D",
                    color=colors[representation], markersize=7, capsize=5)
    ax.set_xticks(range(len(rows)), [representation_label(r["representation"]) for r in rows])
    ax.set_xlabel("input del classificatore")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=150)
    plt.close(fig)


def pick_run(runs: dict, representation: str, seed: int) -> dict:
    """Restituisce il run con il seed richiesto; errore chiaro se quel seed non esiste."""
    for run in runs[representation]:
        if run["summary"]["seed"] == seed:
            return run
    raise ValueError(f"Seed {seed} non trovato per '{representation}'.")


def plot_loss_curves(runs: dict, seed: int, fig_dir: Path) -> None:
    """Un riquadro per input: cross-entropy su training e validation, con la best epoch tratteggiata."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, representation in zip(axes.flat, REPRESENTATIONS):
        run = pick_run(runs, representation, seed)
        h, s = run["history"], run["summary"]
        ax.plot([r["epoch"] for r in h], [r["train_loss"] for r in h], label="training")
        ax.plot([r["epoch"] for r in h], [r["val_loss"] for r in h], label="validation")
        ax.axvline(s["best_epoch"], color="gray", linestyle="--", linewidth=1,
                   label=f"best epoch = {s['best_epoch']}")
        # Scala logaritmica: la loss scende molto nelle prime epoche e poco dopo.
        ax.set_yscale("log")
        ax.set_title(representation_label(representation))
        ax.set_xlabel("epoca (= aggiornamento RProp)")
        ax.legend(fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("loss (cross-entropy)")
    fig.suptitle(f"Curve di loss dei classificatori (seed {seed}, linea tratteggiata = best epoch)")
    fig.tight_layout()
    fig.savefig(fig_dir / "clf_loss_curves.png", dpi=150)
    plt.close(fig)


def plot_val_accuracy(runs: dict, colors: dict, seed: int, fig_dir: Path) -> None:
    """Validation accuracy di tutti gli input nello stesso grafico; pallino = best epoch."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for representation in REPRESENTATIONS:
        run = pick_run(runs, representation, seed)
        # Si parte dall'epoca 1: l'epoca 0 non si può disegnare su un asse logaritmico.
        h, s = run["history"][1:], run["summary"]
        ax.plot([r["epoch"] for r in h], [100 * r["val_accuracy"] for r in h],
                color=colors[representation], label=representation_label(representation))
        ax.plot(s["best_epoch"], 100 * s["best_val_accuracy"], "o", color=colors[representation], markersize=5)
    # Asse x logaritmico: la baseline si ferma dopo decine di epoche, le feature dopo centinaia.
    ax.set_xscale("log")
    # Si mostra solo la parte alta, dove le curve si distinguono.
    ax.set_ylim(80, 97)
    ax.set_xlabel("epoca (= aggiornamento RProp), scala logaritmica")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title(f"Validation accuracy dei classificatori (seed {seed}, pallino = best epoch)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "clf_val_accuracy.png", dpi=150)
    plt.close(fig)


def plot_train_val_test(rows: list, fig_dir: Path) -> None:
    """Accuracy media su training, validation e test: la distanza fra le barre misura l'overfitting."""
    x = list(range(len(rows)))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, key, label in ((-width, "train_accuracy_mean", "training"),
                               (0.0, "val_accuracy_mean", "validation"),
                               (width, "test_accuracy_mean", "test")):
        ax.bar([i + offset for i in x], [100 * r[key] for r in rows], width=width, label=label)
    ax.set_xticks(x, [representation_label(r["representation"]) for r in rows])
    # L'asse non parte da zero: le differenze sono di pochi punti percentuali.
    ax.set_ylim(90, 100)
    ax.set_xlabel("input del classificatore")
    ax.set_ylabel("accuracy media (%), pesi della best epoch")
    ax.set_title("Training, validation e test accuracy (asse da 90%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "clf_train_val_test.png", dpi=150)
    plt.close(fig)


def plot_accuracy_vs_ae_loss(rows: list, colors: dict, fig_dir: Path) -> None:
    """Test accuracy rispetto alla validation MSE dell'encoder: ricostruire meglio non basta."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # La baseline non ha un autoencoder: si disegnano solo i cinque m.
    latent_rows = [r for r in rows if r["representation"] != "raw"]
    ax.plot([r["ae_val_mse"] for r in latent_rows], [100 * r["test_accuracy_mean"] for r in latent_rows],
            color="lightgray", zorder=1)
    for r in latent_rows:
        ax.errorbar(r["ae_val_mse"], 100 * r["test_accuracy_mean"], yerr=100 * r["test_accuracy_std"],
                    fmt="o", color=colors[r["representation"]], capsize=4, markersize=7)
        # L'etichetta di m=25 va a sinistra del punto, altrimenti esce dal grafico.
        offset = (-48, 6) if r["representation"] == "m25" else (8, 6)
        ax.annotate(representation_label(r["representation"]), (r["ae_val_mse"], 100 * r["test_accuracy_mean"]),
                    textcoords="offset points", xytext=offset, fontsize=9)
    # La linea orizzontale è la baseline sui pixel grezzi.
    raw_row = next(r for r in rows if r["representation"] == "raw")
    ax.axhline(100 * raw_row["test_accuracy_mean"], color="gray", linestyle="--", linewidth=1,
               label="baseline sui pixel grezzi")
    # La MSE degli autoencoder copre un ordine di grandezza fra m=125 e m=25.
    ax.set_xscale("log")
    ax.set_xlabel("validation loss dell'autoencoder (MSE, scala logaritmica)")
    ax.set_ylabel("test accuracy media (%)")
    ax.set_title("Qualità della ricostruzione e accuracy del classificatore")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "clf_accuracy_vs_ae_loss.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Legge la campagna da analizzare e il seed da usare per le curve."""
    parser = argparse.ArgumentParser(description="Tabella finale e grafici dei classificatori (giorni 15-16).")
    parser.add_argument("--campaign", default=None,
                        help="run-id in results/campaigns/ (default: i risultati standard in results/).")
    parser.add_argument("--curve-seed", type=int, default=DEFAULT_CURVE_SEED,
                        help="Seed del classificatore mostrato nelle curve per epoca (default: 1).")
    return parser.parse_args()


def main() -> None:
    """Costruisce la tabella finale e tutte le figure dei classificatori."""
    # Avvia il timer del programma: misura tutto, dal lancio alla fine.
    program_start = time.perf_counter()
    args = parse_args()

    # Senza --campaign si leggono i risultati standard; altrimenti quelli di un PC specifico.
    if args.campaign is None:
        input_dir, output_dir = RESULTS_DIR, RESULTS_DIR
    else:
        output_dir = RESULTS_DIR / "campaigns" / args.campaign
        input_dir = output_dir / "results"
        if not input_dir.exists():
            raise FileNotFoundError(f"Campagna non trovata: {input_dir}.")
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Legge tutti i risultati: 30 classificatori e i cinque autoencoder da cui vengono le feature.
    runs = load_classifier_runs(input_dir)
    ae_summaries = load_autoencoder_summaries()
    colors = representation_colors()

    rows = build_table(runs, ae_summaries)
    save_table(rows, output_dir)

    # Le tre metriche riassuntive usano lo stesso tipo di grafico.
    plot_mean_std(runs, rows, colors, "test_accuracy", "test accuracy (%)",
                  "Test accuracy per input (rombo = media ± dev. std., pallini = seed)",
                  "clf_test_accuracy.png", fig_dir, scale=100.0)
    plot_mean_std(runs, rows, colors, "best_epoch", "best epoch",
                  "Epoche alla convergenza dei classificatori (rombo = media ± dev. std.)",
                  "clf_epochs.png", fig_dir)
    plot_mean_std(runs, rows, colors, "training_seconds", "tempo di training (s)",
                  f"Tempo di training ({runs['raw'][0]['summary']['device']})",
                  "clf_time.png", fig_dir)

    plot_loss_curves(runs, args.curve_seed, fig_dir)
    plot_val_accuracy(runs, colors, args.curve_seed, fig_dir)
    plot_train_val_test(rows, fig_dir)
    plot_accuracy_vs_ae_loss(rows, colors, fig_dir)
    print(f"Figure: {fig_dir.relative_to(PROJECT_DIR)}")

    # Tempo effettivo di tutto il programma.
    print(f"\nTempo totale del programma: {format_duration(time.perf_counter() - program_start)}")


if __name__ == "__main__":
    main()
