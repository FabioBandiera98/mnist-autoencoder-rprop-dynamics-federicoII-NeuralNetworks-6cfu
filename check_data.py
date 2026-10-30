"""
check_data.py — Verifica che lo split creato da data.py sia corretto.

COME SI USA
-----------
    python check_data.py

Se lo split non esiste ancora, viene creato automaticamente.

COSA CONTROLLA
--------------
1) Dimensioni degli insiemi e pixel normalizzati in [0, 1]
2) Che i dati salvati coincidano davvero con le immagini MNIST originali
3) Che training e validation non abbiano immagini in comune
4) Che le 10 cifre abbiano proporzioni simili nei tre insiemi (stratificazione)
5) Che rigenerando lo split con lo stesso seed si ottengano le stesse immagini
6) Infine salva in samples.png una griglia con un esempio per cifra e per insieme

Ogni controllo stampa [OK] oppure [ERRORE]; al primo errore lo script si ferma.
"""

import numpy as np
import torch

from data import (N_TEST, N_TRAIN, N_VAL, SEED, SPLIT_FILE, build_split,
                  get_device, load_mnist_dataset, load_splits)


def check(cond: bool, msg: str) -> None:
    """Stampa [OK] o [ERRORE] con il messaggio; se la condizione è falsa termina."""
    print(f"  [{'OK' if cond else 'ERRORE'}] {msg}")
    if not cond:
        raise SystemExit(1)


def main() -> None:
    # --------------------------------------------------------------------- #
    # Ambiente: versione di PyTorch e dispositivo usato (GPU o CPU)
    # --------------------------------------------------------------------- #
    device = get_device()
    print(f"PyTorch {torch.__version__} — device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"File dello split: {SPLIT_FILE}\n")

    # Carica lo split (lo crea se manca) con anche gli indici nei set ufficiali
    splits = load_splits(device=device, return_indices=True)
    idx = splits["indices"]

    # --------------------------------------------------------------------- #
    # 1) Dimensioni e range dei pixel
    # --------------------------------------------------------------------- #
    print("== Split ==")
    for name, n in {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}.items():
        X, y = splits[name]
        print(f"  {name:5s}: X {tuple(X.shape)} {X.dtype}, y {tuple(y.shape)} {y.dtype}, "
              f"pixel in [{X.min():.2f}, {X.max():.2f}], device {X.device}")
        check(tuple(X.shape) == (n, 784) and len(y) == n, f"{name}: {n} campioni da 784 feature")
        check(X.min() >= 0.0 and X.max() <= 1.0, f"{name}: pixel normalizzati in [0,1]")

    # --------------------------------------------------------------------- #
    # 2) Corrispondenza con MNIST ufficiale
    #    Per ogni insieme si riprendono le immagini originali tramite gli indici
    #    salvati e si confrontano con quelle nel file: devono essere identiche.
    #    Così ci si accorge se il file è corrotto o generato con un'altra versione.
    # --------------------------------------------------------------------- #
    print("\n== Corrispondenza con MNIST ufficiale ==")
    x_tr, y_tr, x_te, y_te = load_mnist_dataset()
    originals = {"train": (x_tr, y_tr), "val": (x_tr, y_tr), "test": (x_te, y_te)}
    for name, (Xo, yo) in originals.items():
        X, y = splits[name]
        same = (np.array_equal(X.cpu().numpy(), Xo[idx[name]])
                and np.array_equal(y.cpu().numpy(), yo[idx[name]]))
        check(same, f"{name}: immagini e label coincidono con gli indici salvati")

    # --------------------------------------------------------------------- #
    # 3) Disgiunzione
    #    Train e val vengono dallo stesso set ufficiale: non devono condividere
    #    indici. Il test viene da un altro set, quindi è disgiunto per costruzione.
    # --------------------------------------------------------------------- #
    print("\n== Disgiunzione ==")
    check(len(np.intersect1d(idx["train"], idx["val"])) == 0, "train ∩ val = ∅")
    for k in ("train", "val", "test"):
        check(len(np.unique(idx[k])) == len(idx[k]), f"nessun duplicato in {k}")
    print("  (il test viene dal test set ufficiale: disgiunto per costruzione)")

    # --------------------------------------------------------------------- #
    # 4) Stratificazione
    #    Percentuale di ogni cifra nei tre insiemi: deve essere quasi uguale.
    # --------------------------------------------------------------------- #
    print("\n== Distribuzione delle classi (%) ==")
    print("  cifra " + " ".join(f"{c:>6d}" for c in range(10)))
    perc = {}
    for name in ("train", "val", "test"):
        y = splits[name][1].cpu()
        perc[name] = torch.bincount(y, minlength=10).double() / len(y) * 100
        print(f"  {name:5s} " + " ".join(f"{p:6.2f}" for p in perc[name].tolist()))
    # Tolleranza di mezzo punto percentuale tra train e val
    check((perc["train"] - perc["val"]).abs().max() < 0.5,
          "train e val hanno proporzioni quasi identiche")

    # --------------------------------------------------------------------- #
    # 5) Riproducibilità
    #    Si ricostruisce lo split da zero con lo stesso seed e si confrontano
    #    gli indici con quelli salvati: devono coincidere esattamente.
    # --------------------------------------------------------------------- #
    print("\n== Riproducibilità ==")
    _, again = build_split(SEED)
    check(all(np.array_equal(idx[k], again[k]) for k in idx),
          f"stesso seed ({SEED}) -> stessi split del file salvato")

    print("\nTutto a posto ✔")

    # --------------------------------------------------------------------- #
    # 6) Visualizzazione
    #    Griglia 3 x 10: una riga per insieme, una colonna per cifra.
    # --------------------------------------------------------------------- #
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 10, figsize=(12, 4))
    for row, name in enumerate(("train", "val", "test")):
        X, y = splits[name][0].cpu(), splits[name][1].cpu()
        for c in range(10):
            i = int(torch.nonzero(y == c)[0])            # primo esempio della cifra c
            ax = axes[row, c]
            ax.imshow(X[i].reshape(28, 28), cmap="gray")  # vettore 784 -> immagine 28x28
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, fontsize=11)          # nome dell'insieme a sinistra
            if row == 0:
                ax.set_title(str(c))                      # cifra in alto
    fig.suptitle("MNIST — un esempio per cifra in ogni split")
    fig.tight_layout()
    fig.savefig("samples.png", dpi=120)
    print("Griglia salvata in samples.png")
    plt.show()


if __name__ == "__main__":
    main()
