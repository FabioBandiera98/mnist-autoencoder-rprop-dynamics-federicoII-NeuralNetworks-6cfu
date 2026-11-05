# MNIST Autoencoder + RProp Dynamics

Progetto di Neural Networks per la classificazione delle cifre MNIST tramite autoencoder e rete di classificazione con aggiornamento RProp.

## Obiettivo

L'esperimento studia la capacità di rappresentazioni latenti generate da autoencoder con diverse dimensioni nascoste. In particolare:

- si usa l'immagine grezza MNIST come input
- si costruiscono 5 autoencoder con hidden layer di dimensioni 25, 50, 75, 100, 125
- si usa la rappresentazione codificata come input per un classificatore con un solo hidden layer
- si adotta RProp in batch come regola di aggiornamento dei pesi
- si usa validation per stabilire quando fermare il training e test per la valutazione finale

## Dataset e split

Il progetto usa PyTorch come framework principale.

Suddivisione finale del dataset:

- Training: 20.000 immagini
- Validation: 5.000 immagini
- Test: 5.000 immagini
- Totale: 30.000 immagini

Tutti gli insiemi sono disgiunti e ottenuti con una split stratificata per preservare la distribuzione delle 10 classi:

- training e validation sono estratti a caso dal training set ufficiale MNIST (60.000 immagini);
- il test è estratto dal test set ufficiale MNIST (10.000 immagini), quindi è disgiunto per costruzione e scritto da persone diverse;
- lo split è riproducibile (seed 42) e gli indici delle immagini nei set ufficiali sono salvati insieme ai dati.

## Protocollo sperimentale

### 1. Preparazione dati

- caricamento di MNIST tramite torchvision
- normalizzazione dei pixel in [0, 1]
- flatten da 28x28 a 784 features per immagine
- split stratificata train/validation/test
- controllo non-overlap e distribuzione delle label

### 2. Autoencoder

Per ogni dimensione latente h in {25, 50, 75, 100, 125}:

- encoder: 784 -> h
- decoder: h -> 784
- attivazione sigmoidale nell'encoder e nel decoder; l'uscita resta in [0, 1]
- funzione di perdita: Mean Squared Error (MSE) sui pixel normalizzati
- training solo sul training set
- RProp batch: un gradiente calcolato sull'intero training set per epoca
- monitoraggio della reconstruction MSE su training e validation a ogni epoca
- early stopping basato sulla validation MSE: un'epoca conta come miglioramento solo se abbassa la miglior validation MSE di almeno lo 0,1% (soglia relativa); ci si ferma dopo 100 epoche consecutive senza miglioramento
- massimo 10000 epoche, solo come limite di sicurezza; si conservano i pesi dell'epoca con validation MSE migliore (best epoch = epoche alla convergenza)
- RProp (`torch.optim.Rprop`, variante iRprop-): Δ0 = 0.01, η- = 0.5, η+ = 1.2, Δmin = 1e-6, Δmax = 50
- seed 42 per l'inizializzazione dei pesi (modificabile con `--seed`)
- la MSE sul test viene calcolata una sola volta, alla fine, sui pesi della best epoch

### 3. Classificatore

Per ogni rappresentazione codificata:

- input: rappresentazione ottenuta dall'encoder
- hidden layer singolo
- funzione di attivazione sigmoidale
- output: 10 classi (0..9)
- RProp in batch
- training loss e validation loss monitorati ad ogni epoca
- checkpoint migliore selezionato tramite validation accuracy

### 4. Valutazione

- test set usato solo per la valutazione finale
- metriche principali:
  - accuracy test
  - epochs di convergenza
  - training/validation loss curves
  - tempo di training

## Requisiti

- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- NumPy

## Struttura prevista del progetto

- data.py: caricamento dei dati e split train/validation/test
- autoencoder.py: architettura e addestramento autoencoder
- classifier.py: classificatore con RProp
- experiments.py: esperimenti per i 5 valori di hidden size
- evaluate.py: metriche finali
- plot_results.py: grafici e confronto
- checkpoints/: pesi migliori dei modelli
- results/: log, metriche e grafici
- report/: relazione finale

## Giorno 1: protocollo sperimentale

Nel primo giorno si definisce il metodo di lavoro:

- fixare dataset, split, metriche e criteri di stopping
- definire architetture autoencoder e classificatore
- fissare una seme riproducibile
- preparare il README e la documentazione dell'esperimento

## Giorno 2: caricamento e preparazione di MNIST

Nel secondo giorno si:

- carica MNIST con PyTorch
- normalizza i pixel in [0, 1]
- flattens le immagini 28x28 in vettori 784-dimensionali
- crea i tre insiemi train/validation/test con split stratificata
- salva i dati preprocessati in una cartella dedicata
- verifica forma dei tensori, distribuzione delle classi e assenza di overlap (`check_data.py`)

## Esecuzione

Per preparare il dataset:

```bash
python data.py
```

Il comando scarica MNIST (solo la prima volta), genera lo split e salva `data/processed/mnist_split.npz` e `data/processed/dataset_summary.json`. La cartella `data/` non è versionata: ognuno la rigenera in locale con lo stesso seed e ottiene lo stesso split.

Per verificare lo split (dimensioni, disgiunzione, distribuzione delle classi, riproducibilità) e vedere alcune cifre:

```bash
python check_data.py
```

Per verificare le cinque architetture senza avviare il training:

```bash
python autoencoder.py --check-model
```

Per allenare un singolo modello, per esempio quello con 50 neuroni latenti:

```bash
python autoencoder.py --hidden-size 50
```

Per allenare in sequenza tutti e cinque i modelli con lo stesso protocollo:

```bash
python autoencoder.py --all
```

Il trainer salva un checkpoint, un CSV con le metriche per epoca e un riepilogo JSON separati per ciascuna dimensione latente. Ad esempio, per m=25 usa `checkpoints/autoencoder_m25_best.pt`, `results/autoencoder_m25_history.csv` e `results/autoencoder_m25_summary.json`. Ogni riepilogo specifica se l'addestramento è terminato per patience o per raggiungimento del massimo di epoche.

Ogni log per epoca contiene anche l'epoca 0 (pesi iniziali) e il passo medio di RProp (`mean_step`).

Per confrontare i cinque autoencoder (giorno 6) dopo `python autoencoder.py --all`:

```bash
python compare_autoencoders.py
```

Lo script non addestra niente: produce `results/autoencoder_comparison.csv` (best epoch, convergenza, MSE su train/val/test, statistiche dell'encoder) e le figure in `results/figures/` (curve di loss, MSE ed epoche al variare di m, ricostruzioni di 10 cifre di test).

Per estrarre le rappresentazioni con i checkpoint selezionati:

```bash
python extract_features.py
```

Lo script genera 15 matrici (train, validation e test per ciascun valore di m) nella cartella `data/features/`. Ogni archivio `mnist_features_m{m}.npz` contiene `X_train`, `X_val`, `X_test` e le rispettive label `y_train`, `y_val`, `y_test`. Il file `feature_summary.json` registra forme, checkpoint e relativa epoca e validation MSE.

Il manifest registra anche se ogni autoencoder era converso (`autoencoder_converged`) e, in `all_autoencoders_converged`, se lo erano tutti. Lo script stampa un avviso per ogni autoencoder fermato al limite di epoche. Le feature non sono versionate (`data/` è escluso da git): dopo ogni nuovo training degli autoencoder vanno rigenerate. Per gli autoencoder addestrati con un altro seed:

```bash
python extract_features.py --seed 1
```

In quel caso i file prendono il suffisso `_s1` (`mnist_features_m25_s1.npz`, `feature_summary_s1.json`).

Negli altri script i dati si caricano già come tensori PyTorch:

```python
from data import load_splits, get_device
s = load_splits(device=get_device())
X_train, y_train = s["train"]
```

## Nota metodologica

La validazione non viene usata per aggiornare i pesi, ma solo per valutare la generalizzazione e decidere quando fermare il training. Il test set viene usato soltanto per la valutazione finale, dopo aver scelto tutte le configurazioni del modello.

## Registro delle modifiche

- 2026-11-05, autoencoder: il criterio iniziale (massimo 300 epoche, patience 10, miglioramento minimo assoluto 1e-6) faceva fermare tutti e cinque i modelli al limite di 300 epoche, con la validation MSE ancora in discesa. Una prova con 2000 epoche e soglia dello 0,01% non si fermava comunque, perché la curva entra in una coda molto lenta. Proposta la "convergenza pratica": miglioramento relativo minimo 0,1%, patience 100, limite 10000 epoche. Se confermata, vanno rifatti i cinque training (`python autoencoder.py --all`) e l'estrazione delle feature (`python extract_features.py`).
