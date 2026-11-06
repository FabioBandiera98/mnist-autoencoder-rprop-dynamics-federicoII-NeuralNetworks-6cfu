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
- seed 42 come riferimento; i seed 1 e 2 servono a misurare la variabilità dell'autoencoder (`python autoencoder.py --all-seeds`)
- la MSE sul test viene calcolata una sola volta, alla fine, sui pesi della best epoch

### 3. Classificatore

Si allenano i classificatori sulle rappresentazioni dell'autoencoder con seed 42, così il confronto fra dimensioni latenti non confonde l'effetto di `m` con un encoder diverso. Le altre inizializzazioni degli autoencoder (seed 1 e 2) sono un'analisi di robustezza separata.

- cinque input: feature con `m` in {25, 50, 75, 100, 125}, più baseline sui 784 pixel grezzi
- un hidden layer con 64 neuroni e attivazione sigmoidale, uguale per tutti gli input
- 10 logits in uscita; `CrossEntropyLoss` applica internamente softmax e log-loss, quindi non si aggiunge una softmax al modello
- le feature latenti sono già in [0,1]: non si applica uno scaler
- RProp batch: un gradiente sull'intero training set per ogni epoca
- 5 seed del classificatore: 1, 2, 3, 4, 5 per ciascun input
- a ogni epoca si registrano cross-entropy e accuracy su train e validation
- checkpoint scelto con la validation loss migliore; l'accuracy è registrata come metrica, non usata per lo stopping
- convergenza pratica: la validation loss deve migliorare almeno dello 0,1% rispetto al minimo precedente; patience 100 e massimo 10.000 epoche come limite di sicurezza
- il test è valutato una sola volta, sui pesi del best checkpoint; si riportano loss, accuracy, epoca migliore e tempo

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

Per la valutazione robusta suggerita, allenare ogni dimensione con tutti e tre i seed 42, 1 e 2:

```bash
python autoencoder.py --all-seeds
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

In quel caso i file prendono il suffisso `_s1` (`mnist_features_m25_s1.npz`, `feature_summary_s1.json`). Per i classificatori del confronto principale usare le feature del seed 42.

Per provare rete, cross-entropy e aggiornamento RProp su pochi campioni, senza avviare un esperimento completo:

```bash
python classifier.py --smoke-test
```

Per allenare una configurazione, per esempio il classificatore su `m=25` con seed 1:

```bash
python classifier.py --representation m25 --seed 1
```

La rappresentazione `raw` allena la baseline sui pixel MNIST originali. Per l'esperimento completo (baseline e cinque dimensioni latenti, cinque seed ciascuna):

```bash
python classifier.py --all
```

Il classificatore salva un checkpoint, un CSV per epoca e un riepilogo JSON per ogni coppia input/seed. Il CSV comprende anche l'epoca 0 (prestazioni iniziali); il riepilogo indica chiaramente `converged` e `stop_reason`. Le esecuzioni multi-seed generano inoltre `results/classifier_comparison.csv` con media e deviazione standard di accuracy/loss sul test e best epoch.

Per automatizzare i giorni 11-14, eseguire il runner della campagna invece del comando manuale `classifier.py --all`. Prima devono essere conclusi tutti e cinque gli autoencoder seed 42. Se ne manca uno, completarli con `python autoencoder.py --all --seed 42`. Il runner verifica checkpoint e protocollo, rigenera su CPU le feature obsolete e controlla forme, label e impronte degli input; se manca un prerequisito si ferma senza avviare classificatori. Se il preflight passa, lancia 30 configurazioni in sequenza (sei rappresentazioni per cinque seed classificatore):

```bash
python run_classifier_campaign.py --run-id Fabio-3050 --dry-run
python run_classifier_campaign.py --run-id Fabio-3050
```

Sul secondo computer usare un `run-id` diverso, ad esempio `Yegor-3080`. Il default usa CUDA se disponibile; su CPU il runner si ferma prima dei training, a meno di passare esplicitamente `--allow-cpu`. Ogni PC salva checkpoint, CSV e riepiloghi sotto `results/campaigns/<run-id>/`. Se la campagna viene interrotta, rilanciare lo stesso comando e `run-id`: i run già completi vengono saltati. Se cambiano protocollo, feature o checkpoint, il runner rifiuta il resume per non mescolare esperimenti diversi.

Per confrontare i risultati dopo aver portato la cartella campagna di Yegor `results/campaigns/Yegor-3080/` in questo workspace, eseguire:

```bash
python compare_classifier_campaigns.py Fabio-3050 Yegor-3080
```

Il confronto salva `results/campaigns/combined_comparison.csv`, con una riga per macchina e rappresentazione, media e deviazione standard dei cinque seed, epoche migliori e convergenza. Controlla anche se entrambi i PC hanno usato lo stesso split e le stesse feature. Se le impronte differiscono, i risultati restano separati e non devono essere aggregati fra computer.

Il comparatore riconosce i 30 risultati già completati di Fabio nei percorsi standard `results/classifier_*` e `checkpoints/classifier_*`; non occorre copiarli o riaddestrarli per inserirli in una cartella campagna.

Per confrontare davvero gli hardware a parità di input, condividere fra i PC gli stessi cinque checkpoint degli autoencoder seed 42 (`checkpoints/autoencoder_m*_best.pt`) prima di lanciare la campagna. Le feature sono rigenerate su CPU a partire da quei checkpoint. I checkpoint e i risultati non sono esclusi da `.gitignore`; `data/` resta locale e viene rigenerata.

Negli altri script i dati si caricano già come tensori PyTorch:

```python
from data import load_splits, get_device
s = load_splits(device=get_device())
X_train, y_train = s["train"]
```

## Nota metodologica

La validazione non viene usata per aggiornare i pesi, ma solo per valutare la generalizzazione e decidere quando fermare il training. Il test set viene usato soltanto per la valutazione finale, dopo aver scelto tutte le configurazioni del modello.

## Registro delle modifiche

- 2026-11-05, protocollo adottato: autoencoder con seed 42, 1 e 2; classificatore con 64 neuroni nascosti, seed 1-5, baseline raw e arresto via validation loss con soglia relativa 0,1%, patience 100 e massimo 10.000 epoche. I file in `checkpoints/` e `results/` sono tracciabili da Git; `data/` resta esclusa e va rigenerata localmente. La campagna registra ambiente e impronte dei dati per confrontare due PC.
