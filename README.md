# Conditional GAN for Face Generation (CelebA) 🎭🤖

Questo repository contiene lo sviluppo di un modello generativo basato su **Generative Adversarial Networks (GANs) condizionali**, progettato per sintetizzare volti umani realistici. Il progetto è stato addestrato sul dataset **CelebA** e permette di controllare le caratteristiche visive dell'immagine generata tramite specifici attributi (es. *Uomo/Donna, Sorridente/Non Sorridente, Giovane/Anziano*).

## 🎯 Obiettivo del Progetto

L'obiettivo è dimostrare la capacità di controllare l'output di un modello generativo DCGAN-like superando le classiche sfide di addestramento delle GAN (come il *mode collapse* e l'instabilità del gradiente) attraverso architetture avanzate e tecniche di regolarizzazione.

---

## 🧠 Architettura del Modello (`model.py`)

Il sistema è composto da due reti neurali in competizione:

1. **Generatore (DCGAN-like):**
   * Prende in input un vettore di rumore latente ($z \in \mathbb{R}^{128}$) concatenato a un vettore condizionale multi-hot (3 attributi).
   * Utilizza blocchi di `ConvTranspose2d` e `BatchNorm2d` per effettuare un upsampling progressivo da una mappa spaziale 4x4 fino a un'immagine RGB 64x64.
   * Attivazione finale `Tanh` per mappare l'output nel range $[-1, 1]$.

2. **Discriminatore (Projection Discriminator):**
   * A differenza della concatenazione standard, implementa un **Projection Discriminator**. Questa architettura mappa gli attributi condizionali nello stesso spazio delle feature estratte dal backbone convoluzionale, calcolando un prodotto scalare.
   * *Vantaggio:* Valuta in modo molto più stabile e preciso se l'immagine è reale e se rispetta simultaneamente tutte le condizioni imposte, migliorando la qualità visiva per query multi-attributo.

---

## 🛠️ Tecniche di Stabilizzazione (Training)

L'addestramento di una GAN richiede accorgimenti specifici per evitare che il Discriminatore diventi troppo "forte" rispetto al Generatore. Nel file `train.py` sono state implementate le seguenti best practice:

* **TTUR (Two Time-Scale Update Rule):** Learning rate differenziati ($LR_G = 0.0002$, $LR_D = 0.0001$) per garantire che il Generatore abbia il tempo di apprendere senza essere surclassato immediatamente.
* **Instance Noise con Decay:** Aggiunta di rumore gaussiano decrescente agli input del discriminatore nelle prime epoche. Questo "offusca" le differenze tra reale e fake iniziali, stabilizzando i gradienti.
* **Label Smoothing:** I target reali per il discriminatore sono ridotti da $1.0$ a $0.9$, riducendo l'overconfidence e migliorando il flusso dei gradienti verso il Generatore.
* **Bilanciamento Perfetto delle Classi (`dataset.py`):** Dato che il dataset CelebA è fortemente sbilanciato, è stato implementato un `WeightedRandomSampler` che assegna pesi inversamente proporzionali alla frequenza delle 8 combinazioni possibili dei 3 attributi, garantendo che il modello veda uniformemente tutte le tipologie di volti.

---

## 📂 Struttura del Repository

* `model.py` - Definizione delle classi `Generator` e `Discriminator` in PyTorch.
* `dataset.py` - Logica di preprocessing delle immagini, estrazione attributi e dataloader bilanciato.
* `train.py` - Pipeline di addestramento con salvataggio automatico di pesi, grafici di loss e griglie di validazione.
* `inference.py` - Script ottimizzato per caricare i pesi e generare griglie di volti condizionati (es. "Uomo, Anziano, Sorridente").

---

## 🚀 Utilizzo (Inferenza)

Per testare il modello e generare nuove immagini, assicurati di aver scaricato i pesi pre-addestrati e inseriti nella cartella corretta, dopodiché esegui:

```bash
python inference.py
