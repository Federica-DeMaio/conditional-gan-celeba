import torch
import torch.nn as nn

# ============================================================
# PARAMETRI CONDIVISI
# ============================================================

LATENT_SIZE = 128
# Dimensione del vettore latente z.
# Valore standard nelle GAN scelto sperimentalmente in base a trade_off miglioramenti-complessità.

N_ATTR = 3
# Numero di attributi condizionali (Male, Smiling, Young).
# Ogni immagine è condizionata da un vettore binario di lunghezza 3.

# GENERATOR
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Numero di canali iniziali dopo la proiezione fully-connected.
        # Valore elevato (512) per fornire alta capacità espressiva
        # nella fase iniziale della generazione.
        self.init_channels = 512
        # 1) PROIEZIONE DEL LATENT VECTOR + CONDIZIONE
        # Input: concatenazione [z, c] di dimensione:
        # (LATENT_SIZE + N_ATTR) = 128 + 3 =
        # Output: tensore lineare di dimensione:
        # 512 * 4 * 4 = 8192
        # Questo rappresenta una "immagine latente" 4x4 con 512 canali.
        self.fc = nn.Sequential(
            # Operazione del layer lineare:
            # h = W [z, c] + b
            # con:
            #   [z, c] ∈ R^(B × (LATENT_SIZE + N_ATTR))
            #   W ∈ R^(D × (LATENT_SIZE + N_ATTR))
            #   h ∈ R^(B × D)
            nn.Linear(LATENT_SIZE + N_ATTR, self.init_channels * 4 * 4),
            # Batch Normalization 1D (applicata feature-wise):
            # Per ogni feature j = 1,...,D:
            # μ_j = (1 / B) * Σ_i h_{i,j}
            # σ_j² = (1 / B) * Σ_i (h_{i,j} − μ_j)²
            # ĥ_{i,j} = (h_{i,j} − μ_j) / sqrt(σ_j² + ε)
            # y_{i,j} = γ_j * ĥ_{i,j} + β_j
            # dove γ_j e β_j sono parametri apprendibili.
            # Effetto: normalizza la distribuzione delle attivazioni
            # per migliorare stabilità e velocità di convergenza.
            #Gamma e Beta diversi per ogni canale
            nn.BatchNorm1d(self.init_channels * 4 * 4),
            # ReLU applicata element wise
            # ReLU(y_{i,j}) = max(0, y_{i,j})
            # Introduce non linearità, elimina valori negativi
            # e favorisce gradienti più stabili nel generatore.
            nn.ReLU()
        )
        # 2) BLOCCO CONVOLUZIONALE DI UPSAMPLING
        # Ogni ConvTranspose2d:
        # - raddoppia la risoluzione spaziale
        # - riduce progressivamente il numero di canali
        self.conv_blocks = nn.Sequential(
            # Input: (B, 512, 4, 4)
            nn.ConvTranspose2d(
                self.init_channels, 256,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            # Output size formula ConvTranspose2d:
            # H_out = (H_in - 1) * stride - 2 * padding + kernel_size
            # (4 - 1) * 2 - 2 * 1 + 4 = 8
            # -> (B, 256, 8, 8)
            nn.BatchNorm2d(256),
            # Batch Normalization 2D:
            # Il tensore in input ha forma:
            #   x ∈ R^(B × C × H × W), con C = 256
            # La normalizzazione avviene PER CANALE,
            # aggregando media e varianza su batch e spazio.
            # Per ogni canale c = 1,...,C:
            # μ_c = (1 / (B · H · W)) · Σ_b Σ_h Σ_w x_{b,c,h,w}
            # σ_c² = (1 / (B · H · W)) · Σ_b Σ_h Σ_w (x_{b,c,h,w} − μ_c)²
            # Normalizzazione:
            # x̂_{b,c,h,w} = (x_{b,c,h,w} − μ_c) / sqrt(σ_c² + ε)
            # Scala e shift apprendibili:
            # y_{b,c,h,w} = γ_c · x̂_{b,c,h,w} + β_c
            # con γ_c, β_c ∈ R parametri allenabili.
            # Effetto:
            # - stabilizza le attivazioni convoluzionali
            nn.ReLU(),

            # Input: (B, 256, 8, 8)
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            # Output: (B, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Input: (B, 128, 16, 16)
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            # Output: (B, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Input: (B, 64, 32, 32)
            nn.ConvTranspose2d(
                64, 3,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            # Output: (B, 3, 64, 64)
            # Tanh produce output in [-1, 1], coerente con
            # la normalizzazione delle immagini reali
            #    In architetture DCGAN-like, l’accoppiata:
            #    immagini normalizzate in [-1, 1] + Tanh finale
            #    è una pratica consolidata e empiricamente stabile.
            #    - immagini RGB
            #    - maggiore complessità visiva
            #    - uso estensivo di BatchNorm e convoluzioni profonde
            #    rende preferibile una normalizzazione simmetrica.

            # tanh=(e^x-e^-x)/(e^x+e^-x)
            nn.Tanh()
        )

    def forward(self, z, c):
        """
        z: vettore latente di forma (B, LATENT_SIZE)
        c: vettore di attributi di forma (B, N_ATTR)
        """
        # Conversione esplicita a float per sicurezza numerica
        c = c.float()
        # Concatenazione lungo la dimensione delle feature
        # Risultato: (B, 128 + 3)
        zc = torch.cat((z, c), dim=1)
        # Proiezione fully-connected
        out = self.fc(zc)
        # Reshape in feature map 4x4
        out = out.view(-1, self.init_channels, 4, 4)
        # Upsampling progressivo fino a 64x64
        return self.conv_blocks(out)

# DISCRIMINATOR (Projection Discriminator)
"""
Motivazione dell'approccio Projection Discriminator:

Nei GAN condizionati multi-attributo, 
il discriminatore deve valutare se un'immagine è reale e se soddisfa tutte le condizioni specificate.

Un approccio semplice sarebbe concatenare l'embedding della condizione alle feature estratte 
prima del fully-connected finale. Tuttavia, questo approccio:

- aumenta significativamente la dimensione dell'input al layer finale se ci sono molti attributi,
- può rendere più difficile per il discriminatore distinguere combinazioni di attributi,
- introduce un forte legame tra feature e condizione che può destabilizzare il training.

La Projection Discriminator risolve questi problemi:

- Mappa ogni attributo in un embedding nello stesso spazio delle feature convoluzionali.
- Calcola un prodotto scalare tra feature e embedding.
- L'output finale combina score non condizionato + score condizionale (projection term).
- Vantaggi:
    - Considera tutti gli attributi attivi contemporaneamente (multi-hot).
    - Mantiene dimensione compatta e training stabile.
    - Permette di apprendere combinazioni di attributi in modo naturale.
"""

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # BACKBONE CONVOLUZIONALE
        # Riduce progressivamente la risoluzione spaziale
        # e aumenta il numero di canali (feature extraction).
        self.features = nn.Sequential(
            # Input: (B, 3, 64, 64)
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            # Output: (B, 64, 32, 32)
            nn.LeakyReLU(0.2),
            # Leaky ReLU applicata elemento per elemento:
            # f(x) = x           se x > 0
            # f(x) = alpha * x   se x <= 0, con alpha = 0.2
            # Differenze rispetto a ReLU standard:
            # - ReLU(x) = max(0, x)
            #   -> tutti i valori negativi vengono azzerati
            #   -> rischio di "dead neurons" se molti x < 0
            # - LeakyReLU mantiene una piccola pendenza per x < 0
            #   -> evita che le unità si spengano permanentemente
            #   -> favorisce flusso di gradienti anche per attivazioni negative
            # - Stabilizza il discriminatore, che riceve input sia reali che generati
            # - Empiricamente favorisce convergenza più robusta rispetto alla ReLU pura
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            # Output: (B, 128, 16, 16)
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            # Output: (B, 256, 8, 8)
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            # Output: (B, 512, 4, 4)
            nn.LeakyReLU(0.2),
        )
        # PARTE NON CONDIZIONATA
        # - L’output del backbone convoluzionale è un tensore 4D:
        #   (B, C=512, H=4, W=4)
        # - Per passare a un layer fully-connected (Linear),
        #   dobbiamo avere un tensore 2D di forma (B, D), dove
        #   D = C * H * W = 512 * 4 * 4 = 8192
        # - Quindi Flatten prende ogni feature map e la “appiattisce”
        #   preservando la dimensione batch B.
        self.flatten = nn.Flatten()
        # Fully-connected che produce lo score real/fake base
        # - Riceve in input (B, 8192)
        # - Restituisce un valore scalare per ciascun esempio (B,)
        #   che rappresenta lo score di real/fake non condizionato.
        #   Linear opera solo su
        #   tensori 2D (batch × features).
        self.fc = nn.Linear(512 * 4 * 4, 1)
        # PROJECTION DISCRIMINATOR
        # Embedding che mappa ogni classe condizionale
        # in un vettore dello stesso spazio delle feature.
        #     emb[i] = look-up table[c[i]]
        # - Motivazione:
        #     permette al discriminatore di usare informazioni condizionali
        #     nella stessa "spazio feature" del backbone convoluzionale
        #     tramite prodotto scalare (projection term).
        self.embed = nn.Embedding(N_ATTR, 512 * 4 * 4)
        # Sigmoid per output probabilistico
        # Funzione di attivazione finale che trasforma lo score reale/fake
        # in probabilità nell’intervallo [0,1].
        # Operazione element-wise:
        #   p_i = σ(x_i) = 1 / (1 + exp(-x_i))
        # Input:
        #   - x_i = score finale (somma di score base + projection term)
        #   - shape: (B,)
        # Output:
        #   - p_i ∈ [0,1], interpretabile come probabilità di real/fake
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, c):
        """
        Forward del discriminatore con Projection Discriminator.

        Args:
            x: immagine in input (B, 3, 64, 64)
            c: vettore condizionale (B,) se singolo attributo o (B, N_ATTR) multi-hot
        Returns:
            out: probabilità che l'immagine sia reale e rispetti le condizioni, (B,)
        """
        # 1) Estrazione delle feature convoluzionali
        # x: (B, 3, 64, 64)
        # Dopo backbone conv: h: (B, 512, 4, 4)
        h = self.features(x)
        # 2) Flatten: da 4D (B, C, H, W) a 2D (B, D)
        # D = 512 * 4 * 4 = 8192
        # Serve per passare al fully-connected layer
        h = self.flatten(h)  # (B, 8192)
        # Score base (non condizionato)
        out = self.fc(h).squeeze(1)
        # TERMINE DI PROIEZIONE
        # emb_all: matrice embedding di tutti gli attributi
        # shape: (N_ATTR, 512*4*4)
        # Ogni riga è un embedding appreso per un attributo
        emb_all = self.embed.weight  # (N_ATTR, 512*4*4)
        # Moltiplicazione batch-wise e somma sugli attributi attivi
        #c.float() è il vettore multi-hot degli attributi attivi, quindi 1 per gli attributi presenti, 0 per quelli assenti.
        #Moltiplicando per emb_all e sommando, ottieni un embedding complessivo che rappresenta tutti gli attributi attivi.
        # Prodotto scalare tra feature e embedding:
        # introduce dipendenza esplicita tra immagine e condizione
        emb = c.float() @ emb_all  # (B, 512*4*4)
        # proj: (B,), ogni elemento i = Σ_j h_{i,j} * emb_{i,j}
        # Introduce dipendenza tra immagine e condizione senza concatenazione
        proj = torch.sum(h * emb, dim=1)
        # Score finale è calcolato in due parti in modo tale che il discriminatore possa capire se real o fake e se rispetta le condizioni
        out = out + proj
        return self.sigmoid(out)


