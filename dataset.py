import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import CelebA
from torchvision.transforms import v2

# ============================================================
# CONFIGURAZIONE DATASET E TRASFORMAZIONI
# ============================================================
IMAGE_SIZE = 64

# Indici degli attributi in CelebA: 20=Male, 31=Smiling, 39=Young
ATTR_INDICES = [20, 31, 39]

# Pipeline di trasformazione per le immagini
transform_pipeline = v2.Compose([
    v2.CenterCrop(140),  # Isola la regione centrale del volto
    v2.Resize(IMAGE_SIZE),
    v2.RandomHorizontalFlip(p=0.5),  # Data augmentation
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),  # Converte in [0, 1]
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scala in [-1, 1] per la Tanh
])

def get_balanced_sample_weights(dataset: CelebA) -> torch.Tensor:
    """
    Calcola i pesi di campionamento per bilanciare perfettamente le 8 classi 
    generate dalle combinazioni dei 3 attributi binari (Male, Smiling, Young).

    Args:
        dataset (CelebA): L'istanza del dataset caricato.

    Returns:
        torch.Tensor: Tensore 1D con il peso associato a ciascun sample.
    """
    # Estrazione degli attributi target scelti
    attrs = dataset.attr[:, ATTR_INDICES].long()

    # Creazione ID unico (0-7) tramite codifica binaria
    # Es: Male (4) + Smiling (2) + Young (1) = Classe 7
    labels = attrs[:, 0] * 4 + attrs[:, 1] * 2 + attrs[:, 2]

    # Conteggio delle occorrenze per ciascuna delle 8 classi
    class_counts = torch.bincount(labels, minlength=8)

    print("Distribuzione delle combinazioni (Male/Smiling/Young):")
    for i, count in enumerate(class_counts):
        # Stampa formattata con la rappresentazione binaria della classe
        print(f"  Classe {i} [{i:03b}]: {count.item()} immagini")

    # Il peso è inversamente proporzionale alla frequenza della classe.
    # Aggiungiamo 1e-6 per evitare divisioni per zero se una classe è vuota.
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    class_weights[class_counts == 0] = 0.0

    # Mappiamo il peso della singola classe su ogni specifica immagine del dataset
    sample_weights = class_weights[labels]

    return sample_weights

def get_dataloader(root_path: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    """
    Inizializza il dataset CelebA e restituisce un DataLoader bilanciato 
    tramite WeightedRandomSampler per mitigare lo sbilanciamento degli attributi.

    Args:
        root_path (str): Percorso principale dove è salvato il dataset CelebA.
        batch_size (int): Dimensione del batch.
        num_workers (int): Numero di subprocessi per il caricamento.

    Returns:
        DataLoader: PyTorch DataLoader pronto per l'addestramento.
    """
    # Lambda per restituire solo i tensori degli attributi di interesse
    target_transform = lambda t: t[ATTR_INDICES]

    # Inizializzazione Dataset
    dataset = CelebA(
        root=root_path,
        split='all',
        transform=transform_pipeline,
        target_transform=target_transform,
        download=False
    )

    # Configurazione del campionamento bilanciato
    weights = get_balanced_sample_weights(dataset)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    # Inizializzazione DataLoader
    # NB: Lo shuffle è False perché la selezione randomica è delegata al sampler.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True
    )

    return loader