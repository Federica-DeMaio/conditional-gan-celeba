import os
import torch
import torchvision.utils as vutils

# --- IMPORT DAI NOSTRI FILE ---
# Assicurati che l'import corrisponda al nome effettivo del tuo file (es. model.py o model_finale.py)
from model import Generator

# ============================================================
# CONFIGURAZIONE GENERALE
# ============================================================
CHECKPOINT_PATH = "./outputs/checkpoints/gen_epoch_200.pth"
OUTPUT_FOLDER = "./generated_images"
LATENT_SIZE = 128

def load_pretrained_generator(checkpoint_path: str, device: torch.device) -> Generator:
    """
    Inizializza il Generatore e carica i pesi pre-addestrati.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"⚠️ File dei pesi non trovato in: {checkpoint_path}")

    netG = Generator(latent_size=LATENT_SIZE).to(device)
    
    # map_location gestisce automaticamente il fallback su CPU se CUDA non è disponibile
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    netG.load_state_dict(state_dict)
    
    # CRUCIALE: Modalità eval per bloccare BatchNorm e Dropout durante l'inferenza
    netG.eval()
    print(f"✅ Modello caricato con successo da: {checkpoint_path}")
    
    return netG

def generate_conditional_faces(
    generator: Generator, 
    device: torch.device,
    is_male: bool, 
    is_smiling: bool, 
    is_young: bool, 
    num_samples: int = 16, 
    filename: str = "faces.png"
):
    """
    Genera una griglia di volti basata sulle condizioni specificate.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    with torch.no_grad(): # Nessun calcolo dei gradienti per risparmiare memoria
        # 1. Campionamento del Rumore Latente (z)
        z = torch.randn(num_samples, LATENT_SIZE, device=device)
        
        # 2. Vettore degli attributi (c) -> Ordine: [Male, Smiling, Young]
        attr_vector = torch.tensor([float(is_male), float(is_smiling), float(is_young)], device=device)
        
        # Ripetiamo il vettore per ogni sample nel batch: Shape (num_samples, 3)
        c = attr_vector.repeat(num_samples, 1)

        # 3. Generazione
        fake_images = generator(z, c)

        # 4. Denormalizzazione (da [-1, 1] della Tanh a [0, 1]) e Salvataggio
        fake_images = fake_images * 0.5 + 0.5
        
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        
        # Salviamo come griglia (es. 16 immagini -> nrow=4 crea un 4x4)
        vutils.save_image(fake_images, save_path, nrow=int(num_samples**0.5), padding=2, normalize=False)
        print(f"🖼️  Immagine salvata in: {save_path}")


if __name__ == "__main__":
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Avvio inferenza su: {device}")

    # Caricamento del modello (una sola volta!)
    try:
        generator = load_pretrained_generator(CHECKPOINT_PATH, device)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    print("\nGenerazione delle combinazioni in corso...")

    # Mappatura delle 8 combinazioni possibili per automatizzare la generazione
    # Formato: "Nome_File": (is_male, is_smiling, is_young)
    combinations = {
        "Man_Smiling_Young.png":       (True,  True,  True),
        "Man_Smiling_Old.png":         (True,  True,  False),
        "Man_NotSmiling_Young.png":    (True,  False, True),
        "Man_NotSmiling_Old.png":      (True,  False, False),
        
        "Woman_Smiling_Young.png":     (False, True,  True),
        "Woman_Smiling_Old.png":       (False, True,  False),
        "Woman_NotSmiling_Young.png":  (False, False, True),
        "Woman_NotSmiling_Old.png":    (False, False, False),
    }

    # Esegue la generazione per ogni combinazione
    for filename, (male, smiling, young) in combinations.items():
        generate_conditional_faces(
            generator=generator,
            device=device,
            is_male=male,
            is_smiling=smiling,
            is_young=young,
            num_samples=16, # Genera una griglia 4x4 per ogni categoria
            filename=filename
        )
        
    print("\n✅ Generazione completata con successo!")