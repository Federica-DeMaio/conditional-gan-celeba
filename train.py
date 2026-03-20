import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# --- IMPORT DAI NOSTRI FILE ---
from model import Generator, Discriminator
from dataset import get_dataloader

# ============================================================
# CONFIGURAZIONE E IPERPARAMETRI
# ============================================================
DATA_PATH = './data/celebA'  # <-- INSERIRE QUI IL PATH DEL DATASET
OUTPUT_DIR = './outputs'

BATCH_SIZE = 128
EPOCHS = 200
LATENT_SIZE = 128

# TTUR (Two Time-Scale Update Rule): 
# Learning rates differenziati per bilanciare la convergenza di G e D
LR_G = 0.0002
LR_D = 0.0001
LABEL_SMOOTHING = 0.1

def setup_directories():
    """Crea le cartelle necessarie per salvare i risultati del training."""
    dirs = ['results', 'checkpoints', 'loss_plots']
    for d in dirs:
        os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

# ============================================================
# FUNZIONI DI SUPPORTO (INSTANCE NOISE & PLOTTING)
# ============================================================
def get_noise_std(current_epoch: int, total_epochs: int, start_std: float = 0.1) -> float:
    """Decade linearmente la deviazione standard del rumore col passare delle epoche."""
    progress = current_epoch / total_epochs
    decayed_std = start_std * (1 - progress)
    return max(0.05, decayed_std)

def add_instance_noise(images: torch.Tensor, std: float) -> torch.Tensor:
    """Aggiunge rumore gaussiano per stabilizzare le prime fasi del training."""
    if std <= 0:
        return images
    noise = torch.randn_like(images) * std
    return images + noise

def save_validation_grid(epoch: int, generator: Generator, fixed_z: torch.Tensor, fixed_labels: torch.Tensor):
    """Genera e salva una griglia di immagini per valutare visivamente i progressi."""
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_z, fixed_labels)
        fake_images = fake_images * 0.5 + 0.5  # Denormalizza da [-1, 1] a [0, 1]
        grid = vutils.make_grid(fake_images, nrow=8, padding=2, normalize=False)
        vutils.save_image(grid, os.path.join(OUTPUT_DIR, f"results/epoch_{epoch+1:03d}.png"))
    generator.train()

def save_loss_plot(epoch: int, g_losses: list, d_losses: list):
    """Genera e salva il grafico delle loss cumulative."""
    plt.figure(figsize=(10, 5))
    plt.title(f"Generator and Discriminator Loss (Epoch {epoch+1})")
    plt.plot(g_losses, label="G Loss", color='blue', alpha=0.7)
    plt.plot(d_losses, label="D Loss", color='orange', alpha=0.7)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, f"loss_plots/loss_plot_epoch_{epoch+1:03d}.png"))
    plt.close()

# ============================================================
# TRAINING LOOP PRINCIPALE
# ============================================================
def main():
    plt.switch_backend('agg')  # Backend non interattivo per server/cluster
    setup_directories()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Inizio Training su: {device} ---")

    dataloader = get_dataloader(DATA_PATH, BATCH_SIZE, num_workers=4)

    # Inizializzazione modelli e ottimizzatori
    gen_model = Generator().to(device)
    disc_model = Discriminator().to(device)

    gen_optimizer = optim.Adam(gen_model.parameters(), lr=LR_G, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(disc_model.parameters(), lr=LR_D, betas=(0.5, 0.999))

    # Setup per la griglia di validazione fissa (8 combinazioni di classi)
    print("Preparazione griglia di validazione...")
    FIXED_BATCH_SIZE = 64
    fixed_z = torch.randn(FIXED_BATCH_SIZE, LATENT_SIZE, device=device)
    # Genera le 8 combinazioni binarie possibili (000, 001, ..., 111) x 8 volte
    combinations = [[i >> 2, (i >> 1) & 1, i & 1] for i in range(8)]
    fixed_labels = torch.tensor(combinations * 8, dtype=torch.float32, device=device)

    history_gloss, history_dloss = [], []

    print(">>> Avvio Training Loop...")
    for epoch in range(EPOCHS):
        gen_model.train()
        disc_model.train()

        sum_gloss, sum_dloss = 0.0, 0.0
        batches = 0
        total_batches = len(dataloader)
        
        current_noise_std = get_noise_std(epoch, EPOCHS)

        for i, (x_true, cls) in enumerate(dataloader):
            x_true = x_true.to(device)
            cls = cls.to(device).float()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            disc_optimizer.zero_grad()
            
            # Forward reali con rumore
            x_true_noisy = add_instance_noise(x_true, current_noise_std)
            d_true = disc_model(x_true_noisy, cls)

            # Generazione e forward fake
            z = torch.randn(x_true.shape[0], LATENT_SIZE, device=device)
            x_synth = gen_model(z, cls)
            x_synth_noisy = add_instance_noise(x_synth.detach(), current_noise_std)
            d_synth = disc_model(x_synth_noisy, cls)

            # Label smoothing applicato ai target reali per prevenire l'overconfidence
            t_true = torch.ones_like(d_true) - LABEL_SMOOTHING
            t_synth = torch.zeros_like(d_synth)
            
            dloss = F.binary_cross_entropy(d_true, t_true) + F.binary_cross_entropy(d_synth, t_synth)
            dloss.backward()
            disc_optimizer.step()

            # ---------------------
            #  Train Generator
            # ---------------------
            gen_optimizer.zero_grad()
            
            x_synth_noisy_for_g = add_instance_noise(x_synth, current_noise_std)
            d_synth_for_g = disc_model(x_synth_noisy_for_g, cls)
            
            # Il generatore punta a far classificare le sue immagini come reali (1)
            t_synth_target = torch.ones_like(d_synth_for_g)

            gloss = F.binary_cross_entropy(d_synth_for_g, t_synth_target)
            gloss.backward()
            gen_optimizer.step()

            sum_gloss += gloss.item()
            sum_dloss += dloss.item()
            batches += 1

            if i % 50 == 0:
                print(f"\rEp {epoch+1:03d} [{i:03d}/{total_batches}] | Noise: {current_noise_std:.3f} | DLoss: {dloss.item():.4f} | GLoss: {gloss.item():.4f}", end="")

        # Statistiche di fine epoca
        avg_gloss = sum_gloss / batches
        avg_dloss = sum_dloss / batches
        history_gloss.append(avg_gloss)
        history_dloss.append(avg_dloss)

        print(f"\n>> END EP {epoch+1:03d} | Noise: {current_noise_std:.3f} | GLoss Avg: {avg_gloss:.4f} | DLoss Avg: {avg_dloss:.4f}")

        # Salvataggio artefatti
        save_validation_grid(epoch, gen_model, fixed_z, fixed_labels)

        if (epoch + 1) % 10 == 0:
            save_loss_plot(epoch, history_gloss, history_dloss)
            
        # Salvataggio Pesi
        torch.save(gen_model.state_dict(), os.path.join(OUTPUT_DIR, f'checkpoints/gen_epoch_{epoch+1:03d}.pth'))
        torch.save(disc_model.state_dict(), os.path.join(OUTPUT_DIR, f'checkpoints/disc_epoch_{epoch+1:03d}.pth'))

if __name__ == '__main__':
    main()