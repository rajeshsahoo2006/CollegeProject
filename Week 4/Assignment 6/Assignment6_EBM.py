"""
Assignment 6: Energy-Based Models (EBM) with Contrastive Divergence
Trains an EBM on MNIST using Langevin dynamics sampling.
Reference: Generative Deep Learning 2nd Edition, Chapter 7
Uses PyTorch with MPS (Apple Metal GPU) — all tensors stay on GPU.
"""

import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# All images save here
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(SAVE_DIR, "output"), exist_ok=True)

# ============================================================
# Device setup — use Apple MPS GPU
# ============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">>> Using Apple MPS GPU <<<")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(">>> Using CUDA GPU <<<")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ============================================================
# Hyperparameters
# ============================================================
IMAGE_SIZE = 32
CHANNELS = 1
STEP_SIZE = 10
STEPS = 60          # Langevin steps during training
NOISE = 0.005       # Langevin noise stddev
ALPHA = 0.1         # Regularization weight
GRADIENT_CLIP = 0.03
BATCH_SIZE = 128
BUFFER_SIZE = 8192
LEARNING_RATE = 0.0001
EPOCHS = 120        # Assignment requires 120 epochs (2x the book's 60)

# ============================================================
# 1. Load and preprocess MNIST — preload entire dataset to GPU
# ============================================================
print("Loading MNIST dataset to GPU...")
transform = transforms.Compose([
    transforms.Pad(2),              # 28x28 -> 32x32
    transforms.ToTensor(),          # [0, 1]
    transforms.Normalize([0.5], [0.5]),  # [-1, 1]
])

train_dataset = datasets.MNIST(root=os.path.join(SAVE_DIR, "mnist_data"), train=True, download=True, transform=transform)

# Preload ALL training data to GPU to eliminate CPU->GPU transfer per batch
all_train_imgs = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))]).to(device)
n_train = all_train_imgs.shape[0]
print(f"Loaded {n_train} images to {device} — shape: {all_train_imgs.shape}")

# Display samples
sample_imgs = all_train_imgs[:10].cpu()
fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
for i, ax in enumerate(axes):
    ax.imshow(sample_imgs[i].squeeze().numpy(), cmap="gray")
    ax.axis("off")
plt.suptitle("MNIST Training Samples")
plt.savefig(os.path.join(SAVE_DIR, "output", "training_samples.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 2. Energy Function Network (CNN -> scalar energy) — on GPU
# ============================================================
class EnergyNetwork(nn.Module):
    """Maps 32x32x1 image to scalar energy. Swish (SiLU) activations."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS, 16, kernel_size=5, stride=2, padding=2),  # 16x16x16
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),        # 8x8x32
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),        # 4x4x64
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),        # 2x2x64
            nn.SiLU(),
            nn.Flatten(),                                                   # 256
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

model = EnergyNetwork().to(device)
print(f"Energy Network: {sum(p.numel() for p in model.parameters()):,} params on {device}")

# ============================================================
# 3. Langevin Dynamics — fully on GPU, using torch.autograd.grad
# ============================================================
@torch.no_grad()
def langevin_step_no_grad(model, imgs, step_size, noise_std, grad_clip):
    """Single Langevin step using functional grad (no graph accumulation)."""
    # This is a workaround: we temporarily enable grad just for the input
    pass

def generate_samples(model, inp_imgs, steps, step_size, noise_std, grad_clip, return_per_step=False):
    """
    Langevin dynamics — ALL computation on GPU.
    Uses torch.autograd.grad (functional) to avoid .backward() graph overhead.
    Model params frozen during sampling.
    """
    imgs_per_step = []
    x = inp_imgs.clone().detach()

    # Freeze model weights — only need input gradients
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    for _ in range(steps):
        # Add noise (on GPU)
        x = x + torch.randn_like(x) * noise_std
        x.clamp_(-1.0, 1.0)

        # Compute energy gradient w.r.t. input using functional API
        x.requires_grad_(True)
        energy = model(x).sum()
        grad = torch.autograd.grad(energy, x)[0]

        # Update image via gradient ascent (higher energy = more likely)
        x = x.detach() + step_size * grad.clamp(-grad_clip, grad_clip)
        x.clamp_(-1.0, 1.0)

        if return_per_step:
            imgs_per_step.append(x.clone())

    # Restore model
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()

    if return_per_step:
        return torch.stack(imgs_per_step, dim=0)
    return x.detach()

# ============================================================
# 4. Replay Buffer — stored ENTIRELY on GPU
# ============================================================
class GPUBuffer:
    """Replay buffer that keeps all examples on GPU to avoid transfers."""
    def __init__(self, device):
        self.device = device
        # Initialize with random images on GPU
        self.examples = torch.rand(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device) * 2 - 1

    def sample_batch(self, model, steps, step_size, noise_std, grad_clip):
        n_new = np.random.binomial(BATCH_SIZE, 0.05)
        n_old = BATCH_SIZE - n_new

        # Fresh random images (on GPU)
        rand_imgs = torch.rand(n_new, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=self.device) * 2 - 1

        # Sample from buffer (on GPU — no CPU transfer)
        if n_old > 0:
            idx = torch.randint(0, len(self.examples), (n_old,))
            old_imgs = self.examples[idx]
        else:
            old_imgs = torch.empty(0, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=self.device)

        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0)

        # Langevin refinement (on GPU)
        refined = generate_samples(model, inp_imgs, steps, step_size, noise_std, grad_clip)

        # Push refined samples back into buffer (on GPU)
        self.examples = torch.cat([refined.detach(), self.examples], dim=0)[:BUFFER_SIZE]

        return refined

# ============================================================
# 5. Display helper
# ============================================================
def display_images(images, save_to=None, title=None):
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    n = min(len(images), 10)
    fig, axes = plt.subplots(1, n, figsize=(15, 1.5))
    if n == 1:
        axes = [axes]
    for i in range(n):
        img = images[i].squeeze() if images[i].ndim == 3 else images[i]
        axes[i].imshow(img, cmap="gray", vmin=-1, vmax=1)
        axes[i].axis("off")
    if title:
        plt.suptitle(title)
    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close()

# ============================================================
# 6. Training loop — Contrastive Divergence (all on GPU)
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
buffer = GPUBuffer(device)

loss_history = []
cdiv_history = []
reg_history = []
real_out_history = []
fake_out_history = []

n_batches = n_train // BATCH_SIZE

print(f"\nTraining EBM for {EPOCHS} epochs on {device}")
print(f"Batches per epoch: {n_batches}, Langevin steps/batch: {STEPS}")
print(f"All data + buffer + model on GPU — zero CPU transfers during training\n")

total_start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    epoch_loss = []
    epoch_cdiv = []
    epoch_reg = []
    epoch_real = []
    epoch_fake = []

    # Shuffle indices on GPU
    perm = torch.randperm(n_train, device=device)

    for b in range(n_batches):
        # Get batch directly from GPU tensor (no DataLoader overhead)
        idx = perm[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        real_imgs = all_train_imgs[idx]

        # Add noise to real images (on GPU)
        real_imgs = real_imgs + torch.randn_like(real_imgs) * NOISE
        real_imgs.clamp_(-1.0, 1.0)

        # Generate fake images via Langevin from buffer (on GPU)
        fake_imgs = buffer.sample_batch(model, STEPS, STEP_SIZE, NOISE, GRADIENT_CLIP)

        # Forward pass (on GPU)
        all_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        all_out = model(all_imgs)
        real_out, fake_out = torch.split(all_out, BATCH_SIZE, dim=0)

        # Contrastive divergence + regularization
        cdiv_loss = fake_out.mean() - real_out.mean()
        reg_loss = ALPHA * (real_out ** 2 + fake_out ** 2).mean()
        loss = cdiv_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        epoch_cdiv.append(cdiv_loss.item())
        epoch_reg.append(reg_loss.item())
        epoch_real.append(real_out.mean().item())
        epoch_fake.append(fake_out.mean().item())

    avg_loss = np.mean(epoch_loss)
    avg_cdiv = np.mean(epoch_cdiv)
    avg_reg = np.mean(epoch_reg)
    avg_real = np.mean(epoch_real)
    avg_fake = np.mean(epoch_fake)
    epoch_time = time.time() - epoch_start

    loss_history.append(avg_loss)
    cdiv_history.append(avg_cdiv)
    reg_history.append(avg_reg)
    real_out_history.append(avg_real)
    fake_out_history.append(avg_fake)

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - total_start
        print(f"Epoch {epoch:4d}/{EPOCHS} — loss: {avg_loss:.4f}, "
              f"cdiv: {avg_cdiv:.4f}, reg: {avg_reg:.4f}, "
              f"real: {avg_real:.4f}, fake: {avg_fake:.4f} "
              f"[{epoch_time:.1f}s/epoch, total: {elapsed/60:.1f}min]")

    # Save images at key checkpoints only
    if epoch in [0, 29, 59, 89, 119]:
        gen_steps = 1000 if epoch == 119 else 500
        start_imgs = torch.rand(10, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device) * 2 - 1
        gen_imgs = generate_samples(model, start_imgs, gen_steps, STEP_SIZE, NOISE, GRADIENT_CLIP)
        display_images(gen_imgs, save_to=os.path.join(SAVE_DIR, "output", f"generated_epoch_{epoch:03d}.png"),
                       title=f"Generated — Epoch {epoch}")

        buf_idx = torch.randint(0, len(buffer.examples), (10,))
        display_images(buffer.examples[buf_idx],
                       save_to=os.path.join(SAVE_DIR, "output", f"buffer_epoch_{epoch:03d}.png"),
                       title=f"Buffer Samples — Epoch {epoch}")

total_time = time.time() - total_start
print(f"\nTotal training time: {total_time/60:.1f} minutes")

# ============================================================
# 7. Plot training loss curves
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(loss_history)
axes[0, 0].set_title("Total Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(cdiv_history, color="orange")
axes[0, 1].set_title("Contrastive Divergence Loss")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(real_out_history, label="Real", color="green")
axes[1, 0].plot(fake_out_history, label="Fake", color="red")
axes[1, 0].set_title("Energy Scores (Real vs Fake)")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Mean Energy Score")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(reg_history, color="purple")
axes[1, 1].set_title("Regularization Loss")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle("EBM Training Curves (120 Epochs)", fontsize=14)
plt.tight_layout()
loss_path = os.path.join(SAVE_DIR, "output", "loss_curves_120epochs.png")
plt.savefig(loss_path, dpi=150)
plt.close()
print(f"Loss curves saved to {loss_path}")

# ============================================================
# 8. Final generation: Langevin progression
# ============================================================
print("Generating final samples with 1000 Langevin steps...")
start_imgs = torch.rand(10, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device) * 2 - 1
gen_progression = generate_samples(
    model, start_imgs, steps=1000, step_size=STEP_SIZE, noise_std=NOISE,
    grad_clip=GRADIENT_CLIP, return_per_step=True
)

display_images(start_imgs, save_to=os.path.join(SAVE_DIR, "output", "starting_noise.png"),
               title="Starting Noise")
display_images(gen_progression[-1], save_to=os.path.join(SAVE_DIR, "output", "final_generated.png"),
               title="Final Generated Images (1000 Langevin Steps)")

# Progression for one sample
progression_steps = [0, 1, 3, 5, 10, 30, 50, 100, 300, 999]
fig, axes = plt.subplots(1, len(progression_steps), figsize=(15, 1.5))
for i, step in enumerate(progression_steps):
    img = gen_progression[step][6].squeeze().cpu().numpy()
    axes[i].imshow(img, cmap="gray", vmin=-1, vmax=1)
    axes[i].set_title(f"Step {step}", fontsize=8)
    axes[i].axis("off")
plt.suptitle("Langevin Dynamics Progression (Single Sample)")
plt.savefig(os.path.join(SAVE_DIR, "output", "langevin_progression.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved langevin_progression.png")

# ============================================================
# 9. Comparison summary
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON WITH BOOK RESULTS")
print("=" * 60)
print(f"""
Book configuration:
  - Epochs: 60
  - Langevin steps (training): 60
  - Langevin steps (generation): 1000
  - Step size: 10, Noise: 0.005, Gradient clip: 0.03
  - Alpha (reg): 0.1, Batch size: 128, Buffer: 8192
  - Learning rate: 0.0001 (Adam)
  - Framework: TensorFlow/Keras

Our configuration:
  - Epochs: 120 (2x the book)
  - Framework: PyTorch with MPS GPU
  - ALL data, buffer, and model on GPU — zero CPU<->GPU transfers
  - All other hyperparameters identical

Total training time: {total_time/60:.1f} minutes

Final training loss: {loss_history[-1]:.4f}
Final cdiv loss: {cdiv_history[-1]:.4f}
Final real score: {real_out_history[-1]:.4f}
Final fake score: {fake_out_history[-1]:.4f}
""")
