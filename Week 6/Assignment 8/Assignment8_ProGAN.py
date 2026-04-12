"""
Assignment 8: Progressive Growing Generative Adversarial Networks (ProGAN)
Generates realistic facial images using progressive resolution training.
Reference: Karras et al. "Progressive Growing of GANs" (ICLR 2018)
           https://blog.paperspace.com/progan/
Uses PyTorch with MPS (Apple Metal GPU).
Trains on LFW (Labeled Faces in the Wild) — limited to 10,000 images per assignment constraints.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from sklearn.datasets import fetch_lfw_people
from PIL import Image

# ============================================================
# Paths and device
# ============================================================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SAVE_DIR, "output")
DATA_DIR = os.path.join(SAVE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

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
LATENT_DIM = 128
MAX_FILTERS = 128
N_IMAGES = 10000          # use only 10k images (assignment constraint)
LEARNING_RATE = 1e-3
BETAS = (0.0, 0.99)
EPSILON = 1e-8
DRIFT_PENALTY = 0.001     # keeps discriminator outputs small

# Progressive schedule: (resolution, epochs, batch_size)
# 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
SCHEDULE = [
    (4,   8,  16),
    (8,   10, 16),
    (16,  10, 16),
    (32,  10,  8),
    (64,  10,  4),
    (128, 10,  4),
]


# ============================================================
# Custom layers
# ============================================================
class PixelNorm(nn.Module):
    """Per-pixel feature vector normalization (replaces batch norm in ProGAN)."""
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + EPSILON)


class MinibatchStddev(nn.Module):
    """Appends minibatch stddev as extra feature map to improve variety."""
    def forward(self, x):
        batch, c, h, w = x.shape
        stddev = torch.sqrt(torch.mean((x - torch.mean(x, dim=0, keepdim=True)) ** 2,
                                        dim=0, keepdim=True) + EPSILON)
        mean_std = torch.mean(stddev, dim=[1, 2, 3], keepdim=True)
        extra = mean_std.expand(batch, 1, h, w)
        return torch.cat([x, extra], dim=1)


class EqualizedConv2d(nn.Module):
    """Conv2d with equalized learning rate (He init at runtime)."""
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        nn.init.normal_(self.conv.weight, 0.0, 1.0)
        nn.init.zeros_(self.conv.bias)
        fan_in = in_ch * kernel_size * kernel_size
        self.scale = math.sqrt(2.0 / fan_in)

    def forward(self, x):
        return self.conv(x * self.scale)


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.normal_(self.linear.weight, 0.0, 1.0)
        nn.init.zeros_(self.linear.bias)
        self.scale = math.sqrt(2.0 / in_features)

    def forward(self, x):
        return self.linear(x * self.scale)


# ============================================================
# Generator
# ============================================================
class GenInitBlock(nn.Module):
    """Initial 4x4 block: latent -> 4x4 feature maps."""
    def __init__(self, latent_dim, n_filters):
        super().__init__()
        self.dense = EqualizedLinear(latent_dim, n_filters * 4 * 4)
        self.conv = EqualizedConv2d(n_filters, n_filters, 3, padding=1)
        self.pn = PixelNorm()
        self.act = nn.LeakyReLU(0.2)

    def forward(self, z):
        x = self.dense(z)
        x = x.view(-1, MAX_FILTERS, 4, 4)
        x = self.act(self.pn(x))
        x = self.act(self.pn(self.conv(x)))
        return x


class GenBlock(nn.Module):
    """Upsampling block: doubles spatial resolution."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = EqualizedConv2d(out_ch, out_ch, 3, padding=1)
        self.pn = PixelNorm()
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.act(self.pn(self.conv1(x)))
        x = self.act(self.pn(self.conv2(x)))
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim, max_filters):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_block = GenInitBlock(latent_dim, max_filters)

        # Blocks for each progressive step (8x8, 16x16, 32x32, 64x64, 128x128)
        self.blocks = nn.ModuleList([
            GenBlock(128, 128),   # -> 8x8
            GenBlock(128, 128),   # -> 16x16
            GenBlock(128, 128),   # -> 32x32
            GenBlock(128, 64),    # -> 64x64
            GenBlock(64,  32),    # -> 128x128
        ])

        # toRGB layers for each resolution level
        self.to_rgb = nn.ModuleList([
            EqualizedConv2d(128, 3, 1),  # 4x4
            EqualizedConv2d(128, 3, 1),  # 8x8
            EqualizedConv2d(128, 3, 1),  # 16x16
            EqualizedConv2d(128, 3, 1),  # 32x32
            EqualizedConv2d(64,  3, 1),  # 64x64
            EqualizedConv2d(32,  3, 1),  # 128x128
        ])

        self.current_depth = 0   # 0 = 4x4, 1 = 8x8, ...
        self.alpha = 1.0         # fade-in blending factor

    def forward(self, z):
        x = self.init_block(z)

        if self.current_depth == 0:
            return self.to_rgb[0](x)

        # Pass through all blocks up to current depth
        for i in range(self.current_depth - 1):
            x = self.blocks[i](x)

        # Fade-in: blend upsampled old RGB with new block RGB
        prev = x
        x = self.blocks[self.current_depth - 1](x)

        # Old path: upsample previous features, convert to RGB
        old_rgb = F.interpolate(self.to_rgb[self.current_depth - 1](prev),
                                scale_factor=2, mode='nearest')
        # New path: new block output to RGB
        new_rgb = self.to_rgb[self.current_depth](x)

        return (1 - self.alpha) * old_rgb + self.alpha * new_rgb


# ============================================================
# Discriminator
# ============================================================
class DiscFinalBlock(nn.Module):
    """Final 4x4 block with minibatch stddev -> score."""
    def __init__(self, n_filters):
        super().__init__()
        self.mbstd = MinibatchStddev()
        self.conv1 = EqualizedConv2d(n_filters + 1, n_filters, 3, padding=1)
        self.conv2 = EqualizedConv2d(n_filters, n_filters, 4)
        self.act = nn.LeakyReLU(0.2)
        self.out = EqualizedLinear(n_filters, 1)

    def forward(self, x):
        x = self.mbstd(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.out(x)


class DiscBlock(nn.Module):
    """Downsampling block: halves spatial resolution."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = EqualizedConv2d(in_ch, out_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, max_filters):
        super().__init__()
        self.final_block = DiscFinalBlock(max_filters)

        # Blocks for each progressive step (reverse order of generator)
        self.blocks = nn.ModuleList([
            DiscBlock(128, 128),   # 8x8 -> 4x4
            DiscBlock(128, 128),   # 16x16 -> 8x8
            DiscBlock(128, 128),   # 32x32 -> 16x16
            DiscBlock(64,  128),   # 64x64 -> 32x32
            DiscBlock(32,  64),    # 128x128 -> 64x64
        ])

        # fromRGB layers for each resolution level
        self.from_rgb = nn.ModuleList([
            EqualizedConv2d(3, 128, 1),  # 4x4
            EqualizedConv2d(3, 128, 1),  # 8x8
            EqualizedConv2d(3, 128, 1),  # 16x16
            EqualizedConv2d(3, 128, 1),  # 32x32
            EqualizedConv2d(3, 64,  1),  # 64x64
            EqualizedConv2d(3, 32,  1),  # 128x128
        ])

        self.act = nn.LeakyReLU(0.2)
        self.current_depth = 0
        self.alpha = 1.0

    def forward(self, img):
        if self.current_depth == 0:
            x = self.act(self.from_rgb[0](img))
            return self.final_block(x)

        # New path: fromRGB at current resolution -> process through new block
        new_x = self.act(self.from_rgb[self.current_depth](img))
        new_x = self.blocks[self.current_depth - 1](new_x)

        # Old path: downsample image, then fromRGB at previous resolution
        old_x = F.avg_pool2d(img, 2)
        old_x = self.act(self.from_rgb[self.current_depth - 1](old_x))

        # Fade-in blend
        x = (1 - self.alpha) * old_x + self.alpha * new_x

        # Pass through remaining blocks
        for i in range(self.current_depth - 2, -1, -1):
            x = self.blocks[i](x)

        return self.final_block(x)


# ============================================================
# Wasserstein loss + gradient penalty (WGAN-GP)
# ============================================================
def gradient_penalty(disc, real, fake, device):
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interp = disc(interpolated)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.view(batch_size, -1)
    penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


# ============================================================
# Data loading — LFW faces (auto-downloads via sklearn)
# ============================================================
_lfw_cache = {}  # cache resized tensors per resolution

def get_face_loader(resolution, batch_size, n_images=N_IMAGES):
    """Returns a DataLoader for LFW faces at the given resolution."""
    if resolution in _lfw_cache:
        tensor_data = _lfw_cache[resolution]
    else:
        print(f"  Loading LFW faces at {resolution}x{resolution}...")
        lfw = fetch_lfw_people(min_faces_per_person=1, resize=1.0, color=True,
                               data_home=DATA_DIR)
        images = lfw.images[:n_images]  # limit to n_images

        # Resize each image to target resolution and normalize to [-1, 1]
        resized = []
        for img_np in images:
            pil_img = Image.fromarray(img_np.astype(np.uint8))
            # Center-crop to square, then resize
            w, h = pil_img.size
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            pil_img = pil_img.crop((left, top, left + crop_size, top + crop_size))
            pil_img = pil_img.resize((resolution, resolution), Image.BILINEAR)
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
            resized.append(arr)

        tensor_data = torch.tensor(np.array(resized), dtype=torch.float32)
        _lfw_cache[resolution] = tensor_data
        print(f"  Loaded {tensor_data.shape[0]} faces at {resolution}x{resolution}")

    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, pin_memory=False)
    return loader


# ============================================================
# Training utilities
# ============================================================
def save_generated_images(gen, epoch, resolution, status, n_samples=16):
    """Generate and save a grid of sample images."""
    gen.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM, device=device)
        imgs = gen(z).cpu()
    gen.train()

    # De-normalize from [-1,1] to [0,1]
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
    grid_size = int(math.sqrt(n_samples))

    fig, axes = plt.subplots(grid_size, grid_size,
                             figsize=(grid_size * 2, grid_size * 2))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            img = imgs[idx].permute(1, 2, 0).numpy()
            axes[i][j].imshow(img)
            axes[i][j].axis('off')

    title = f"ProGAN {resolution}x{resolution} — Epoch {epoch} ({status})"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR,
                         f"progan_{resolution:03d}x{resolution:03d}_e{epoch:02d}_{status}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")
    return fname


def save_loss_curves(d_losses, g_losses, resolution):
    """Plot and save loss curves for a training phase."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d_losses, label='Discriminator', alpha=0.7)
    ax.plot(g_losses, label='Generator', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'ProGAN Training Loss — {resolution}x{resolution}')
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"loss_{resolution:03d}x{resolution:03d}.png")
    plt.savefig(fname, dpi=150)
    plt.close()


# ============================================================
# Main training loop
# ============================================================
def train():
    print("=" * 60)
    print("ProGAN: Progressive Growing of GANs — Face Generation")
    print("=" * 60)

    gen = Generator(LATENT_DIM, MAX_FILTERS).to(device)
    disc = Discriminator(MAX_FILTERS).to(device)

    all_output_images = []

    for stage, (resolution, n_epochs, batch_size) in enumerate(SCHEDULE):
        print(f"\n{'='*60}")
        print(f"Stage {stage}: Resolution {resolution}x{resolution}")
        print(f"  Epochs: {n_epochs}  |  Batch size: {batch_size}")
        print(f"{'='*60}")

        # Set progressive depth
        gen.current_depth = stage
        disc.current_depth = stage

        # Fresh optimizers per stage (as in original ProGAN)
        opt_g = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE,
                                 betas=BETAS, eps=EPSILON)
        opt_d = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE,
                                 betas=BETAS, eps=EPSILON)

        loader = get_face_loader(resolution, batch_size)
        total_batches = len(loader)

        d_losses_stage = []
        g_losses_stage = []

        for epoch in range(1, n_epochs + 1):
            d_loss_epoch = []
            g_loss_epoch = []

            for batch_idx, (real_imgs,) in enumerate(loader):
                real_imgs = real_imgs.to(device)
                cur_batch = real_imgs.size(0)

                # Update alpha for fade-in (ramp over first half of epochs)
                total_steps = total_batches * n_epochs
                current_step = (epoch - 1) * total_batches + batch_idx
                if stage == 0:
                    alpha = 1.0
                else:
                    # Fade in over first half, then stay at 1.0
                    fade_steps = total_steps // 2
                    alpha = min(1.0, current_step / max(fade_steps, 1))
                gen.alpha = alpha
                disc.alpha = alpha

                # ------ Train Discriminator ------
                opt_d.zero_grad()
                z = torch.randn(cur_batch, LATENT_DIM, device=device)
                fake_imgs = gen(z).detach()

                d_real = disc(real_imgs)
                d_fake = disc(fake_imgs)

                # Wasserstein loss + gradient penalty
                gp = gradient_penalty(disc, real_imgs, fake_imgs, device)
                d_loss = d_fake.mean() - d_real.mean() + 10.0 * gp
                d_loss += DRIFT_PENALTY * (d_real ** 2).mean()

                d_loss.backward()
                opt_d.step()

                # ------ Train Generator ------
                opt_g.zero_grad()
                z = torch.randn(cur_batch, LATENT_DIM, device=device)
                fake_imgs = gen(z)
                g_loss = -disc(fake_imgs).mean()

                g_loss.backward()
                opt_g.step()

                d_losses_stage.append(d_loss.item())
                g_losses_stage.append(g_loss.item())
                d_loss_epoch.append(d_loss.item())
                g_loss_epoch.append(g_loss.item())

            avg_d = np.mean(d_loss_epoch)
            avg_g = np.mean(g_loss_epoch)
            print(f"  Epoch {epoch}/{n_epochs} | alpha={alpha:.3f} "
                  f"| D_loss={avg_d:.4f} | G_loss={avg_g:.4f}")

            # Save sample images at key epochs
            if epoch == 1 or epoch == n_epochs or epoch % max(1, n_epochs // 3) == 0:
                status = "fadein" if alpha < 1.0 else "stable"
                fname = save_generated_images(gen, epoch, resolution, status)
                all_output_images.append(fname)

        # Save loss curve for this stage
        save_loss_curves(d_losses_stage, g_losses_stage, resolution)
        print(f"  Stage {stage} complete ({resolution}x{resolution}).")

    # Save final model
    model_path = os.path.join(OUTPUT_DIR, "progan_generator_final.pt")
    torch.save(gen.state_dict(), model_path)
    print(f"\nFinal generator saved to {model_path}")

    # Generate final high-res output grid
    print("\nGenerating final output images...")
    gen.eval()
    with torch.no_grad():
        z = torch.randn(25, LATENT_DIM, device=device)
        final_imgs = gen(z).cpu()
    final_imgs = (final_imgs * 0.5 + 0.5).clamp(0, 1)

    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for i in range(5):
        for j in range(5):
            img = final_imgs[i * 5 + j].permute(1, 2, 0).numpy()
            axes[i][j].imshow(img)
            axes[i][j].axis('off')
    fig.suptitle("ProGAN — Final Generated Faces (128x128)", fontsize=16)
    plt.tight_layout()
    final_path = os.path.join(OUTPUT_DIR, "progan_final_faces.png")
    plt.savefig(final_path, dpi=200)
    plt.close()
    print(f"Final output saved to {final_path}")

    # Generate progression comparison
    print("Generating resolution progression comparison...")
    gen_depths = list(range(len(SCHEDULE)))
    fig, axes = plt.subplots(1, len(gen_depths), figsize=(3 * len(gen_depths), 3))
    z_fixed = torch.randn(1, LATENT_DIM, device=device)
    for idx, depth in enumerate(gen_depths):
        gen.current_depth = depth
        gen.alpha = 1.0
        with torch.no_grad():
            img = gen(z_fixed).cpu()
        img = (img[0] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
        res = SCHEDULE[depth][0]
        axes[idx].imshow(img)
        axes[idx].set_title(f"{res}x{res}")
        axes[idx].axis('off')
    fig.suptitle("Progressive Resolution Growth", fontsize=14)
    plt.tight_layout()
    prog_path = os.path.join(OUTPUT_DIR, "progan_progression.png")
    plt.savefig(prog_path, dpi=150)
    plt.close()
    print(f"Progression comparison saved to {prog_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    train()
