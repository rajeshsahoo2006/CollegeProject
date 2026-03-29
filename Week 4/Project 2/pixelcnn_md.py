"""
Project 2: Generate Images with PixelCNN (Mixture Distribution)
================================================================
PixelCNN with mixture of logistic distributions for Fashion MNIST.
Accepts integer pixel values in range [0, 255].
Uses PyTorch with MPS (Apple GPU) / CUDA acceleration.

Reference: Generative Deep Learning, 2nd Edition - Chapter 5 (p.162)
PixelCNN++: https://arxiv.org/abs/1701.05517
"""

import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# 0. Device Setup
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Apple MPS GPU detected")
else:
    DEVICE = torch.device("cpu")
    print("No GPU detected. Running on CPU.")

print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")

# ---------------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 28  # Fashion MNIST native size
N_COMPONENTS = 5  # Number of logistic mixture components
EPOCHS = 15
BATCH_SIZE = 128
LR = 0.001
NUM_FILTERS = 64
NUM_LAYERS = 7
NUM_GENERATE = 10

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Prepare the Data - Integer pixel values [0, 255]
# ---------------------------------------------------------------------------
print("\n--- Loading Fashion MNIST ---")

train_dataset = datasets.FashionMNIST(
    root=os.path.join(OUTPUT_DIR, "data"), train=True, download=True,
    transform=transforms.ToTensor(),
)
test_dataset = datasets.FashionMNIST(
    root=os.path.join(OUTPUT_DIR, "data"), train=False, download=True,
    transform=transforms.ToTensor(),
)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


print("Preparing data...")
x_train = (train_dataset.data.unsqueeze(1).float())  # (60000, 1, 28, 28) in [0, 255]
y_train = train_dataset.targets
x_test = (test_dataset.data.unsqueeze(1).float())
y_test = test_dataset.targets

print(f"Training data shape: {x_train.shape}")
print(f"Pixel value range: [{x_train.min():.0f}, {x_train.max():.0f}]")

train_loader = DataLoader(
    TensorDataset(x_train), batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(x_test), batch_size=BATCH_SIZE, shuffle=False
)

# Display sample training images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i, 0].numpy(), cmap="gray")
    ax.set_title(CLASS_NAMES[y_train[i]])
    ax.axis("off")
plt.suptitle("Sample Training Images (Fashion MNIST)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_samples.png"), dpi=150)
plt.close()
print("Saved training_samples.png")


# ---------------------------------------------------------------------------
# 3. Masked Convolution Layer
# ---------------------------------------------------------------------------
class MaskedConv2d(nn.Conv2d):
    """
    Masked convolution for autoregressive ordering.
    Type A: masks the center pixel (used for the first layer).
    Type B: allows the center pixel (used for subsequent layers).
    """

    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, h, w = self.weight.shape
        # Zero out the bottom half
        self.mask[:, :, h // 2 + 1 :, :] = 0
        # Zero out the right half of the center row
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# ---------------------------------------------------------------------------
# 4. Residual Block
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = MaskedConv2d("B", filters, filters, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = MaskedConv2d("B", filters, filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv3 = MaskedConv2d("B", filters, filters, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(filters)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(x + out)


# ---------------------------------------------------------------------------
# 5. PixelCNN with Mixture of Logistics Output
# ---------------------------------------------------------------------------
class PixelCNNMixtureDistribution(nn.Module):
    """
    PixelCNN that outputs parameters for a mixture of logistic distributions.
    For each pixel, outputs:
      - K mixture logits (weights)
      - K means
      - K log-scales
    Total output channels per pixel: 3 * K
    """

    def __init__(self, in_channels=1, num_filters=64, num_layers=7,
                 num_components=5):
        super().__init__()
        self.num_components = num_components
        out_channels = 3 * num_components  # logits, means, log_scales

        layers_list = []
        # First layer: Type A mask (don't see current pixel)
        layers_list.append(MaskedConv2d("A", in_channels, num_filters, 7, padding=3))
        layers_list.append(nn.BatchNorm2d(num_filters))
        layers_list.append(nn.ReLU())

        # Residual blocks with Type B masks
        for _ in range(num_layers):
            layers_list.append(ResidualBlock(num_filters))

        # Output layers
        layers_list.append(MaskedConv2d("B", num_filters, num_filters, 1))
        layers_list.append(nn.ReLU())
        layers_list.append(MaskedConv2d("B", num_filters, out_channels, 1))

        self.network = nn.Sequential(*layers_list)

    def forward(self, x):
        """
        Input: x of shape (B, 1, H, W) with values in [0, 255]
        Output: params of shape (B, 3*K, H, W)
        """
        return self.network(x)

    def get_mixture_params(self, logits_raw):
        """Split network output into mixture parameters."""
        K = self.num_components
        B, _, H, W = logits_raw.shape

        # Reshape to (B, 3, K, H, W)
        params = logits_raw.view(B, 3, K, H, W)

        mix_logits = params[:, 0, :, :, :]   # (B, K, H, W) - mixture weights
        means = params[:, 1, :, :, :]         # (B, K, H, W) - component means
        log_scales = params[:, 2, :, :, :]    # (B, K, H, W) - log scales
        log_scales = torch.clamp(log_scales, min=-7.0)  # numerical stability

        return mix_logits, means, log_scales

    def mixture_log_prob(self, x, logits_raw):
        """
        Compute log probability of x under the mixture of logistics.
        x: (B, 1, H, W) with values in [0, 255]
        """
        mix_logits, means, log_scales = self.get_mixture_params(logits_raw)
        K = self.num_components

        # Expand x to match K components: (B, 1, H, W) -> (B, K, H, W)
        x_expanded = x.expand(-1, K, -1, -1)

        # Compute CDF values for discretized logistic
        scales = torch.exp(log_scales)
        centered = (x_expanded - means) / scales

        # For discretized logistic on [0, 255]:
        # P(x) = sigmoid((x + 0.5 - mean) / scale) - sigmoid((x - 0.5 - mean) / scale)
        upper = torch.sigmoid((x_expanded + 0.5 - means) / scales)
        lower = torch.sigmoid((x_expanded - 0.5 - means) / scales)

        # Edge cases: clamp at 0 and 255
        cdf_upper = torch.where(x_expanded >= 255.0, torch.ones_like(upper), upper)
        cdf_lower = torch.where(x_expanded <= 0.0, torch.zeros_like(lower), lower)

        # Log probability of each component
        mid_prob = torch.clamp(cdf_upper - cdf_lower, min=1e-12)
        log_probs_components = torch.log(mid_prob)  # (B, K, H, W)

        # Log mixture weights
        log_mix_weights = F.log_softmax(mix_logits, dim=1)  # (B, K, H, W)

        # Log-sum-exp over mixture components
        log_probs = torch.logsumexp(
            log_mix_weights + log_probs_components, dim=1
        )  # (B, H, W)

        return log_probs.sum(dim=[1, 2])  # Sum over pixels -> (B,)

    @torch.no_grad()
    def sample(self, num_samples, device, image_size=28):
        """
        Generate images autoregressively, pixel by pixel.
        Samples from the mixture of logistic distributions.
        """
        self.eval()
        samples = torch.zeros(num_samples, 1, image_size, image_size, device=device)

        for h in range(image_size):
            for w in range(image_size):
                logits_raw = self.forward(samples)
                mix_logits, means, log_scales = self.get_mixture_params(logits_raw)

                # Sample mixture component
                mix_weights = F.softmax(mix_logits[:, :, h, w], dim=1)  # (B, K)
                component = torch.multinomial(mix_weights, 1).squeeze(1)  # (B,)

                # Get parameters for selected component
                batch_idx = torch.arange(num_samples, device=device)
                mu = means[batch_idx, component, h, w]
                log_s = log_scales[batch_idx, component, h, w]

                # Sample from logistic distribution
                u = torch.rand_like(mu).clamp(1e-5, 1 - 1e-5)
                x_sample = mu + torch.exp(log_s) * (torch.log(u) - torch.log(1 - u))

                # Clamp to valid pixel range [0, 255]
                x_sample = torch.clamp(x_sample.round(), 0, 255)
                samples[:, 0, h, w] = x_sample

        self.train()
        return samples


# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------
print("\n--- Building PixelCNN Model ---")
model = PixelCNNMixtureDistribution(
    in_channels=1,
    num_filters=NUM_FILTERS,
    num_layers=NUM_LAYERS,
    num_components=N_COMPONENTS,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Architecture: {NUM_LAYERS} residual blocks, {NUM_FILTERS} filters, "
      f"{N_COMPONENTS} mixture components")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
train_losses = []
test_losses = []

print(f"\n--- Training PixelCNN for {EPOCHS} epochs ---")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    epoch_loss = 0
    num_batches = 0
    for (batch,) in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        logits_raw = model(batch)
        log_prob = model.mixture_log_prob(batch, logits_raw)
        loss = -log_prob.mean()  # Negative log-likelihood
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1

    avg_train_loss = epoch_loss / num_batches
    train_losses.append(avg_train_loss)

    # Evaluate on test set
    model.eval()
    test_loss = 0
    test_batches = 0
    with torch.no_grad():
        for (batch,) in test_loader:
            batch = batch.to(DEVICE)
            logits_raw = model(batch)
            log_prob = model.mixture_log_prob(batch, logits_raw)
            test_loss += (-log_prob.mean()).item()
            test_batches += 1
    avg_test_loss = test_loss / test_batches
    test_losses.append(avg_test_loss)

    elapsed = time.time() - start_time
    print(f"Epoch {epoch:2d}/{EPOCHS} | Train NLL: {avg_train_loss:.2f} | "
          f"Test NLL: {avg_test_loss:.2f} | Time: {elapsed:.0f}s")

    # Generate sample images at key epochs (sampling is slow: 784 sequential steps per image)
    if epoch in [1, 5, 10, 15] or epoch == EPOCHS:
        print(f"  Generating sample images for epoch {epoch}...")
        gen_imgs = model.sample(3, DEVICE, IMAGE_SIZE).cpu().numpy()
        n_gen = gen_imgs.shape[0]
        fig, axes = plt.subplots(1, n_gen, figsize=(n_gen * 2.5, 2.5))
        if n_gen == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.imshow(gen_imgs[i, 0], cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
        plt.suptitle(f"Generated Images - Epoch {epoch}", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"generated_epoch_{epoch:03d}.png"), dpi=150
        )
        plt.close()
        print(f"  Saved generated_epoch_{epoch:03d}.png")

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time:.0f}s")

# ---------------------------------------------------------------------------
# 7. Plot Training & Test Loss
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, "b-o", label="Train NLL", linewidth=2)
plt.plot(range(1, EPOCHS + 1), test_losses, "r-s", label="Test NLL", linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Negative Log-Likelihood", fontsize=12)
plt.title("PixelCNN Training & Test Loss", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"), dpi=150)
plt.close()
print("Saved training_loss.png")

# ---------------------------------------------------------------------------
# 8. Generate Final Images
# ---------------------------------------------------------------------------
print(f"\n--- Generating {NUM_GENERATE} Final Images ---")
generated_images = model.sample(NUM_GENERATE, DEVICE, IMAGE_SIZE).cpu().numpy()

print(f"Generated image shape: {generated_images.shape}")
print(f"Generated pixel range: [{generated_images.min():.0f}, {generated_images.max():.0f}]")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_images[i, 0], cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Sample {i + 1}")
    ax.axis("off")
plt.suptitle("PixelCNN Generated Images (Final)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "final_generated_images.png"), dpi=150)
plt.close()
print("Saved final_generated_images.png")

# ---------------------------------------------------------------------------
# 9. Real vs Generated Comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(5):
    axes[0, i].imshow(x_train[i, 0].numpy(), cmap="gray", vmin=0, vmax=255)
    axes[0, i].set_title("Real")
    axes[0, i].axis("off")
    axes[1, i].imshow(generated_images[i, 0], cmap="gray", vmin=0, vmax=255)
    axes[1, i].set_title("Generated")
    axes[1, i].axis("off")
plt.suptitle("Real vs Generated Images", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "real_vs_generated.png"), dpi=150)
plt.close()
print("Saved real_vs_generated.png")

# ---------------------------------------------------------------------------
# 10. Print Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Model: PixelCNN with Mixture of Logistic Distributions")
print(f"Dataset: Fashion MNIST ({IMAGE_SIZE}x{IMAGE_SIZE})")
print(f"Input pixel range: [0, 255] (integer values)")
print(f"Mixture components: {N_COMPONENTS}")
print(f"Architecture: {NUM_LAYERS} residual blocks, {NUM_FILTERS} filters")
print(f"Total parameters: {total_params:,}")
print(f"Epochs trained: {EPOCHS}")
print(f"Final training NLL: {train_losses[-1]:.2f}")
print(f"Final test NLL: {test_losses[-1]:.2f}")
print(f"Device used: {DEVICE}")
print(f"Total training time: {total_time:.0f}s")
print(f"Output saved to: {OUTPUT_DIR}")
print("=" * 60)
