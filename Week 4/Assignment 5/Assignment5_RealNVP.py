"""
Assignment 5: Normalizing Flow Models (RealNVP)
Transforms the make_moons distribution into a Gaussian using a RealNVP network.
Reference: Generative Deep Learning 2nd Edition, Chapter 6
Uses PyTorch with MPS (Apple Metal GPU) for acceleration.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# All images save here
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Device setup — use Apple MPS GPU if available
# ============================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ============================================================
# Hyperparameters
# ============================================================
COUPLING_DIM = 256
COUPLING_LAYERS = 2
INPUT_DIM = 2
REGULARIZATION = 0.01
BATCH_SIZE = 256
EPOCHS = 600
LEARNING_RATE = 0.0001

# ============================================================
# 1. Load and normalize the make_moons dataset
# ============================================================
data, _ = datasets.make_moons(30000, noise=0.05)
data = data.astype("float32")
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data).astype("float32")

plt.figure()
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c="green", s=1)
plt.title("Normalized make_moons Dataset")
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.savefig(os.path.join(SAVE_DIR, "dataset_visualization.png"), dpi=150)
plt.close()
print("Saved dataset_visualization.png")

# Convert to PyTorch dataset
tensor_data = torch.tensor(normalized_data)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================
# 2. Coupling Layer: separate networks for scale (s) and translation (t)
# ============================================================
class CouplingNetwork(nn.Module):
    def __init__(self, input_dim, coupling_dim):
        super().__init__()
        # Scale network (s) — tanh output
        self.s_net = nn.Sequential(
            nn.Linear(input_dim, coupling_dim), nn.ReLU(),
            nn.Linear(coupling_dim, coupling_dim), nn.ReLU(),
            nn.Linear(coupling_dim, coupling_dim), nn.ReLU(),
            nn.Linear(coupling_dim, coupling_dim), nn.ReLU(),
            nn.Linear(coupling_dim, input_dim), nn.Tanh(),
        )
        # Translation network (t) — linear output
        self.t_net = nn.Sequential(
            nn.Linear(input_dim, coupling_dim), nn.ReLU(),
            nn.Linear(coupling_dim, coupling_dim), nn.ReLU(),
            nn.Linear(coupling_dim, coupling_dim), nn.ReLU(),
            nn.Linear(coupling_dim, coupling_dim), nn.ReLU(),
            nn.Linear(coupling_dim, input_dim),
        )

    def forward(self, x):
        return self.s_net(x), self.t_net(x)


# ============================================================
# 3. RealNVP Model
# ============================================================
class RealNVP(nn.Module):
    def __init__(self, input_dim, coupling_layers, coupling_dim):
        super().__init__()
        self.coupling_layers = coupling_layers
        # Alternating binary masks
        self.masks = torch.tensor(
            [[0, 1], [1, 0]] * (coupling_layers // 2), dtype=torch.float32
        )
        self.layers_list = nn.ModuleList([
            CouplingNetwork(input_dim, coupling_dim)
            for _ in range(coupling_layers)
        ])

    def forward(self, x, reverse=False):
        """Forward: data -> latent (training). Reverse: latent -> data (generation)."""
        log_det_inv = torch.zeros(x.shape[0], device=x.device)
        if reverse:
            direction = 1
        else:
            direction = -1

        indices = list(range(self.coupling_layers))
        if direction == -1:
            indices = indices[::-1]

        for i in indices:
            mask = self.masks[i].to(x.device)
            reversed_mask = 1 - mask
            x_masked = x * mask
            s, t = self.layers_list[i](x_masked)
            s = s * reversed_mask
            t = t * reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * torch.exp(direction * s) + direction * t * torch.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * torch.sum(s, dim=1)
        return x, log_det_inv

    def log_loss(self, x):
        y, logdet = self.forward(x)
        # Base distribution: 2D standard Gaussian
        log_prob = -0.5 * (y.shape[1] * np.log(2 * np.pi) + torch.sum(y ** 2, dim=1))
        log_likelihood = log_prob + logdet
        return -torch.mean(log_likelihood)


# ============================================================
# 4. Instantiate model and optimizer
# ============================================================
model = RealNVP(
    input_dim=INPUT_DIM,
    coupling_layers=COUPLING_LAYERS,
    coupling_dim=COUPLING_DIM,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

# ============================================================
# 5. Helper: generate and visualize results
# ============================================================
def generate_and_display(model, normalized_data_np, num_samples=3000, save_to=None):
    model.eval()
    with torch.no_grad():
        # Data -> Latent
        data_tensor = torch.tensor(normalized_data_np, device=device)
        z, _ = model(data_tensor)
        z = z.cpu().numpy()

        # Latent -> Data (reverse pass)
        samples = torch.randn(num_samples, INPUT_DIM, device=device)
        x_gen, _ = model(samples, reverse=True)
        x_gen = x_gen.cpu().numpy()
        samples_np = samples.cpu().numpy()

    f, axes = plt.subplots(2, 2)
    f.set_size_inches(8, 5)

    axes[0, 0].scatter(normalized_data_np[:, 0], normalized_data_np[:, 1], color="r", s=1)
    axes[0, 0].set(title="Data space X", xlabel="x_1", ylabel="x_2")
    axes[0, 0].set_xlim([-2, 2]); axes[0, 0].set_ylim([-2, 2])

    axes[0, 1].scatter(z[:, 0], z[:, 1], color="r", s=1)
    axes[0, 1].set(title="f(X) — Data to Latent", xlabel="z_1", ylabel="z_2")
    axes[0, 1].set_xlim([-2, 2]); axes[0, 1].set_ylim([-2, 2])

    axes[1, 0].scatter(samples_np[:, 0], samples_np[:, 1], color="g", s=1)
    axes[1, 0].set(title="Latent space Z", xlabel="z_1", ylabel="z_2")
    axes[1, 0].set_xlim([-2, 2]); axes[1, 0].set_ylim([-2, 2])

    axes[1, 1].scatter(x_gen[:, 0], x_gen[:, 1], color="g", s=1)
    axes[1, 1].set(title="g(Z) — Generated Data", xlabel="x_1", ylabel="x_2")
    axes[1, 1].set_xlim([-2, 2]); axes[1, 1].set_ylim([-2, 2])

    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    if save_to:
        plt.savefig(save_to, dpi=150)
        print(f"Saved to {save_to}")
    plt.close()
    model.train()

# ============================================================
# 6. Train for 600 epochs
# ============================================================
print(f"\nTraining RealNVP for {EPOCHS} epochs on {device}...")
loss_history = []

for epoch in range(EPOCHS):
    epoch_losses = []
    for (batch,) in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = model.log_loss(batch)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    avg_loss = np.mean(epoch_losses)
    loss_history.append(avg_loss)

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:4d}/{EPOCHS} — Loss: {avg_loss:.4f}")

    if epoch % 100 == 0:
        generate_and_display(
            model, normalized_data,
            save_to=os.path.join(SAVE_DIR, f"generated_epoch_{epoch:03d}.png")
        )

# ============================================================
# 7. Plot the training loss curve
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(loss_history)
plt.title("Training Loss Curve (600 Epochs)")
plt.xlabel("Epoch")
plt.ylabel("Negative Log-Likelihood Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
loss_path = os.path.join(SAVE_DIR, "loss_curve_600epochs.png")
plt.savefig(loss_path, dpi=150)
plt.close()
print(f"\nLoss curve saved to {loss_path}")

# ============================================================
# 8. Final results visualization
# ============================================================
generate_and_display(
    model, normalized_data,
    save_to=os.path.join(SAVE_DIR, "final_results.png")
)

# ============================================================
# 9. Comparison summary
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON WITH BOOK RESULTS")
print("=" * 60)
print(f"""
Book configuration:
  - Epochs: 300
  - Coupling layers: 2
  - Coupling dim: 256
  - Learning rate: 0.0001
  - Framework: TensorFlow/Keras

Our configuration:
  - Epochs: 600 (2x the book)
  - Framework: PyTorch with MPS GPU acceleration
  - All other hyperparameters identical

Key observations:
  1. The loss curve should show convergence, with the loss
     decreasing rapidly in the first ~100 epochs and then
     gradually stabilizing.

  2. With 600 epochs (vs 300 in the book), the model has
     more time to refine the transformation, potentially
     achieving a tighter Gaussian in latent space and
     sharper generated moons.

  3. The forward pass f(X) should map the two-moon shape
     into an approximately circular Gaussian blob.

  4. The inverse pass g(Z) should transform Gaussian
     samples back into a recognizable two-moon shape.

  5. Overfitting risk: with 2x epochs, watch for the loss
     curve plateauing or oscillating — if so, the extra
     epochs provide diminishing returns but shouldn't hurt
     due to the L2 regularization (weight_decay).

Final training loss: {loss_history[-1]:.4f}
""")
