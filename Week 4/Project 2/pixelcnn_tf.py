"""
Project 2: Generate Images with PixelCNN (Mixture Distribution)
================================================================
PixelCNN with mixture of logistic distributions using TensorFlow Probability.
Uses tfp.distributions.PixelCNN as required by the assignment.
Trained on Fashion MNIST with integer pixel values in range [0, 255].

Reference: Generative Deep Learning, 2nd Edition - Chapter 5 (p.162)
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_keras import datasets, layers, models, optimizers, callbacks
import tensorflow_probability as tfp

# Force flush for real-time output
def log(msg):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# 0. Device Setup
# ---------------------------------------------------------------------------
log(f"TensorFlow version: {tf.__version__}")
log(f"TensorFlow Probability version: {tfp.__version__}")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    log(f"GPU detected: {gpus}")
    DEVICE_INFO = "Apple Metal GPU (M2)"
else:
    log("No GPU detected. Running on CPU.")
    DEVICE_INFO = "CPU"

# ---------------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 28
N_COMPONENTS = 5  # Number of logistic mixture components
EPOCHS = 5
BATCH_SIZE = 128
NUM_GENERATE = 10

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output tf")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Prepare the Data - Integer pixel values [0, 255]
# ---------------------------------------------------------------------------
log("\n--- Loading Fashion MNIST ---")
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# Add channel dimension, keep as float in [0, 255]
x_train = np.expand_dims(x_train, -1).astype("float32")
x_test = np.expand_dims(x_test, -1).astype("float32")

log(f"Training data shape: {x_train.shape}")
log(f"Test data shape: {x_test.shape}")
log(f"Pixel value range: [{x_train.min():.0f}, {x_train.max():.0f}]")

# Display sample training images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i, :, :, 0], cmap="gray")
    ax.set_title(CLASS_NAMES[y_train[i]])
    ax.axis("off")
plt.suptitle("Sample Training Images (Fashion MNIST)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_samples.png"), dpi=150)
plt.close()
log("Saved training_samples.png")

# ---------------------------------------------------------------------------
# 3. Build the PixelCNN using tfp.distributions.PixelCNN
# ---------------------------------------------------------------------------
log("\n--- Building PixelCNN Model (tfp.distributions.PixelCNN) ---")

# Create PixelCNN distribution with mixture of logistic output
# This is the TensorFlow function specified in the assignment
dist = tfp.distributions.PixelCNN(
    image_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
    num_resnet=1,           # ResNet blocks per hierarchy
    num_hierarchies=2,      # Resolution levels
    num_filters=32,         # Convolutional filters
    num_logistic_mix=N_COMPONENTS,  # Mixture components (p.162 in textbook)
    dropout_p=0.3,          # Dropout for regularization
)

# Wrap in a Keras model
image_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
log_prob = dist.log_prob(image_input)

pixelcnn = models.Model(inputs=image_input, outputs=log_prob)
pixelcnn.add_loss(-tf.reduce_mean(log_prob))  # Negative log-likelihood loss

# Use legacy Adam optimizer (recommended by TF for Apple Silicon)
pixelcnn.compile(optimizer=optimizers.legacy.Adam(learning_rate=0.001))

pixelcnn.summary(print_fn=log)

# ---------------------------------------------------------------------------
# 4. Train the PixelCNN
# ---------------------------------------------------------------------------
log(f"\n--- Training PixelCNN for {EPOCHS} epochs ---")

train_losses = []
test_losses = []
start_time = time.time()


class EpochLogger(callbacks.Callback):
    """Log epoch results and generate sample images."""

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        train_losses.append(loss)

        # Evaluate on test set
        test_loss = self.model.evaluate(x_test, batch_size=BATCH_SIZE, verbose=0)
        test_losses.append(test_loss)

        elapsed = time.time() - start_time
        log(f"Epoch {epoch+1}/{EPOCHS} | Train NLL: {loss:.2f} | "
            f"Test NLL: {test_loss:.2f} | Time: {elapsed:.0f}s")

        # Generate sample images at first and last epoch
        if epoch == 0 or epoch == EPOCHS - 1:
            log(f"  Generating sample images for epoch {epoch+1}...")
            gen_start = time.time()
            generated = dist.sample(3).numpy()
            gen_time = time.time() - gen_start
            log(f"  Sampling took {gen_time:.0f}s")

            fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
            for i, ax in enumerate(axes):
                ax.imshow(generated[i, :, :, 0], cmap="gray")
                ax.axis("off")
            plt.suptitle(f"Generated Images - Epoch {epoch+1}", fontsize=12)
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, f"generated_epoch_{epoch+1:03d}.png"),
                dpi=150,
            )
            plt.close()
            log(f"  Saved generated_epoch_{epoch+1:03d}.png")


pixelcnn.fit(
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[EpochLogger()],
)

total_time = time.time() - start_time
log(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f} min)")

# ---------------------------------------------------------------------------
# 5. Plot Training & Test Loss
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, "b-o", label="Train NLL", linewidth=2)
plt.plot(range(1, EPOCHS + 1), test_losses, "r-s", label="Test NLL", linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Negative Log-Likelihood", fontsize=12)
plt.title("PixelCNN Training & Test Loss (TensorFlow)", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"), dpi=150)
plt.close()
log("Saved training_loss.png")

# ---------------------------------------------------------------------------
# 6. Generate Final Images using dist.sample()
# ---------------------------------------------------------------------------
log(f"\n--- Generating {NUM_GENERATE} Final Images using dist.sample() ---")
gen_start = time.time()
generated_images = dist.sample(NUM_GENERATE).numpy()
gen_time = time.time() - gen_start
log(f"Sampling {NUM_GENERATE} images took {gen_time:.0f}s")

log(f"Generated image shape: {generated_images.shape}")
log(f"Generated pixel range: [{generated_images.min():.0f}, {generated_images.max():.0f}]")

# Display generated images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_images[i, :, :, 0], cmap="gray")
    ax.set_title(f"Sample {i + 1}")
    ax.axis("off")
plt.suptitle("PixelCNN Generated Images (Final) - TensorFlow", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "final_generated_images.png"), dpi=150)
plt.close()
log("Saved final_generated_images.png")

# ---------------------------------------------------------------------------
# 7. Real vs Generated Comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(5):
    axes[0, i].imshow(x_train[i, :, :, 0], cmap="gray")
    axes[0, i].set_title("Real")
    axes[0, i].axis("off")
    axes[1, i].imshow(generated_images[i, :, :, 0], cmap="gray")
    axes[1, i].set_title("Generated")
    axes[1, i].axis("off")
plt.suptitle("Real vs Generated Images", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "real_vs_generated.png"), dpi=150)
plt.close()
log("Saved real_vs_generated.png")

# ---------------------------------------------------------------------------
# 8. Print Summary
# ---------------------------------------------------------------------------
log("\n" + "=" * 60)
log("SUMMARY")
log("=" * 60)
log(f"Model: tfp.distributions.PixelCNN (Mixture of Logistics)")
log(f"Dataset: Fashion MNIST ({IMAGE_SIZE}x{IMAGE_SIZE})")
log(f"Input pixel range: [0, 255] (integer values)")
log(f"Mixture components (num_logistic_mix): {N_COMPONENTS}")
log(f"Architecture: 1 ResNet block, 2 hierarchies, 32 filters")
log(f"Optimizer: Adam (legacy, lr=0.001)")
log(f"Epochs trained: {EPOCHS}")
log(f"Final training NLL: {train_losses[-1]:.2f}")
log(f"Final test NLL: {test_losses[-1]:.2f}")
log(f"Device: {DEVICE_INFO}")
log(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
log(f"Output saved to: {OUTPUT_DIR}")
log("=" * 60)
