#!/usr/bin/env python3
"""Run autoencoder on Fashion-MNIST and save output images to outputimage folder."""
import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, datasets
import tensorflow.keras.backend as K

OUTPUT_DIR = "outputimage"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
IMAGE_SIZE = 32
CHANNELS = 1
BATCH_SIZE = 100
EMBEDDING_DIM = 2
EPOCHS = 3  # Use 3 for faster run; increase to 10 for better quality
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
SUBMISSION_INDICES = [0, 100, 500, 1000, 2500]

print("Loading Fashion-MNIST...")
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()


def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs


x_train = preprocess(x_train)
x_test = preprocess(x_test)
# Use subset for faster training (full: 60k train, 10k test)
TRAIN_SUBSET = 60000  # Use 5000 for faster run (~1 min), 60000 for full training
x_train = x_train[:TRAIN_SUBSET]
y_train = y_train[:TRAIN_SUBSET]
print("Training shape:", x_train.shape, "| Test shape:", x_test.shape)

# 1. Sample training images
print("\nSaving sample training images...")
fig, axes = plt.subplots(1, 10, figsize=(20, 3))
for i in range(10):
    axes[i].imshow(x_train[i].squeeze(), cmap="gray_r")
    axes[i].set_title(CLASS_NAMES[y_train[i]], fontsize=8)
    axes[i].axis("off")
plt.suptitle("Sample Fashion-MNIST Training Images", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_sample_training.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {OUTPUT_DIR}/01_sample_training.png")

# 2. Build encoder
print("\nBuilding encoder...")
encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="encoder_input")
x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]
x = layers.Flatten()(x)
encoder_output = layers.Dense(EMBEDDING_DIM, name="encoder_output")(x)
encoder = models.Model(encoder_input, encoder_output)

# 3. Build decoder
print("Building decoder...")
decoder_input = layers.Input(shape=(EMBEDDING_DIM,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
decoder_output = layers.Conv2D(
    CHANNELS, (3, 3), strides=1, activation="sigmoid", padding="same", name="decoder_output"
)(x)
decoder = models.Model(decoder_input, decoder_output)

# 4. Build and train autoencoder
autoencoder = models.Model(encoder_input, decoder(encoder_output))
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

print(f"\nTraining autoencoder ({EPOCHS} epochs)...")
history = autoencoder.fit(
    x_train, x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=1,
)

# 5. Training history plot
print("\nSaving training history...")
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.legend()
plt.title("Autoencoder Training History")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_training_history.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {OUTPUT_DIR}/02_training_history.png")

# 6. Reconstruct and save original vs reconstruction (10 samples)
print("\nGenerating reconstructions...")
predictions = autoencoder.predict(x_test)

fig, axes = plt.subplots(2, 10, figsize=(20, 4))
for i in range(10):
    axes[0, i].imshow(x_test[i].squeeze(), cmap="gray_r")
    axes[0, i].set_title(CLASS_NAMES[y_test[i]], fontsize=8)
    axes[0, i].axis("off")
    axes[1, i].imshow(predictions[i].squeeze(), cmap="gray_r")
    axes[1, i].set_title("Recon", fontsize=8)
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("Original", fontsize=10)
axes[1, 0].set_ylabel("Reconstructed", fontsize=10)
plt.suptitle("Original vs Reconstructed (First 10 Test Images)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_original_vs_reconstructed.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {OUTPUT_DIR}/03_original_vs_reconstructed.png")

# 7. Submission: 5 chosen images - side by side comparison
print("\nSaving submission images (5 chosen)...")
labels_subset = y_test[SUBMISSION_INDICES]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, idx in enumerate(SUBMISSION_INDICES):
    axes[0, i].imshow(x_test[idx].squeeze(), cmap="gray_r")
    axes[0, i].set_title(f"Original: {CLASS_NAMES[labels_subset[i]]}")
    axes[0, i].axis("off")
    axes[1, i].imshow(predictions[idx].squeeze(), cmap="gray_r")
    axes[1, i].set_title("Reconstructed")
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("Original", fontsize=12)
axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
plt.suptitle("Submission: 5 Images - Original vs Reconstructed", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_submission_5_images.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {OUTPUT_DIR}/04_submission_5_images.png")

# 8. Individual images for each of the 5 (original | reconstructed)
for i, idx in enumerate(SUBMISSION_INDICES):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(x_test[idx].squeeze(), cmap="gray_r")
    axes[0].set_title(f"Original: {CLASS_NAMES[labels_subset[i]]}")
    axes[0].axis("off")
    axes[1].imshow(predictions[idx].squeeze(), cmap="gray_r")
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")
    plt.suptitle(f"Image {i+1} (Index {idx})", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"05_image_{i+1}_index_{idx}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {OUTPUT_DIR}/05_image_{i+1}_index_{idx}.png")

print(f"\nDone! All images saved to {OUTPUT_DIR}/")
