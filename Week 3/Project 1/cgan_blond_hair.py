"""
Project 1: Conditional GAN (CGAN) conditioned on the Blond_Hair attribute
of the CelebA faces dataset.

Based on: https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/
          blob/main/notebooks/04_gan/03_cgan/cgan.ipynb

Key assignment changes in train_step:
  - The generator accepts 1D one-hot labels: shape (batch, CLASSES)
  - The critic accepts 2D spatial labels:   shape (batch, IMAGE_SIZE, IMAGE_SIZE, CLASSES)
  The train_step must expand and tile the flat one-hot labels into a full
  spatial label map before passing them to the critic.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils, metrics, optimizers

# ---------------------------------------------------------------------------
# 0. Hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE    = 64
CHANNELS      = 3
CLASSES       = 2          # blond (1) vs non-blond (0)
BATCH_SIZE    = 128
Z_DIM         = 32
LEARNING_RATE = 0.00005
ADAM_BETA_1   = 0.5
ADAM_BETA_2   = 0.9
EPOCHS        = 20
CRITIC_STEPS  = 3
GP_WEIGHT     = 10.0
LOAD_MODEL    = False
LABEL         = "Blond_Hair"

# Paths – adjust DATA_DIR to wherever the CelebA dataset lives
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))   # Week 3/Project 1/
DATA_DIR      = "/app/data/celeba-dataset"                   # default Docker path
IMG_DIR       = os.path.join(DATA_DIR, "img_align_celeba")
ATTR_CSV      = os.path.join(DATA_DIR, "list_attr_celeba.csv")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output")
CKPT_PATH     = os.path.join(BASE_DIR, "checkpoint", "checkpoint.ckpt")
LOG_DIR       = os.path.join(BASE_DIR, "logs")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "checkpoint"), exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Prepare the data
# ---------------------------------------------------------------------------
attributes = pd.read_csv(ATTR_CSV)
print("Columns:", attributes.columns.tolist())
print(attributes.head())

# CelebA uses {-1, 1}; convert to {0, 1}
labels = attributes[LABEL].tolist()
int_labels = [1 if x == 1 else 0 for x in labels]

print(f"\nBlond samples   : {sum(int_labels)}")
print(f"Non-blond samples: {len(int_labels) - sum(int_labels)}")

train_data = utils.image_dataset_from_directory(
    IMG_DIR,
    labels=int_labels,
    color_mode="rgb",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)


def preprocess(img):
    """Rescale pixel values from [0, 255] to [-1, 1]."""
    return (tf.cast(img, "float32") - 127.5) / 127.5


# Map to (normalised_image, one_hot_label)
train = train_data.map(
    lambda x, y: (preprocess(x), tf.one_hot(y, depth=CLASSES))
)

# ---------------------------------------------------------------------------
# 2. Build critic and generator
# ---------------------------------------------------------------------------

# --- Critic ---
# Input 1: the image  (IMAGE_SIZE x IMAGE_SIZE x CHANNELS)
# Input 2: the label  (IMAGE_SIZE x IMAGE_SIZE x CLASSES) – spatial label map
critic_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
label_input  = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CLASSES))

x = layers.Concatenate(axis=-1)([critic_input, label_input])   # depth = CHANNELS + CLASSES
x = layers.Conv2D(64,  kernel_size=4, strides=2, padding="same")(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(1,   kernel_size=4, strides=1, padding="valid")(x)
critic_output = layers.Flatten()(x)

critic = models.Model([critic_input, label_input], critic_output)
critic.summary()

# --- Generator ---
# Input 1: latent vector  (Z_DIM,)
# Input 2: one-hot label  (CLASSES,)  – flat, NOT spatial
generator_input = layers.Input(shape=(Z_DIM,))
label_input_gen = layers.Input(shape=(CLASSES,))

x = layers.Concatenate(axis=-1)([generator_input, label_input_gen])
x = layers.Reshape((1, 1, Z_DIM + CLASSES))(x)
x = layers.Conv2DTranspose(128, kernel_size=4, strides=1, padding="valid",  use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same",   use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same",   use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(64,  kernel_size=4, strides=2, padding="same",   use_bias=False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
generator_output = layers.Conv2DTranspose(
    CHANNELS, kernel_size=4, strides=2, padding="same", activation="tanh"
)(x)

generator = models.Model([generator_input, label_input_gen], generator_output)
generator.summary()

# ---------------------------------------------------------------------------
# 3. Conditional WGAN-GP model
# ---------------------------------------------------------------------------

class ConditionalWGAN(models.Model):
    """
    Wasserstein GAN with Gradient Penalty, conditioned on a class label.

    Key format difference between generator and critic inputs:
      Generator  – expects a *flat* one-hot vector  shape (batch, CLASSES)
      Critic     – expects a *spatial* label map     shape (batch, H, W, CLASSES)

    The train_step resolves this mismatch by tiling the flat vector across
    the spatial dimensions before feeding it to the critic.
    """

    def __init__(self, critic, generator, latent_dim, critic_steps, gp_weight):
        super().__init__()
        self.critic       = critic
        self.generator    = generator
        self.latent_dim   = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight    = gp_weight

    def compile(self, c_optimizer, g_optimizer):
        super().compile(run_eagerly=True)
        self.c_optimizer         = c_optimizer
        self.g_optimizer         = g_optimizer
        self.c_wass_loss_metric  = metrics.Mean(name="c_wass_loss")
        self.c_gp_metric         = metrics.Mean(name="c_gp")
        self.c_loss_metric       = metrics.Mean(name="c_loss")
        self.g_loss_metric       = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [
            self.c_loss_metric,
            self.c_wass_loss_metric,
            self.c_gp_metric,
            self.g_loss_metric,
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images, image_one_hot_labels):
        """WGAN-GP gradient penalty on interpolated images."""
        alpha        = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = real_images + alpha * (fake_images - real_images)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic([interpolated, image_one_hot_labels], training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm  = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, data):
        """
        Custom training step.

        *** Assignment change – matching input formats ***
        The dataset provides one_hot_labels with shape (batch, CLASSES).

        - Generator input  : one_hot_labels as-is → (batch, CLASSES)
        - Critic input     : one_hot_labels must be expanded to a spatial map
                             → (batch, IMAGE_SIZE, IMAGE_SIZE, CLASSES)

        We do this by:
          1. Adding two singleton spatial dimensions: (batch, 1, 1, CLASSES)
          2. tf.repeat along axis=1 IMAGE_SIZE times  → (batch, IMAGE_SIZE, 1, CLASSES)
          3. tf.repeat along axis=2 IMAGE_SIZE times  → (batch, IMAGE_SIZE, IMAGE_SIZE, CLASSES)
        """
        real_images, one_hot_labels = data

        # --- Build spatial label map for the critic ---
        # Step 1: (batch, CLASSES) → (batch, 1, 1, CLASSES)
        image_one_hot_labels = one_hot_labels[:, None, None, :]
        # Step 2: tile over height
        image_one_hot_labels = tf.repeat(image_one_hot_labels, repeats=IMAGE_SIZE, axis=1)
        # Step 3: tile over width
        image_one_hot_labels = tf.repeat(image_one_hot_labels, repeats=IMAGE_SIZE, axis=2)
        # Shape is now (batch, IMAGE_SIZE, IMAGE_SIZE, CLASSES) ✓

        batch_size = tf.shape(real_images)[0]

        # ---- Train critic for critic_steps iterations ----
        for _ in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                # Generator gets flat one_hot_labels
                fake_images = self.generator(
                    [random_latent_vectors, one_hot_labels], training=True
                )
                # Critic gets spatial image_one_hot_labels
                fake_predictions = self.critic(
                    [fake_images,  image_one_hot_labels], training=True
                )
                real_predictions = self.critic(
                    [real_images,  image_one_hot_labels], training=True
                )

                c_wass_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
                c_gp   = self.gradient_penalty(
                    batch_size, real_images, fake_images, image_one_hot_labels
                )
                c_loss = c_wass_loss + c_gp * self.gp_weight

            c_gradients = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(c_gradients, self.critic.trainable_variables)
            )

        # ---- Train generator ----
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            fake_images      = self.generator(
                [random_latent_vectors, one_hot_labels], training=True
            )
            fake_predictions = self.critic(
                [fake_images, image_one_hot_labels], training=True
            )
            g_loss = -tf.reduce_mean(fake_predictions)

        gen_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        self.c_loss_metric.update_state(c_loss)
        self.c_wass_loss_metric.update_state(c_wass_loss)
        self.c_gp_metric.update_state(c_gp)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}


# ---------------------------------------------------------------------------
# 4. Instantiate and compile
# ---------------------------------------------------------------------------
cgan = ConditionalWGAN(
    critic=critic,
    generator=generator,
    latent_dim=Z_DIM,
    critic_steps=CRITIC_STEPS,
    gp_weight=GP_WEIGHT,
)

if LOAD_MODEL:
    cgan.load_weights(CKPT_PATH)

cgan.compile(
    c_optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2),
    g_optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1, beta_2=ADAM_BETA_2),
)

# ---------------------------------------------------------------------------
# 5. Callbacks
# ---------------------------------------------------------------------------

def display_and_save(images, save_path, title=""):
    """Display a grid of images and save to file."""
    images = np.clip(images, 0, 255).astype("uint8")
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    if n == 1:
        axes = [axes]
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved: {save_path}")


model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=CKPT_PATH,
    save_weights_only=True,
    save_freq="epoch",
    verbose=0,
)

tensorboard_callback = callbacks.TensorBoard(log_dir=LOG_DIR)


class ImageGenerator(callbacks.Callback):
    """At the end of each epoch, generate images for both class labels."""

    def __init__(self, num_img, latent_dim):
        self.num_img    = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        z = tf.random.normal(shape=(self.num_img, self.latent_dim))

        # Label 0 → non-blond  [1, 0]
        label_0 = np.repeat([[1, 0]], self.num_img, axis=0)
        imgs_0  = self.model.generator([z, label_0], training=False)
        imgs_0  = (imgs_0.numpy() * 127.5 + 127.5)
        display_and_save(
            imgs_0,
            save_path=os.path.join(OUTPUT_DIR, f"epoch_{epoch:03d}_non_blond.png"),
            title=f"Epoch {epoch} – Non-Blond (label=0)",
        )

        # Label 1 → blond  [0, 1]
        label_1 = np.repeat([[0, 1]], self.num_img, axis=0)
        imgs_1  = self.model.generator([z, label_1], training=False)
        imgs_1  = (imgs_1.numpy() * 127.5 + 127.5)
        display_and_save(
            imgs_1,
            save_path=os.path.join(OUTPUT_DIR, f"epoch_{epoch:03d}_blond.png"),
            title=f"Epoch {epoch} – Blond (label=1)",
        )

        # Print losses
        if logs:
            print(
                f"  c_loss={logs.get('c_loss', 0):.4f}  "
                f"c_wass={logs.get('c_wass_loss', 0):.4f}  "
                f"c_gp={logs.get('c_gp', 0):.4f}  "
                f"g_loss={logs.get('g_loss', 0):.4f}"
            )


# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
print("\n=== Training Conditional WGAN-GP ===")
print(f"Conditioning on attribute: {LABEL}")
print(f"Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  Z_DIM: {Z_DIM}\n")

history = cgan.fit(
    train,
    epochs=EPOCHS,
    callbacks=[
        model_checkpoint_callback,
        tensorboard_callback,
        ImageGenerator(num_img=10, latent_dim=Z_DIM),
    ],
)

# ---------------------------------------------------------------------------
# 7. Final generation
# ---------------------------------------------------------------------------
print("\n=== Generating final images ===")
z_sample = np.random.normal(size=(10, Z_DIM))

# Non-blond faces
label_0 = np.repeat([[1, 0]], 10, axis=0)
imgs_0   = cgan.generator.predict([z_sample, label_0])
imgs_0   = (imgs_0 * 127.5 + 127.5)
display_and_save(imgs_0, os.path.join(OUTPUT_DIR, "final_non_blond.png"), "Final – Non-Blond")

# Blond faces
label_1 = np.repeat([[0, 1]], 10, axis=0)
imgs_1   = cgan.generator.predict([z_sample, label_1])
imgs_1   = (imgs_1 * 127.5 + 127.5)
display_and_save(imgs_1, os.path.join(OUTPUT_DIR, "final_blond.png"), "Final – Blond")

# Save models
os.makedirs(MODELS_DIR, exist_ok=True)
generator.save(os.path.join(MODELS_DIR, "generator"))
critic.save(os.path.join(MODELS_DIR, "critic"))
print("Models saved to ./models/")

# ---------------------------------------------------------------------------
# 8. Plot training curves
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["c_loss"],      label="critic loss")
axes[0].plot(history.history["c_wass_loss"], label="wasserstein")
axes[0].plot(history.history["c_gp"],        label="gradient penalty")
axes[0].set_title("Critic Losses")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["g_loss"], label="generator loss", color="orange")
axes[1].set_title("Generator Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=100, bbox_inches="tight")
plt.show()
print("Training curves saved.")
