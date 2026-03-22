"""
Project 1 – Conditional WGAN-GP conditioned on the Blond_Hair attribute
of the CelebA faces dataset.

Demo run: reads real Blond_Hair labels from list_attr_celeba.csv and uses
synthetic images so the full pipeline can be exercised and captured on any
machine (no GPU / no img_align_celeba download required for this demo).

For the full run on a GPU machine with the images downloaded, switch
USE_SYNTHETIC_IMAGES = False and set IMG_DIR to the images folder.
"""

import os, sys, csv, time

# ---- venv site-packages path (must come before any third-party imports) -----
_VENV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "venv", "lib", "python3.9", "site-packages"
)
sys.path.insert(0, os.path.abspath(_VENV))

import numpy as np

from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, metrics, optimizers

# ---- Output / logging -------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_FILE   = os.path.join(OUTPUT_DIR, "training_output.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "checkpoint"), exist_ok=True)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_fh = open(LOG_FILE, "w")
sys.stdout = Tee(sys.__stdout__, log_fh)

# =============================================================================
# 0. Hyperparameters
# =============================================================================
IMAGE_SIZE  = 64
CHANNELS    = 3
CLASSES     = 2          # [1,0] = non-blond,  [0,1] = blond
BATCH_SIZE  = 32
Z_DIM       = 32
LR          = 0.00005
BETA_1      = 0.5
BETA_2      = 0.9
EPOCHS      = 5          # demo: 5 epochs  (use 20+ for real training)
CRITIC_STEPS = 3
GP_WEIGHT   = 10.0
LABEL       = "Blond_Hair"

ATTR_CSV               = os.path.join(BASE_DIR, "list_attr_celeba.csv")
IMG_DIR                = os.path.join(BASE_DIR, "img_align_celeba")   # real images
USE_SYNTHETIC_IMAGES   = not os.path.isdir(IMG_DIR)                   # auto-detect

print("=" * 70)
print("  Conditional WGAN-GP – Blond_Hair attribute")
print("=" * 70)
print(f"TensorFlow        : {tf.__version__}")
print(f"GPUs available    : {tf.config.list_physical_devices('GPU')}")
print(f"Conditioning on   : {LABEL}")
print(f"Image size        : {IMAGE_SIZE}x{IMAGE_SIZE}x{CHANNELS}")
print(f"Z_DIM             : {Z_DIM}   Batch: {BATCH_SIZE}   Epochs: {EPOCHS}")
print(f"Synthetic images  : {USE_SYNTHETIC_IMAGES}")
print()

# =============================================================================
# 1. Load labels from list_attr_celeba.csv
# =============================================================================
print(">>> Reading Blond_Hair labels from list_attr_celeba.csv ...")
with open(ATTR_CSV, newline="") as f:
    reader     = csv.DictReader(f)
    all_labels = [int(row[LABEL]) for row in reader]

# CelebA uses {-1, +1} → convert to {0, 1}
int_labels = [1 if v == 1 else 0 for v in all_labels]
N_TOTAL    = len(int_labels)
N_BLOND    = sum(int_labels)
print(f"    Total images in CSV  : {N_TOTAL:,}")
print(f"    Blond  (label = 1)   : {N_BLOND:,}  ({100*N_BLOND/N_TOTAL:.1f}%)")
print(f"    Non-blond (label = 0): {N_TOTAL - N_BLOND:,}  ({100*(N_TOTAL-N_BLOND)/N_TOTAL:.1f}%)")

# For the demo, use only the first N_DEMO samples
N_DEMO     = 512
demo_labels = int_labels[:N_DEMO]
print(f"\n    Using first {N_DEMO} samples for demo run.")
print(f"    Blond in demo    : {sum(demo_labels)}")
print(f"    Non-blond in demo: {N_DEMO - sum(demo_labels)}")

# =============================================================================
# 2. Build tf.data pipeline
# =============================================================================
print("\n>>> Building tf.data pipeline ...")

if USE_SYNTHETIC_IMAGES:
    print("    (img_align_celeba not found – using synthetic random images)")
    rng        = np.random.default_rng(42)
    pixel_data = rng.integers(0, 256, (N_DEMO, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.uint8)
    image_ds   = tf.data.Dataset.from_tensor_slices(pixel_data.astype("float32"))
else:
    # Real images path (used when img_align_celeba is present)
    fnames    = [f"{i+1:06d}.jpg" for i in range(N_DEMO)]
    full_paths = [os.path.join(IMG_DIR, fn) for fn in fnames]
    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=CHANNELS)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        return img
    image_ds = tf.data.Dataset.from_tensor_slices(full_paths).map(load_image)

label_ds = tf.data.Dataset.from_tensor_slices(
    np.array(demo_labels, dtype=np.int32)
)

def preprocess(img, lbl):
    img = (tf.cast(img, tf.float32) - 127.5) / 127.5   # [-1, 1]
    lbl = tf.one_hot(lbl, depth=CLASSES)
    return img, lbl

train = (
    tf.data.Dataset.zip((image_ds, label_ds))
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(512, seed=42)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)
print("    Pipeline ready.\n")

# =============================================================================
# 3. Build critic and generator
# =============================================================================
print(">>> Building Critic ...")
print("-" * 70)

# Critic: image (64,64,3)  +  spatial label map (64,64,2)  → score
ci = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="critic_image")
cl = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CLASSES),  name="critic_label_map")
x  = layers.Concatenate(axis=-1)([ci, cl])          # → (64,64,5)
x  = layers.Conv2D(64,  4, strides=2, padding="same")(x)
x  = layers.LeakyReLU(0.2)(x)
x  = layers.Conv2D(128, 4, strides=2, padding="same")(x)
x  = layers.LeakyReLU(0.2)(x);   x = layers.Dropout(0.3)(x)
x  = layers.Conv2D(128, 4, strides=2, padding="same")(x)
x  = layers.LeakyReLU(0.2)(x);   x = layers.Dropout(0.3)(x)
x  = layers.Conv2D(128, 4, strides=2, padding="same")(x)
x  = layers.LeakyReLU(0.2)(x);   x = layers.Dropout(0.3)(x)
x  = layers.Conv2D(1,   4, strides=1, padding="valid")(x)
co = layers.Flatten()(x)
critic = models.Model([ci, cl], co, name="Critic")
critic.summary()

print("\n>>> Building Generator ...")
print("-" * 70)

# Generator: latent vector (32,)  +  flat one-hot (2,)  → image (64,64,3)
gi = layers.Input(shape=(Z_DIM,),   name="gen_latent")
gl = layers.Input(shape=(CLASSES,), name="gen_label")
x  = layers.Concatenate(axis=-1)([gi, gl])          # → (34,)
x  = layers.Reshape((1, 1, Z_DIM + CLASSES))(x)
x  = layers.Conv2DTranspose(128, 4, strides=1, padding="valid",  use_bias=False)(x)
x  = layers.BatchNormalization(momentum=0.9)(x); x = layers.LeakyReLU(0.2)(x)
x  = layers.Conv2DTranspose(128, 4, strides=2, padding="same",   use_bias=False)(x)
x  = layers.BatchNormalization(momentum=0.9)(x); x = layers.LeakyReLU(0.2)(x)
x  = layers.Conv2DTranspose(128, 4, strides=2, padding="same",   use_bias=False)(x)
x  = layers.BatchNormalization(momentum=0.9)(x); x = layers.LeakyReLU(0.2)(x)
x  = layers.Conv2DTranspose(64,  4, strides=2, padding="same",   use_bias=False)(x)
x  = layers.BatchNormalization(momentum=0.9)(x); x = layers.LeakyReLU(0.2)(x)
go = layers.Conv2DTranspose(CHANNELS, 4, strides=2, padding="same", activation="tanh")(x)
generator = models.Model([gi, gl], go, name="Generator")
generator.summary()

# =============================================================================
# 4. Conditional WGAN-GP
# =============================================================================

class ConditionalWGAN(models.Model):
    """
    Conditional WGAN-GP conditioned on Blond_Hair attribute.

    *** Key train_step change – matching input formats ***

    The dataset yields one_hot_labels of shape (batch, CLASSES=2).

    - Generator expects FLAT one-hot  : (batch, 2)
    - Critic expects a SPATIAL map    : (batch, 64, 64, 2)

    The train_step builds image_one_hot_labels by tiling the flat vector
    across spatial dimensions (H and W) before passing it to the critic:

        image_one_hot_labels = one_hot_labels[:, None, None, :]
        image_one_hot_labels = tf.repeat(..., IMAGE_SIZE, axis=1)   # tile H
        image_one_hot_labels = tf.repeat(..., IMAGE_SIZE, axis=2)   # tile W

    The flat one_hot_labels are fed to the generator unchanged.
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
        self.c_optimizer        = c_optimizer
        self.g_optimizer        = g_optimizer
        self.c_wass_loss_metric = metrics.Mean(name="c_wass_loss")
        self.c_gp_metric        = metrics.Mean(name="c_gp")
        self.c_loss_metric      = metrics.Mean(name="c_loss")
        self.g_loss_metric      = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.c_loss_metric, self.c_wass_loss_metric,
                self.c_gp_metric,  self.g_loss_metric]

    def gradient_penalty(self, batch_size, real_images, fake_images, image_one_hot_labels):
        alpha        = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = real_images + alpha * (fake_images - real_images)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic([interpolated, image_one_hot_labels], training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm  = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, data):
        real_images, one_hot_labels = data

        # ------------------------------------------------------------------ #
        #  INPUT FORMAT CHANGE                                                 #
        #  one_hot_labels shape : (batch, 2)                                  #
        #                                                                      #
        #  Generator  → needs flat labels      : (batch, 2)  [unchanged]      #
        #  Critic     → needs spatial label map: (batch, 64, 64, 2)           #
        #                                                                      #
        #  Build image_one_hot_labels by adding H,W dims and tiling:          #
        # ------------------------------------------------------------------ #
        image_one_hot_labels = one_hot_labels[:, None, None, :]              # (B,1,1,2)
        image_one_hot_labels = tf.repeat(image_one_hot_labels, IMAGE_SIZE, 1) # (B,64,1,2)
        image_one_hot_labels = tf.repeat(image_one_hot_labels, IMAGE_SIZE, 2) # (B,64,64,2)

        batch_size = tf.shape(real_images)[0]

        # ---- Train critic ------------------------------------------------- #
        for _ in range(self.critic_steps):
            z = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator(
                    [z, one_hot_labels], training=True)          # flat labels
                fake_pred   = self.critic(
                    [fake_images,  image_one_hot_labels], training=True)  # spatial
                real_pred   = self.critic(
                    [real_images,  image_one_hot_labels], training=True)
                c_wass = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
                c_gp   = self.gradient_penalty(
                    batch_size, real_images, fake_images, image_one_hot_labels)
                c_loss = c_wass + c_gp * self.gp_weight
            cg = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(cg, self.critic.trainable_variables))

        # ---- Train generator --------------------------------------------- #
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            fake_images = self.generator([z, one_hot_labels], training=True)
            fake_pred   = self.critic([fake_images, image_one_hot_labels], training=True)
            g_loss      = -tf.reduce_mean(fake_pred)
        gg = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gg, self.generator.trainable_variables))

        self.c_loss_metric.update_state(c_loss)
        self.c_wass_loss_metric.update_state(c_wass)
        self.c_gp_metric.update_state(c_gp)
        self.g_loss_metric.update_state(g_loss)
        return {m.name: m.result() for m in self.metrics}


# =============================================================================
# 5. Compile and callbacks
# =============================================================================
cgan = ConditionalWGAN(
    critic=critic, generator=generator,
    latent_dim=Z_DIM, critic_steps=CRITIC_STEPS, gp_weight=GP_WEIGHT,
)
cgan.compile(
    c_optimizer=optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2),
    g_optimizer=optimizers.Adam(learning_rate=LR, beta_1=BETA_1, beta_2=BETA_2),
)

CKPT_PATH = os.path.join(BASE_DIR, "checkpoint", "checkpoint.ckpt")
ckpt_cb   = callbacks.ModelCheckpoint(
    filepath=CKPT_PATH, save_weights_only=True, save_freq="epoch", verbose=0)


def save_image_grid(images_nhwc, path, label_str):
    """Save a row of images as a single PNG using PIL only (no matplotlib)."""
    images_nhwc = np.clip(images_nhwc, 0, 255).astype(np.uint8)
    n, h, w, c  = images_nhwc.shape
    grid        = Image.new("RGB", (w * n, h))
    for i, img in enumerate(images_nhwc):
        grid.paste(Image.fromarray(img), (i * w, 0))
    grid.save(path)
    print(f"    Saved: {os.path.basename(path)}  [{label_str}]")


class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img, latent_dim):
        self.num_img    = num_img
        self.latent_dim = latent_dim
        self.z_fixed    = tf.random.normal(shape=(num_img, latent_dim))

    def on_epoch_end(self, epoch, logs=None):
        for vec, tag, label_str in [
            ([1, 0], "non_blond", "Non-Blond (label=0)"),
            ([0, 1], "blond",     "Blond     (label=1)"),
        ]:
            lbl  = np.repeat([vec], self.num_img, axis=0)
            imgs = self.model.generator([self.z_fixed, lbl], training=False).numpy()
            imgs = (imgs * 127.5 + 127.5)
            save_image_grid(
                imgs,
                os.path.join(OUTPUT_DIR, f"epoch_{epoch+1:03d}_{tag}.png"),
                label_str,
            )
        if logs:
            print(
                f"  c_loss={logs.get('c_loss',0):.4f}  "
                f"c_wass={logs.get('c_wass_loss',0):.4f}  "
                f"c_gp={logs.get('c_gp',0):.4f}  "
                f"g_loss={logs.get('g_loss',0):.4f}"
            )

img_gen_cb = ImageGenerator(num_img=8, latent_dim=Z_DIM)

# =============================================================================
# 6. Train
# =============================================================================
print("\n" + "=" * 70)
print("  Training")
print("=" * 70)
t0 = time.time()

history = cgan.fit(train, epochs=EPOCHS, callbacks=[ckpt_cb, img_gen_cb], verbose=1)

elapsed = time.time() - t0
print(f"\nTraining completed in {elapsed:.1f}s")

# =============================================================================
# 7. Loss table
# =============================================================================
print("\n" + "=" * 70)
print("  Per-epoch loss summary")
print("=" * 70)
print(f"{'Epoch':>6}  {'c_loss':>10}  {'c_wass_loss':>12}  {'c_gp':>8}  {'g_loss':>10}")
print("-" * 56)
for ep in range(EPOCHS):
    print(
        f"{ep+1:>6}  "
        f"{history.history['c_loss'][ep]:>10.4f}  "
        f"{history.history['c_wass_loss'][ep]:>12.4f}  "
        f"{history.history['c_gp'][ep]:>8.4f}  "
        f"{history.history['g_loss'][ep]:>10.4f}"
    )

# =============================================================================
# 8. Final side-by-side comparison
# =============================================================================
print("\n>>> Generating final comparison (same latent z, two labels) ...")
z_fixed  = np.random.normal(size=(8, Z_DIM)).astype(np.float32)

label_0  = np.repeat([[1, 0]], 8, axis=0)
label_1  = np.repeat([[0, 1]], 8, axis=0)

imgs_0   = cgan.generator.predict([z_fixed, label_0], verbose=0)
imgs_0   = (imgs_0 * 127.5 + 127.5)
imgs_1   = cgan.generator.predict([z_fixed, label_1], verbose=0)
imgs_1   = (imgs_1 * 127.5 + 127.5)

# Stack non-blond row on top of blond row
both     = np.clip(np.concatenate([imgs_0, imgs_1], axis=0), 0, 255).astype(np.uint8)
n, h, w, _ = both.shape
grid     = Image.new("RGB", (w * 8, h * 2))
for i in range(8):
    grid.paste(Image.fromarray(both[i]),     (i * w, 0))      # row 0: non-blond
    grid.paste(Image.fromarray(both[i + 8]), (i * w, h))      # row 1: blond
comp_path = os.path.join(OUTPUT_DIR, "final_comparison.png")
grid.save(comp_path)
print(f"    Saved: final_comparison.png  (top=Non-Blond, bottom=Blond)")

# =============================================================================
# 9. Simple loss curve image (using PIL – no matplotlib)
# =============================================================================
print("\n>>> Saving loss curve plot ...")
try:
    import math

    def make_curve_png(series_dict, title, path, width=600, height=300):
        """Draw a minimal line chart with PIL."""
        pad    = 50
        W, H   = width + 2*pad, height + 2*pad
        img    = Image.new("RGB", (W, H), "white")
        pix    = img.load()

        all_vals = [v for s in series_dict.values() for v in s]
        mn, mx   = min(all_vals), max(all_vals)
        rng_v    = mx - mn if mx != mn else 1.0
        n_pts    = len(next(iter(series_dict.values())))

        colors = [(31,119,180),(255,127,14),(44,160,44),(214,39,40)]

        for (name, series), color in zip(series_dict.items(), colors):
            pts = []
            for i, v in enumerate(series):
                px = pad + int(i / max(n_pts - 1, 1) * width)
                py = pad + height - int((v - mn) / rng_v * height)
                pts.append((px, py))
            for a, b in zip(pts, pts[1:]):
                # Bresenham line
                x0,y0,x1,y1 = a[0],a[1],b[0],b[1]
                dx,dy = abs(x1-x0), abs(y1-y0)
                sx    = 1 if x0<x1 else -1
                sy    = 1 if y0<y1 else -1
                err   = dx - dy
                while True:
                    if 0<=x0<W and 0<=y0<H: pix[x0,y0] = color
                    if x0==x1 and y0==y1: break
                    e2 = 2*err
                    if e2 > -dy: err -= dy; x0 += sx
                    if e2 <  dx: err += dx; y0 += sy

        img.save(path)

    make_curve_png(
        {"c_loss"      : history.history["c_loss"],
         "c_wass_loss" : history.history["c_wass_loss"],
         "c_gp"        : history.history["c_gp"]},
        "Critic Losses",
        os.path.join(OUTPUT_DIR, "critic_loss_curve.png"),
    )
    make_curve_png(
        {"g_loss": history.history["g_loss"]},
        "Generator Loss",
        os.path.join(OUTPUT_DIR, "generator_loss_curve.png"),
    )
    print("    Saved: critic_loss_curve.png, generator_loss_curve.png")
except Exception as e:
    print(f"    (curve plot skipped: {e})")

# =============================================================================
# 10. Artefact summary
# =============================================================================
print("\n" + "=" * 70)
print("  Output artefacts  →  Week 3/Project 1/output/")
print("=" * 70)
for f in sorted(os.listdir(OUTPUT_DIR)):
    p = os.path.join(OUTPUT_DIR, f)
    print(f"    {f:<45}  {os.path.getsize(p):>8,} bytes")

print("\nDone.")
sys.stdout = sys.__stdout__
log_fh.close()
print(f"\nFull log saved → {LOG_FILE}")
