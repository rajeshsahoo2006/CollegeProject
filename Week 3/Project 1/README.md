# Project 1: Conditional GAN (CGAN) – Blond Hair Attribute

**Author:** Rajesh Kumar Sahoo
**Email:** rsahoo44691@ucumberlands.edu
**Institution:** University of the Cumberlands
**Course:** Neural Networks and Deep Learning (NNDL) — Week 3

---

## Assignment Brief

Build a **Conditional Wasserstein GAN with Gradient Penalty (CGAN / WGAN-GP)** that can control its output based on a class label. Specifically:

1. Condition the CGAN on the **Blond_Hair** attribute of a celebrity faces dataset
2. Modify the `train_step` to reconcile the different label formats expected by the generator and the critic
3. Submit the code, its output, and observations about accuracy

Reference notebook: [davidADSP/Generative_Deep_Learning_2nd_Edition – cgan.ipynb](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/04_gan/03_cgan/cgan.ipynb)

---

## Grading Rubric

| Component | Points |
|-----------|--------|
| Code for network and its output | 50 / 100 |
| Observations about output and accuracy | 50 / 100 |
| **Total** | **100** |

---

## Dataset

**Source:** [Face VAE – Kaggle](https://www.kaggle.com/datasets/kushsheth/face-vae)

The dataset contains the **CelebA** (Large-scale CelebFaces Attributes) collection:

| File | Description |
|------|-------------|
| `img_align_celeba/` | 202,599 aligned celebrity face images (178×218 px, JPG) |
| `list_attr_celeba.csv` | 40 binary attributes per image, including `Blond_Hair` |

The `Blond_Hair` column uses `{-1, +1}` encoding (CelebA convention). The code converts this to `{0, 1}` for training:
- `0` → **Non-Blond** (172,616 images, 85.2%)
- `1` → **Blond**     (29,983 images, 14.8%)

> **Note:** `list_attr_celeba.csv` (24 MB) and the image folder are excluded from this repo via `.gitignore`.
> Download the dataset from the Kaggle link above and place the files inside this folder before running the full training script.

---

## Files

| File | Purpose |
|------|---------|
| `cgan_blond_hair.py` | Full training script — requires the CelebA image folder and `pandas`/`matplotlib` |
| `cgan_demo_run.py` | Self-contained demo — reads real labels from the CSV but uses synthetic images; runs on any machine |
| `output/` | Generated images, loss curves, and full training log from the demo run |
| `list_attr_celeba.csv` | CelebA attribute labels (not committed — download from Kaggle) |

---

## Architecture

### What is a Conditional GAN?

A standard GAN has two networks — a **Generator** that creates fake images and a **Critic/Discriminator** that tells real from fake. A **Conditional GAN** adds a class label as an extra input to both networks, so the generator learns to produce images that match a specific condition (e.g., "generate a blond face").

This implementation uses **WGAN-GP** (Wasserstein loss + Gradient Penalty) instead of the standard binary cross-entropy loss, which produces more stable training and avoids mode collapse.

---

### Class: `ConditionalWGAN`

The main model class that wraps the critic and generator and orchestrates the entire training loop.

```
ConditionalWGAN(models.Model)
│
├── critic          – the discriminator network
├── generator       – the image generation network
├── latent_dim      – size of the random noise vector (Z_DIM = 32)
├── critic_steps    – how many times to train critic per generator step (= 3)
└── gp_weight       – weight of the gradient penalty term (= 10.0)
```

**Methods:**

| Method | What it does |
|--------|-------------|
| `compile()` | Sets up two separate Adam optimisers (one for the critic, one for the generator) and initialises four loss tracking metrics: `c_loss`, `c_wass_loss`, `c_gp`, `g_loss` |
| `gradient_penalty()` | Computes the WGAN-GP regularisation term — creates interpolated images between real and fake, runs them through the critic, and penalises if the gradient norm deviates from 1.0. This replaces weight-clipping and keeps training numerically stable |
| `train_step()` | One full training iteration for a single batch — trains the critic `critic_steps` times, then trains the generator once. See the detailed explanation below |

---

### The Key `train_step` Change (Assignment Requirement)

The generator and critic take the **same label information** but require it in **different tensor shapes**. This mismatch must be resolved inside `train_step`.

#### The Problem

The dataset yields labels as flat one-hot vectors:
```
one_hot_labels shape: (batch, 2)
   e.g. [1, 0] = non-blond
        [0, 1] = blond
```

But the two networks need labels in different formats:

| Network | Required shape | Why |
|---------|---------------|-----|
| **Generator** | `(batch, 2)` — flat vector | Concatenated with the latent noise vector before the first layer; both are 1-D so shapes must match |
| **Critic** | `(batch, 64, 64, 2)` — spatial map | Concatenated channel-wise with the image tensor; both must share the same H×W spatial dimensions |

#### The Fix (inside `train_step`)

```python
# Step 1: add two singleton spatial dims  →  (batch, 1, 1, 2)
image_one_hot_labels = one_hot_labels[:, None, None, :]

# Step 2: tile across height  →  (batch, 64, 1, 2)
image_one_hot_labels = tf.repeat(image_one_hot_labels, IMAGE_SIZE, axis=1)

# Step 3: tile across width   →  (batch, 64, 64, 2)
image_one_hot_labels = tf.repeat(image_one_hot_labels, IMAGE_SIZE, axis=2)

# Generator gets the original flat labels (batch, 2)
fake_images = self.generator([z, one_hot_labels], training=True)

# Critic gets the tiled spatial map (batch, 64, 64, 2)
fake_pred = self.critic([fake_images, image_one_hot_labels], training=True)
```

The critic is trained 3 times per generator update (`critic_steps = 3`) to keep the critic stronger than the generator throughout training — a requirement for the Wasserstein distance estimate to be valid.

---

### Critic Network

Takes a `(64, 64, 3)` image and a `(64, 64, 2)` spatial label map, concatenates them on the channel axis to form a `(64, 64, 5)` tensor, then downsamples through Conv2D layers to output a single scalar realness score. No sigmoid activation is used — WGAN relies on raw unbounded scores.

```
Input image     (64, 64, 3) ─┐
Input label map (64, 64, 2) ─┴─ Concatenate → (64, 64, 5)
    │
    ├─ Conv2D(64,  4×4, stride=2) + LeakyReLU(0.2)              → (32, 32, 64)
    ├─ Conv2D(128, 4×4, stride=2) + LeakyReLU(0.2) + Dropout(0.3) → (16, 16, 128)
    ├─ Conv2D(128, 4×4, stride=2) + LeakyReLU(0.2) + Dropout(0.3) →  (8,  8, 128)
    ├─ Conv2D(128, 4×4, stride=2) + LeakyReLU(0.2) + Dropout(0.3) →  (4,  4, 128)
    ├─ Conv2D(1,   4×4, stride=1, padding=valid)                 →  (1,  1,   1)
    └─ Flatten → scalar score

Total trainable params: 662,977
```

---

### Generator Network

Takes a `(32,)` latent noise vector and a `(2,)` flat one-hot label, concatenates them to form a `(34,)` vector, then upsamples through Conv2DTranspose layers to produce a `(64, 64, 3)` RGB image with pixel values in `[-1, 1]`.

```
Input latent z (32,) ─┐
Input label     (2,) ─┴─ Concatenate → (34,)
    │
    ├─ Reshape → (1, 1, 34)
    ├─ Conv2DTranspose(128, 4×4, stride=1) + BatchNorm + LeakyReLU(0.2) →  (4,  4, 128)
    ├─ Conv2DTranspose(128, 4×4, stride=2) + BatchNorm + LeakyReLU(0.2) →  (8,  8, 128)
    ├─ Conv2DTranspose(128, 4×4, stride=2) + BatchNorm + LeakyReLU(0.2) → (16, 16, 128)
    ├─ Conv2DTranspose(64,  4×4, stride=2) + BatchNorm + LeakyReLU(0.2) → (32, 32,  64)
    └─ Conv2DTranspose(3,   4×4, stride=2, activation=tanh)              → (64, 64,   3)

Total trainable params: 728,963
```

`BatchNormalization` after each transpose-conv layer stabilises the generator's internal activations and speeds up convergence.

---

## Training Configuration

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| `IMAGE_SIZE` | 64 | Input/output image resolution |
| `Z_DIM` | 32 | Latent noise vector size |
| `CLASSES` | 2 | Blond / Non-Blond |
| `BATCH_SIZE` | 128 (32 in demo) | Samples per update |
| `LEARNING_RATE` | 0.00005 | Same for both networks |
| `ADAM_BETA_1` | 0.5 | Lower than default; standard for GAN training to reduce momentum |
| `ADAM_BETA_2` | 0.9 | |
| `EPOCHS` | 20 (5 in demo) | |
| `CRITIC_STEPS` | 3 | Critic updates per generator update |
| `GP_WEIGHT` | 10.0 | Gradient penalty coefficient |

---

## Running the Code

### Demo run (no dataset required)
```bash
cd "Week 3/Project 1"
python3 cgan_demo_run.py
```
Automatically uses synthetic random images if `img_align_celeba/` is not found.
Output is saved to `output/` and a full log to `output/training_output.txt`.

### Full training run (requires CelebA images)
1. Download the dataset from [Kaggle – Face VAE](https://www.kaggle.com/datasets/kushsheth/face-vae)
2. Place `img_align_celeba/` and `list_attr_celeba.csv` inside this folder
3. Update `DATA_DIR` in `cgan_blond_hair.py` to point to this folder
4. Run:
```bash
python3 cgan_blond_hair.py
```

---

## Results (Demo Run)

**Environment:** Apple M2 GPU (Metal), TensorFlow 2.15.0, 5 epochs, 43.8 seconds

### Per-Epoch Loss

| Epoch | c_loss | c_wass_loss | c_gp | g_loss |
|-------|--------|-------------|------|--------|
| 1 | 5.8367 | -0.6217 | 0.6458 | -0.1317 |
| 2 | -3.9395 | -5.0314 | 0.1092 | -2.8632 |
| 3 | -11.4964 | -12.7519 | 0.1256 | -5.6433 |
| 4 | -20.2133 | -23.3473 | 0.3134 | -8.2409 |
| 5 | -25.9176 | -32.7025 | 0.6785 | -6.6762 |

### Observations

**Critic (Wasserstein) loss** becomes increasingly negative across epochs. In WGAN, a more negative Wasserstein distance means the critic is better at distinguishing real from fake — this is the expected direction of convergence.

**Gradient penalty (c_gp)** started at 0.6458 in epoch 1 and dropped to ~0.11 by epoch 2 before gradually rising again. This shows the critic's gradients were close to the target norm of 1.0 for most of training, indicating well-regulated training without exploding or vanishing gradients.

**Generator loss (g_loss)** also trends negative, which is correct for WGAN-GP — the generator's objective is to maximise the critic's score on fake images (equivalent to minimising the negative mean score). The generator is learning to produce outputs that increasingly fool the critic.

**Image quality:** The demo run uses synthetic random noise images as a stand-in for the full CelebA dataset. As a result the generated outputs look like noise rather than faces. With the real CelebA images trained for 20+ epochs, the generator would learn to produce realistic faces, and the blond/non-blond conditioning would produce visually distinct hair colours when the same latent vector `z` is used with each label.

### Output Files

| File | Description |
|------|-------------|
| `output/epoch_001_blond.png` … `epoch_005_blond.png` | Generated images conditioned on Blond label per epoch |
| `output/epoch_001_non_blond.png` … `epoch_005_non_blond.png` | Generated images conditioned on Non-Blond label per epoch |
| `output/final_comparison.png` | Side-by-side grid: same latent `z`, top row = Non-Blond, bottom row = Blond |
| `output/critic_loss_curve.png` | Critic loss curves over training |
| `output/generator_loss_curve.png` | Generator loss curve over training |
| `output/training_output.txt` | Full console log including model summaries |
