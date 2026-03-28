# Assignment 6: Energy-Based Models (EBM) with Contrastive Divergence

**Author:** Rajesh Sahoo
**Email:** rsahoo44691@ucumberlands.edu
**Institution:** University of the Cumberlands
**Course:** Neural Networks and Deep Learning (NNDL) — Week 4

---

## Assignment Brief

Build an **Energy-Based Model (EBM)** that learns to generate handwritten digits from the MNIST dataset using **contrastive divergence** training and **Langevin dynamics** sampling. Specifically:

1. Build the neural network that represents the **energy function**
2. Perform the training step of the **contrastive divergence algorithm** within a custom model for **120 epochs**
3. Submit the loss curve and compare results with the textbook

Reference notebook: [davidADSP/Generative_Deep_Learning_2nd_Edition — ebm.ipynb](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/07_ebm/01_ebm/ebm.ipynb)

---

## Grading Rubric

| Component | Points |
|-----------|--------|
| Code for network and its output | 50 / 100 |
| Loss curve and comparison with book results | 50 / 100 |
| **Total** | **100** |

---

## Why PyTorch Instead of Keras/TensorFlow?

| Factor | TensorFlow/Keras | PyTorch |
|--------|-------------------|---------|
| **Apple Silicon GPU** | `tensorflow-metal` unavailable for Python 3.13; TF 2.20 segfaults | PyTorch MPS backend works natively on Apple Silicon |
| **GPU acceleration** | Not functional in this environment | Full MPS GPU — all data, model, and buffer on GPU |
| **Data transfers** | Standard DataLoader moves batches CPU→GPU each step | Entire MNIST dataset preloaded to GPU — zero transfers |
| **Langevin gradients** | `tf.GradientTape` per step | `torch.autograd.grad` (functional API) — no graph accumulation overhead |
| **Training speed** | CPU fallback would be 5-10x slower | 120 epochs completed in **134 minutes** on MPS GPU |

> **Bottom line:** PyTorch with MPS provides reliable, fast GPU training on macOS. The entire training pipeline — data, replay buffer, model, and Langevin sampling — runs on GPU with zero CPU↔GPU transfers.

---

## Dataset

**Source:** `torchvision.datasets.MNIST`

| Parameter | Value |
|-----------|-------|
| Training samples | 60,000 |
| Image size | 28×28 → padded to 32×32 |
| Channels | 1 (grayscale) |
| Normalization | `[-1, 1]` range |
| Storage | Entire dataset preloaded to MPS GPU |

---

## Files

| File | Purpose |
|------|---------|
| `Assignment6_EBM.py` | Full training script — EBM with PyTorch on MPS GPU |
| `Instruction` | Original assignment instructions |
| `output/` | All generated images and loss curves |

### Output Files

| File | Description |
|------|-------------|
| `output/training_samples.png` | MNIST training samples for reference |
| `output/loss_curves_120epochs.png` | 2×2 grid: total loss, cdiv loss, energy scores, regularization |
| `output/final_generated.png` | Final generated digits (1000 Langevin steps) |
| `output/starting_noise.png` | Random noise before Langevin sampling |
| `output/langevin_progression.png` | Step-by-step progression: noise → digit |
| `output/generated_epoch_*.png` | Generated samples at epochs 0, 29, 59, 89, 119 |
| `output/buffer_epoch_*.png` | Replay buffer samples at key epochs |

---

## Architecture

### What is an Energy-Based Model?

An **Energy-Based Model** assigns a scalar energy value to each input. Low energy = high probability (real data), high energy = low probability (noise/fake). The model learns an energy landscape where real data sits in low-energy valleys.

Unlike GANs (which use a separate generator), EBMs generate new samples by starting from random noise and iteratively moving toward low-energy regions using **Langevin dynamics** (gradient-based MCMC sampling).

---

### Energy Function Network

A CNN that maps a 32×32×1 grayscale image to a single scalar energy score:

```
Input (1, 32, 32)
    │
    ├─ Conv2d(1→16, 5×5, stride=2, pad=2) + SiLU   → (16, 16, 16)
    ├─ Conv2d(16→32, 3×3, stride=2, pad=1) + SiLU   → (32, 8, 8)
    ├─ Conv2d(32→64, 3×3, stride=2, pad=1) + SiLU   → (64, 4, 4)
    ├─ Conv2d(64→64, 3×3, stride=2, pad=1) + SiLU   → (64, 2, 2)
    ├─ Flatten                                        → (256)
    ├─ Linear(256→64) + SiLU                          → (64)
    └─ Linear(64→1)                                   → scalar energy

Total trainable parameters: 76,993
```

**SiLU (Swish)** activation: `x * sigmoid(x)` — smooth, non-monotonic, better gradient flow than ReLU.

---

### Langevin Dynamics Sampler

Generates samples by iteratively refining random noise using the energy function's gradient:

```
For each step:
    1. x = x + N(0, noise)                    # Add small Gaussian noise
    2. x = clamp(x, -1, 1)                    # Keep in valid range
    3. grad = d/dx [E(x)]                     # Gradient of energy w.r.t. input
    4. grad = clamp(grad, -clip, +clip)        # Prevent explosion
    5. x = x + step_size * grad               # Move toward high-energy region
    6. x = clamp(x, -1, 1)
```

- **Training:** 60 Langevin steps per batch (refine buffer samples)
- **Generation:** 1000 Langevin steps (produce final images from random noise)
- Uses `torch.autograd.grad` (functional API) — avoids `.backward()` graph accumulation

---

### Replay Buffer (GPU-Resident)

Stabilizes training by maintaining a pool of previously generated samples on the GPU:

```
GPUBuffer (stored as single GPU tensor, up to 8192 samples)
    │
    ├── Each batch: ~5% fresh random noise + ~95% from buffer
    ├── Refine via 60 Langevin steps
    └── Push refined samples back into buffer
```

This avoids running full Langevin chains from scratch every iteration and keeps all data on the GPU to eliminate CPU↔GPU transfers.

---

### Contrastive Divergence Training

The loss function has two components:

```
cdiv_loss = mean(E(fake)) - mean(E(real))     # Push real energy up, fake energy down
reg_loss  = alpha * mean(E(real)^2 + E(fake)^2)  # Prevent score divergence
total_loss = cdiv_loss + reg_loss
```

The model learns to assign **higher scores** to real data and **lower scores** to generated/fake data. The regularization term prevents the energy scores from growing unboundedly.

---

## Training Configuration

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| `IMAGE_SIZE` | 32 | MNIST padded from 28 |
| `STEP_SIZE` | 10 | Langevin step size |
| `STEPS` | 60 | Langevin steps per training batch |
| `NOISE` | 0.005 | Langevin noise stddev |
| `GRADIENT_CLIP` | 0.03 | Langevin gradient clipping |
| `ALPHA` | 0.1 | Regularization weight |
| `BATCH_SIZE` | 128 | |
| `BUFFER_SIZE` | 8192 | GPU-resident replay buffer |
| `LEARNING_RATE` | 0.0001 | Adam optimizer |
| `EPOCHS` | 120 | 2× the textbook (60) |
| `Device` | Apple MPS GPU | All tensors on GPU |

---

## Running the Code

```bash
cd "Week 4/Assignment 6"
python3 Assignment6_EBM.py
```

**Requirements:**
```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

The script automatically detects and uses the best available device (MPS GPU > CUDA GPU > CPU). The entire MNIST dataset is preloaded to GPU memory.

---

## Results

**Environment:** Apple MPS GPU, PyTorch 2.11.0, Python 3.13.5, 120 epochs
**Total training time:** 134.2 minutes

### Training Loss Per Epoch

| Epoch | Total Loss | Cdiv Loss | Reg Loss | Real Score | Fake Score |
|-------|-----------|-----------|----------|------------|------------|
| 0 | -2.3513 | -4.5843 | 2.2330 | 2.4993 | -2.0850 |
| 10 | -4.9931 | -9.9888 | 4.9957 | 4.9956 | -4.9932 |
| 20 | -4.9995 | -9.9994 | 4.9999 | 4.9997 | -4.9998 |
| 30 | -4.9999 | -9.9999 | 5.0000 | 4.9999 | -5.0000 |
| 40 | -5.0000 | -10.0000 | 5.0000 | 5.0000 | -5.0000 |
| 50 | -5.0000 | -10.0000 | 5.0000 | 5.0000 | -5.0000 |
| 60 | -4.9994 | -9.9990 | 4.9996 | 4.9995 | -4.9995 |
| 70 | -5.0000 | -10.0000 | 5.0000 | 5.0000 | -5.0000 |
| 80 | -5.0000 | -10.0000 | 5.0000 | 5.0000 | -5.0000 |
| 90 | -5.0000 | -10.0000 | 5.0000 | 5.0000 | -5.0000 |
| 100 | -4.9989 | -9.9976 | 4.9987 | 4.9988 | -4.9988 |
| 110 | -5.0000 | -10.0000 | 5.0000 | 5.0000 | -5.0000 |
| 119 | -5.0000 | -10.0000 | 5.0000 | 5.0000 | -5.0000 |

---

## Comparison with Book Results

| Aspect | Book (TensorFlow) | This Implementation (PyTorch) |
|--------|-------------------|-------------------------------|
| Framework | TensorFlow/Keras | PyTorch with MPS GPU |
| Epochs | 60 | 120 |
| GPU usage | CPU or TF-Metal | All data + buffer + model on MPS GPU |
| Langevin steps (train) | 60 | 60 |
| Langevin steps (gen) | 1000 | 1000 |
| Buffer | CPU list of tensors | Single GPU tensor (zero transfers) |
| All other hyperparameters | Identical | Identical |

---

## Observations

1. **Rapid convergence:** The model converges quickly within the first 10-20 epochs. The contrastive divergence loss drops from -4.58 (epoch 0) to -10.00 (epoch 20), and the energy scores saturate at ±5.0 due to the L2 regularization (alpha=0.1) which balances the cdiv term.

2. **Score equilibrium:** The real energy score stabilizes at +5.0 and fake at -5.0. This equilibrium is determined by the regularization weight — the `reg_loss = 0.1 * mean(real^2 + fake^2)` term penalizes large scores, creating a natural ceiling. The model correctly assigns higher energy to real data than to generated samples.

3. **Langevin progression:** The `langevin_progression.png` shows how random noise gradually transforms into recognizable digit shapes over 1000 steps. Early steps (0-10) show rapid structure formation, while later steps (100-999) refine details and sharpen edges.

4. **Replay buffer effect:** The buffer samples (`buffer_epoch_*.png`) show increasingly digit-like patterns across epochs, demonstrating that the buffer serves as a persistent memory of the learned energy landscape. By reusing ~95% buffer samples each batch, training avoids expensive full Langevin chains from pure noise.

5. **120 vs 60 epochs:** Since the model converges by epoch 20-30, the additional 60 epochs (beyond the book's 60) provide diminishing returns in terms of loss improvement. However, the extended training allows the replay buffer to accumulate higher-quality samples, which can improve the diversity and sharpness of generated digits.

6. **GPU acceleration:** By keeping the entire dataset, replay buffer, and model on the MPS GPU with zero CPU↔GPU transfers, training completed in 134 minutes (~49 seconds/epoch after warmup). The `torch.autograd.grad` functional API avoids computation graph accumulation during Langevin sampling, further improving GPU efficiency.
