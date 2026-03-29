# Project 2: Generate Images with PixelCNN (Mixture Distribution)

## Overview
This project implements a **PixelCNN** with a **mixture of logistic distributions** output using `tfp.distributions.PixelCNN` from TensorFlow Probability. The model learns the joint pixel distribution autoregressively and generates new Fashion MNIST images pixel-by-pixel via ancestral sampling.

## How to Run

**Requirements:** Python 3.11, TensorFlow 2.16.2, TensorFlow Probability 0.24.0

```bash
pip install tensorflow==2.16.2 tensorflow-probability==0.24.0 tf-keras==2.16.0 matplotlib numpy
python pixelcnn_tf.py
```

Generated images and plots are saved to the `output tf/` directory.

**Note:** TensorFlow 2.20+ has a known segmentation fault on Python 3.13 with Apple Silicon. The compatible versions above (TF 2.16.2 + TFP 0.24.0 + Python 3.11) were used to ensure stable execution.

## GPU vs CPU Justification

**The assignment requests GPU execution for faster processing.**

The code runs on **Apple Metal GPU (M2)** via TensorFlow's Metal plugin (`tensorflow-metal`). GPU detection is automatic and printed at startup:

```
GPU detected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
Metal device set to: Apple M2
```

The legacy Adam optimizer (`tf.keras.optimizers.legacy.Adam`) is used because TensorFlow's newer Adam optimizer runs slowly on M1/M2 Macs, as noted in TensorFlow's own warning message.

| Aspect | Detail |
|--------|--------|
| GPU used | Apple Metal (M2) |
| Optimizer | Legacy Adam (recommended by TF for Apple Silicon) |
| Training time | ~38 min for 5 epochs |
| Sampling time | ~78-94s for 3-10 images |

## Model Architecture

- **Model**: `tfp.distributions.PixelCNN` (TensorFlow Probability)
- **Dataset**: Fashion MNIST (28x28), pixel values in [0, 255]
- **Mixture Components** (`num_logistic_mix`): 5 logistic distributions per pixel
- **ResNet Blocks** (`num_resnet`): 1 per hierarchy
- **Hierarchies** (`num_hierarchies`): 2 resolution levels
- **Filters** (`num_filters`): 32 convolutional filters
- **Dropout** (`dropout_p`): 0.3
- **Optimizer**: Adam (legacy, lr=0.001)
- **Epochs**: 5
- **Loss**: Negative log-likelihood (`-tf.reduce_mean(log_prob)`)
- **Total parameters**: 625,026

## Why Mixture of Logistics?

Traditional PixelCNN uses a 256-way categorical distribution per pixel, which:
- Requires 256 output channels per pixel
- Treats nearby values (e.g., 127 and 128) as unrelated classes
- Is parameter-inefficient

The **mixture of logistic distributions** (from PixelCNN++, as described on p.162 of the textbook):
- Uses only K=5 components, each with mean, scale, and weight parameters
- Naturally captures that neighboring pixel values are similar
- Provides smoother gradients during training
- Is far more parameter-efficient
- Discretizes the continuous logistic CDF to compute P(x) for integer pixel values

## Output Interpretation

### Training Loss (Negative Log-Likelihood)
- NLL decreased from **1837** (epoch 1) to **1615** (epoch 5)
- Test NLL decreased from **1678** to **1596**
- Train and test losses track closely — no overfitting observed

### Generated Images
- **Epoch 1**: Noisy images with faint structural patterns emerging
- **Epoch 5**: Recognizable fashion item shapes — shirts, trousers, bags, shoes visible
- Quality would improve further with more training epochs

### Autoregressive Sampling
Images are generated one pixel at a time, left-to-right, top-to-bottom using `dist.sample()`. Each pixel is sampled from its conditional mixture distribution given all previously generated pixels. This is slower than parallel generation (like GANs/VAEs) but ensures proper modeling of spatial dependencies.

## Output Files

| File | Description |
|------|-------------|
| `output tf/training_samples.png` | Sample real Fashion MNIST images |
| `output tf/generated_epoch_001.png` | Generated images after epoch 1 |
| `output tf/generated_epoch_005.png` | Generated images after epoch 5 |
| `output tf/training_loss.png` | Train and test NLL loss curves |
| `output tf/final_generated_images.png` | 10 generated images after full training |
| `output tf/real_vs_generated.png` | Side-by-side real vs generated comparison |
