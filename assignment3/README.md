# Assignment 3: Autoencoders on Fashion-MNIST

Based on **Generative Deep Learning 2nd Edition** by David Foster:
- [autoencoder.ipynb](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/03_vae/01_autoencoder/autoencoder.ipynb)

## Setup

```bash
pip install -r requirements.txt
```

## Contents

1. **Build** encoder and decoder (convolutional architecture)
2. **Train** on Fashion-MNIST
3. **Test** reconstruction quality
4. **Submission**: 5 chosen images with discrepancy descriptions

## Run

**Option 1 - Notebook:**
```bash
jupyter notebook autoencoder_assignment.ipynb
# or
jupyter lab autoencoder_assignment.ipynb
```

**Option 2 - Script (saves images to outputimage/):**
```bash
python3 run_autoencoder.py
```

## Output Images

The `outputimage/` folder contains:
- `01_sample_training.png` - Sample Fashion-MNIST images
- `02_training_history.png` - Training loss curve
- `03_original_vs_reconstructed.png` - First 10 test images comparison
- `04_submission_5_images.png` - 5 chosen images (original vs reconstructed)
- `05_image_1_index_0.png` through `05_image_5_index_2500.png` - Individual comparisons
