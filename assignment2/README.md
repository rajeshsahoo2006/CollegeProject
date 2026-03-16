# Assignment 2: Convolutions - Stride Comparison

Based on **Generative Deep Learning 2nd Edition** by David Foster  
Source: [convolutions.ipynb](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/02_deeplearning/02_cnn/convolutions.ipynb)

## Setup

```bash
pip install -r requirements.txt
```

## Run the Notebook

```bash
jupyter notebook convolutions_assignment.ipynb
# or
jupyter lab convolutions_assignment.ipynb
```

## Contents

1. **Part 1**: Textbook convolution code (horizontal edge filter)
   - Stride 1 (full resolution)
   - Stride 2 (original textbook)
   - Stride 3 (custom, different from 2)

2. **Part 2**: Two CNN models on MNIST
   - Model 1: Stride=2 (original)
   - Model 2: Stride=3 (custom)

3. **Part 3**: Training both models (5 epochs)

4. **Part 4**: Evaluation and comparison

5. **Part 5**: Interpretation for submission
