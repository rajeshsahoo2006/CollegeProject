"""
Assignment 7: Variation of GPT Model (Transformer-based Text Generator)
========================================================================
Author: Rajesh Sahoo
Course: Neural Networks and Deep Learning
University: University of the Cumberlands

Based on: https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/09_transformer/gpt/gpt.ipynb

Steps:
  1. Load & clean the Wine Reviews dataset (Kaggle)
  2. Tokenize the text using TextVectorization
  3. Build a GPT-style transformer model with:
     - TokenAndPositionEmbedding (custom layer)
     - TransformerBlock with causal (autoregressive) attention
  4. Train the model on next-token prediction
  5. Generate text using temperature-based sampling at 0.3, 0.5, 1.0, 1.2
  6. Compare and interpret outputs across temperatures

Dataset: Wine Reviews (winemag-data-130k-v2.json) — download from Kaggle:
  https://www.kaggle.com/datasets/zynicide/wine-reviews
Place the file at: ./data/winemag-data-130k-v2.json
"""

import os
import json
import re
import string

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, losses

# ──────────────────────────────────────────────
# 0. Hyperparameters
# ──────────────────────────────────────────────
VOCAB_SIZE       = 10000          # Reduced from 20K (smaller softmax layer)
MAX_LEN          = 60             # Reduced from 80 (shorter sequences)
EMBEDDING_DIM    = 128            # Reduced from 256 (smaller model)
KEY_DIM          = 128            # Reduced from 256
N_HEADS          = 2
FEED_FORWARD_DIM = 128            # Reduced from 256
VALIDATION_SPLIT = 0.2
SEED             = 42
BATCH_SIZE       = 128            # Increased from 64 (fewer gradient steps)
EPOCHS           = 5              # Reduced from 10
MAX_REVIEWS      = 30000          # Use subset of dataset (full: ~120K)
LOAD_MODEL       = False          # Set True after first run to skip training

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH        = os.path.join(SCRIPT_DIR, "data", "winemag-data-130k-v2.json")
MODEL_SAVE_PATH  = os.path.join(SCRIPT_DIR, "models", "gpt_wine.keras")
CHECKPOINT_PATH  = os.path.join(SCRIPT_DIR, "checkpoint", "checkpoint.weights.h5")

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ──────────────────────────────────────────────
# 1. Load the Data
# ──────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading data...")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nDataset not found at:\n  {DATA_PATH}\n"
        "Download 'winemag-data-130k-v2.json' from Kaggle:\n"
        "  https://www.kaggle.com/datasets/zynicide/wine-reviews\n"
        "and place it in the ./data/ folder."
    )

with open(DATA_PATH) as f:
    wine_data = json.load(f)

# Filter entries with all required fields present
filtered_data = [
    "wine review : "
    + x["country"]
    + " : "
    + x["province"]
    + " : "
    + x["variety"]
    + " : "
    + x["description"]
    for x in wine_data
    if x.get("country")
    and x.get("province")
    and x.get("variety")
    and x.get("description")
]

# Limit dataset size for faster training
np.random.shuffle(filtered_data)
filtered_data = filtered_data[:MAX_REVIEWS]

n_wines = len(filtered_data)
print(f"{n_wines} wine reviews loaded (limited from full dataset for speed)")
print(f"\nExample:\n{filtered_data[0]}\n")

# ──────────────────────────────────────────────
# 2. Tokenize the Data
# ──────────────────────────────────────────────
print("=" * 60)
print("Step 2: Tokenizing data...")
print("=" * 60)


def pad_punctuation(s):
    """Pad punctuation with spaces so each symbol becomes a separate token."""
    s = re.sub(f"([{string.punctuation}, '\n'])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s


text_data = [pad_punctuation(x) for x in filtered_data]

# Build TF dataset
text_ds = (
    tf.data.Dataset.from_tensor_slices(text_data)
    .batch(BATCH_SIZE)
    .shuffle(1000)
)

# TextVectorization layer
vectorize_layer = layers.TextVectorization(
    standardize="lower",
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()

print(f"Vocabulary size: {len(vocab)}")
print("Sample token mappings:")
for i, word in enumerate(vocab[:10]):
    print(f"  {i}: '{word}'")

# ──────────────────────────────────────────────
# 3. Create the Training Set
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Creating training set (shifted sequences)...")
print("=" * 60)


def prepare_inputs(text):
    """Create input-target pairs: x = tokens[:-1], y = tokens[1:]."""
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


train_ds = text_ds.map(prepare_inputs)

# ──────────────────────────────────────────────
# 4. Causal Attention Mask
# ──────────────────────────────────────────────


def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """Lower-triangular mask so position i can only attend to positions <= i."""
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


# ──────────────────────────────────────────────
# 5. Custom Layers
# ──────────────────────────────────────────────


class TokenAndPositionEmbedding(layers.Layer):
    """Learns token and positional embeddings, then sums them."""

    def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


class TransformerBlock(layers.Layer):
    """Single decoder-only transformer block with causal masking."""

    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attn = layers.MultiHeadAttention(
            num_heads, key_dim, output_shape=embed_dim
        )
        self.dropout_1 = layers.Dropout(dropout_rate)
        self.ln_1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_2 = layers.Dense(embed_dim)
        self.dropout_2 = layers.Dropout(dropout_rate)
        self.ln_2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output, attention_scores = self.attn(
            inputs, inputs,
            attention_mask=causal_mask,
            return_attention_scores=True,
        )
        attention_output = self.dropout_1(attention_output)
        out1 = self.ln_1(inputs + attention_output)
        ffn_1 = self.ffn_1(out1)
        ffn_2 = self.ffn_2(ffn_1)
        ffn_output = self.dropout_2(ffn_2)
        return self.ln_2(out1 + ffn_output), attention_scores

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ──────────────────────────────────────────────
# 6. Build the GPT Model
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Building GPT model...")
print("=" * 60)

inputs = layers.Input(shape=(None,), dtype=tf.int32)
x = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM)(inputs)
x, attention_scores = TransformerBlock(
    N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM
)(x)
outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)
gpt = models.Model(inputs=inputs, outputs=[outputs, attention_scores])
gpt.compile("adam", loss=[losses.SparseCategoricalCrossentropy(), None])

gpt.summary()

# ──────────────────────────────────────────────
# 7. Text Generation Callback
# ──────────────────────────────────────────────


class TextGenerator(callbacks.Callback):
    """Generates sample text at the end of each epoch."""

    def __init__(self, index_to_word, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {word: index for index, word in enumerate(index_to_word)}

    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y, att = self.model.predict(x, verbose=0)
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            info.append({
                "prompt": start_prompt,
                "word_probs": probs,
                "atts": att[0, :, -1, :],
            })
            start_tokens.append(sample_token)
            start_prompt = start_prompt + " " + self.index_to_word[sample_token]
        print(f"\nGenerated text:\n{start_prompt}\n")
        return info

    def on_epoch_end(self, epoch, logs=None):
        self.generate("wine review", max_tokens=MAX_LEN, temperature=1.0)


# ──────────────────────────────────────────────
# 8. Train the Model
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: Training GPT model...")
print("=" * 60)

os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    save_weights_only=True,
    save_freq="epoch",
    verbose=0,
)

tensorboard_callback = callbacks.TensorBoard(
    log_dir=os.path.join(SCRIPT_DIR, "logs")
)

text_generator = TextGenerator(vocab)

if LOAD_MODEL and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading weights from {CHECKPOINT_PATH}")
    gpt.load_weights(CHECKPOINT_PATH)
else:
    gpt.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback, tensorboard_callback, text_generator],
    )
    gpt.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

# ──────────────────────────────────────────────
# 9. Generate Text at Different Temperatures
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6: Generating text at different temperatures...")
print("=" * 60)

text_generator.set_model(gpt)

TEMPERATURES = [0.3, 0.5, 1.0, 1.2]
PROMPTS = [
    "wine review : us",
    "wine review : italy",
    "wine review : france",
]

all_outputs = {}

for temp in TEMPERATURES:
    print(f"\n{'─' * 50}")
    print(f"  Temperature = {temp}")
    print(f"{'─' * 50}")
    all_outputs[temp] = []
    for prompt in PROMPTS:
        print(f"\n  Prompt: \"{prompt}\"")
        info = text_generator.generate(prompt, max_tokens=MAX_LEN, temperature=temp)
        all_outputs[temp].append(info)

# ──────────────────────────────────────────────
# 10. Temperature Comparison & Interpretation
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 7: Temperature Comparison & Interpretation")
print("=" * 60)

print("""
Temperature Controls the Randomness of Text Generation
-------------------------------------------------------

How it works:
  - Probabilities are raised to the power of (1 / temperature)
  - Then renormalized before sampling

Results observed:

  Temperature = 0.3 (Low randomness)
  → Very predictable, repetitive text
  → Tends to pick the highest-probability word every time
  → Good coherence but lacks creativity and diversity
  → May get stuck in loops of common phrases

  Temperature = 0.5 (Moderate-low randomness)
  → Mostly coherent with occasional variety
  → A good balance for readable, structured wine reviews
  → Still favors common words but allows some exploration

  Temperature = 1.0 (Standard / no scaling)
  → Uses the raw learned probability distribution
  → Produces diverse, creative text
  → Some grammatical errors may appear
  → Good for exploring the model's learned vocabulary

  Temperature = 1.2 (High randomness)
  → Very creative and diverse word choices
  → Often produces unusual or nonsensical combinations
  → Poor coherence — sentences may not make sense
  → Useful for seeing the full range of the model's vocabulary

Summary:
  Lower temperature → more repetitive, more coherent
  Higher temperature → more creative, less stable
  The trade-off is between predictability and diversity.
""")

print("=" * 60)
print("Assignment 7 complete.")
print("=" * 60)
