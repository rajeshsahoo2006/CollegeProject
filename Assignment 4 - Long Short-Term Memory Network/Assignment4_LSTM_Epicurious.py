"""
Assignment 4: LSTM Text Generation on Epicurious Recipes Dataset
================================================================
Based on: https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/05_autoregressive/01_lstm/lstm.ipynb

Steps:
  1. Load & clean the Epicurious Recipes dataset
  2. Tokenize the text
  3. Build and train an LSTM network (next-word prediction)
  4. Generate text using temperature sampling at two different temperatures
     (temperatures used: 0.5 and 0.8 — as required, neither 1.0 nor 0.2)

Dataset: Epicurious Recipes (full_format_recipes.json) — download from Kaggle:
  https://www.kaggle.com/datasets/hugodarwood/epirecipes
Place the file at: ./data/full_format_recipes.json
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
VOCAB_SIZE       = 10000
MAX_LEN          = 200
EMBEDDING_DIM    = 100
N_UNITS          = 128
VALIDATION_SPLIT = 0.2
SEED             = 42
BATCH_SIZE       = 32
EPOCHS           = 5
LOAD_MODEL       = True           # Set True to skip training and load a saved model
DATA_PATH        = "/Users/sahoo/Desktop/NNDL/Code/Deep-Learning-with-TensorFlow-2-and-Keras/Week 3/Epicurious Recipes - Kaggle/full_format_recipes.json"
MODEL_SAVE_PATH  = "./models/lstm_recipe.keras"
CHECKPOINT_PATH  = "./checkpoint/checkpoint.weights.h5"

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
        "Please ensure 'full_format_recipes.json' exists at that path."
    )

with open(DATA_PATH, encoding="utf-8") as f:
    recipe_data = json.load(f)

# Format each recipe as: "Recipe for <title> | <directions>"
filtered_data = [
    "Recipe for " + x["title"] + " | " + " ".join(x["directions"])
    for x in recipe_data
    if "title" in x
    and x["title"] is not None
    and "directions" in x
    and x["directions"] is not None
]

n_recipes = len(filtered_data)
print(f"{n_recipes} recipes loaded.\n")
print("Example recipe (raw):")
print(filtered_data[9][:300], "...\n")

# ──────────────────────────────────────────────
# 2. Tokenize the Data
# ──────────────────────────────────────────────
print("=" * 60)
print("Step 2: Tokenizing...")
print("=" * 60)


def pad_punctuation(s: str) -> str:
    """Pad punctuation so each symbol becomes a separate token."""
    s = re.sub(f"([{re.escape(string.punctuation)}])", r" \1 ", s)
    s = re.sub(r" +", " ", s)
    return s.strip()


text_data = [pad_punctuation(x) for x in filtered_data]

print("Example recipe (tokenized):")
print(text_data[9][:300], "...\n")

# Build a tf.data pipeline
text_ds = (
    tf.data.Dataset.from_tensor_slices(text_data)
    .batch(BATCH_SIZE)
    .shuffle(1000, seed=SEED)
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
print("Sample vocab (index: word):")
for i, word in enumerate(vocab[:10]):
    print(f"  {i}: {word}")

print("\nExample recipe tokenized (first 20 tokens):")
example_tokenised = vectorize_layer(text_data[9])
print(example_tokenised.numpy()[:20], "\n")

# ──────────────────────────────────────────────
# 3. Create Training Dataset (next-token pairs)
# ──────────────────────────────────────────────
print("=" * 60)
print("Step 3: Building training dataset...")
print("=" * 60)


def prepare_inputs(text):
    """
    Given a batch of raw text strings, vectorize and split into
    (input sequence, target sequence shifted by one token).
    """
    text = tf.expand_dims(text, -1)
    tokenized = vectorize_layer(text)
    x = tokenized[:, :-1]   # all tokens except the last
    y = tokenized[:, 1:]    # all tokens except the first (next-token labels)
    return x, y


train_ds = text_ds.map(prepare_inputs)
print("Training dataset ready.\n")

# ──────────────────────────────────────────────
# 4. Build the LSTM Model
# ──────────────────────────────────────────────
print("=" * 60)
print("Step 4: Building LSTM model...")
print("=" * 60)

inputs  = layers.Input(shape=(None,), dtype="int32")
x       = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)
x       = layers.LSTM(N_UNITS, return_sequences=True)(x)
outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)

lstm = models.Model(inputs, outputs)
lstm.summary()

# ──────────────────────────────────────────────
# 5. TextGenerator Callback
# ──────────────────────────────────────────────

class TextGenerator(callbacks.Callback):
    """
    Custom Keras callback that generates sample text at the end of each epoch.

    Temperature controls randomness of sampling:
      - Low  temperature (e.g. 0.5) → more deterministic / repetitive
      - High temperature (e.g. 0.8) → more diverse / creative
    """

    def __init__(self, index_to_word: list, top_k: int = 10):
        super().__init__()
        self.index_to_word = index_to_word
        self.word_to_index = {word: idx for idx, word in enumerate(index_to_word)}
        self._lstm_model = None  # set via set_lstm_model() when loading from checkpoint

    def set_lstm_model(self, model):
        self._lstm_model = model

    def _get_model(self):
        return self._lstm_model if self._lstm_model is not None else self.model

    def sample_from(self, probs: np.ndarray, temperature: float):
        """Apply temperature scaling and sample the next token index."""
        probs = probs ** (1.0 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt: str, max_tokens: int, temperature: float) -> list:
        """
        Auto-regressively generate text from start_prompt.
        Returns per-step probability info for inspection.
        """
        start_tokens = [
            self.word_to_index.get(w, 1) for w in start_prompt.split()
        ]
        sample_token = None
        info = []

        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y = self._get_model().predict(x, verbose=0)  # (1, seq_len, vocab)
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            info.append({"prompt": start_prompt, "word_probs": probs})
            start_tokens.append(sample_token)
            start_prompt = start_prompt + " " + self.index_to_word[sample_token]

        print(f"\n[Generated text at temperature={temperature}]:\n{start_prompt}\n")
        return info

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n--- Epoch {epoch + 1} sample ---")
        self.generate("recipe for", max_tokens=100, temperature=0.5)


def print_probs(info: list, vocab: list, top_k: int = 5):
    """Pretty-print the top-k word probabilities at each generation step."""
    for step in info:
        print(f"\nPROMPT: {step['prompt']}")
        word_probs = step["word_probs"]
        top_indices = np.argsort(word_probs)[::-1][:top_k]
        for i in top_indices:
            print(f"  {vocab[i]:<20s} {np.round(100 * word_probs[i], 2):6.2f}%")
        print("--------")

# ──────────────────────────────────────────────
# 6. Train the LSTM
# ──────────────────────────────────────────────
print("=" * 60)
print("Step 5: Training the LSTM...")
print("=" * 60)

loss_fn = losses.SparseCategoricalCrossentropy()
lstm.compile(optimizer="adam", loss=loss_fn)

os.makedirs("./checkpoint", exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

text_generator = TextGenerator(vocab)

if LOAD_MODEL and os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading saved model from {MODEL_SAVE_PATH} ...")
    lstm = models.load_model(MODEL_SAVE_PATH, compile=False)
    lstm.compile(optimizer="adam", loss=loss_fn)
    text_generator.model = lstm
    print("Model loaded.\n")
elif LOAD_MODEL and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint weights from {CHECKPOINT_PATH} ...")
    lstm.load_weights(CHECKPOINT_PATH)
    text_generator.set_lstm_model(lstm)
    print("Checkpoint weights loaded.\n")
else:
    model_checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_weights_only=True,
        save_freq="epoch",
        verbose=0,
    )
    lstm.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_cb, text_generator],
    )

    lstm.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

# ──────────────────────────────────────────────
# 7. Generate Text at Two Different Temperatures
#    (Assignment requirement: not 1.0 or 0.2)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6: Generating text at two temperatures")
print("  Temperature 1: 0.5  (more focused / conservative)")
print("  Temperature 2: 0.8  (more creative / diverse)")
print("=" * 60)

PROMPTS = [
    "recipe for roasted vegetables | chop 1 /",
    "recipe for chocolate cake |",
]

for prompt in PROMPTS:
    print(f"\n{'─' * 60}")
    print(f"PROMPT: \"{prompt}\"")
    print(f"{'─' * 60}")

    # ── Temperature = 0.5 ──────────────────────────────────────
    print("\n[Temperature = 0.5]  (lower → more deterministic)")
    info_05 = text_generator.generate(
        start_prompt=prompt,
        max_tokens=50,
        temperature=0.5,
    )
    print_probs(info_05, vocab, top_k=5)

    # ── Temperature = 0.8 ──────────────────────────────────────
    print("\n[Temperature = 0.8]  (higher → more creative)")
    info_08 = text_generator.generate(
        start_prompt=prompt,
        max_tokens=50,
        temperature=0.8,
    )
    print_probs(info_08, vocab, top_k=5)

print("\n" + "=" * 60)
print("Assignment complete.")
print("=" * 60)
