"""
Assignment 9: Music-Generating Transformer
Generates Bach-style cello music using a Transformer trained on MIDI files.
Reference: Generative Deep Learning 2nd Edition, Chapter 11
           https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition
Uses PyTorch with MPS (Apple Metal GPU).
Trains on Bach MIDI files (Cello Suites / Chorales) from music21 corpus.
"""

import os
import glob
import math
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fractions import Fraction
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import music21

# ============================================================
# Paths and device
# ============================================================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SAVE_DIR, "output")
DATA_DIR = os.path.join(SAVE_DIR, "data", "bach-cello")
PARSED_DIR = os.path.join(SAVE_DIR, "parsed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PARSED_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">>> Using Apple MPS GPU <<<")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(">>> Using CUDA GPU <<<")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ============================================================
# Hyperparameters
# ============================================================
SEQ_LEN = 50
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 4
DROPOUT_RATE = 0.3
FEED_FORWARD_DIM = 256

EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
GENERATE_LEN = 100


# ============================================================
# MIDI parsing utilities
# ============================================================
def parse_midi_files(file_list, seq_len):
    """Parse MIDI files into note and duration sequences."""
    notes = []
    durations = []

    for i, filepath in enumerate(file_list):
        print(f"  {i+1}/{len(file_list)} Parsing {os.path.basename(filepath)}")
        try:
            score = music21.converter.parse(filepath).chordify()
        except Exception as e:
            print(f"    Skipping (parse error): {e}")
            continue

        notes.append("START")
        durations.append("0.0")

        for element in score.flatten():
            note_name = None
            duration_name = None

            if isinstance(element, music21.key.Key):
                note_name = str(element.tonic.name) + ":" + str(element.mode)
                duration_name = "0.0"
            elif isinstance(element, music21.meter.TimeSignature):
                note_name = str(element.ratioString) + "TS"
                duration_name = "0.0"
            elif isinstance(element, music21.chord.Chord):
                note_name = element.pitches[-1].nameWithOctave
                duration_name = str(element.duration.quarterLength)
            elif isinstance(element, music21.note.Rest):
                note_name = str(element.name)
                duration_name = str(element.duration.quarterLength)
            elif isinstance(element, music21.note.Note):
                note_name = str(element.nameWithOctave)
                duration_name = str(element.duration.quarterLength)

            if note_name and duration_name:
                notes.append(note_name)
                durations.append(duration_name)

    print(f"  Total tokens parsed: {len(notes)}")

    # Build overlapping sequences
    notes_seqs = []
    duration_seqs = []
    for i in range(len(notes) - seq_len):
        notes_seqs.append(notes[i:i + seq_len])
        duration_seqs.append(durations[i:i + seq_len])

    print(f"  Training sequences: {len(notes_seqs)}")
    return notes_seqs, duration_seqs, notes, durations


def build_vocab(sequences):
    """Build token-to-index and index-to-token mappings."""
    all_tokens = set()
    for seq in sequences:
        all_tokens.update(seq)
    vocab = ["<PAD>", "<UNK>"] + sorted(all_tokens)
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    return vocab, token_to_idx


def encode_sequences(sequences, token_to_idx):
    """Convert token sequences to integer indices."""
    encoded = []
    for seq in sequences:
        encoded.append([token_to_idx.get(t, 1) for t in seq])  # 1 = <UNK>
    return np.array(encoded, dtype=np.int64)


def get_midi_note(sample_note, sample_duration):
    """Convert note/duration tokens back to music21 objects."""
    new_note = None

    if "TS" in sample_note:
        new_note = music21.meter.TimeSignature(sample_note.split("TS")[0])
    elif "major" in sample_note or "minor" in sample_note:
        tonic, mode = sample_note.split(":")
        new_note = music21.key.Key(tonic, mode)
    elif sample_note == "rest":
        new_note = music21.note.Rest()
        new_note.duration = music21.duration.Duration(float(Fraction(sample_duration)))
        new_note.storedInstrument = music21.instrument.Violoncello()
    elif sample_note == "START" or sample_note in ("<PAD>", "<UNK>"):
        pass
    else:
        try:
            new_note = music21.note.Note(sample_note)
            new_note.duration = music21.duration.Duration(float(Fraction(sample_duration)))
            new_note.storedInstrument = music21.instrument.Violoncello()
        except Exception:
            pass

    return new_note


# ============================================================
# Sinusoidal Positional Encoding
# ============================================================
class SinePositionEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""
    def __init__(self, max_wavelength=10000):
        super().__init__()
        self.max_wavelength = max_wavelength

    def forward(self, x):
        seq_len, d_model = x.size(1), x.size(2)
        position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=x.device)
            * -(math.log(self.max_wavelength) / d_model)
        )
        pe = torch.zeros(seq_len, d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


# ============================================================
# Token + Position Embedding
# ============================================================
class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = SinePositionEncoding()

    def forward(self, x):
        emb = self.token_emb(x)
        pos = self.pos_enc(emb)
        return emb + pos


# ============================================================
# Transformer Block (Causal Self-Attention)
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, key_dim, ff_dim, dropout_rate):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            dropout=dropout_rate, batch_first=True
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        seq_len = x.size(1)
        # Causal mask: prevent attending to future tokens
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()

        attn_out, attn_weights = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.ln1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout2(ffn_out))
        return x, attn_weights


# ============================================================
# Music Transformer Model
# ============================================================
class MusicTransformer(nn.Module):
    def __init__(self, notes_vocab_size, durations_vocab_size,
                 embed_dim, n_heads, key_dim, ff_dim, dropout_rate):
        super().__init__()
        half_dim = embed_dim // 2

        self.note_embedding = TokenAndPositionEmbedding(notes_vocab_size, half_dim)
        self.duration_embedding = TokenAndPositionEmbedding(durations_vocab_size, half_dim)

        self.transformer = TransformerBlock(embed_dim, n_heads, key_dim, ff_dim, dropout_rate)

        self.note_head = nn.Linear(embed_dim, notes_vocab_size)
        self.duration_head = nn.Linear(embed_dim, durations_vocab_size)

    def forward(self, note_tokens, duration_tokens):
        note_emb = self.note_embedding(note_tokens)
        dur_emb = self.duration_embedding(duration_tokens)
        x = torch.cat([note_emb, dur_emb], dim=-1)

        x, attn_weights = self.transformer(x)

        note_logits = self.note_head(x)
        dur_logits = self.duration_head(x)
        return note_logits, dur_logits, attn_weights


# ============================================================
# Music generation
# ============================================================
def sample_from(probs, temperature=1.0):
    """Sample from probability distribution with temperature."""
    probs = probs ** (1.0 / temperature)
    probs = probs / probs.sum()
    return np.random.choice(len(probs), p=probs.numpy() if isinstance(probs, torch.Tensor) else probs)


def generate_music(model, notes_vocab, durations_vocab,
                   note_to_idx, dur_to_idx,
                   start_notes=None, start_durations=None,
                   max_tokens=GENERATE_LEN, temperature=0.5):
    """Generate music autoregressively from seed tokens."""
    model.eval()

    if start_notes is None:
        start_notes = ["START"]
    if start_durations is None:
        start_durations = ["0.0"]

    note_tokens = [note_to_idx.get(n, 1) for n in start_notes]
    dur_tokens = [dur_to_idx.get(d, 1) for d in start_durations]
    generated_notes = list(start_notes)
    generated_durations = list(start_durations)

    midi_stream = music21.stream.Stream()
    midi_stream.append(music21.clef.BassClef())

    # Add seed notes to stream
    for note, dur in zip(start_notes, start_durations):
        midi_note = get_midi_note(note, dur)
        if midi_note is not None:
            midi_stream.append(midi_note)

    with torch.no_grad():
        while len(note_tokens) < max_tokens:
            # Prepare input (keep last SEQ_LEN-1 tokens)
            input_notes = torch.tensor([note_tokens[-(SEQ_LEN-1):]],
                                       dtype=torch.long, device=device)
            input_durs = torch.tensor([dur_tokens[-(SEQ_LEN-1):]],
                                      dtype=torch.long, device=device)

            note_logits, dur_logits, _ = model(input_notes, input_durs)

            # Get predictions for last position
            note_probs = F.softmax(note_logits[0, -1], dim=-1).cpu()
            dur_probs = F.softmax(dur_logits[0, -1], dim=-1).cpu()

            # Sample (skip <PAD>=0 and <UNK>=1)
            note_probs_adj = note_probs.clone()
            note_probs_adj[0] = 0
            note_probs_adj[1] = 0
            sample_note_idx = sample_from(note_probs_adj, temperature)

            dur_probs_adj = dur_probs.clone()
            dur_probs_adj[0] = 0
            dur_probs_adj[1] = 0
            sample_dur_idx = sample_from(dur_probs_adj, temperature)

            sample_note = notes_vocab[sample_note_idx]
            sample_dur = durations_vocab[sample_dur_idx]

            # Skip zero-duration pitched notes
            if sample_dur == "0.0" and sample_note not in ("START",) and "TS" not in sample_note and ":" not in sample_note:
                continue

            midi_note = get_midi_note(sample_note, sample_dur)
            if midi_note is not None:
                midi_stream.append(midi_note)

            note_tokens.append(sample_note_idx)
            dur_tokens.append(sample_dur_idx)
            generated_notes.append(sample_note)
            generated_durations.append(sample_dur)

            if sample_note == "START":
                break

    model.train()
    return midi_stream, generated_notes, generated_durations


# ============================================================
# Visualization
# ============================================================
def plot_training_loss(losses, save_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Music Transformer — Training Loss")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_attention(attn_weights, generated_notes, save_path, max_show=40):
    """Visualize attention weights from the last generation."""
    attn = attn_weights[0].cpu().numpy()  # first head
    n = min(max_show, attn.shape[-1])
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attn[:n, :n], cmap='viridis', aspect='auto')
    labels = generated_notes[:n]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Transformer Attention Weights (Head 0)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_generated_piano_roll(notes_list, durations_list, save_path):
    """Simple piano-roll visualization of generated music."""
    pitches = []
    times = []
    durs = []
    current_time = 0.0

    for note, dur in zip(notes_list, durations_list):
        try:
            dur_val = float(Fraction(dur))
        except (ValueError, ZeroDivisionError):
            dur_val = 0.0

        if note not in ("START", "rest", "<PAD>", "<UNK>") and "TS" not in note and ":" not in note:
            try:
                p = music21.pitch.Pitch(note)
                pitches.append(p.midi)
                times.append(current_time)
                durs.append(max(dur_val, 0.1))
            except Exception:
                pass

        current_time += dur_val

    if not pitches:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    for p, t, d in zip(pitches, times, durs):
        ax.barh(p, d, left=t, height=0.8, color='steelblue', edgecolor='navy', linewidth=0.5)
    ax.set_xlabel("Time (quarter notes)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title("Generated Music — Piano Roll")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# Main training loop
# ============================================================
def main():
    print("=" * 60)
    print("Music Transformer — Bach-Style Music Generation")
    print("=" * 60)

    # ------ 1. Parse MIDI files ------
    print("\n--- Parsing MIDI files ---")
    file_list = sorted(glob.glob(os.path.join(DATA_DIR, "*.mid")))
    print(f"Found {len(file_list)} MIDI files")

    if not file_list:
        print("ERROR: No MIDI files found. Please add .mid files to:")
        print(f"  {DATA_DIR}")
        return

    notes_seqs, dur_seqs, all_notes, all_durations = parse_midi_files(file_list, SEQ_LEN + 1)

    # ------ 2. Build vocabularies ------
    print("\n--- Building vocabularies ---")
    notes_vocab, note_to_idx = build_vocab(notes_seqs)
    dur_vocab, dur_to_idx = build_vocab(dur_seqs)
    print(f"  Notes vocabulary: {len(notes_vocab)} tokens")
    print(f"  Duration vocabulary: {len(dur_vocab)} tokens")

    # Save vocabs for inference
    with open(os.path.join(PARSED_DIR, "vocabs.pkl"), "wb") as f:
        pickle.dump({
            "notes_vocab": notes_vocab, "note_to_idx": note_to_idx,
            "dur_vocab": dur_vocab, "dur_to_idx": dur_to_idx,
        }, f)

    # ------ 3. Create training dataset ------
    print("\n--- Creating training dataset ---")
    note_encoded = encode_sequences(notes_seqs, note_to_idx)
    dur_encoded = encode_sequences(dur_seqs, dur_to_idx)

    # Input: tokens 0..N-2, Target: tokens 1..N-1
    note_input = torch.tensor(note_encoded[:, :-1], dtype=torch.long)
    note_target = torch.tensor(note_encoded[:, 1:], dtype=torch.long)
    dur_input = torch.tensor(dur_encoded[:, :-1], dtype=torch.long)
    dur_target = torch.tensor(dur_encoded[:, 1:], dtype=torch.long)

    dataset = TensorDataset(note_input, dur_input, note_target, dur_target)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"  Sequences: {len(dataset)} | Batches: {len(loader)}")

    # ------ 4. Build model ------
    print("\n--- Building Transformer model ---")
    model = MusicTransformer(
        notes_vocab_size=len(notes_vocab),
        durations_vocab_size=len(dur_vocab),
        embed_dim=EMBEDDING_DIM,
        n_heads=N_HEADS,
        key_dim=KEY_DIM,
        ff_dim=FEED_FORWARD_DIM,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    note_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    dur_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # ------ 5. Train ------
    print(f"\n--- Training for {EPOCHS} epochs ---")
    losses = []
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for ni, di, nt, dt in loader:
            ni, di = ni.to(device), di.to(device)
            nt, dt = nt.to(device), dt.to(device)

            note_logits, dur_logits, _ = model(ni, di)

            # Reshape for cross-entropy: (batch * seq_len, vocab_size)
            n_loss = note_loss_fn(note_logits.reshape(-1, len(notes_vocab)), nt.reshape(-1))
            d_loss = dur_loss_fn(dur_logits.reshape(-1, len(dur_vocab)), dt.reshape(-1))
            loss = n_loss + d_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{EPOCHS} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

        # Generate sample every 50 epochs
        if epoch % 50 == 0 or epoch == EPOCHS:
            midi_stream, gen_notes, gen_durs = generate_music(
                model, notes_vocab, dur_vocab, note_to_idx, dur_to_idx,
                start_notes=["START"], start_durations=["0.0"],
                max_tokens=GENERATE_LEN, temperature=0.5,
            )
            midi_path = os.path.join(OUTPUT_DIR, f"generated_epoch_{epoch:04d}.mid")
            midi_stream.write("midi", fp=midi_path)
            print(f"    Generated MIDI saved: {midi_path}")

    # ------ 6. Save training loss plot ------
    plot_training_loss(losses, os.path.join(OUTPUT_DIR, "training_loss.png"))
    print(f"\nTraining loss plot saved.")

    # ------ 7. Final generation with multiple temperatures ------
    print("\n--- Generating final music samples ---")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"),
                                     map_location=device, weights_only=True))

    for temp in [0.3, 0.5, 0.8, 1.0]:
        print(f"\n  Temperature = {temp}")
        midi_stream, gen_notes, gen_durs = generate_music(
            model, notes_vocab, dur_vocab, note_to_idx, dur_to_idx,
            start_notes=["START"], start_durations=["0.0"],
            max_tokens=GENERATE_LEN, temperature=temp,
        )

        # Save MIDI
        midi_path = os.path.join(OUTPUT_DIR, f"final_temp_{temp:.1f}.mid")
        midi_stream.write("midi", fp=midi_path)
        print(f"    MIDI: {midi_path}")

        # Piano roll
        roll_path = os.path.join(OUTPUT_DIR, f"piano_roll_temp_{temp:.1f}.png")
        plot_generated_piano_roll(gen_notes, gen_durs, roll_path)
        print(f"    Piano roll: {roll_path}")

        # Print generated tokens
        print(f"    Notes: {' '.join(gen_notes[:30])}...")
        print(f"    Durs:  {' '.join(gen_durs[:30])}...")

    # ------ 8. Attention visualization ------
    print("\n--- Attention visualization ---")
    model.eval()
    with torch.no_grad():
        # Run one more generation to get attention weights
        sample_notes = [note_to_idx.get(n, 1) for n in gen_notes[:SEQ_LEN-1]]
        sample_durs = [dur_to_idx.get(d, 1) for d in gen_durs[:SEQ_LEN-1]]
        inp_n = torch.tensor([sample_notes], dtype=torch.long, device=device)
        inp_d = torch.tensor([sample_durs], dtype=torch.long, device=device)
        _, _, attn_weights = model(inp_n, inp_d)

    plot_attention(attn_weights, gen_notes[:SEQ_LEN-1],
                   os.path.join(OUTPUT_DIR, "attention_weights.png"))
    print("  Attention heatmap saved.")

    # ------ 9. Vocab distribution plot ------
    note_counts = Counter(all_notes)
    top_notes = note_counts.most_common(20)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar([n for n, _ in top_notes], [c for _, c in top_notes], color='steelblue')
    ax.set_xlabel("Note")
    ax.set_ylabel("Count")
    ax.set_title("Top 20 Notes in Training Data")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "note_distribution.png"), dpi=150)
    plt.close()
    print("  Note distribution plot saved.")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
