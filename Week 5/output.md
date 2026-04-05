# Assignment 7: GPT Wine Review Generator — Output & Approach

## Approach

This implementation builds a simplified GPT-style transformer model trained on the
Wine Reviews dataset, following the reference notebook from
[Generative Deep Learning 2nd Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/09_transformer/gpt/gpt.ipynb).

### Architecture (Unchanged from Reference)

The core architecture follows the reference exactly:

| Component | Description |
|-----------|-------------|
| **TokenAndPositionEmbedding** | Custom Keras layer — sums learned token embeddings + positional embeddings |
| **TransformerBlock** | Decoder-only block: MultiHeadAttention (causal mask) → Dropout → LayerNorm → FFN (Dense-ReLU-Dense) → Dropout → LayerNorm |
| **GPT Model** | Input → TokenAndPositionEmbedding → TransformerBlock → Dense(softmax) |
| **Loss** | SparseCategoricalCrossentropy (next-token prediction) |
| **Optimizer** | Adam |
| **Text Generation** | Autoregressive sampling with temperature scaling: `probs^(1/T)` then renormalize |

### Data Pipeline (Unchanged)

- Wine reviews formatted as: `"wine review : {country} : {province} : {variety} : {description}"`
- Punctuation padded with spaces (each symbol becomes a separate token)
- `TextVectorization` layer maps words to integer IDs
- Training pairs: `(tokens[:-1], tokens[1:])` — standard next-token prediction

---

## Deviations from the Assignment Requirements (for Speed)

The following hyperparameters were reduced to bring total runtime from **~2.5-5 hours** down to **~10-15 minutes** on a MacBook (CPU/MPS). None of these changes affect the architecture, the training approach, or the generation logic.

| Parameter | Assignment / Reference | Our Value | Reason |
|-----------|----------------------|-----------|--------|
| **Dataset size** | Full (~120K reviews) | 30,000 reviews | ~4x fewer training samples → 4x faster per epoch |
| **Epochs** | 5-20 (suggested) | 5 | Still enough to show learning trend; saves ~50% time vs 10 |
| **Embedding dim** | 256 | 128 | Halves embedding matrix size and attention computation |
| **Key dim** | 256 | 128 | Smaller attention key/query projections |
| **Feed-forward dim** | 256 | 128 | Smaller FFN inside transformer block |
| **Batch size** | 32-64 | 128 | Fewer gradient update steps per epoch |
| **Sequence length** | 80 | 60 | Shorter sequences → less memory, faster attention (O(n^2)) |
| **Vocab size** | 20,000 | 10,000 | Smaller softmax output layer, faster tokenization |

### What Was NOT Changed

- **Architecture**: Same custom `TokenAndPositionEmbedding`, `TransformerBlock`, causal mask, model structure
- **Training method**: Same `SparseCategoricalCrossentropy` + Adam, same shifted-sequence approach
- **Generation logic**: Same autoregressive temperature-based sampling
- **Temperature values**: 0.3, 0.5, 1.0, 1.2 — all four as required
- **Multiple prompts**: US, Italy, France — to show variety in generation

### Impact on Quality

With 30K reviews and 5 epochs at reduced dimensions, the model will:
- Learn basic wine vocabulary and sentence structure
- Distinguish between country-specific wine styles
- Show clear temperature effects (the primary deliverable)

The generated text will be less polished than a full-scale model, but the **temperature comparison** — the core requirement — will be clearly demonstrable.

---

## Actual Output

### Training Loss

| Epoch | Loss |
|-------|------|
| 1 | 5.13 |
| 2 | 3.83 |
| 3 | 3.36 |
| 4 | 3.12 |
| 5 | 2.98 |

Loss dropped from 9.2 → 2.98 over 5 epochs, showing strong learning. Total training time: ~3 minutes on Apple M2 (Metal GPU).

### Generated Text at Different Temperatures

#### Temperature = 0.3 (Low randomness)

**Prompt: "wine review : us"**
> wine review : us : california : pinot noir : this is a rich , dry and fruity wine , with a good structure , but it ' s full - bodied and a full - bodied wine . it ' s a bit of cherry and sweet , with a fine , but it ' s a bit of

**Prompt: "wine review : italy"**
> wine review : italy : tuscany : sangiovese : this opens with aromas of black cherry , leather and spice . the palate offers dried black cherry , clove and clove flavors of crushed black pepper , clove and licorice alongside firm tannins .

**Prompt: "wine review : france"**
> wine review : france : bordeaux : bordeaux - style red blend : this is a blend of cabernet sauvignon , syrah and cabernet sauvignon , this is a dense wine , full - bodied wine . it ' s a full - bodied wine , with a fine - grained tannins .

**Observations:** Very coherent and readable, but highly repetitive. Phrases like "full-bodied wine" and "it's a bit of" loop. Correctly associates Italy→Sangiovese, France→Bordeaux, US→Pinot Noir.

---

#### Temperature = 0.5 (Moderate-low randomness)

**Prompt: "wine review : us"**
> wine review : us : california : zinfandel : a rich , rich , concentrated and ripe and ripe , with cherry fruit flavors , it ' s a little hot , with a touch of vanilla and well .

**Prompt: "wine review : italy"**
> wine review : italy : northeastern italy : pinot grigio : this opens with aromas of mature red flower and a whiff of dried stone . the palate offers the palate offers peach , black cherry , peach and a hint of honey alongside bright acidity .

**Prompt: "wine review : france"**
> wine review : france : burgundy : pinot noir : this is a [UNK] - syrah - based wine , with its full - bodied red fruits , this wine is a ripe wine that is dense and full of ripe black currant fruit and spice . the wine is ready to drink .

**Observations:** Good balance of structure and variety. More diverse grape/region choices (Zinfandel, Pinot Grigio, Burgundy). Some repetition ("rich, rich") but overall readable.

---

#### Temperature = 1.0 (Standard — no scaling)

**Prompt: "wine review : us"**
> wine review : us : california : zinfandel : there ' s lots of botrytis on the eye , this newish bottling . the dense , soft palate presents crisp morello wide - plum fruit , tobacco and ultraripe pear flavors that linger long acidity .

**Prompt: "wine review : italy"**
> wine review : italy : tuscany : red blend : made predominantly merlot , this blend of pinotage , nebbiolo opens with dried red plum , olive , pressed leather and menthol . smooth , easy drinking , and personality . it shows a big creamy , focused mix of black chocolate and exotic clove . lively , angular ,

**Prompt: "wine review : france"**
> wine review : france : bordeaux : bordeaux - style red fruit , so 93 spicy lime and everything is an champoux . the aromatic elements , this luscious wine is very delicious . the wine is nothing of the first , too much in flavor , this has its acidity and blackberry flavors . it ' s not overly

**Observations:** Creative and diverse vocabulary — "ultraripe pear", "pressed leather and menthol", "exotic clove". Some grammatical errors appear. Interesting wine knowledge surfaces (botrytis, pinotage, nebbiolo).

---

#### Temperature = 1.2 (High randomness)

**Prompt: "wine review : us"**
> wine review : us : california : petite sirah : grenache also smoke , soft , cedar and oak aromas meet of pencil baron as iodine . this just - , aroma seems small 82 displays scents pop on through black smoky on the palate of bell pepper in the remainder , given beginning for eight years , so thick

**Prompt: "wine review : italy"**
> wine review : italy : piedmont : nebbiolo : this barolo is a packaged from the linear , exciting st for which in taylor is wiry , a flavorful , [UNK] vineyard delicate style suggest aromas . aromas of overwhelms the orchard grapes can ' ve quite ripe best at things and puglia . fruit , it ' s cedary

**Prompt: "wine review : france"**
> wine review : france : burgundy : pinot noir : a light two parcels almost - are shaped single - this one intensity for selection is becoming irresistibly port out flavors . there are balanced by fine muscadet being flabby . still flavors are developing late - aftertaste has at it provide plenty of brooding tannins . drink now .

**Observations:** Very creative but chaotic. Unusual combinations ("pencil baron as iodine", "exciting st for which in taylor is wiry"). Grammar breaks down. Still picks real wine terms (Petite Sirah, Nebbiolo, Barolo, Muscadet) but combines them nonsensically.

---

### Interpretation

| Aspect | Temp 0.3 | Temp 0.5 | Temp 1.0 | Temp 1.2 |
|--------|----------|----------|----------|----------|
| **Coherence** | High — reads like a real review | Good — minor repetition | Moderate — creative but uneven | Low — fragmented sentences |
| **Creativity** | Low — same phrases recycled | Moderate — some variety | High — rich vocabulary | Very High — unusual combos |
| **Repetition** | High — loops on common phrases | Some — occasional repeats | Low — diverse word choices | Very Low — almost random |
| **Wine Knowledge** | Correct but generic | Good variety/region pairing | Impressive domain terms | Real terms, wrong context |

The fundamental trade-off: **lower temperature sharpens the probability distribution** (making the model more confident/repetitive), while **higher temperature flattens it** (making the model more exploratory/chaotic).

---

## How to Run

```bash
# 1. Download dataset from Kaggle and place in data/
#    https://www.kaggle.com/datasets/zynicide/wine-reviews
#    File: winemag-data-130k-v2.json → Week 5/data/

# 2. Run the script
python "Week 5/Assignment7_GPT_WineReviews.py"

# 3. After first run, set LOAD_MODEL = True to skip retraining
```

Estimated runtime: **~10-15 minutes** on MacBook (CPU/MPS).
