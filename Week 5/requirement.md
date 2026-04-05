Good catch—that _should definitely be included_ since the assignment explicitly references it.

Here’s the **updated Programming Requirement Document** with the GitHub link properly integrated (clean and submission-ready).

---

# **Programming Requirement Document**

## **Assignment 7: Variation of GPT Model (Transformer-based Text Generator)**

---

## **1. Objective (Programming Scope Only)**

The objective of this assignment is to implement a simplified GPT-style transformer model using the Wine Reviews dataset. The implementation must follow the structure and principles demonstrated in the official textbook code repository.

---

## **2. Reference Implementation (Mandatory)**

The implementation must align with the following reference notebook:

- **GitHub Source (Textbook Code):**
  [https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/09_transformer/gpt/gpt.ipynb](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/09_transformer/gpt/gpt.ipynb)

### **Usage Requirement**

- Use this notebook as the **baseline architecture**
- Modify it to:
  - Work with the Wine Reviews dataset
  - Adjust preprocessing for natural language input
  - Train on reduced dataset size if needed

---

## **3. Dataset Requirements**

### **Dataset Source**

- Wine Reviews Dataset (Kaggle)

### **Programming Tasks**

- Load dataset using Pandas
- Select relevant text column (`description`)
- Remove null values
- Optionally limit dataset size for faster training

---

## **4. Data Preprocessing Requirements**

### **Text Processing**

- Convert text to lowercase
- Remove unnecessary symbols (optional)
- Normalize whitespace

### **Tokenization**

- Use `TextVectorization` layer from Keras
- Define:
  - `max_tokens` (e.g., 20,000)
  - `output_sequence_length` (e.g., 80–100)

### **Sequence Preparation**

- Create shifted sequences:
  - Input → sequence of tokens
  - Target → next-token prediction

---

## **5. Custom Layer Requirement**

### **TokenAndPositionEmbedding Layer**

You must implement a custom embedding layer similar to the reference GitHub code.

```python
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(maxlen, embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
```

---

## **6. Transformer Model Requirements**

### **Architecture (Based on GitHub Notebook)**

- Input Layer
- Token + Position Embedding
- Transformer Block(s):
  - Multi-Head Attention
  - Feed Forward Network
  - Layer Normalization
  - Dropout

### **Output Layer**

- Dense + Softmax
- Vocabulary-sized output

---

## **7. Model Compilation**

- **Loss Function:** SparseCategoricalCrossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy (optional)

---

## **8. Training Requirements**

- Train the model using the processed dataset
- Suggested:
  - Epochs: 5–20
  - Batch size: 32 or 64

### **Expected Output**

- Training loss trend
- Evidence of learning

---

## **9. Text Generation Requirements**

### **Generation Logic**

- Provide a seed sentence
- Predict next token iteratively
- Append predicted tokens to sequence

---

## **10. Temperature-Based Sampling (Critical)**

### **Required Temperatures**

- Must use values **different from 1.0 and 0.5**

### **Suggested**

- 0.3 (low randomness)
- 1.2 (high randomness)

---

## **11. Output Deliverables**

### **You Must Provide**

- Generated text for:
  - Temperature = 0.3
  - Temperature = 1.2

- Comparison with:
  - Temperature = 0.5
  - Temperature = 1.0

---

## **12. Interpretation Requirements**

Explain differences in:

- Coherence
- Creativity
- Repetition

### **Expected Behavior**

- Lower temperature → more predictable text
- Higher temperature → more diverse but less stable text

---

## **13. Constraints**

- Follow GPT autoregressive modeling principle
- Use GitHub notebook as structural reference
- No need for large-scale training

As covered in your course materials, GPT models rely on transformer architectures for high-quality text generation .

---

## **14. Success Criteria**

The implementation is complete if:

- GitHub structure is correctly adapted
- Dataset preprocessing is functional
- Custom embedding layer works
- Model trains successfully
- Temperature-based outputs are generated and compared

---
