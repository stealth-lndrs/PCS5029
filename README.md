# Word Vectors via Co-occurrence → PMI → Context-Expansion

This repository provides a **toy implementation** of word vector learning using
a small, interpretable corpus. It shows how to go from raw text to co-occurrence
counts, then to **PMI/PPMI matrices**, and finally to a simple but clever
**dimensionality reduction technique (context-expansion)** for visualization.

---

## 📖 What the script does

1. **Takes a tiny corpus** of daily facts:
   ```text
   alice likes cheese and bread
   bob likes fish and rice
   cheese is dairy
   fish is seafood
   bread and rice are carbs
   ```

2. **Builds a co-occurrence matrix**  
   Each entry `C[w,c]` counts how often word `w` appears near context `c`
   (using a symmetric window of 2 words).

   \[
   C[w,c] = \#\{\text{occurrences of $w$ within window of $c$}\}
   \]

3. **Computes PMI and PPMI**  
   Probabilities:

   \[
   p(w,c) = \frac{C[w,c]}{\sum_{u,v} C[u,v]}, \quad
   p(w) = \sum_c p(w,c), \quad
   p(c) = \sum_w p(w,c)
   \]

   PMI (Pointwise Mutual Information):

   \[
   \mathrm{PMI}(w,c) = \log_2 \frac{p(w,c)}{p(w)\,p(c)}
   \]

   PPMI (Positive PMI):

   \[
   \mathrm{PPMI}(w,c) = \max(\mathrm{PMI}(w,c),0)
   \]

   This highlights word–context pairs that occur **more often than chance**.

4. **Applies a “smarter” 2D dimensionality reduction**  
   Instead of SVD/PCA, we use *context-expansion bundles*.  
   - Anchors = the words in the question, here: `{likes, dairy}`.  
   - For each anchor $A$, find all **neighbors** $N(A)$ with $\mathrm{PPMI}(c,A)>0$.  
   - Build an aggregated axis:

     \[
     \mathrm{Axis}_A(w) = \sum_{c \in N(A)} M_{w,c} \cdot M_{c,A}
     \]

     where $M$ is the PPMI matrix.  
     (Think of it as a weighted “bundle” of contexts related to the anchor.)

   - Normalize (z-score) for readability:

     \[
     \widetilde{\mathrm{Axis}}_A(w) =
     \frac{\mathrm{Axis}_A(w) - \mu_A}{\sigma_A}
     \]

   - Final 2D embedding:

     \[
     \mathbf v(w) =
     \big[ \widetilde{\mathrm{Axis}}_{\text{likes}}(w),
            \widetilde{\mathrm{Axis}}_{\text{dairy}}(w) \big]
     \]

5. **Vector composition experiment**  
   Form the query vector:

   \[
   \mathbf q = \mathbf v(\text{likes}) + \mathbf v(\text{dairy})
   \]

   Then compare cosine similarities:

   \[
   \cos(\mathbf q, \mathbf v(s)) =
   \frac{\mathbf q \cdot \mathbf v(s)}{\|\mathbf q\|\;\|\mathbf v(s)\|}
   \]

   This allows answering *out-of-corpus* logical questions like:

   > **“___ likes dairy?”**  
   > → Correct answer: **alice**

6. **Plots results with non-overlapping labels**  
   - A 2D scatter of all word vectors.  
   - The composition experiment with arrows for `likes`, `dairy`, and `q`, and dashed rays for `alice` and `bob`.

---

## 🖥️ How to run

1. Clone/download this repo.
2. Install requirements:
   ```bash
   pip install numpy pandas matplotlib
   ```
3. Run the script:
   ```bash
   python word_vectors_demo.py
   ```
4. It will print cosine similarity scores and generate figures:
   - `context_bundle_vectors_2d.png` — word vectors in 2D (bundled axes).  
   - `context_bundle_comp_2d.png` — composition experiment plot.

---

## 🎓 Theory Recap

- **Co-occurrence → PPMI** encodes the **distributional hypothesis**:  
  *“You shall know a word by the company it keeps.”*

- **Context-expansion bundling** is a simple, explainable alternative to SVD/PCA.  
  It captures **multi-hop logic** (e.g., *alice likes cheese*, *cheese is dairy* → *alice likes dairy*) while staying 2D for visualization.

- **Modern methods** (Word2Vec, GloVe) scale this idea:  
  - Word2Vec skip-gram with negative sampling ≈ implicit factorization of shifted PMI.  
  - Transformers (BERT, GPT) go further, producing **contextual embeddings** (vector of a word depends on its sentence).

---

## 📂 Outputs

- `cooccurrence.csv` — co-occurrence counts  
- `pmi.csv`, `ppmi.csv` — PMI/PPMI matrices  
- `context_bundle_reduced_2d.csv` — reduced 2D embedding table  
- `context_bundle_vectors_2d.png` — 2D scatter of words  
- `context_bundle_comp_2d.png` — composition experiment

---

## ✅ Example Output (cosine scores)

```
Cosine scores: {'alice': 0.051, 'bob': -0.918}
```

Interpretation: Query vector is much closer to **alice** than **bob** →  
answer to “___ likes dairy?” is **alice**.

---

## ✍️ Authors

- **Your Name** — code implementation, experiments, and report preparation.

---

## 🤖 LLM Usage Acknowledgement

This README and parts of the Python code were drafted with the help of an
AI language model (ChatGPT by OpenAI). The model assisted in:

- Structuring explanations and math formulas in Markdown/LaTeX.  
- Drafting didactic descriptions for slides and README.  
- Suggesting the context-expansion dimensionality reduction technique.  
- Iteratively refining figures and outputs for clarity.

All results and code have been reviewed and validated by the author.
