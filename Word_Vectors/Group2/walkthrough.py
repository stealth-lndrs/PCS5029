import itertools
import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt

# ---------------- 1) Corpus ----------------
corpus = [
    "alice likes cheese and bread",
    "bob likes fish and rice",
    "cheese is dairy",
    "fish is seafood",
    "bread and rice are carbs"
]

def tokenize(s): return s.lower().split()
tokenized = [tokenize(s) for s in corpus]
vocab = sorted(set(itertools.chain.from_iterable(tokenized)))
V = len(vocab)
idx = {w: i for i, w in enumerate(vocab)}

# ---------------- 2) Co-occurrence (symmetric, window=2) ----------------
window = 2
C = np.zeros((V, V), float)
for sent in tokenized:
    for i, w in enumerate(sent):
        wi = idx[w]
        for j in range(max(0, i - window), min(len(sent), i + window + 1)):
            if j == i: 
                continue
            C[wi, idx[sent[j]]] += 1.0

C_df = pd.DataFrame(C.astype(int), index=vocab, columns=vocab)

# ---------------- 3) PMI / PPMI ----------------
total = C.sum()
row = C.sum(axis=1, keepdims=True)
col = C.sum(axis=0, keepdims=True)

with np.errstate(divide='ignore', invalid='ignore'):
    p_wc = C / total if total > 0 else np.zeros_like(C)
    p_w = row / total if total > 0 else np.zeros_like(row)
    p_c = col / total if total > 0 else np.zeros_like(col)
    PMI = np.log2(p_wc / (p_w @ p_c))
PMI[~np.isfinite(PMI)] = 0.0
PPMI = np.maximum(PMI, 0.0)

PMI_df = pd.DataFrame(np.round(PMI, 3), index=vocab, columns=vocab)
PPMI_df = pd.DataFrame(np.round(PPMI, 3), index=vocab, columns=vocab)

# ---------------- 4) Context-expansion dimensionality reduction ----------------
def build_axis(anchor):
    col = PPMI_df[anchor]
    neighbors = col[col > 0]
    if neighbors.empty:
        return PPMI_df[anchor].copy(), [anchor]
    weights = neighbors
    bundle = pd.Series(0.0, index=vocab)
    for c, w in weights.items():
        bundle += PPMI_df[c] * float(w)
    return bundle, list(neighbors.index)

axis_likes, N_likes = build_axis("likes")
axis_dairy, N_dairy = build_axis("dairy")

W2_bundle = pd.DataFrame({"likes_bundle": axis_likes,
                          "dairy_bundle": axis_dairy}, index=vocab)

# z-score normalization for readability
def zscore(col):
    s = col.std()
    return (col - col.mean()) / s if s > 0 else col * 0

W2 = pd.DataFrame({"likes_ctx": zscore(W2_bundle["likes_bundle"]),
                   "dairy_ctx": zscore(W2_bundle["dairy_bundle"])},
                   index=vocab).round(3)

# ---------------- 5) Vector composition ----------------
w2v = {w: W2.loc[w].values for w in vocab}
def cos(a, b):
    na, nb = norm(a), norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

q = w2v["likes"] + w2v["dairy"]
cos_scores = {cand: round(cos(w2v[cand], q), 3) for cand in ["alice", "bob"]}
print("Cosine scores:", cos_scores)

# ---------------- 6) Plotting with epsilon-grouping for stacked labels ----------------
def grouped_scatter_stacked(ax, df2d, xlab, ylab, title, eps=0.05, stack_dy=0.07):
    pts = df2d[[xlab, ylab]].values
    words = list(df2d.index)
    groups = []
    for i, (x, y) in enumerate(pts):
        assigned = False
        for g in groups:
            dx = x - g["cx"]; dy = y - g["cy"]
            if (dx*dx + dy*dy) <= eps*eps:
                g["items"].append((words[i], x, y))
                g["cx"] = np.mean([it[1] for it in g["items"]])
                g["cy"] = np.mean([it[2] for it in g["items"]])
                assigned = True
                break
        if not assigned:
            groups.append({"cx": x, "cy": y, "items": [(words[i], x, y)]})
    for g in groups:
        cx, cy = g["cx"], g["cy"]
        ax.plot(cx, cy, marker='x', markersize=6, linestyle='None')
        items_sorted = sorted(g["items"], key=lambda t: t[0])
        for k, (w, _, _) in enumerate(items_sorted):
            ax.text(cx, cy + (k+1)*stack_dy, w, ha='center', va='bottom', fontsize=10)
    ax.axhline(0, linewidth=0.8); ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(title)
    return groups

# Word vectors plot
figA, axA = plt.subplots(figsize=(8,7))
grouped_scatter_stacked(
    axA,
    W2.rename(columns={"likes_ctx":"likes bundle (z)",
                       "dairy_ctx":"dairy bundle (z)"}),
    "likes bundle (z)", "dairy bundle (z)",
    "2D Word Vectors via Context-Expansion Bundles"
)
plt.show()

# Composition plot
figB, axB = plt.subplots(figsize=(8,7))
grouped_scatter_stacked(
    axB,
    W2.rename(columns={"likes_ctx":"likes bundle (z)",
                       "dairy_ctx":"dairy bundle (z)"}),
    "likes bundle (z)", "dairy bundle (z)",
    "“___ likes dairy?” — Context-Expansion Bundles"
)
# arrows
for name in ["likes", "dairy"]:
    vx, vy = w2v[name]
    axB.arrow(0, 0, vx, vy, head_width=0.05, length_includes_head=True)
qx, qy = q
axB.arrow(0, 0, qx, qy, head_width=0.05, length_includes_head=True)
# candidate rays
for who in ["alice", "bob"]:
    vx, vy = w2v[who]
    axB.plot([0, vx], [0, vy], linestyle="--", linewidth=1.0)
    axB.text(0.6*vx, 0.6*vy, f"cos(q,{who})={cos_scores[who]:.3f}", fontsize=10)
plt.show()
