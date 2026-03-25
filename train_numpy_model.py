import numpy as np
import collections
import time
import sys

# Parameters
filename = 'clean_corpus.txt'
EMBED_SIZE = 300
WINDOW_SIZE = 5
NEGATIVE_SAMPLES = 5
LR = 0.05
MAX_STEPS = 50000 

print("Loading corpus...", flush=True)
try:
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().lower().split()
except FileNotFoundError:
    print("clean_corpus.txt not found. Exiting.")
    sys.exit(1)

print(f"Total tokens: {len(text)}", flush=True)

if not text:
    print("Corpus is empty.")
    sys.exit(1)

# Build Vocab
counts = collections.Counter(text)
vocab = sorted(counts, key=counts.get, reverse=True)
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}", flush=True)

# Init Weights
np.random.seed(42)  # For reproducibility
# Initialize with small random values
scale = 1.0 / np.sqrt(EMBED_SIZE)
W_in = np.random.uniform(-scale, scale, (vocab_size, EMBED_SIZE))
W_out = np.random.uniform(-scale, scale, (vocab_size, EMBED_SIZE))

# Sigmoid
def sigmoid(x):
    # Clip to avoid overflow
    x = np.clip(x, -6, 6)
    return 1.0 / (1.0 + np.exp(-x))

print("Training (Simplified Skip-Gram)...", flush=True)

indices = [word2idx[w] for w in text if w in word2idx]
step = 0
start_time = time.time()

# Training loop
for i, center_idx in enumerate(indices):
    if step >= MAX_STEPS:
        break
    
    if i % 1000 == 0 and i > 0:
        print(f"Step {step}/{MAX_STEPS}", end='\r')

    # context window
    start = max(0, i - WINDOW_SIZE)
    end = min(len(indices), i + WINDOW_SIZE + 1)
    context_indices = indices[start:i] + indices[i+1:end]
    
    if not context_indices:
        continue
        
    for context_idx in context_indices:
        if step >= MAX_STEPS: break
        
        # 1. Positive sample
        emb_center = W_in[center_idx]
        emb_context = W_out[context_idx]
        
        score = np.dot(emb_center, emb_context)
        p = sigmoid(score)
        
        # Gradient of Loss wrt score: (p - target)
        # target = 1
        g = (p - 1.0) * LR
        
        # Gradients wrt vectors
        grad_center = g * emb_context
        grad_context = g * emb_center
        
        # Update context (W_out)
        W_out[context_idx] -= grad_context
        
        # 2. Negative samples
        bad_indices = np.random.randint(0, vocab_size, NEGATIVE_SAMPLES)
        
        emb_negs = W_out[bad_indices]
        scores_neg = np.dot(emb_negs, emb_center)
        p_neg = sigmoid(scores_neg)
        
        # target = 0
        g_neg = (p_neg - 0.0) * LR
        
        # Accumulate gradient for center from negatives
        # dot(g_neg, emb_negs) performs sum(scalar * vector)
        grad_center += np.dot(g_neg, emb_negs)
        
        # Update negatives (W_out)
        # We need to subtract g_neg * emb_center from each bad_index row
        # W_out[bad_indices] -= g_neg[:, None] * emb_center
        for bi, gi in zip(bad_indices, g_neg):
            W_out[bi] -= gi * emb_center

        # Update center (W_in)
        W_in[center_idx] -= grad_center
        
        step += 1

print(f"\nTraining finished after {step} steps. Duration: {time.time() - start_time:.2f}s", flush=True)

# Select a word
if len(sys.argv) > 1:
    target_word = sys.argv[1].lower()
else:
    target_word = "research"

if target_word not in word2idx:
    # Fallback for default if not found
    target_word = "student"

if target_word in word2idx:
    vector = W_in[word2idx[target_word]]
    vector_str = ", ".join([f"{x:.4f}" for x in vector])
    # print(f"{target_word} - {vector_str}")

# --- Potential Analogies ---
def get_embedding(word):
    if word in word2idx:
        return W_in[word2idx[word]]
    return None

def find_analogy(w1, w2, w3):
    if w1 not in word2idx or w2 not in word2idx or w3 not in word2idx:
        return None
    
    vec1 = W_in[word2idx[w1]]
    vec2 = W_in[word2idx[w2]]
    vec3 = W_in[word2idx[w3]]
    
    target_vec = vec2 - vec1 + vec3
    
    # Calculate cosine similarity with all vectors
    # W_in shape (vocab_size, EMBED_SIZE)
    # target_vec shape (EMBED_SIZE,)
    
    # Dot product
    dots = np.dot(W_in, target_vec)
    
    # Norms
    norm_w = np.linalg.norm(W_in, axis=1)
    norm_t = np.linalg.norm(target_vec)
    
    # Avoid div by zero
    norm_w[norm_w == 0] = 1e-10
    if norm_t == 0: norm_t = 1e-10

    sims = dots / (norm_w * norm_t)
    
    # Find most similar words (excluding w1, w2, w3)
    indices = np.argsort(sims)[::-1]
    
    results = []
    count = 0
    for idx in indices:
        word = vocab[idx]
        if word not in [w1, w2, w3]:
            results.append((word, sims[idx]))
            count += 1
            if count >= 3:
                break
    return results

print("\n--- Analogy Search Results ---")
candidates = [
    # General
    ("student", "learning", "teacher"),
    ("student", "study", "teacher"),
    ("good", "better", "bad"),
    # Domain
    ("student", "research", "faculty"),
    ("teaching", "learning", "research"),
    ("computer", "science", "electrical"),
    ("text", "image", "visual"),
    ("vision", "image", "text"),
    ("learning", "knowledge", "research"),
    ("iit", "jodhpur", "institute"), 
    ("paper", "conference", "journal")
]

for w1, w2, w3 in candidates:
    res = find_analogy(w1, w2, w3)
    if res:
        best_match = res[0]
        print(f"{w1} : {w2} :: {w3} : {best_match[0]} (score: {best_match[1]:.4f})")
    else:
        pass
