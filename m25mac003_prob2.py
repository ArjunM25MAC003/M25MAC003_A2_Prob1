"""PROBLEM 2: CHARACTER-LEVEL NAME GENERATION USING RNN
VARIANTS
Objective: The objective of this assignment is to design and 
compare sequence models for character-level name generation
using recurrent neural architectures."""

# TASK-0: Dataset
import torch
import torch.nn as nn
import random

# Initial dataset of names
# Read dataset from file
with open("TrainingNames.txt", "r") as f:
    names = [line.strip() for line in f if line.strip()]

# Function to augment data with noise
def augment_name(name):
    if random.random() < 0.3:
        i = random.randint(0, len(name)-1)
        name = name[:i] + name[i].lower() + name[i+1:]
    return name

training_names = []
for name in names:
    training_names.append(name.lower())

# Character vocabulary
chars = sorted(set("." + "".join(training_names)))  # create set of unique characters
stoi = {ch: i for i, ch in enumerate(chars)}  # character to index mapping
itos = {i: ch for ch, i in stoi.items()}      # index to character mapping
vocab_size = len(chars)

# Create training data
# each word is converted into input-output character pairs
data = []
for word in training_names:
    word = "." + word + "."
    x = [stoi[ch] for ch in word[:-1]]
    y = [stoi[ch] for ch in word[1:]]

    x = torch.tensor([x])
    y = torch.tensor([y])
    data.append((x,y))

# Task-1: Model Implementation

# (1) Vanilla RNN : predict next character based on previous characters 
class fn_vanilla_rnn(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size) # converts character indices to dense vectors
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True) # processes the sequence of embeddings
        self.fc = nn.Linear(hidden_size,vocab_size)  # maps RNN output to character probabilities

    def forward(self,x):
        x= self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out


# (2) BLSTM : captures both forward and backward context in the sequence, improving prediction of next character
class fn_blstm(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.blstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True) # processes the sequence in both directions
        self.fc = nn.Linear(hidden_size*2, vocab_size)  

    def forward(self,x):
        x= self.embedding(x)
        out, _ = self.blstm(x)
        out = self.fc(out)
        return out

# (3) RNN with Attention : allows the model to focus on relevant parts of the input sequence when 
      #predicting the next character, improving performance on longer sequences
class fn_rnn_attention(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1) # computes attention weights for each time step in the RNN output
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self,x):
        x= self.embedding(x)
        out, _ = self.rnn(x) # out: (batch, seq_len, hidden)
        
        attn_weights = torch.softmax(self.attention(out), dim=1) # (batch, seq_len, 1)
        context = attn_weights * out  # (batch, seq_len, hidden) - keep sequence info

        out = self.fc(context)     # (batch, seq_len, vocab_size)
        return out


# Training function : trains the model using cross-entropy loss and Adam optimizer, iterating 
# over the training data for a specified number of epochs
def fn_train(model, data, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        for x,y in data:
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data)}")

# Sampling function : samples the next character from the model's output probabilities, 
# using top-k sampling to select from the most likely characters and adding temperature scaling to control randomness in the generation process
def sample_top_k(probs, k=5):
    topk_probs, topk_idx = torch.topk(probs, k)
    topk_probs = topk_probs / torch.sum(topk_probs)  # normalize
    idx = torch.multinomial(topk_probs, 1).item()
    return topk_idx[idx].item()

# NAME GENERATION FUNCTION : generates new names by sampling characters from the 
# model's output probabilities, starting with a random character and iteratively predicting the next character until a specified maximum length is reached
def fn_generate_name(model, max_len=12, temperature=0.8, k=5):
    model.eval()
    name = ""
    ch = "."

    for _ in range(max_len):
        seq = [stoi[c] for c in "." + name]
        x = torch.tensor([seq])

        out = model(x)
        logits = out[0][-1] / temperature

        logits[stoi["."]] -= 1.5  # reduce early stop

        probs = torch.softmax(logits, dim=0)
        next_idx = sample_top_k(probs, k)

        ch = itos[next_idx]

        if ch == "." and len(name) > 2:
            break

        name += ch

    return name.capitalize() 

# Generate unique names by sampling from the model and ensuring they are not present 
# in the training dataset, allowing for a specified number of unique names to be generated
def generate_unique_names(model, training_names, n=200):
    generated = []
    training_set = set(training_names)

    while len(generated) < n:
        name = fn_generate_name(model)
        if name.lower() not in training_set:
            generated.append(name)

    return generated

# TASK-3: QUANTITATIVE EVALUATION
# Novelty Rate: measures the proportion of generated names that are not present in 
# the training dataset, indicating the model's ability to create new names.
def fn_novelty_rate(generated, training):
    training = set([w.lower() for w in training])
    generated = [w.lower() for w in generated]
    
    new = [w for w in generated if w not in training]
    return len(new) / len(generated)
# Diversity: measures the variety of generated names by calculating the ratio of unique 
# names to total generated names, indicating the model's ability to produce a wide range of names.
def fn_diversity(generated):
    return len(set(generated)) / len(generated)   

# RUN ALL MODELS

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

hidden_size = 128

models = {
    "Vanilla RNN": fn_vanilla_rnn(vocab_size, hidden_size),
    "BLSTM": fn_blstm(vocab_size, hidden_size),
    "Attention RNN": fn_rnn_attention(vocab_size, hidden_size)
}

results = {}

for model_name, model in models.items():
    print(f"\n{model_name}")

    # Parameter count
    params = count_parameters(model)
    size_mb = params * 4 / (1024 ** 2)  # Assuming float32 (4 bytes)
    print(f"Total Parameters: {params}")
    print(f"Model Size: {size_mb:.4f} MB")

    # Train model
    fn_train(model, data, epochs=20)

    # Generate names
    generated_names = generate_unique_names(model, training_names, n=200)

    # Metrics
    novelty = fn_novelty_rate(generated_names, training_names)
    diversity_score = fn_diversity(generated_names)

    # Store
    results[model_name] = (novelty, diversity_score)

    print(f"Novelty Rate: {novelty:.4f}")
    print(f"Diversity: {diversity_score:.4f}")
    print("Sample Names:", generated_names[:10])

# FINAL COMPARISON

print("\nFINAL COMPARISON")
for model_name, (n, d) in results.items():
    print(f"{model_name}: Novelty = {n:.4f}, Diversity = {d:.4f}")

# TASK-4: QUALITATIVE ANALYSIS

print("\n  QUALITATIVE ANALYSIS \n")

# Show sample generated names
print("Sample Generated Names (First 20):")
print(generated_names[:20])

# Failure modes detection
print("\nCommon Failure Modes")

repetition_errors = [name for name in generated_names if any(name.count(c) > 2 for c in set(name))]
short_names = [name for name in generated_names if len(name) <= 2]
duplicates = [name for name in generated_names if generated_names.count(name) > 1]

print(f"Repetition Errors (examples): {repetition_errors[:5]}")
print(f"Too Short Names (examples): {short_names[:5]}")
print(f"Duplicate Names (examples): {duplicates[:5]}")