import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Device Configuration
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# Dataset (Expanded slightly for better learning)
# -----------------------------
text = """
hello world
hello python
hello machine learning
hello artificial intelligence
language models predict tokens
deep learning is powerful
transformers changed NLP
recurrent neural networks remember sequential context
long short term memory networks handle long dependencies
generative artificial intelligence creates new content
python is the primary language for deep learning
""".strip().lower()

# -----------------------------
# Character Vocabulary & Encoders
# -----------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

# -----------------------------
# Model Architecture (LSTM)
# -----------------------------
class CharLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        out = self.embedding(x)
        out, hidden = self.lstm(out, hidden)
        logits = self.fc(out)
        return logits, hidden

model = CharLSTMModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

# -----------------------------
# Training Setup
# -----------------------------
block_size = 16  # Increased sequence context window
batch_size = 1

print(f"Training on {len(data)} characters with vocabulary size {vocab_size}...\n")

model.train()
for epoch in range(3001):
    # Pick a random starting index for training sequence
    start = torch.randint(0, len(data) - block_size - 1, (1,)).item()
    
    x = data[start : start + block_size].unsqueeze(0).to(device)
    y = data[start + 1 : start + block_size + 1].unsqueeze(0).to(device)

    logits, _ = model(x)

    # Flatten tensors for Cross Entropy Loss
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

# -----------------------------
# Text Generation with Temperature
# -----------------------------
def generate(start_text, max_new_chars=120, temperature=0.6):
    """
    temperature:
      - 0.2 to 0.5: Very focused, deterministic output (fewer typos)
      - 0.6 to 0.8: Balanced creativity and structure
      - 1.0+: Highly creative/random output
    """
    model.eval()
    
    encoded_prompt = encode(start_text.lower())
    if not encoded_prompt:
        encoded_prompt = [0]
        
    ids = torch.tensor([encoded_prompt], dtype=torch.long).to(device)
    
    hidden = None

    with torch.no_grad():
        for _ in range(max_new_chars):
            # Crop context to block_size
            cond_ids = ids[:, -block_size:]
            
            logits, hidden = model(cond_ids)
            
            # Extract last character predictions
            logits = logits[:, -1, :]
            
            # Apply Temperature scaling before Softmax
            logits = logits / temperature
            probabilities = F.softmax(logits, dim=-1)
            
            # Sample next token based on scaled probabilities
            next_token = torch.multinomial(probabilities, num_samples=1)
            
            # Append predicted token to sequence
            ids = torch.cat([ids, next_token], dim=1)

    return decode(ids[0].tolist())

# -----------------------------
# Testing Generation
# -----------------------------
print("\n--- Generated Output (Temperature = 0.5) ---")
print(generate("hello", max_new_chars=100, temperature=0.5))

print("\n--- Generated Output (Temperature = 0.8) ---")
print(generate("hello", max_new_chars=100, temperature=0.8))