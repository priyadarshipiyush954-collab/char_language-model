import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# -----------------------------
# Dataset
# -----------------------------
text = """
hello world
hello python
hello machine learning
hello artificial intelligence
language models predict tokens
deep learning is powerful
transformers changed NLP
""".strip().lower()  # .strip() removes accidental leading/trailing newlines

# -----------------------------
# Character vocabulary
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
# Model
# -----------------------------
class CharLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.linear1 = nn.Linear(32, 64)
        self.linear2 = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.linear1(x))
        logits = self.linear2(x)
        return logits

model = CharLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# -----------------------------
# Training
# -----------------------------
block_size = 8

for epoch in range(5000):
    start = torch.randint(0, len(data) - block_size - 1, (1,)).item()
    x = data[start:start + block_size].unsqueeze(0).to(device)
    y = data[start + 1:start + block_size + 1].unsqueeze(0).to(device)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -----------------------------
# Generate text
# -----------------------------
def generate(start_text, max_new_chars=100):
    model.eval()
    
    # Ensure start_text is lowercase and mapped correctly
    encoded_prompt = encode(start_text.lower())
    ids = torch.tensor([encoded_prompt], dtype=torch.long).to(device)

    with torch.no_grad():  # Turn off gradients during inference
        for _ in range(max_new_chars):
            # Crop context to block_size if it exceeds it
            cond_ids = ids[:, -block_size:]
            
            logits = model(cond_ids)
            # Focus only on the last time-step
            logits = logits[:, -1, :]
            
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            
            ids = torch.cat([ids, next_token], dim=1)

    return decode(ids[0].tolist())

print("\nGenerated text:")
print(generate("hello", 100))