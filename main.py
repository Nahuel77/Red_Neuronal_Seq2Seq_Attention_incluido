import torch
import torch.nn as nn
import random
import numpy as np

# ----------------------------
# 1. Configuración
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 10         # Tokens posibles (0..9)
seq_length = 5          # Longitud fija de las secuencias
embedding_dim = 16
hidden_dim = 32
num_epochs = 2000
batch_size = 64

# ----------------------------
# 2. Generar dataset sintético
# ----------------------------
def generate_batch(batch_size, seq_length, vocab_size):
    X = np.random.randint(1, vocab_size, (batch_size, seq_length)) # 
    Y = X.copy()  # salida igual a la entrada
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

# ----------------------------
# 3. Definir modelo Seq2Seq
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        outputs, (h, c) = self.lstm(emb)
        return h, c

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h, c):
        emb = self.embedding(x)
        outputs, (h, c) = self.lstm(emb, (h, c))
        logits = self.fc(outputs)
        return logits, h, c

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(device)

        h, c = self.encoder(src)
        input = trg[:, 0].unsqueeze(1)  # primer token (podría ser un <SOS>)

        for t in range(1, trg_len):
            output, h, c = self.decoder(input, h, c)
            outputs[:, t] = output.squeeze(1)
            top1 = output.argmax(2)
            input = trg[:, t].unsqueeze(1) if random.random() < teacher_forcing_ratio else top1

        return outputs

# ----------------------------
# 4. Entrenamiento
# ----------------------------
encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
decoder = Decoder(vocab_size, embedding_dim, hidden_dim)
model = Seq2Seq(encoder, decoder).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(num_epochs):
    X, Y = generate_batch(batch_size, seq_length, vocab_size)
    X, Y = X.to(device), Y.to(device)

    optimizer.zero_grad()
    output = model(X, Y)
    loss = criterion(output[:, 1:].reshape(-1, vocab_size), Y[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# ----------------------------
# 5. Evaluación
# ----------------------------
model.eval()
with torch.no_grad():
    X_test, Y_test = generate_batch(5, seq_length, vocab_size)
    X_test = X_test.to(device)
    output = model(X_test, Y_test, teacher_forcing_ratio=0.0)
    preds = output.argmax(2).cpu().numpy()

    for i in range(5):
        print(f"\nInput:    {X_test[i].cpu().numpy()}")
        print(f"Predicho: {preds[i]}")
        print(f"Esperado: {Y_test[i].cpu().numpy()}")
