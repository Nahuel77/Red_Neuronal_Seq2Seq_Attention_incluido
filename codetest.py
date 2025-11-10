import torch
import torch.nn as nn
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 10         # Tokens posibles (0..9)
seq_length = 5          # Longitud fija de las secuencias
embedding_dim = 16
hidden_dim = 32
num_epochs = 2000
batch_size = 64

def generate_batch(batch_size, seq_length, vocab_size):
    X = np.random.randint(1, vocab_size, (batch_size, seq_length)) # 
    Y = X.copy()  # salida igual a la entrada
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

print(generate_batch(batch_size, seq_length, vocab_size))