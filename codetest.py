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

fc = nn.Linear(hidden_dim, vocab_size)
