import torch

# Training parameters
batch_size = 32
seq_len = 25
num_epochs = 5
lr = 0.0035
gradient_accumulation_steps = 4
wordlist_path = "wordlists/top100k.txt"

# Model parameters
embedding_size = 128
hidden_size = 512
num_layers = 8
dropout = 0.5
model_path = "models/trained_model.pth"

# Password generation parameters
password_length = 10
num_passwords = 20
temperature = 0.5

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")