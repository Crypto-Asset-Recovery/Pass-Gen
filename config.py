import torch

# Training parameters
batch_size = 32
seq_len = 20
num_epochs = 20
lr = 0.0025
gradient_accumulation_steps = 2
wordlist_path = "wordlists/top10m-10chars.txt"

# Model parameters
embedding_size = 128
hidden_size = 512
num_layers = 4
dropout = 0.3
model_path = "models/trained_model.pth"

# Password generation parameters
password_length = 10
num_passwords = 20
temperature = 0.5

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
