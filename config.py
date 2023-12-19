import torch

EOS_TOKEN = '<eos>'

# Training parameters
batch_size = 32
seq_len = 20
num_epochs = 20
lr = 0.0035
gradient_accumulation_steps = 2
wordlist_path = "wordlists/top100k.txt"

# Model parameters
embedding_size = 256
hidden_size = 512
num_layers = 4
dropout = 0.2
model_path = "models/trained_model.pth"

# Password generation parameters
password_length = 15
num_passwords = 50
temperature = 0.8

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
