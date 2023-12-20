import torch

EOS_TOKEN = '<eos>'

# Training parameters
batch_size = 32
seq_len = 16
num_epochs = 20
lr = 0.0035
gradient_accumulation_steps = 2
wordlist_path = "wordlists/rockyou.txt"

# Fine tuning parameters
batch_size_ft = 16
num_epochs_ft = 5
lr_ft = [0.0035, 0.0035, 0.005]
model_path_ft = "models/finetuned_model.pth"

# Model parameters
embedding_size = 256
hidden_size = 512
num_layers = 4
dropout = 0.2
model_path = "models/trained_model_1m.pth"

# Password generation parameters
password_length = 15
num_passwords = 100
temperature = 0.4

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
