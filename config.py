import torch

EOS_TOKEN = '<eos>'

# Training parameters
batch_size = 32
seq_len = 16
num_epochs = 35
lr = 0.0035
gradient_accumulation_steps = 2
wordlist_path = "wordlists/top100k.txt"

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
model_path = "models/trained_model.pth"

# Password generation parameters
password_length = 15
num_passwords = 35
temperature = 0.5

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
