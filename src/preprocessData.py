import torch
from torch.utils.data import Dataset, DataLoader

from config import *

class PasswordDataset(Dataset):
    def __init__(self, passwords, seq_length):
        self.passwords = passwords
        self.seq_length = seq_length

    def __len__(self):
        return len(self.passwords)

    def __getitem__(self, idx):
        password = self.passwords[idx]

        # If password is shorter than seq_length, pad with zeros
        if len(password) < self.seq_length:
            password = torch.cat((password, torch.zeros(self.seq_length - len(password)).long()))

        # Truncate password if longer than seq_length
        password = password[:self.seq_length]

        return password[:-1], password[1:]

def get_dataloaders(train_data, val_data, batch_size, seq_length):
    train_dataset = PasswordDataset(train_data, seq_length)
    val_dataset = PasswordDataset(val_data, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    return train_loader, val_loader
