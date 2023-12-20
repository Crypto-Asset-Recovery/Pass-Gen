from src.prepareData import prepareData
import argparse, torch
from src.preprocessData import get_dataloaders

from torch import nn
from src.model import RNN
from src.train import train
from src.generate import generate_password

from config import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune a model')
    parser.add_argument('--model_path', default=model_path, type=str)
    parser.add_argument('--embedding_size', default=embedding_size, type=int)
    parser.add_argument('--hidden_size', default=hidden_size, type=int)
    parser.add_argument('--num_layers', default=num_layers, type=int)
    parser.add_argument('--dropout', default=dropout, type=float)
    parser.add_argument('--batch_size', default=batch_size_ft, type=int)
    parser.add_argument('--seq_len', default=seq_len, type=int)
    parser.add_argument('--num_epochs', default=num_epochs_ft, type=int)
    parser.add_argument('--lr', default=lr_ft, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=gradient_accumulation_steps, type=int)
    parser.add_argument('--temperature', default=temperature, type=float)
    parser.add_argument('--password_length', default=password_length, type=int)
    parser.add_argument('--num_passwords', default=num_passwords, type=int)
    args = parser.parse_args()
    
    train_data, val_data, existing_vocab = prepareData()
    new_train_loader, new_val_loader = get_dataloaders(train_data, val_data, args.batch_size, args.seq_len) # batch_size, seq_len

    new_output_dim = len(existing_vocab)  # The size of the original vocabulary

    pretrained_model = RNN(len(existing_vocab), args.embedding_size, args.hidden_size, args.num_layers, args.dropout)
    pretrained_model.load_state_dict(torch.load(args.model_path))

    pretrained_model.fc = nn.Linear(pretrained_model.hidden_size, new_output_dim)

    # Set requires_grad to False for all layers except the output layer
    for name, param in pretrained_model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    trained_model = train(
        pretrained_model,
        new_train_loader,
        new_val_loader,
        args.num_epochs,
        args.lr,
        args.gradient_accumulation_steps
    )  # Train the model

    # Save the fine-tuned model
    torch.save(trained_model.state_dict(), model_path_ft)

