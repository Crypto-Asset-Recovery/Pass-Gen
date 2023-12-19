import argparse, torch
from src.prepareData import prepareData
from src.preprocessData import get_dataloaders

from src.model import RNN
from src.train import train
from src.generate import generate_password

from config import *

def main(args):
    # Preprocess data and create dataloaders
    train_data, val_data, vocab = prepareData()
    train_loader, val_loader = get_dataloaders(train_data, val_data, args.batch_size, args.seq_len)

    # Create RNN model
    model = RNN(len(vocab), args.embedding_size, args.hidden_size, args.num_layers, args.dropout)

    # Train model
    trained_model = train(model, train_loader, val_loader, args.num_epochs, args.lr, args.gradient_accumulation_steps)

    # Save the trained model
    torch.save(trained_model.state_dict(), args.model_path)

    # Load the trained model
    loaded_model = RNN(len(vocab), args.embedding_size, args.hidden_size, args.num_layers, args.dropout)
    loaded_model.load_state_dict(torch.load(args.model_path))

    # Generate passwords
    for i in range(args.num_passwords):
        password = generate_password(loaded_model, vocab, args.password_length, args.temperature)
        print(f"Password {i + 1}: {password}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Password Generator")
    parser.add_argument("--batch_size", default=batch_size, type=int)
    parser.add_argument("--seq_len", default=seq_len, type=int)
    parser.add_argument("--num_epochs", default=num_epochs, type=int)
    parser.add_argument("--lr", default=lr, type=float)
    parser.add_argument("--password_length", default=password_length, type=int)
    parser.add_argument("--num_passwords", default=num_passwords, type=int)
    parser.add_argument("--temperature", default=temperature, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=gradient_accumulation_steps, type=int)
    parser.add_argument("--embedding_size", default=embedding_size, type=int)
    parser.add_argument("--hidden_size", default=hidden_size, type=int)
    parser.add_argument("--num_layers", default=num_layers, type=int)
    parser.add_argument("--dropout", default=dropout, type=float)
    parser.add_argument("--model_path", default=model_path, type=str)

    args = parser.parse_args()
    main(args)
