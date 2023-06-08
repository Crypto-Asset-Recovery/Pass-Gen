import argparse, torch
from src.prepareData import prepareData
from src.preprocessData import get_dataloaders

from src.model import RNN
from src.train import train
from src.generate import generate_password

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
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--seq_len", default=20, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--lr", default=0.0035, type=float)
    parser.add_argument("--password_length", default=10, type=int)
    parser.add_argument("--num_passwords", default=20, type=int)
    parser.add_argument("--temperature", default=0.5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--embedding_size", default=128, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--num_layers", default=8, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--model_path", default="trained_model.pth", type=str)

    args = parser.parse_args()
    main(args)
