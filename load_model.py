import argparse, torch
from src.prepareData import prepareData
from src.preprocessData import get_dataloaders

from src.model import RNN
from src.train import train
from src.generate import generate_password


def main(args):
    # Preprocess data and create dataloaders
    _, _, vocab = prepareData()

    # Create RNN model
    num_layers = args.num_layers
    if args.load_num_layers:
        num_layers = args.load_num_layers

    model = RNN(len(vocab), args.embedding_size, args.hidden_size, num_layers, args.dropout)

    # Load the trained model
    model_path = 'models/100k/fine-tuned-5-epochs-4-gradient-accumulation.pth'

    if args.model_path:
        model_path = args.model_path

    model.load_state_dict(torch.load(model_path))

    # Generate passwords
    passwords = []
    count = 0
    for i in range(args.num_passwords):
        password = generate_password(model, vocab, args.password_length, args.temperature)
        print(f"{count}/{args.num_passwords}: {password.replace('.', '')}")
        passwords.append(password.replace('.', ''))
        count += 1
    
    # Check how many of the generated passwords are in the file
    percentage = check_passwords_in_file(passwords, 'wordlists/rockyou.txt')
    print(f"{percentage}% of the generated passwords found in rockyou.txt")

    unique = unique_percentage(passwords)
    print(f"{unique}% of the generated passwords are unique")

def check_passwords_in_file(passwords, file_path):
    # Read the passwords from the given file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        file_passwords = f.readlines()
        
    # Clean and store the passwords in a set
    file_passwords = {password.strip() for password in file_passwords}
    
    # Check how many generated passwords are in the file
    matching_passwords = sum(1 for password in passwords if password in file_passwords)

    # Calculate the percentage of matching passwords
    percentage = (matching_passwords / len(passwords)) * 100

    return percentage

def unique_percentage(input_list):
    if not input_list:
        return 0
    
    unique_items = set(input_list)
    unique_count = len(unique_items)
    total_count = len(input_list)

    percentage = (unique_count / total_count) * 100
    return percentage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Password Generator")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--seq_len", default=20, type=int)
    parser.add_argument("--password_length", default=10, type=int)
    parser.add_argument("--num_passwords", default=20, type=int)
    parser.add_argument("--temperature", default=0.5, type=float)
    parser.add_argument("--embedding_size", default=128, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--load_num_layers", default=None, type=int)

    args = parser.parse_args()
    main(args)
