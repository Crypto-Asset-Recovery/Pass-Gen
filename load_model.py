import argparse, torch
from src.prepareData import prepareData
from src.preprocessData import get_dataloaders

from src.model import RNN
from src.train import train
from src.generate import generate_password

from config import *

def main(args):
    checkpoint = torch.load(model_path, map_location=device)

    model = RNN(
        len(checkpoint['vocab']),
        checkpoint['embedding_size'],
        checkpoint['hidden_size'],
        checkpoint['num_layers'],
        checkpoint['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    vocab = checkpoint['vocab']
    
    # Generate passwords
    passwords = []
    count = 0
    for i in range(args.num_passwords):
        password = generate_password(model, vocab, args.password_length, args.temperature, seed=args.seed)
        print(f"{count}/{args.num_passwords}: {password.replace('.', '')}")
        passwords.append(password.replace('.', ''))
        count += 1
    
    # Check how many of the generated passwords are in the file
    percentage = check_passwords_in_file(passwords, 'wordlists/top10m.txt')
    print(f"{percentage}% of the generated passwords found in top1m.txt")

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
    parser.add_argument("--batch_size", default=batch_size, type=int)
    parser.add_argument("--seq_len", default=seq_len, type=int)
    parser.add_argument("--password_length", default=password_length, type=int)
    parser.add_argument("--num_passwords", default=num_passwords, type=int)
    parser.add_argument("--temperature", default=temperature, type=float)
    parser.add_argument("--embedding_size", default=embedding_size, type=int)
    parser.add_argument("--hidden_size", default=hidden_size, type=int)
    parser.add_argument("--num_layers", default=num_layers, type=int)
    parser.add_argument("--dropout", default=dropout, type=float)
    parser.add_argument("--model_path", default=model_path, type=str)
    parser.add_argument("--load_num_layers", default=num_layers, type=int)
    parser.add_argument("--seed", default=None, type=str, help="Seed for password generation")

    args = parser.parse_args()
    main(args)
