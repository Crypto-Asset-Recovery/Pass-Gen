import torch
import random

import string

def generate_password(model, vocab, length, temperature=1.0):
    # Helper function to count character types
    def count_char_types(password):
        count_alpha = sum(c in string.ascii_letters for c in password)
        count_digits = sum(c.isdigit() for c in password)
        count_symbols = sum(c in string.punctuation for c in password)
        
        return count_alpha, count_digits, count_symbols
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while True:
        # Initialize hidden state
        hidden = model.init_hidden(1)
        password = []

        # Choose a random starting character
        start_char = random.choice(list(vocab.get_stoi().keys()))
        password.append(start_char)

        # Generate password
        while len(password) < length:
            # Convert character to tensor
            input_tensor = torch.tensor([vocab.get_stoi()[password[-1]]]).unsqueeze(0).to(device)

            # Forward pass
            output, hidden = model(input_tensor, hidden)

            # Sample from output distribution
            distribution = output.squeeze().div(temperature).exp()
            char_idx = torch.multinomial(distribution, 1).item()

            # Convert tensor back to character and add to password
            char = vocab.get_itos()[char_idx]

            # Break if <eos> token is encountered
            if char == '<eos>':
                break

            password.append(char)

        password_str = "".join(password)

        # Check password length and character types
        if len(password_str) >= 10:
            count_alpha, count_digits, count_symbols = count_char_types(password_str)
            if sum([count_alpha > 0, count_digits > 0, count_symbols > 0]) >= 2:
                break

    return password_str

