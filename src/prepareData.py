import re, torch, torchtext
from sklearn.model_selection import train_test_split

import string

from config import *

def has_alpha_numeric_symbol(password):
    has_alpha = any(c.isalpha() for c in password)
    has_numeric = any(c.isdigit() for c in password)
    has_symbol = any(c in string.punctuation for c in password)

    count = 0
    if has_alpha:
        count += 1
    if has_numeric:
        count += 1
    if has_symbol:
        count += 1

    return count >= 2


def prepareData(data = None):
    if data is None:
        with open(wordlist_path, 'r', encoding='utf-8', errors='ignore') as f:
            passwords = f.readlines()
    else:
        passwords = data

    # Remove passwords that have more than 2 non-alphanumeric characters (symbols) or start with a symbol
    passwords = [password for password in passwords if sum(not char.isalnum() for char in password) <= 2 and password[0].isalnum()]

    # Add end-of-sequence token
    passwords = [password.strip() + EOS_TOKEN for password in passwords]

    # Define regular expression to remove non-ASCII characters
    regex = re.compile(r'[^\x00-\x7F]+')

    # Clean passwords
    passwords = [regex.sub('', password.strip()) for password in passwords]

    # Define tokenizer
    #tokenizer = torchtext.data.utils.get_tokenizer('char')

    # Tokenize passwords
    passwords = [char_tokenizer(password) for password in passwords]

    # Define vocabulary
    vocab = torchtext.vocab.build_vocab_from_iterator(passwords, specials=[EOS_TOKEN])

    # Convert passwords to tensors
    passwords = [torch.tensor([vocab[token] for token in password], dtype=torch.long) for password in passwords]

    if data is None:
        train_data, val_data = train_test_split(passwords, test_size=0.2, random_state=42)
        return train_data, val_data, vocab
    else:
        train_data = passwords
        val_data = []
    return train_data, val_data, vocab

def char_tokenizer(password):
    return list(password)

if __name__ == '__main__':
    print('Preparing data...')
    train_data, val_data = prepareData()
    print('Done!')
    print(train_data[0])
    print(val_data[0])
    
