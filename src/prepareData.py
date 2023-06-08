import re, torch, torchtext
from sklearn.model_selection import train_test_split

import string

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
        with open('wordlists/top100k.txt', 'r', encoding='utf-8', errors='ignore') as f:
            passwords = f.readlines()
    else:
        passwords = data

    # Define regular expression to remove non-ASCII characters
    regex = re.compile(r'[^\x00-\x7F]+')

    # Clean passwords
    passwords = [regex.sub('', password.strip()) for password in passwords]

    # Define tokenizer
    #tokenizer = torchtext.data.utils.get_tokenizer('char')

    # Tokenize passwords
    passwords = [char_tokenizer(password) for password in passwords]

    # Define vocabulary
    vocab = torchtext.vocab.build_vocab_from_iterator(passwords)

    # Convert passwords to tensors
    passwords = [torch.tensor([vocab[token] for token in password]) for password in passwords]

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
    