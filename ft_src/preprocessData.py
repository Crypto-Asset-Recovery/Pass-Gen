import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from prepareData import prepare_data

def encode_sequences(inputs, targets):
    all_chars = sorted(set(''.join(inputs + targets)))
    char_to_index = {char: index for index, char in enumerate(all_chars)}

    encoded_inputs = [[char_to_index[char] for char in sequence] for sequence in inputs]
    encoded_targets = [[char_to_index[char] for char in sequence] for sequence in targets]

    return encoded_inputs, encoded_targets, char_to_index, all_chars

def one_hot_encode(encoded_inputs, encoded_targets, num_classes):
    one_hot_inputs = np.zeros((len(encoded_inputs), len(encoded_inputs[0]), num_classes), dtype=np.float32)
    one_hot_targets = np.zeros((len(encoded_targets), len(encoded_targets[0]), num_classes), dtype=np.float32)

    for i, sequence in enumerate(encoded_inputs):
        for j, value in enumerate(sequence):
            one_hot_inputs[i, j, value] = 1

    for i, sequence in enumerate(encoded_targets):
        for j, value in enumerate(sequence):
            one_hot_targets[i, j, value] = 1

    return one_hot_inputs, one_hot_targets

def preprocess_data(inputs, targets):
    encoded_inputs, encoded_targets, char_to_index, all_chars = encode_sequences(inputs, targets)
    num_classes = len(all_chars)
    one_hot_inputs, one_hot_targets = one_hot_encode(encoded_inputs, encoded_targets, num_classes)
    
    return one_hot_inputs, one_hot_targets, char_to_index, all_chars

if __name__ == '__main__':
    # Assuming the 'inputs' and 'targets' variables are already obtained from the data preparation step
    file_path = 'wordlists/user.txt'
    sequence_length = 10
    inputs, targets = prepare_data(file_path, sequence_length)
    one_hot_inputs, one_hot_targets, char_to_index, all_chars = preprocess_data(inputs, targets)
