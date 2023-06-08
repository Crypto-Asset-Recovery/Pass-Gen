import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.num_layers = num_layers
        self.hidden_size = hidden_size  # add this line

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.zeros_(param)


    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                  weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
        return hidden

class FineTuneModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FineTuneModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x