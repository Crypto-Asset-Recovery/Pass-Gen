import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                  weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
        return hidden