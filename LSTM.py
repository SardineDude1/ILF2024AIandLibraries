import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # Add padding_idx for efficient padding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden):
        embed = self.embedding(x)
        output, hidden = cp.checkpoint(self.lstm, embed, hidden, use_reentrant=False)  # Use gradient checkpointing for LSTM
        output = self.dropout(output)
        logits = self.fc(output)
        return logits.view(-1, logits.size(-1)), hidden

    def init_hidden(self, batch_size, device):
        # Initialize hidden state with zeros
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device))