import math
import numpy as np
from segtok import tokenizer
import torch as th
from torch import nn


# Using a basic RNN/LSTM for Language modeling
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, rnn_size, num_layers=1, dropout=0):
        super().__init__()

        # Create an embedding layer of shape [vocab_size, rnn_size]
        # Use nn.Embedding
        # That will map each word in our vocab into a vector of rnn_size size.
        self.embedding = nn.Embedding(vocab_size, rnn_size)
        self.embed_dropout = nn.Dropout(dropout * 0.7)  # Specific embedding dropout

        # Create an LSTM layer of rnn_size size. Use any features you wish.
        # We will be using batch_first convention
        self.lstm = nn.LSTM(
            rnn_size,
            rnn_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # Add LayerNorm for better convergence
        self.layernorm = nn.LayerNorm(rnn_size)
        # LSTM layer does not add dropout to the last hidden output.
        # Add this if you wish.
        self.dropout = nn.Dropout(dropout)
        # Use a dense layer to project the outputs of the RNN cell into logits of
        # the size of vocabulary (vocab_size).
        self.output = nn.Linear(rnn_size, vocab_size)
        # Weight tying between embedding and output layer – helps reduce overfitting.
        self.output.weight = self.embedding.weight
        self.temp = math.sqrt(rnn_size)  # Temperature scaling

    def forward(self, x):
        embeds = self.embedding(x)
        embeds = self.embed_dropout(embeds)  # Apply embedding dropout
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.layernorm(lstm_out)  # Layer normalization
        lstm_drop = self.dropout(lstm_out)
        #         logits = your_code
        # Apply temperature scaling to logits
        logits = self.output(lstm_drop) / self.temp
        return logits
