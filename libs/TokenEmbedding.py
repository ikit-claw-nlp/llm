import torch
from torch import nn as nn
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, pad_idx, seq_length, d_model, dropout=0.0):
        # seq_length here is the maximal sequence length.
        super(TokenEmbedding, self).__init__()
        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim = d_model,
            padding_idx=pad_idx,
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings = seq_length,
            embedding_dim = d_model
        )
        self.embedding_dropout = nn.Dropout(dropout)
    def forward(self, embedding_idx):
        # embedding_idx => [batch_size, token_length]
        batch_size, token_length = embedding_idx.shape
        token_embedding = self.token_embeddings(embedding_idx)
        pos_embedding = self.pos_embedding(torch.arange(token_length, device=token_embedding.device))
        return self.embedding_dropout(token_embedding + pos_embedding)
