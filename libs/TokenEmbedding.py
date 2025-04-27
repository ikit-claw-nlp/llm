import torch
from torch import nn as nn
class TokenEmbedding(nn.Module):
    def __init__(self, tokenizer, seq_length, d_model):
        super(TokenEmbedding, self).__init__()
        self.token_embeddings = nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim = d_model,
            padding_idx=tokenizer.convert_tokens_to_ids("<pad>")
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings = seq_length,
            embedding_dim = d_model
        )
    def forward(self, embedding_idx):
        # embedding_idx => [batch_size, token_length]
        batch_size, token_length = embedding_idx.shape
        token_embedding = self.token_embeddings(embedding_idx)
        pos_embedding = self.pos_embedding(torch.arange(token_length, device=token_embedding.device))
        return token_embedding + pos_embedding
