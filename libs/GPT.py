import torch
from torch import nn
from .TransformerBlock import TransformerBlock, LayerNorm
from .TokenEmbedding import TokenEmbedding
class GPT(nn.Module):
    def __init__(self, cfg):
        super(GPT, self).__init__()
        self.token_embeddings = TokenEmbedding(vocab_size=cfg['vocab_size'],
                                               pad_idx=cfg['pad_idx'],
                                               seq_length=cfg['context_length'],
                                               d_model = cfg['emb_dim'],
                                               dropout=cfg['drop_rate'])
        self.drop_embedding = nn.Dropout(cfg['drop_rate'])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_proj = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
    def forward(self, token_idx):
        token_embeddings = self.token_embeddings(token_idx)
        token_embeddings = self.drop_embedding(token_embeddings)
        x = self.transformer_blocks(token_embeddings)
        x = self.final_norm(x)
        logits = self.out_proj(x)
        return logits

