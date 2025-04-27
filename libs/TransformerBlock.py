from .MHA import MultiHeadAttention
from torch import nn
import torch


# Layer normalization
class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))
    def forward(self, x):
        # x shape [batch_size, seq_len, model_dim]
        # var shape [batch_size, seq_len, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # mean shape [batch_size, seq_len, 1]
        mean = x.mean(dim=-1, keepdim=True)
        # use eps to avoid divided by 0
        # x shape [batch_size, seq_len, model_dim]
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        self.scale * norm_x + self.shift
        return self.scale * norm_x + self.shift
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(
                torch.Tensor([2.0 / torch.pi])).to(x.device) * (x + 0.044715 * torch.pow(x, 3)) 
        )
    )
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super(FeedForward, self).__init__()
        emb_dim = cfg['emb_dim']
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )
    def forward(self, x):
        x = self.layers(x)
        return x
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super(TransformerBlock, self).__init__()
        assert cfg['emb_dim'] % cfg['n_heads'] == 0, "Embedding Dim must be integer multiple of the n_heads!"
        head_dim = int(cfg['emb_dim'] / cfg['n_heads'])
        self.mha_layer = MultiHeadAttention(cfg['n_heads'], head_dim, cfg['emb_dim'],
                                             cfg['context_length'], cfg['drop_rate'],
                                             use_qkv_bias=cfg['qkv_bias'],use_mask=True)
        self.before_mha_norm = LayerNorm(cfg['emb_dim'])
        self.after_mha_norm = LayerNorm(cfg['emb_dim'])
        self.ff = FeedForward(cfg)
        self.drop_residual = nn.Dropout(cfg['drop_rate'])
    def forward(self, x):
        raw_input = x
        x = self.before_mha_norm(x)
        x, _ = self.mha_layer(x, x, x)
        x = self.drop_residual(x)
        # Residual connection.
        x = x + raw_input
        raw_input = x
        x = self.after_mha_norm(x)
        x = self.ff(x)
        x = self.drop_residual(x)
        return x + raw_input
