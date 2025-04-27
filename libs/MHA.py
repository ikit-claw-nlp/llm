import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_dim, d_model, seq_len, dropout =0.0, use_qkv_bias=False, use_mask=False):
        # if use_mask=False => self-attention
        # use_mask = True => casuality attention.
        # head_dim = d_k
        # seq_len here is the maximal sequence length.
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)
        self.W_Q = nn.Linear(d_model, n_heads * head_dim, bias=use_qkv_bias)
        self.W_K = nn.Linear(d_model, n_heads * head_dim, bias=use_qkv_bias)
        self.W_V = nn.Linear(d_model, n_heads * head_dim, bias=use_qkv_bias)
        # The book added a linear projection of the final output!
        self.output_proj = nn.Linear(n_heads * head_dim, n_heads * head_dim)
        # The book doesn't coding W_O in the attention paper!
        # self.W_O = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.use_mask = use_mask
        if self.use_mask:
            self.register_buffer(
                "mask",
                torch.triu(
                    torch.ones(self.seq_len, self.seq_len),
                    diagonal=1
                )
            )
    def forward(self, Q, K, V):
        batch_size, token_len, d_model = Q.shape
        # Q size: from [batch_size, seq_len, d_model]
        Q = self.W_Q(Q) # => [batch_size, seq_len, n_heads * n_dim]
        K = self.W_K(K) # => [batch_size, seq_len, n_heads * n_dim]
        V = self.W_V(V) # => [batch_size, seq_len, n_heads * n_dim]

        # Modify the view of each tensor.
        Q = Q.view(batch_size, token_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, token_len, self.n_heads, self.head_dim)
        V = V.view(batch_size, token_len, self.n_heads, self.head_dim)

        # Q, K, V => [batch_size, num_head, token_len, head_dim]
        Q = Q.transpose(2, 1)
        K = K.transpose(2, 1)
        V = V.transpose(2, 1)
        # attention_weights => [batch_size, num_head, token_len, token_len]
        attention_weights = torch.matmul(Q, K.transpose(-2, -1))
        # Use Q.shape[-1] ** 0.5 to avoid
        # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! 
        attention_weights = attention_weights / Q.shape[-1]**0.5
        if self.use_mask:
            mask = self.mask.bool()[:token_len, :token_len]
            attention_weights.masked_fill_(mask, -torch.inf)
        # attention_weights = [batch_size, num_head, token_len, token_len]
        attention_weights = nn.functional.softmax(attention_weights, dim=-1)
        # drop out some attention_weights.
        attention_weights = self.dropout(attention_weights)
        attention_score = attention_weights
        # context_vec = [batch_size, num_head, token_len, head_dim]
        context_vec = torch.matmul(attention_weights, V)
        # context_vec = [batch_size, token_len, num_head, head_dim]
        context_vec = context_vec.transpose(2, 1)
        context_vec = context_vec.contiguous().view(
            batch_size, token_len, self.n_heads * self.head_dim
        )
        context_vec = self.output_proj(context_vec)
        return context_vec, attention_score
