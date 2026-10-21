import torch
import torch.nn as nn
from einops import einsum, rearrange

import math
from lm.nn_utils import softmax


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return einsum(x, self.weight, "... i, o i -> ... o")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x / rms * self.weight
        return out.to(in_dtype)

def silu(z):
    return z * torch.sigmoid(z)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        # theta_i = Theta^(-2i/d_k), shape (d_k/2,)
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device).float() / d_k)
        positions = torch.arange(max_seq_len, device=device).float()
        angles = positions[:, None] * inv_freq[None, :]  # (max_seq_len, d_k/2)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        cos = self.cos[token_positions]  
        sin = self.sin[token_positions]

        while cos.dim() < x.dim():
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = einsum(q, k, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        # True = may attend, False = blocked
        scores = scores.masked_fill(~mask, float("-inf"))
    # Softmax in fp32, then back to the original dtype
    weights = softmax(scores.float(), dim=-1).to(q.dtype)
    return einsum(weights, v, "... q k, ... k d -> ... q d")


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None, rope=None,
                 device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert d_model % num_heads == 0
        assert num_heads % self.num_kv_heads == 0
        self.d_k = d_model // num_heads
        self.groups = num_heads // self.num_kv_heads
        self.rope = rope

        kw = dict(device=device, dtype=dtype)
        self.q_proj = Linear(d_model, num_heads * self.d_k, **kw)
        self.k_proj = Linear(d_model, self.num_kv_heads * self.d_k, **kw)
        self.v_proj = Linear(d_model, self.num_kv_heads * self.d_k, **kw)
        # self.o_proj = Linear(num_heads * self.d_k, d_model, **kw)
        self.output_proj = Linear(num_heads * self.d_k, d_model, **kw)

    def forward(self, x, token_positions=None):
        seq = x.shape[-2]

        q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=self.num_kv_heads)
        v = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=self.num_kv_heads)

        # RoPE on q and k only (before repeating k, so it runs on fewer heads)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # GQA: each kv head serves `groups` consecutive query heads
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=-3)
            v = v.repeat_interleave(self.groups, dim=-3)

        # Causal mask: True on and below the diagonal
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))

        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "... h s d -> ... s (h d)")
        # return self.o_proj(out)
        return self.output_proj(out)




class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_kv_heads=None, rope=None,
                 device=None, dtype=None):
        super().__init__()
        kw = dict(device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, **kw)
        self.attn = MultiheadSelfAttention(d_model, num_heads, num_kv_heads, rope=rope, **kw)
        self.ln2 = RMSNorm(d_model, **kw)
        self.ffn = SwiGLU(d_model, d_ff, **kw)

    def forward(self, x, token_positions=None):
        h = x + self.attn(self.ln1(x), token_positions)
        return h + self.ffn(self.ln2(h))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads,
                 d_ff, rope_theta, num_kv_heads=None, device=None, dtype=None):
        super().__init__()
        kw = dict(device=device, dtype=dtype)
        self.context_length = context_length

        rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads,
                                         context_length, device=device)

        self.token_embeddings = Embedding(vocab_size, d_model, **kw)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, num_kv_heads, rope=rope, **kw)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, **kw)
        self.lm_head = Linear(d_model, vocab_size, **kw)

    def forward(self, ids):
        seq = ids.shape[-1]
        token_positions = torch.arange(seq, device=ids.device)
        x = self.token_embeddings(ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        return self.lm_head(self.ln_final(x)).float()