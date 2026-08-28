"""Small causal Transformer shared by the starter trainer and Candidate ABI."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = int(config["n_heads"])
        self.head_dim = int(config["d_model"]) // self.n_heads
        self.dropout = float(config["dropout"])
        self.qkv = nn.Linear(config["d_model"], 3 * config["d_model"], bias=False)
        self.proj = nn.Linear(config["d_model"], config["d_model"], bias=False)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(output)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = int(config["d_model"])
        self.ln_1 = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * width, width, bias=False),
            nn.Dropout(float(config["dropout"])),
        )

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class ByteTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = int(config["d_model"])
        self.max_context = int(config["context_length"])
        self.token = nn.Embedding(int(config["vocab_size"]), width)
        self.position = nn.Embedding(self.max_context, width)
        self.dropout = nn.Dropout(float(config["dropout"]))
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(int(config["n_layers"]))]
        )
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, int(config["vocab_size"]), bias=False)
        self.output.weight = self.token.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        length = tokens.shape[1]
        if length > self.max_context:
            raise ValueError("input exceeds configured context length")
        positions = torch.arange(length, device=tokens.device)
        x = self.dropout(self.token(tokens) + self.position(positions)[None, :, :])
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))
