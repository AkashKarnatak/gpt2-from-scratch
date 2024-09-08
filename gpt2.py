import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from transformers import GPT2LMHeadModel


@dataclass
class GPT2Config:
    n_ctx: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimensions


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        head_size = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_head * head_size)
        self.c_proj = nn.Linear(config.n_head * head_size, config.n_embd)
        # will not be considered model parameter
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(1, 1, config.n_ctx, config.n_ctx)) == 0,
            persistent=False,
        )

    def forward(self, x: torch.Tensor):
        b, t, c = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        q = q.view(b, t, self.n_head, -1)  # b, t, n_head, head_size
        k = k.view(b, t, self.n_head, -1)  # b, t, n_head, head_size
        v = v.view(b, t, self.n_head, -1)  # b, t, n_head, head_size
        q = q.transpose(1, 2)  # b, n_head, t, head_size
        k = k.transpose(1, 2)  # b, n_head, t, head_size
        v = v.transpose(1, 2)  # b, n_head, t, head_size

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = (q @ k.transpose(2, 3)) * (
                1 / np.sqrt(k.shape[-1])
            )  # b, n_head, t, t
            attn = attn.masked_fill(
                self.bias[:, :, :t, :t], float("-inf")
            )  # mask upper triangular elements
            attn = F.softmax(attn, dim=-1)
            y = attn @ v  # b, n_head, t, head_size

        y = y.transpose(1, 2)  # b, t, n_head, head_size
        y = y.reshape(b, t, -1)  # b, t, c
        proj = self.c_proj(y)

        return proj


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.act(self.c_fc(x))
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        return x


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.n_ctx, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # tie token embedding weight with lm_head
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx: torch.Tensor):
        b, t = idx.shape
        assert t <= self.config.n_ctx, "Number of tokens should not exceed block size"

        te = self.transformer.wte(idx)  # b, t, n_embd
        pe = self.transformer.wpe(torch.arange(t).to(idx.device))  # t, n_embd
        x = te + pe
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # b, t, vocab_size
        return logits

    @classmethod
    def from_pretrained(cls, model: str = "gpt2"):
        hf_model = GPT2LMHeadModel.from_pretrained(model)
        state = hf_model.state_dict()
        for k, v in state.items():
            if "c_" not in k:
                continue
            if len(v.shape) != 2:
                continue
            state[k] = v.T
        config = GPT2Config(
            n_ctx=hf_model.config.n_ctx,
            vocab_size=hf_model.config.vocab_size,
            n_layer=hf_model.config.n_layer,
            n_head=hf_model.config.n_head,
            n_embd=hf_model.config.n_embd,
        )
        gpt = GPT2Model(config)
        gpt.load_state_dict(state)
        return gpt
