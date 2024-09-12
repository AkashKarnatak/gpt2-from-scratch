import math
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BertConfig:
    n_ctx: int = 512  # max sequence length
    vocab_size: int = 30522
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimensions


class BertSdpaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), "Embedding dimension is not divisible by num_heads"
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        b, t, c = x.shape
        q = (
            self.query(x).view(b, t, self.n_head, self.head_size).transpose(1, 2)
        )  # b, n_head, t, head_size
        k = (
            self.key(x).view(b, t, self.n_head, self.head_size).transpose(1, 2)
        )  # b, n_head, t, head_size
        v = (
            self.value(x).view(b, t, self.n_head, self.head_size).transpose(1, 2)
        )  # b, n_head, t, head_size
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            wei = (q @ k.transpose(-1, -2)) * (
                1 / math.sqrt(k.shape[-1])
            )  # b, n_head, t, t
            wei = F.softmax(wei, dim=-1)
            y = wei @ v
        y = y.transpose(1, 2).reshape(b, t, c)
        return y


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.LayerNorm = nn.LayerNorm(config.n_embd, eps=1e-12)

    def forward(self, x, orig):
        x = self.dense(x)
        x = self.LayerNorm(orig + x)
        return x


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSdpaSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, x):
        orig, x = x, self.self(x)
        x = self.output(x, orig=orig)
        return x


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, x):
        x = self.dense(x)
        x = self.intermediate_act_fn(x)
        return x


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(4 * config.n_embd, config.n_embd)
        self.LayerNorm = nn.LayerNorm(config.n_embd, eps=1e-12)

    def forward(self, x, orig):
        x = self.dense(x)
        x = self.LayerNorm(orig + x)
        return x


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, x):
        x = self.attention(x)
        orig, x = x, self.intermediate(x)
        x = self.output(x, orig=orig)
        return x


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_ctx = config.n_ctx
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.n_ctx, config.n_embd)
        self.token_type_embeddings = nn.Embedding(2, config.n_embd)
        self.LayerNorm = nn.LayerNorm(config.n_embd, eps=1e-12)

    def forward(self, ids, token_type_ids=None):
        b, t = ids.shape
        assert t <= self.n_ctx, "Length of ids is greater than max context length"
        idx = torch.arange(t, device=ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(t, dtype=torch.long, device=ids.device)
        we = self.word_embeddings(ids)  # b, t, n_embd
        pe = self.position_embeddings(idx)  # t, n_embd
        te = self.token_type_embeddings(token_type_ids)  # b, t, n_embd
        x = self.LayerNorm(we + pe + te)
        return x


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.n_layer)])

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.dense(x[:, 0])
        x = self.activation(x)
        return x


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, x, token_type_ids=None):
        x = self.embeddings(x, token_type_ids)
        x = self.encoder(x)
        pool = self.pooler(x)
        return x, pool

    @classmethod
    def from_pretrained(cls, model="bert-base-uncased"):
        bert = BertModel(BertConfig())
        from transformers import BertModel as BertModelHF

        hf_bert = BertModelHF.from_pretrained(model, attn_implementation="sdpa")
        bert.load_state_dict(hf_bert.state_dict())
        return bert
