"""Causal transformer over (kind, x, y) slots with two heads:
    next   one next-token softmax at every slot (actions after state slots, cells after action slots, END);
           trained by teacher forcing, it is both the policy and the dynamics model
    value  categorical over the K return bins, read at state slots
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn

from .env import N_ACTIONS
from .tokens import Tokenizer, N_TYPES


@dataclass
class ModelConfig:
    n_kind: int
    n_x: int
    n_y: int
    max_len: int
    K: int
    n_out: int
    n_types: int = N_TYPES
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    pos_enc: str = "learned"   # "learned": a free vector per slot index (the original).
                               # "rope": rotary positions inside attention (relative offsets, so "the most recent
                               #   state slot" is one pattern at every position) plus a FIXED sinusoidal absolute
                               #   embedding of the slot index, which is elapsed time -- the value head needs it.

    @staticmethod
    def for_tokenizer(tok: Tokenizer, **kw) -> "ModelConfig":
        return ModelConfig(n_kind=tok.n_kind, n_x=tok.n_x, n_y=tok.n_y, max_len=tok.L, K=tok.K, n_out=tok.n_out, **kw)


def sinusoidal(L, D, base=10000.0):
    """Fixed absolute embedding [L, D] of slot index: sin/cos at geometrically spaced frequencies."""
    pos = jnp.arange(L)[:, None].astype(jnp.float32)
    freq = base ** (-jnp.arange(0, D, 2).astype(jnp.float32) / D)
    ang = pos * freq[None]
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], -1)[:, :D]


def rope(x, base=10000.0):
    """Rotary position embedding on [B, L, H, d]: rotate each (even, odd) pair of features by an angle that
    grows linearly with the slot index, so q.k depends on the offset between slots, not their absolute index."""
    B, L, H, d = x.shape
    half = d // 2
    freq = base ** (-jnp.arange(half).astype(jnp.float32) / half)
    ang = jnp.arange(L).astype(jnp.float32)[:, None] * freq[None]              # [L, half]
    cos, sin = jnp.cos(ang)[None, :, None, :], jnp.sin(ang)[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:2 * half]
    out = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    return out if 2 * half == d else jnp.concatenate([out, x[..., 2 * half:]], -1)


class Block(nn.Module):
    d_model: int
    n_heads: int
    use_rope: bool = False

    @nn.compact
    def __call__(self, x, mask):
        B, L, D = x.shape
        H = self.n_heads
        y = nn.LayerNorm()(x)
        qkv = nn.Dense(3 * D)(y).reshape(B, L, 3, H, D // H)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.use_rope:
            q, k = rope(q), rope(k)
        att = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(D // H)
        att = jax.nn.softmax(jnp.where(mask, att, jnp.finfo(att.dtype).min), axis=-1)
        x = x + nn.Dense(D)(jnp.einsum("bhqk,bkhd->bqhd", att, v).reshape(B, L, D))
        y = nn.LayerNorm()(x)
        return x + nn.Dense(D)(nn.gelu(nn.Dense(4 * D)(y)))


class MazeTransformer(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, tokens, types):
        """tokens int32 [B, L, 3] = (kind, x, y); returns next-token logits [B, L, n_out], value logits [B, L, K]."""
        c = self.cfg
        B, L, _ = tokens.shape
        if c.pos_enc not in ("learned", "rope"):
            raise ValueError(f"pos_enc={c.pos_enc!r} not in learned|rope")
        x = (nn.Embed(c.n_kind, c.d_model, name="kind_emb")(tokens[..., 0])
             + nn.Embed(c.n_x, c.d_model, name="x_emb")(tokens[..., 1])
             + nn.Embed(c.n_y, c.d_model, name="y_emb")(tokens[..., 2])
             + nn.Embed(c.n_types, c.d_model, name="type_emb")(types)[None])
        if c.pos_enc == "learned":
            x = x + nn.Embed(c.max_len, c.d_model, name="pos_emb")(jnp.arange(L))[None]
        else:
            x = x + sinusoidal(L, c.d_model)[None]                     # absolute slot index = elapsed time, fixed
        mask = jnp.tril(jnp.ones((L, L), dtype=bool))[None, None]
        for i in range(c.n_layers):
            x = Block(c.d_model, c.n_heads, use_rope=(c.pos_enc == "rope"), name=f"block{i}")(x, mask)
        x = nn.LayerNorm(name="ln_f")(x)
        return {"next": nn.Dense(c.n_out, name="next_head")(x),
                "value": nn.Dense(c.K, name="value_head")(x)}


def make_forward(model: MazeTransformer, tok: Tokenizer):
    """jit'd forward. Returns the raw next-token logits over all slots plus the three readouts:
        pi_logits   [B, T+1, 4]        next-token logits at state slots, actions only (pi, or pi_R in R mode)
        dyn_logits  [B, T, n_cells]    next-token logits at action slots, cells only (P(s' | s, a))
        v_logits    [B, T+1, K]        value logits at state slots
    """
    types, sidx, aidx = jnp.asarray(tok.types), jnp.asarray(tok.sidx), jnp.asarray(tok.aidx)
    a0, c0, c1 = tok.OUT_ACT0, tok.OUT_CELL0, tok.OUT_END

    @jax.jit
    def fwd(params, tokens):
        out = model.apply({"params": params}, tokens, types)
        nxt = out["next"]
        return {"next": nxt, "pi_logits": nxt[:, sidx, a0:a0 + N_ACTIONS], "dyn_logits": nxt[:, aidx, c0:c1],
                "v_logits": out["value"][:, sidx]}

    return fwd


def next_token_loss(next_logits, targets, mask):
    """Teacher forcing: mean cross-entropy of the next-token head at slot i against the token at slot i + 1.
    next_logits [B, L, n_out]; targets, mask [B, L - 1] from Tokenizer.next_targets."""
    logp = jax.nn.log_softmax(next_logits[:, :-1], -1)
    nll = -jnp.take_along_axis(logp, jnp.maximum(targets, 0)[..., None], -1)[..., 0]
    m = mask.astype(nll.dtype)
    return (nll * m).sum() / jnp.maximum(m.sum(), 1.0)


def count_params(params) -> int:
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
