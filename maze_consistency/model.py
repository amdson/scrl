"""Causal transformer with three heads (action, categorical value, scalar Q) on the residual stream."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.linen as nn


@dataclass
class ModelConfig:
    vocab: int
    n_types: int
    max_len: int
    K: int
    n_actions: int = 4
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.0


class Block(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, mask):
        B, L, D = x.shape
        H = self.n_heads
        dh = D // H
        y = nn.LayerNorm()(x)
        qkv = nn.Dense(3 * D, name="qkv")(y).reshape(B, L, 3, H, dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        att = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(dh)
        att = jnp.where(mask, att, jnp.finfo(att.dtype).min)
        att = jax.nn.softmax(att, axis=-1)
        o = jnp.einsum("bhqk,bkhd->bqhd", att, v).reshape(B, L, D)
        x = x + nn.Dense(D, name="proj")(o)
        y = nn.LayerNorm()(x)
        x = x + nn.Dense(D, name="fc2")(nn.gelu(nn.Dense(4 * D, name="fc1")(y)))
        return x


class MazeTransformer(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, tokens, types):
        c = self.cfg
        B, L = tokens.shape
        x = (nn.Embed(c.vocab, c.d_model, name="tok_emb")(tokens)
             + nn.Embed(c.n_types, c.d_model, name="type_emb")(types)[None]
             + nn.Embed(c.max_len, c.d_model, name="pos_emb")(jnp.arange(L))[None])
        mask = jnp.tril(jnp.ones((L, L), dtype=bool))[None, None]
        for i in range(c.n_layers):
            x = Block(c.d_model, c.n_heads, name=f"block{i}")(x, mask)
        x = nn.LayerNorm(name="ln_f")(x)
        return {
            "action": nn.Dense(c.n_actions, name="action_head")(x),
            "value": nn.Dense(c.K, name="value_head")(x),
            "q": nn.Dense(c.n_actions, name="q_head")(x),
        }


def make_forward(model: MazeTransformer, types, sidx):
    """jit'd forward that returns heads gathered at the state tokens: [B, T+1, .]."""
    types = jnp.asarray(types)
    sidx = jnp.asarray(sidx)

    @jax.jit
    def fwd(params, tokens):
        out = model.apply({"params": params}, tokens, types)
        return {k: v[:, sidx] for k, v in out.items()}

    return fwd


def count_params(params) -> int:
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))
