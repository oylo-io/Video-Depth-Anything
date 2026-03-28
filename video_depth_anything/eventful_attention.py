"""
Eventful attention for DINOv2: skip attention for unchanged tokens.

Based on "Eventful Transformers" (Dutson et al., ICCV 2023, MIT license).
For video streams, detects which tokens changed between frames and only
computes attention for those queries. Unchanged tokens reuse cached
attention output. K and V always use all tokens for correctness.

Change detection uses relative L2 delta per token. The number of tokens
to recompute adapts dynamically each frame — few on static scenes,
all on scene changes.
"""

import torch
import torch.nn as nn
from torch.linalg import vector_norm


class EventfulDINOv2Block(nn.Module):
    """
    Wraps a DINOv2 NestedTensorBlock with selective attention.

    First frame: full computation, cache attention output.
    Subsequent frames:
      1. Relative L2 delta per token vs previous frame
      2. Count tokens above threshold (dynamic k)
      3. If k >= 90% of N: full recompute (scene change)
      4. If k == 0: reuse everything, just run MLP
      5. Otherwise: topk(k) changed indices, subset attention + cache blend
    """

    def __init__(self, block, num_heads, embed_dim, change_threshold=0.1):
        super().__init__()
        self.blk = block
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.change_threshold = change_threshold

        self.first = True
        self.prev_x = None
        self.cached_attn_output = None

    def forward(self, x):
        if self.first:
            return self._forward_full(x)
        return self._forward_eventful(x)

    def _forward_full(self, x):
        """Full computation + cache priming."""
        self.first = False
        self.prev_x = x.clone()
        B, N, D = x.shape

        residual = x
        x_normed = self.blk.norm1(x)
        qkv = self.blk.attn.qkv(x_normed).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn_out = attn @ v
        self.cached_attn_output = attn_out.clone()

        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        attn_out = self.blk.attn.proj(attn_out)
        if hasattr(self.blk, 'ls1'):
            attn_out = self.blk.ls1(attn_out)
        x = residual + attn_out

        residual2 = x
        x = self.blk.mlp(self.blk.norm2(x))
        if hasattr(self.blk, 'ls2'):
            x = self.blk.ls2(x)
        return residual2 + x

    def _forward_eventful(self, x):
        """Selective attention: recompute only changed tokens."""
        B, N, D = x.shape

        # Detect changes
        delta = x - self.prev_x
        token_delta = vector_norm(delta[0], dim=-1)
        token_norm = vector_norm(x[0], dim=-1)
        rel_change = token_delta / (token_norm + 1e-8)

        # Dynamic k via threshold count (one GPU→CPU sync, ~0.1ms)
        n_changed = (rel_change > self.change_threshold).sum().item()

        self.prev_x = x.clone()

        # Scene change: full recompute
        if n_changed >= int(N * 0.9):
            return self._forward_full(x)

        # Static: skip attention, just run MLP on cached
        if n_changed == 0:
            return self._forward_mlp_only(x)

        # Selective: topk changed indices
        _, changed_idx = rel_change.topk(n_changed, sorted=False)

        # Full QKV (need all K, V)
        residual = x
        x_normed = self.blk.norm1(x)
        qkv = self.blk.attn.qkv(x_normed).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k_full, v_full = qkv[0], qkv[1], qkv[2]

        # Attention: changed queries × all keys
        q_changed = q[:, :, changed_idx]
        attn = (q_changed * self.scale) @ k_full.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn_out_changed = attn @ v_full

        # Blend with cache
        full_attn = self.cached_attn_output.clone()
        full_attn[:, :, changed_idx] = attn_out_changed
        self.cached_attn_output = full_attn

        # Recombine + projection
        attn_out = full_attn.transpose(1, 2).reshape(B, N, D)
        attn_out = self.blk.attn.proj(attn_out)
        if hasattr(self.blk, 'ls1'):
            attn_out = self.blk.ls1(attn_out)
        x = residual + attn_out

        # MLP (full — only 8% of block compute)
        residual2 = x
        x = self.blk.mlp(self.blk.norm2(x))
        if hasattr(self.blk, 'ls2'):
            x = self.blk.ls2(x)
        return residual2 + x

    def _forward_mlp_only(self, x):
        """Nothing changed — reuse cached attention output, just run MLP."""
        B, N, D = x.shape
        residual = x
        attn_out = self.cached_attn_output.transpose(1, 2).reshape(B, N, D)
        attn_out = self.blk.attn.proj(attn_out)
        if hasattr(self.blk, 'ls1'):
            attn_out = self.blk.ls1(attn_out)
        x = residual + attn_out

        residual2 = x
        x = self.blk.mlp(self.blk.norm2(x))
        if hasattr(self.blk, 'ls2'):
            x = self.blk.ls2(x)
        return residual2 + x

    def reset(self):
        """Reset caches."""
        self.first = True
        self.prev_x = None
        self.cached_attn_output = None


def wrap_dino_blocks(dino, change_threshold=0.1):
    """Wrap all DINOv2 blocks with eventful attention."""
    return [
        EventfulDINOv2Block(
            blk, num_heads=dino.num_heads, embed_dim=dino.embed_dim,
            change_threshold=change_threshold
        )
        for blk in dino.blocks
    ]
