"""
Eventful attention for DINOv2: skip attention for unchanged tokens.

Based on "Eventful Transformers" (Dutson et al., ICCV 2023, MIT license).
Simplified for DINOv2 ViT blocks: detect which tokens changed from previous
frame, only compute attention for those queries, reuse cached attention
output for unchanged queries.

Tokens that changed are detected by comparing to the previous frame's
input — tokens whose relative L2 delta exceeds a threshold are recomputed.
On scene changes, most/all tokens exceed the threshold so the full
attention is computed automatically.
"""

import torch
import torch.nn as nn
from torch.linalg import vector_norm


class EventfulDINOv2Block(nn.Module):
    """
    Wraps a DINOv2 NestedTensorBlock with selective attention.

    On first frame: full computation, cache attention output.
    On subsequent frames:
      1. Compare input tokens to previous frame (relative L2 delta)
      2. Tokens above change_threshold → recompute attention
      3. Tokens below → reuse cached attention output
      4. K and V always use all tokens (needed for correct attention)
      5. MLP runs on all tokens (cheap, 8% of block)
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
        self.cached_attn_output = None  # [B, heads, N, head_dim]

    def forward(self, x):
        if self.first:
            return self._forward_full(x)
        return self._forward_eventful(x)

    def _forward_full(self, x):
        """First frame: full computation, prime caches."""
        self.first = False
        self.prev_x = x.clone()

        B, N, D = x.shape

        residual = x
        x_normed = self.blk.norm1(x)
        qkv = self.blk.attn.qkv(x_normed).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = (q * self.scale) @ k.transpose(-2, -1)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_out = attn_weights @ v
        self.cached_attn_output = attn_out.clone()

        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        attn_out = self.blk.attn.proj(attn_out)
        if hasattr(self.blk, 'ls1'):
            attn_out = self.blk.ls1(attn_out)
        x = residual + attn_out

        # MLP
        residual2 = x
        x = self.blk.mlp(self.blk.norm2(x))
        if hasattr(self.blk, 'ls2'):
            x = self.blk.ls2(x)
        x = residual2 + x

        return x

    def _forward_eventful(self, x):
        """Subsequent frames: selective attention for changed tokens."""
        B, N, D = x.shape

        # Detect changed tokens by relative L2 delta
        delta = x - self.prev_x
        token_delta = vector_norm(delta[0], dim=-1)  # [N]
        token_norm = vector_norm(x[0], dim=-1)  # [N]
        rel_change = token_delta / (token_norm + 1e-8)  # [N]

        changed_mask = rel_change > self.change_threshold
        n_changed = changed_mask.sum().item()

        self.prev_x = x.clone()

        # If all or nearly all changed, just do full forward
        if n_changed >= N * 0.95:
            return self._forward_full_update(x)

        # If nothing changed, reuse everything
        if n_changed == 0:
            return self._forward_cached(x)

        changed_idx = changed_mask.nonzero(as_tuple=True)[0]

        # Full QKV projection (need all K, V)
        residual = x
        x_normed = self.blk.norm1(x)
        qkv = self.blk.attn.qkv(x_normed).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k_full, v_full = qkv[0], qkv[1], qkv[2]

        # Only compute attention for changed queries
        q_changed = q[:, :, changed_idx]  # [B, heads, n_changed, head_dim]
        attn_weights = (q_changed * self.scale) @ k_full.transpose(-2, -1)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_out_changed = attn_weights @ v_full  # [B, heads, n_changed, head_dim]

        # Assemble: cached for unchanged, fresh for changed
        full_attn = self.cached_attn_output.clone()
        full_attn[:, :, changed_idx] = attn_out_changed
        self.cached_attn_output = full_attn

        # Recombine heads + projection
        attn_out = full_attn.transpose(1, 2).reshape(B, N, D)
        attn_out = self.blk.attn.proj(attn_out)
        if hasattr(self.blk, 'ls1'):
            attn_out = self.blk.ls1(attn_out)
        x = residual + attn_out

        # MLP (full)
        residual2 = x
        x = self.blk.mlp(self.blk.norm2(x))
        if hasattr(self.blk, 'ls2'):
            x = self.blk.ls2(x)
        x = residual2 + x

        return x

    def _forward_full_update(self, x):
        """Full forward that also updates caches (for scene changes)."""
        out = self._forward_full(x)
        # _forward_full already updates caches and sets first=False
        return out

    def _forward_cached(self, x):
        """Nothing changed — reuse cached attention, just run MLP."""
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
        x = residual2 + x

        return x

    def reset(self):
        """Reset caches (call on scene change)."""
        self.first = True
        self.prev_x = None
        self.cached_attn_output = None


def wrap_dino_blocks(dino, change_threshold=0.1):
    """
    Wrap all DINOv2 blocks with eventful attention.

    Args:
        dino: DINOv2 backbone (DinoVisionTransformer)
        change_threshold: relative L2 change threshold per token (0.1 = 10% change triggers recompute)

    Returns:
        list of EventfulDINOv2Block wrapping the original blocks
    """
    return [
        EventfulDINOv2Block(
            blk, num_heads=dino.num_heads, embed_dim=dino.embed_dim,
            change_threshold=change_threshold
        )
        for blk in dino.blocks
    ]
