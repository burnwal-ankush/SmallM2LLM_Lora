"""
=============================================================================
model.py — GPT-Style Decoder-Only Transformer
=============================================================================

Implements a modern decoder-only transformer language model from scratch,
using the same architectural choices as LLaMA and Mistral:

  - RMSNorm (instead of LayerNorm) for faster, simpler normalization
  - RoPE (Rotary Position Embeddings) for position encoding
  - SwiGLU activation in the feed-forward network
  - Pre-norm architecture (normalize before attention/FFN, not after)
  - Weight tying between token embeddings and the output head
  - Causal (autoregressive) attention mask

Architecture per transformer block:
  Input → RMSNorm → Multi-Head Attention (with RoPE) → Residual Add
        → RMSNorm → SwiGLU Feed-Forward → Residual Add → Output

Full model:
  Token Embedding → N x TransformerBlock → RMSNorm → Linear Head → Logits

=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


# =============================================================================
# RMSNorm — Root Mean Square Layer Normalization
# =============================================================================
# A simpler alternative to LayerNorm used in LLaMA/Mistral.
# Instead of computing mean and variance, it normalizes by the root-mean-square
# of the input, which is faster and works just as well in practice.
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scale parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS: sqrt(mean(x^2) + eps), then normalize
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# =============================================================================
# Rotary Position Embeddings (RoPE)
# =============================================================================
# RoPE encodes position information by rotating query and key vectors in the
# attention mechanism. Unlike learned position embeddings, RoPE:
#   - Doesn't add parameters to the model
#   - Generalizes better to sequence lengths not seen during training
#   - Encodes relative position naturally (attention between tokens depends
#     on their distance, not absolute position)
#
# How it works:
#   1. Precompute rotation frequencies for each position and dimension
#   2. Split each vector into pairs of dimensions
#   3. Apply 2D rotation using cos/sin of the position-dependent angle
# =============================================================================

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute cosine and sine frequencies for RoPE.

    Args:
        dim:         Dimension of each attention head (hidden_size // num_heads)
        max_seq_len: Maximum sequence length to precompute for
        theta:       Base frequency (higher = slower rotation, longer-range patterns)

    Returns:
        Tuple of (cos_freqs, sin_freqs), each of shape (max_seq_len, dim // 2)
    """
    # Compute frequency for each dimension pair: 1 / (theta^(2i/dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # Create position indices: [0, 1, 2, ..., max_seq_len-1]
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # Outer product: (max_seq_len, dim//2) — angle for each position and dimension
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply rotary position embeddings to query or key tensors.

    Args:
        x:   Tensor of shape (batch, heads, seq_len, head_dim)
        cos: Precomputed cosine frequencies
        sin: Precomputed sine frequencies

    Returns:
        Rotated tensor of same shape as x
    """
    # Split the head dimension in half for rotation pairs
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # Broadcast cos/sin to match batch and head dimensions
    cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    # Apply 2D rotation: [x1*cos - x2*sin, x2*cos + x1*sin]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# =============================================================================
# Multi-Head Self-Attention
# =============================================================================
# The core mechanism that allows each token to "look at" other tokens in the
# sequence. Uses a causal mask so tokens can only attend to previous tokens
# (autoregressive property — essential for text generation).
#
# Steps:
#   1. Project input into Query, Key, Value using a single fused linear layer
#   2. Split into multiple heads (parallel attention computations)
#   3. Apply RoPE to Q and K (inject position information)
#   4. Compute attention scores: softmax(Q @ K^T / sqrt(d))
#   5. Apply causal mask (prevent attending to future tokens)
#   6. Weighted sum of Values using attention scores
#   7. Concatenate heads and project back to hidden_size
# =============================================================================

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads

        # Fused QKV projection: one matrix multiply instead of three
        self.qkv_proj = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        # Output projection: combines multi-head outputs back to hidden_size
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, mask=None):
        B, T, C = x.shape  # Batch, Sequence Length, Hidden Size

        # Project to Q, K, V and reshape for multi-head attention
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, num_heads, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply RoPE to queries and keys (not values — they don't need position info)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # Causal mask: set future positions to -inf so softmax gives them 0 weight
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values, then reshape back to (B, T, C)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


# =============================================================================
# SwiGLU Feed-Forward Network
# =============================================================================
# A gated feed-forward network using the SiLU (Swish) activation function.
# Instead of the traditional ReLU MLP: FFN(x) = ReLU(xW1)W2
# SwiGLU uses:  FFN(x) = (SiLU(xW_gate) * xW_up) W_down
#
# The gating mechanism (SiLU(gate) * up) has been shown to improve training
# dynamics and final model quality. Used in LLaMA, Mistral, and PaLM.
# =============================================================================

class FeedForward(nn.Module):
    """SwiGLU-style feed-forward network."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Gate projection: controls which features pass through
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        # Up projection: expands features to intermediate size
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        # Down projection: compresses back to hidden size
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # SwiGLU: SiLU(gate) * up, then project down
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# =============================================================================
# Transformer Block
# =============================================================================
# One complete transformer layer combining attention and feed-forward.
# Uses pre-norm architecture: normalize BEFORE the operation, not after.
# Residual connections (x + sublayer(x)) ensure gradients flow through
# the entire network during training.
#
# Flow: x → Norm → Attention → + x → Norm → FFN → + x
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size)  # Pre-attention normalization
        self.attn = Attention(cfg)
        self.ff_norm = RMSNorm(cfg.hidden_size)     # Pre-FFN normalization
        self.ff = FeedForward(cfg)

    def forward(self, x, cos, sin, mask=None):
        # Attention with residual connection
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        # Feed-forward with residual connection
        x = x + self.ff(self.ff_norm(x))
        return x


# =============================================================================
# GPTModel — Full Language Model
# =============================================================================
# Assembles the complete model:
#   1. Token embedding: converts token IDs to dense vectors
#   2. N transformer blocks: process the sequence
#   3. Final RMSNorm: normalize before output
#   4. Language model head: project to vocabulary size for next-token prediction
#
# Weight tying: the embedding matrix and LM head share the same weights,
# reducing parameter count and improving training (the model learns a single
# mapping between token IDs and their semantic representations).
#
# Returns:
#   - logits: raw scores for each token in the vocabulary (B, T, vocab_size)
#   - loss: cross-entropy loss if targets are provided, None otherwise
# =============================================================================

class GPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding: maps token IDs → dense vectors of size hidden_size
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        # Stack of transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])

        # Final normalization before the output head
        self.norm = RMSNorm(cfg.hidden_size)

        # Language model head: projects hidden states → vocabulary logits
        self.lm_head = nn.Linear(cfg.vocab_size, cfg.hidden_size, bias=False)

        # Weight tying: embedding and LM head share the same weight matrix
        self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE rotation frequencies (stored as non-trainable buffers)
        cos, sin = precompute_rope_freqs(
            cfg.hidden_size // cfg.num_heads, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # Initialize all weights with small random values
        self.apply(self._init_weights)
        print(f"Model parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.1f}M")

    def _init_weights(self, module):
        """Initialize weights with small normal distribution (std=0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            targets:   Target token IDs for loss computation (optional)

        Returns:
            logits: Prediction scores of shape (batch_size, seq_len, vocab_size)
            loss:   Cross-entropy loss if targets provided, else None
        """
        B, T = input_ids.shape

        # Convert token IDs to embeddings
        x = self.tok_emb(input_ids)

        # Create causal mask: lower triangular matrix prevents attending to future tokens
        # Shape: (1, 1, T, T) — broadcasts across batch and heads
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        # Pass through all transformer blocks
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, mask)

        # Final normalization and project to vocabulary
        x = self.norm(x)
        logits = self.lm_head(x)

        # Compute loss if targets are provided (training mode)
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
