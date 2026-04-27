"""
=============================================================================
config.py — Model and Training Configuration
=============================================================================

Central configuration file for the LLM prototype. Contains all hyperparameters
for both the model architecture and the training process.

Two dataclasses:
  - ModelConfig: Defines the transformer architecture (size, layers, heads, etc.)
  - TrainConfig: Controls the training loop (batch size, learning rate, steps, etc.)

Adjust these values to scale the model up or down:
  - For quick testing on CPU: use defaults (small model, ~12M params)
  - For better quality on GPU: increase hidden_size, num_layers, num_heads
  - For production: hidden_size=4096, num_layers=32+, num_heads=32+

=============================================================================
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Transformer model architecture configuration.

    Attributes:
        vocab_size:        Number of tokens in the vocabulary (set by tokenizer at runtime)
        hidden_size:       Dimensionality of token embeddings and hidden states
        num_layers:        Number of stacked transformer blocks
        num_heads:         Number of attention heads (hidden_size must be divisible by this)
        intermediate_size: Size of the feed-forward network's inner layer (typically 4x hidden_size)
        max_seq_len:       Maximum sequence length the model can process
        dropout:           Dropout rate applied in attention and feed-forward layers
        rope_theta:        Base frequency for Rotary Position Embeddings (RoPE)
    """
    vocab_size: int = 32000
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 1024  # 4x hidden_size
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    """
    Training loop configuration.

    Attributes:
        batch_size:                  Number of samples per training step
        gradient_accumulation_steps: Accumulate gradients over N steps before updating
                                     (effective batch = batch_size * gradient_accumulation_steps)
        learning_rate:               Peak learning rate for the optimizer
        weight_decay:                L2 regularization strength (applied to non-bias, non-norm params)
        warmup_steps:                Number of steps to linearly ramp up learning rate
        max_steps:                   Total training steps before stopping
        eval_interval:               Log training loss every N steps
        save_interval:               Save a model checkpoint every N steps
        max_grad_norm:               Maximum gradient norm for gradient clipping
        output_dir:                  Directory to save model checkpoints
        log_wandb:                   Whether to log metrics to Weights & Biases
    """
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 500
    eval_interval: int = 50
    save_interval: int = 250
    max_grad_norm: float = 1.0
    output_dir: str = "checkpoints"
    log_wandb: bool = False
