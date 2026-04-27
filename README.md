# LLM Prototype — GPT-style Language Model

A minimal but complete prototype for training a decoder-only transformer language model from scratch, covering the full pipeline: data → pre-training → fine-tuning → inference.

## Architecture
- Decoder-only Transformer (GPT-style)
- ~125M parameters (configurable)
- RoPE positional encoding
- Multi-head self-attention with causal mask
- Pre-norm (RMSNorm) architecture

## Project Structure
```
llm-prototype/
├── config.py          # Model and training hyperparameters
├── model.py           # Transformer model implementation
├── tokenizer_utils.py # Tokenizer wrapper (uses HuggingFace tokenizers)
├── dataset.py         # Data loading and preprocessing
├── train.py           # Pre-training loop
├── finetune.py        # Supervised fine-tuning on instruction data
├── generate.py        # Text generation / inference
├── requirements.txt   # Dependencies
└── README.md
```

## Quick Start
```bash
pip install -r requirements.txt
# Pre-train on sample data
python train.py
# Fine-tune on instruction data
python finetune.py
# Generate text
python generate.py --prompt "Explain quantum computing"
```

## Requirements
- Python 3.10+
- PyTorch 2.0+
- 1x GPU with 16GB+ VRAM (for 125M model)
- For larger models, use multiple GPUs with FSDP/DeepSpeed
