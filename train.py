"""
=============================================================================
train.py — Pre-training Loop (Phase 1)
=============================================================================

Pre-trains the GPT model on raw text using next-token prediction.
This is the foundational phase where the model learns language patterns,
grammar, facts, and reasoning from a large text corpus.

Pipeline:
  1. Load tokenizer (Mistral's 32K vocab BPE tokenizer)
  2. Load dataset (WikiText-2 for demo — 2M tokens of Wikipedia text)
  3. Create the model and move to best available device (CUDA > MPS > CPU)
  4. Set up AdamW optimizer with weight decay (skip bias and norm params)
  5. Run training loop with gradient accumulation and checkpointing

The training objective is simple: given tokens [1, 2, 3, 4], predict [2, 3, 4, 5].
This is called "causal language modeling" or "next-token prediction".

Usage:
    python3 train.py

Output:
    checkpoints/model_final.pt — the pre-trained model weights

=============================================================================
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset

from config import ModelConfig, TrainConfig
from model import GPTModel
from dataset import get_tokenizer, TextDataset


def train():
    # =========================================================================
    # Setup: Load configs and select compute device
    # =========================================================================
    mcfg = ModelConfig()
    tcfg = TrainConfig()

    # Device priority: CUDA GPU > Apple Silicon MPS > CPU
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # =========================================================================
    # Step 1: Tokenizer
    # =========================================================================
    # Load a pre-trained tokenizer and update vocab_size to match
    tokenizer = get_tokenizer()
    mcfg.vocab_size = len(tokenizer)

    # =========================================================================
    # Step 2: Dataset
    # =========================================================================
    # WikiText-2: ~2M tokens of clean Wikipedia text, good for prototyping.
    # For production, use FineWeb (15T tokens) or RedPajama (1T+ tokens).
    print("Loading dataset...")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in raw["text"] if t.strip()]  # filter empty lines

    # TextDataset tokenizes all text and chunks it into fixed-length sequences
    dataset = TextDataset(texts, tokenizer, max_len=mcfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=tcfg.batch_size, shuffle=True, drop_last=True)
    print(f"Dataset: {len(dataset)} chunks of {mcfg.max_seq_len} tokens")

    # =========================================================================
    # Step 3: Model
    # =========================================================================
    model = GPTModel(mcfg).to(device)

    # =========================================================================
    # Step 4: Optimizer
    # =========================================================================
    # AdamW with weight decay applied only to weight matrices (not biases or norms).
    # This is standard practice — regularizing biases and norms hurts performance.
    no_decay = {"bias", "norm"}
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": tcfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=tcfg.learning_rate, betas=(0.9, 0.95))

    # =========================================================================
    # Step 5: Training Loop
    # =========================================================================
    # Uses gradient accumulation: instead of updating weights every step,
    # we accumulate gradients over N steps and then update. This simulates
    # a larger batch size without needing more memory.
    os.makedirs(tcfg.output_dir, exist_ok=True)
    model.train()
    step = 0
    optimizer.zero_grad()

    for epoch in range(100):  # Loop enough epochs to reach max_steps
        for input_ids, targets in tqdm(loader, desc=f"Epoch {epoch}"):
            # Move data to device (GPU/MPS/CPU)
            input_ids, targets = input_ids.to(device), targets.to(device)

            # Forward pass: compute logits and loss
            _, loss = model(input_ids, targets)

            # Scale loss by accumulation steps (so the total gradient is correct)
            loss = loss / tcfg.gradient_accumulation_steps
            loss.backward()

            # Update weights every gradient_accumulation_steps
            if (step + 1) % tcfg.gradient_accumulation_steps == 0:
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Log training loss periodically
            if step % tcfg.eval_interval == 0:
                print(f"Step {step} | Loss: {loss.item() * tcfg.gradient_accumulation_steps:.4f}")

            # Save checkpoint periodically
            if step > 0 and step % tcfg.save_interval == 0:
                path = os.path.join(tcfg.output_dir, f"model_step_{step}.pt")
                torch.save(model.state_dict(), path)
                print(f"Saved checkpoint: {path}")

            step += 1
            if step >= tcfg.max_steps:
                break
        if step >= tcfg.max_steps:
            break

    # Save the final model
    final_path = os.path.join(tcfg.output_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    train()
