"""
=============================================================================
finetune_llama.py — Fine-Tune Pre-trained Models with LoRA on Mac
=============================================================================

Fine-tunes a pre-trained HuggingFace model (SmolLM2, LLaMA, Mistral, etc.)
using LoRA (Low-Rank Adaptation) — a parameter-efficient technique that only
trains ~1-2% of the model's parameters while keeping the rest frozen.

Why LoRA?
  - Full fine-tuning of a 1.7B model requires ~7GB just for gradients
  - LoRA adds small trainable matrices to attention layers (~18M params)
  - This makes fine-tuning feasible on consumer hardware (Mac, single GPU)
  - Quality is nearly as good as full fine-tuning for most tasks

Supported dataset formats:
  - Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
  - OpenHermes format: {"conversations": [{"from": "human", "value": "..."}]}
  - Custom JSON: any list of dicts with instruction/output fields

Requirements:
    pip install torch transformers trl peft datasets accelerate

Usage:
    # Default: fine-tune SmolLM2-360M on Alpaca
    python3 finetune_llama.py

    # Use larger model (needs 16GB+ RAM)
    python3 finetune_llama.py --model HuggingFaceTB/SmolLM2-1.7B --batch_size 1

    # Use a code-specific dataset
    python3 finetune_llama.py --dataset sahil2801/CodeAlpaca-20k

    # Use a custom JSON dataset
    python3 finetune_llama.py --dataset my_data.json

=============================================================================
"""

import argparse
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


# =============================================================================
# Dataset Formatting
# =============================================================================
# Different datasets use different schemas. This function normalizes them
# all into a single "text" field using the ### Instruction / ### Response
# template that the model learns to follow.
# =============================================================================

def format_instruction(example: dict) -> dict:
    """
    Format a single example into the instruction/response template.
    Handles multiple dataset formats automatically.

    Supported formats:
      - OpenHermes: {"conversations": [{"from": "human", "value": "..."}, ...]}
      - Alpaca: {"instruction": "...", "input": "...", "output": "..."}

    Returns:
        Dict with a single "text" field containing the formatted example
    """
    # --- OpenHermes format: list of conversation turns ---
    if "conversations" in example:
        parts = []
        for turn in example["conversations"]:
            role = turn.get("from", turn.get("role", ""))
            text = turn.get("value", turn.get("content", ""))
            if role in ("human", "user"):
                parts.append(f"### Instruction:\n{text}")
            elif role in ("gpt", "assistant"):
                parts.append(f"### Response:\n{text}")
        return {"text": "\n\n".join(parts)}

    # --- Alpaca format: instruction/input/output fields ---
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    if example.get("input"):
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{output}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return {"text": text}


def load_custom_dataset(path: str) -> Dataset:
    """Load a custom JSON dataset from a local file."""
    with open(path) as f:
        data = json.load(f)
    return Dataset.from_list(data)


def get_device():
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    # =========================================================================
    # Parse command-line arguments
    # =========================================================================
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained LLM with LoRA")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-360M",
                        help="HuggingFace model name (e.g., HuggingFaceTB/SmolLM2-1.7B)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name or path to local JSON file")
    parser.add_argument("--output_dir", type=str, default="smol-finetuned",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Total number of training steps")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size (reduce to 1 for larger models)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Peak learning rate")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length for training examples")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (higher = more trainable params, more capacity)")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # =========================================================================
    # Step 1: Load the pre-trained base model
    # =========================================================================
    # We load in float32 for MPS compatibility (Apple Silicon doesn't support
    # bf16 well). On CUDA, you could use torch.bfloat16 for faster training.
    # device_map=None means we manually move to device (required for MPS).
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # MPS works best with float32
        device_map=None,            # Manual device placement for MPS
    )
    model = model.to(device)
    model.config.use_cache = False  # Disable KV cache during training (incompatible with gradient checkpointing)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =========================================================================
    # Step 2: Configure LoRA (Low-Rank Adaptation)
    # =========================================================================
    # LoRA works by injecting small trainable matrices into the model's
    # attention layers. Instead of updating the full weight matrix W (huge),
    # it learns two small matrices A and B such that: W' = W + A @ B
    #
    # Parameters:
    #   r (rank):       Size of the low-rank matrices (higher = more capacity)
    #   lora_alpha:     Scaling factor (typically 2x rank)
    #   target_modules: Which layers to apply LoRA to (all attention + FFN projections)
    #   lora_dropout:   Dropout on LoRA layers for regularization
    # =========================================================================
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",    # Attention projections
                        "gate_proj", "up_proj", "down_proj"],        # FFN projections
        lora_dropout=0.05,
        bias="none",           # Don't train bias terms
        task_type="CAUSAL_LM", # Causal language modeling task
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Shows how many params are actually being trained

    # =========================================================================
    # Step 3: Load and format the dataset
    # =========================================================================
    if args.dataset and args.dataset.endswith(".json"):
        print(f"Loading custom dataset: {args.dataset}")
        dataset = load_custom_dataset(args.dataset)
    else:
        dataset_name = args.dataset or "tatsu-lab/alpaca"
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")

    # Format all examples into the ### Instruction / ### Response template
    dataset = dataset.map(format_instruction)

    # Drop all columns except 'text' to avoid type casting issues
    # (some datasets like OpenHermes have messy mixed-type columns)
    keep_cols = {"text"}
    drop_cols = [c for c in dataset.column_names if c not in keep_cols]
    if drop_cols:
        dataset = dataset.remove_columns(drop_cols)

    # Remove any empty examples
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    print(f"Dataset size: {len(dataset)} examples")

    # =========================================================================
    # Step 4: Training configuration (optimized for Mac / MPS)
    # =========================================================================
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,    # Effective batch = batch_size * 4
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",       # Cosine decay: high → low learning rate
        warmup_steps=50,                  # Gradually ramp up LR at the start
        logging_steps=25,                 # Log metrics every 25 steps
        save_steps=250,                   # Save checkpoint every 250 steps
        fp16=False,                       # Disabled: MPS doesn't support fp16 training well
        bf16=False,                       # Disabled: MPS doesn't support bf16
        max_length=args.max_seq_len,      # Truncate sequences to this length
        dataset_text_field="text",        # Column name containing the training text
        gradient_checkpointing=True,      # Trade compute for memory (essential for large models)
        optim="adamw_torch",              # Standard AdamW (paged_adamw requires CUDA)
        dataloader_pin_memory=False,      # Must be False for MPS compatibility
    )

    # =========================================================================
    # Step 5: Train with SFTTrainer (from HuggingFace TRL library)
    # =========================================================================
    # SFTTrainer handles tokenization, batching, loss computation, and
    # gradient updates automatically. It's built on top of HuggingFace Trainer.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting fine-tuning...")
    trainer.train()

    # =========================================================================
    # Step 6: Save the fine-tuned LoRA adapter and tokenizer
    # =========================================================================
    # Only the LoRA weights are saved (small, ~35MB for r=16).
    # At inference time, the base model is loaded and the adapter is applied on top.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nDone! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
