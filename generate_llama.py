"""
=============================================================================
generate_llama.py — Inference with Fine-Tuned LoRA Models
=============================================================================

Loads a fine-tuned LoRA adapter (from finetune_llama.py) on top of the
base model and generates text responses to prompts.

How LoRA inference works:
  1. Load the base model (e.g., SmolLM2-1.7B)
  2. Load the LoRA adapter weights (small, ~35MB)
  3. The adapter modifies the base model's attention layers
  4. Generate text using the combined model

The prompt is formatted as "### Instruction: ... ### Response: ..." to match
the template used during fine-tuning.

Usage:
    python3 generate_llama.py --prompt "Explain what a neural network is"
    python3 generate_llama.py --prompt "Write a Python sort function" --temperature 0.3
    python3 generate_llama.py --model_dir smol-finetuned --max_tokens 512

=============================================================================
"""

import argparse
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM


def get_device():
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    # =========================================================================
    # Parse arguments
    # =========================================================================
    parser = argparse.ArgumentParser(description="Generate text with a fine-tuned LoRA model")
    parser.add_argument("--prompt", type=str, required=True,
                        help="The instruction/question to send to the model")
    parser.add_argument("--model_dir", type=str, default="smol-finetuned",
                        help="Directory containing the fine-tuned LoRA adapter")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0.1=focused, 1.5=creative)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model from {args.model_dir}...")

    # =========================================================================
    # Load model: base model + LoRA adapter combined automatically
    # =========================================================================
    # AutoPeftModelForCausalLM loads the base model specified in the adapter
    # config and applies the LoRA weights on top.
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,  # float32 for MPS compatibility
        device_map=None,
    )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # =========================================================================
    # Format prompt using the instruction template
    # =========================================================================
    # Must match the template used during fine-tuning so the model knows
    # where the instruction ends and where to start generating the response.
    formatted = f"### Instruction:\n{args.prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    # =========================================================================
    # Generate response
    # =========================================================================
    print("Generating...\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,              # Enable sampling (vs greedy decoding)
            repetition_penalty=1.15,     # Penalize repeating tokens
        )

    # Decode and extract just the response portion
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    print("=" * 60)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    main()
