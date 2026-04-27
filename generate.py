"""
=============================================================================
generate.py — Text Generation / Inference
=============================================================================

Generates text from the trained model using autoregressive decoding.
The model predicts one token at a time, appending each prediction to the
input and repeating until it hits the max length or an EOS token.

Sampling strategies implemented:
  - Temperature: Controls randomness (lower = deterministic, higher = creative)
  - Top-k: Only consider the k most likely next tokens
  - Top-p (nucleus): Only consider tokens whose cumulative probability < p

These can be combined for fine-grained control over generation quality.

Usage:
    python3 generate.py --prompt "Once upon a time"
    python3 generate.py --prompt "### Instruction:\\nExplain gravity\\n\\n### Response:\\n"
    python3 generate.py --prompt "The meaning of life" --temperature 0.3

=============================================================================
"""

import argparse
import torch
from config import ModelConfig
from model import GPTModel
from dataset import get_tokenizer


@torch.no_grad()  # Disable gradient computation for faster inference
def generate(
    model: GPTModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cpu",
) -> str:
    """
    Generate text autoregressively from a prompt.

    Args:
        model:          Trained GPTModel instance
        tokenizer:      Tokenizer for encoding/decoding text
        prompt:         Input text to continue from
        max_new_tokens: Maximum number of tokens to generate
        temperature:    Sampling temperature (0.1 = focused, 1.5 = creative)
        top_k:          Keep only top-k most likely tokens (0 = disabled)
        top_p:          Nucleus sampling threshold (1.0 = disabled)
        device:         Device to run inference on

    Returns:
        Generated text string (including the original prompt)
    """
    model.eval()

    # Encode the prompt into token IDs
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate one token at a time
    for _ in range(max_new_tokens):
        # Crop input to max sequence length (sliding window)
        ids = input_ids[:, -model.cfg.max_seq_len :]

        # Get model predictions (logits for every position)
        logits, _ = model(ids)

        # We only care about the LAST position's predictions (next token)
        logits = logits[:, -1, :] / temperature

        # =====================================================================
        # Top-k filtering: zero out all tokens except the top-k most likely
        # This prevents the model from choosing very unlikely tokens
        # =====================================================================
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[:, -1:]] = float("-inf")

        # =====================================================================
        # Top-p (nucleus) filtering: keep the smallest set of tokens whose
        # cumulative probability exceeds p. This adapts the number of
        # candidates based on the model's confidence.
        # =====================================================================
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative_probs > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False  # Always keep at least one token
        sorted_logits[remove] = float("-inf")
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        # Sample from the filtered probability distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append the new token to the sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stop if we generated an end-of-sequence token
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode the full sequence back to text
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained LLM")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_finetuned.pt")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # Select best available device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = get_tokenizer()
    mcfg = ModelConfig()
    mcfg.vocab_size = len(tokenizer)
    model = GPTModel(mcfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Generate and print
    output = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, device=device)
    print(f"\n{'='*60}")
    print(output)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
