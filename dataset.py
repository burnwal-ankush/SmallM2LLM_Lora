"""
=============================================================================
dataset.py — Data Loading and Preprocessing
=============================================================================

Provides dataset classes for both phases of LLM training:

  1. TextDataset (Pre-training):
     - Takes raw text, tokenizes it into one long stream of token IDs
     - Chunks the stream into fixed-length sequences
     - Each sample returns (input_ids, targets) shifted by one position
     - This teaches the model next-token prediction on raw text

  2. InstructDataset (Fine-tuning):
     - Takes instruction/response pairs
     - Formats them with a template: "### Instruction: ... ### Response: ..."
     - Tokenizes, pads to fixed length, and returns shifted pairs
     - This teaches the model to follow instructions

Also provides get_tokenizer() which loads a pre-trained tokenizer from
HuggingFace (Mistral's 32K vocab tokenizer by default).

=============================================================================
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# =============================================================================
# Tokenizer
# =============================================================================
# We reuse an existing pre-trained tokenizer rather than training one from
# scratch. Mistral's tokenizer uses BPE (Byte-Pair Encoding) with a 32K
# vocabulary, which is a good balance between coverage and efficiency.
# =============================================================================

def get_tokenizer(name: str = "mistralai/Mistral-7B-v0.1"):
    """
    Load a pre-trained tokenizer from HuggingFace.

    Args:
        name: HuggingFace model name to load the tokenizer from

    Returns:
        AutoTokenizer instance with pad_token set
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    # Ensure pad token exists (some tokenizers don't define one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# =============================================================================
# Pre-training Dataset
# =============================================================================
# For pre-training, we concatenate all text into one long token sequence,
# then chop it into fixed-length chunks. Each chunk becomes one training
# sample where:
#   - input_ids = tokens[0:N]     (what the model sees)
#   - targets   = tokens[1:N+1]   (what the model should predict)
#
# This "shifted by one" pattern is how the model learns next-token prediction.
# Example: "The cat sat" → input: [The, cat, sat] → target: [cat, sat, <eos>]
# =============================================================================

class TextDataset(Dataset):
    """Pre-training dataset: chunks tokenized text into fixed-length sequences."""

    def __init__(self, texts: list[str], tokenizer, max_len: int = 1024):
        """
        Args:
            texts:     List of raw text strings to tokenize
            tokenizer: Tokenizer instance for encoding text
            max_len:   Length of each training sequence (in tokens)
        """
        self.max_len = max_len

        # Step 1: Tokenize all texts and concatenate into one long sequence
        # We add EOS token between documents to mark boundaries
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)

        # Step 2: Chunk into sequences of max_len + 1 (the +1 gives us the target)
        self.chunks = []
        for i in range(0, len(all_ids) - max_len, max_len):
            self.chunks.append(torch.tensor(all_ids[i : i + max_len + 1], dtype=torch.long))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        # input_ids: all tokens except the last
        # targets: all tokens except the first (shifted by 1)
        return chunk[:-1], chunk[1:]


# =============================================================================
# Instruction Fine-tuning Dataset
# =============================================================================
# For supervised fine-tuning (SFT), we format instruction/response pairs
# using a template that the model learns to follow:
#
#   ### Instruction:
#   <user's question or task>
#
#   ### Response:
#   <model's answer>
#
# The model learns to generate the response portion given the instruction.
# Same shifted input/target pattern as pre-training.
# =============================================================================

class InstructDataset(Dataset):
    """Fine-tuning dataset for instruction-following."""

    # Template that structures each training example
    TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"

    def __init__(self, examples: list[dict], tokenizer, max_len: int = 1024):
        """
        Args:
            examples:  List of dicts with 'instruction' and 'response' keys
            tokenizer: Tokenizer instance for encoding text
            max_len:   Maximum sequence length (padded/truncated to this)
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Format the instruction/response pair using the template
        text = self.TEMPLATE.format(
            instruction=ex["instruction"], response=ex["response"]
        )

        # Tokenize with truncation to max_len + 1
        ids = self.tokenizer.encode(text, max_length=self.max_len + 1, truncation=True)
        ids = torch.tensor(ids, dtype=torch.long)

        # Pad shorter sequences to fixed length (needed for batching)
        if len(ids) < self.max_len + 1:
            pad = torch.full((self.max_len + 1 - len(ids),), self.tokenizer.pad_token_id)
            ids = torch.cat([ids, pad])

        # Return shifted input/target pairs
        return ids[: self.max_len], ids[1 : self.max_len + 1]
