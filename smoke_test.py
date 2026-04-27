"""
=============================================================================
smoke_test.py — Quick Validation of All Components
=============================================================================

Runs a suite of lightweight tests to verify every component of the LLM
pipeline works correctly, without needing a GPU or downloading datasets.

Tests:
  1. Model forward pass — correct output shapes and valid loss
  2. Inference mode — works without targets (generation scenario)
  3. TextDataset — pre-training data chunking works correctly
  4. InstructDataset — fine-tuning data formatting and padding works
  5. Backward pass — gradients flow through all layers
  6. Generation — token-by-token generation loop runs end-to-end

All tests use tiny model configs (64-dim, 2 layers) and fake tokenizers
so they run in seconds on CPU.

Usage:
    python3 smoke_test.py

=============================================================================
"""

import torch
from config import ModelConfig, TrainConfig
from model import GPTModel
from dataset import TextDataset, InstructDataset
from generate import generate


# =============================================================================
# Test 1: Forward Pass
# =============================================================================
# Verifies the model produces the correct output shape and a valid loss.
# Expected: logits shape (batch, seq_len, vocab_size), loss > 0
# =============================================================================

def test_model_forward():
    """Test that the model runs a forward pass and produces correct shapes."""
    print("1. Testing model forward pass...")
    cfg = ModelConfig(vocab_size=256, hidden_size=64, num_layers=2, num_heads=4,
                      intermediate_size=128, max_seq_len=32)
    model = GPTModel(cfg)

    input_ids = torch.randint(0, 256, (2, 32))  # batch=2, seq_len=32
    targets = torch.randint(0, 256, (2, 32))

    logits, loss = model(input_ids, targets)
    assert logits.shape == (2, 32, 256), f"Bad logits shape: {logits.shape}"
    assert loss is not None and loss.item() > 0, "Loss should be positive"
    print(f"   Logits shape: {logits.shape}, Loss: {loss.item():.4f} ✓")


# =============================================================================
# Test 2: Inference Mode (No Targets)
# =============================================================================
# During generation, we don't have targets — the model should return
# logits without computing loss.
# =============================================================================

def test_model_no_targets():
    """Test inference mode (no targets, no loss)."""
    print("2. Testing inference mode (no targets)...")
    cfg = ModelConfig(vocab_size=256, hidden_size=64, num_layers=2, num_heads=4,
                      intermediate_size=128, max_seq_len=32)
    model = GPTModel(cfg)

    input_ids = torch.randint(0, 256, (1, 10))
    logits, loss = model(input_ids)
    assert logits.shape == (1, 10, 256)
    assert loss is None
    print(f"   Logits shape: {logits.shape}, Loss: None ✓")


# =============================================================================
# Test 3: Pre-training Dataset
# =============================================================================
# Verifies that TextDataset correctly tokenizes text, concatenates it,
# and chunks it into fixed-length sequences with proper input/target pairs.
# =============================================================================

def test_text_dataset():
    """Test pre-training dataset chunking."""
    print("3. Testing TextDataset...")

    # Fake tokenizer that just converts characters to indices
    class FakeTokenizer:
        eos_token_id = 0
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text)))

    texts = ["a" * 100, "b" * 100]
    ds = TextDataset(texts, FakeTokenizer(), max_len=16)
    assert len(ds) > 0, "Dataset should have chunks"
    x, y = ds[0]
    assert x.shape == (16,) and y.shape == (16,), f"Bad shapes: {x.shape}, {y.shape}"
    print(f"   Chunks: {len(ds)}, input shape: {x.shape}, target shape: {y.shape} ✓")


# =============================================================================
# Test 4: Instruction Fine-tuning Dataset
# =============================================================================
# Verifies that InstructDataset correctly formats instruction/response pairs,
# tokenizes them, and pads to the expected fixed length.
# =============================================================================

def test_instruct_dataset():
    """Test fine-tuning dataset formatting."""
    print("4. Testing InstructDataset...")

    class FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0
        def encode(self, text, max_length=None, truncation=False):
            return list(range(min(len(text), max_length or len(text))))

    examples = [
        {"instruction": "Say hello", "response": "Hello!"},
        {"instruction": "Count to 3", "response": "1, 2, 3"},
    ]
    ds = InstructDataset(examples, FakeTokenizer(), max_len=32)
    x, y = ds[0]
    assert x.shape == (32,) and y.shape == (32,)
    print(f"   Samples: {len(ds)}, input shape: {x.shape}, target shape: {y.shape} ✓")


# =============================================================================
# Test 5: Backward Pass (Gradient Flow)
# =============================================================================
# Verifies that gradients propagate through the entire model during
# backpropagation. If gradients don't flow, the model can't learn.
# =============================================================================

def test_backward_pass():
    """Test that gradients flow correctly."""
    print("5. Testing backward pass (gradient flow)...")
    cfg = ModelConfig(vocab_size=256, hidden_size=64, num_layers=2, num_heads=4,
                      intermediate_size=128, max_seq_len=32)
    model = GPTModel(cfg)

    input_ids = torch.randint(0, 256, (2, 32))
    targets = torch.randint(0, 256, (2, 32))
    _, loss = model(input_ids, targets)
    loss.backward()

    # Check that at least some parameters received gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients found!"
    print(f"   Gradients flowing ✓")


# =============================================================================
# Test 6: Text Generation
# =============================================================================
# Verifies the full generation pipeline: encode prompt → generate tokens
# one at a time → decode back to text.
# =============================================================================

def test_generation():
    """Test text generation loop."""
    print("6. Testing text generation...")
    cfg = ModelConfig(vocab_size=256, hidden_size=64, num_layers=2, num_heads=4,
                      intermediate_size=128, max_seq_len=32)
    model = GPTModel(cfg)

    class FakeTokenizer:
        eos_token_id = 0
        def encode(self, text, return_tensors=None):
            ids = list(range(1, min(len(text) + 1, 10)))
            if return_tensors == "pt":
                return torch.tensor([ids])
            return ids
        def decode(self, ids, skip_special_tokens=False):
            return "generated_text_ok"

    result = generate(model, FakeTokenizer(), "test", max_new_tokens=5, device="cpu")
    assert isinstance(result, str) and len(result) > 0
    print(f"   Generated: '{result}' ✓")


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("LLM Prototype Smoke Test")
    print("=" * 50)

    tests = [
        test_model_forward,
        test_model_no_targets,
        test_text_dataset,
        test_instruct_dataset,
        test_backward_pass,
        test_generation,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   FAILED: {e}")

    print("=" * 50)
    print(f"Results: {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("All good — your pipeline is working! 🚀")
    else:
        print("Some tests failed — check the errors above.")
    print("=" * 50)
