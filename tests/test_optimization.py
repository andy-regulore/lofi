"""Tests for optimization utilities."""

import pytest
import torch
import torch.nn as nn


def test_generation_cache():
    """Test generation cache."""
    from src.optimization import GenerationCache

    cache = GenerationCache(max_cache_size=10)

    # Test put and get
    params = {'tempo': 75, 'key': 'Am', 'seed': 42}
    tokens = [1, 2, 3, 4, 5]

    cache.put(params, tokens)
    cached = cache.get(params)

    assert cached == tokens

    # Test cache miss
    different_params = {'tempo': 80, 'key': 'C', 'seed': 42}
    assert cache.get(different_params) is None

    # Test cache eviction
    for i in range(15):
        cache.put({'seed': i}, [i])

    stats = cache.get_stats()
    assert stats['size'] == 10  # Max size
    assert stats['utilization'] == 100.0


def test_model_quantization():
    """Test model quantization."""
    from src.optimization import ModelQuantizer

    # Create simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )

    # Test INT8 quantization
    quantized = ModelQuantizer.quantize_int8(model)
    assert quantized is not None

    # Test inference still works
    x = torch.randn(1, 100)
    output = quantized(x)
    assert output.shape == (1, 10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fp16_conversion():
    """Test FP16 conversion."""
    from src.optimization import ModelQuantizer

    model = nn.Linear(100, 50)

    # Convert to FP16
    fp16_model = ModelQuantizer.convert_to_fp16(model, device='cuda')

    # Check dtype
    for param in fp16_model.parameters():
        assert param.dtype == torch.float16


def test_beam_search_generator():
    """Test beam search generation."""
    from src.optimization import BeamSearchGenerator

    # Note: This test would require a full model
    # Just test that the class exists and has the method
    assert hasattr(BeamSearchGenerator, 'beam_search')


def test_constrained_decoder():
    """Test constrained decoding."""
    from src.optimization import ConstrainedDecoder
    from src.music_theory import MusicTheoryEngine
    from unittest.mock import Mock

    theory = MusicTheoryEngine()
    tokenizer = Mock()

    decoder = ConstrainedDecoder(theory, tokenizer)

    # Test setting constraints
    decoder.set_constraints(key='C', allow_chromatic=True)
    assert decoder.allowed_tokens is not None


def test_batch_inference_optimizer():
    """Test batch inference optimization."""
    from src.optimization import BatchInferenceOptimizer

    requests = [{'id': i} for i in range(100)]

    batches = BatchInferenceOptimizer.dynamic_batching(
        requests,
        max_batch_size=32,
    )

    assert len(batches) == 4  # 100 / 32 = 3.125 -> 4 batches
    assert len(batches[0]) == 32
    assert len(batches[-1]) == 4  # Remaining


def test_kv_cache_optimizer():
    """Test KV-cache optimization."""
    from src.optimization import KVCacheOptimizer
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=100,
        n_positions=512,
        n_embd=256,
        n_layer=2,
        n_head=4,
    )

    model = GPT2LMHeadModel(config)

    # Enable KV-cache
    KVCacheOptimizer.enable_kv_cache(model)
    assert model.config.use_cache is True
