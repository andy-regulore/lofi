"""Unit tests for model module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.model import ConditionedLoFiModel, LoFiMusicModel


@pytest.mark.unit
class TestLoFiMusicModel:
    """Tests for LoFiMusicModel class."""

    def test_init(self, test_config):
        """Test model initialization."""
        vocab_size = 1000
        model = LoFiMusicModel(test_config, vocab_size)

        assert model.config == test_config
        assert model.vocab_size == vocab_size
        assert model.model is not None
        assert model.model_config["vocab_size"] == vocab_size

    def test_model_parameters(self, test_config):
        """Test that model has expected architecture."""
        vocab_size = 1000
        model = LoFiMusicModel(test_config, vocab_size)

        # Check GPT-2 config
        assert model.gpt2_config.vocab_size == vocab_size
        assert model.gpt2_config.n_positions == test_config["model"]["context_length"]
        assert model.gpt2_config.n_embd == test_config["model"]["embedding_dim"]
        assert model.gpt2_config.n_layer == test_config["model"]["num_layers"]
        assert model.gpt2_config.n_head == test_config["model"]["num_heads"]

    def test_parameter_count(self, test_config):
        """Test that model has expected number of parameters."""
        vocab_size = 1000
        model = LoFiMusicModel(test_config, vocab_size)

        total_params = sum(p.numel() for p in model.model.parameters())

        # Should be around 117M parameters (allow some variance)
        assert 100_000_000 < total_params < 150_000_000

    def test_get_model(self, test_config):
        """Test get_model method."""
        vocab_size = 1000
        model = LoFiMusicModel(test_config, vocab_size)

        gpt2_model = model.get_model()
        assert gpt2_model is not None
        assert hasattr(gpt2_model, "generate")

    def test_save_and_load(self, test_config, temp_dir):
        """Test model save and load."""
        vocab_size = 1000
        model = LoFiMusicModel(test_config, vocab_size)

        # Save model
        save_path = temp_dir / "test_model"
        model.save(str(save_path))

        assert save_path.exists()
        assert (save_path / "pytorch_model.bin").exists() or (
            save_path / "model.safetensors"
        ).exists()
        assert (save_path / "config.json").exists()
        assert (save_path / "lofi_config.json").exists()

        # Load model
        new_model = LoFiMusicModel(test_config, vocab_size)
        new_model.load(str(save_path))

        assert new_model.model is not None

    def test_load_nonexistent_path(self, test_config):
        """Test loading from nonexistent path raises error."""
        vocab_size = 1000
        model = LoFiMusicModel(test_config, vocab_size)

        with pytest.raises(ValueError):
            model.load("/nonexistent/path")

    def test_to_device(self, test_config):
        """Test moving model to device."""
        vocab_size = 1000
        model = LoFiMusicModel(test_config, vocab_size)

        # Move to CPU
        model = model.to("cpu")
        assert next(model.model.parameters()).device.type == "cpu"

        # Move to CUDA if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            assert next(model.model.parameters()).device.type == "cuda"

    def test_generate(self, test_config, device):
        """Test token generation."""
        vocab_size = 1000
        model = LoFiMusicModel(test_config, vocab_size)
        model.to(device)

        # Create input
        input_ids = torch.randint(0, vocab_size, (1, 10)).to(device)

        # Generate
        output = model.generate(
            input_ids,
            max_length=50,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
        )

        assert output.shape[0] == 1
        assert output.shape[1] <= 50
        assert output.device.type == device

    def test_generate_multiple_sequences(self, test_config, device):
        """Test generating multiple sequences."""
        vocab_size = 1000
        model = LoFiMusicModel(test_config, vocab_size)
        model.to(device)

        input_ids = torch.randint(0, vocab_size, (1, 10)).to(device)

        # Generate 3 sequences
        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=3,
        )

        assert output.shape[0] == 3
        assert output.shape[1] <= 50

    def test_generate_with_different_temperatures(self, test_config, device):
        """Test generation with different temperatures."""
        vocab_size = 1000
        model = LoFiMusicModel(test_config, vocab_size)
        model.to(device)

        input_ids = torch.randint(0, vocab_size, (1, 10)).to(device)

        # Test different temperatures
        for temperature in [0.5, 0.9, 1.2]:
            output = model.generate(
                input_ids,
                max_length=30,
                temperature=temperature,
            )
            assert output is not None

    def test_get_model_info(self, test_config):
        """Test model information retrieval."""
        vocab_size = 1000
        model = LoFiMusicModel(test_config, vocab_size)

        info = model.get_model_info()

        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "vocab_size" in info
        assert "embedding_dim" in info
        assert "num_layers" in info
        assert "num_heads" in info
        assert "context_length" in info

        assert info["vocab_size"] == vocab_size
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0


@pytest.mark.unit
class TestConditionedLoFiModel:
    """Tests for ConditionedLoFiModel class."""

    def test_init(self, test_config):
        """Test conditioned model initialization."""
        base_vocab_size = 1000
        model = ConditionedLoFiModel(test_config, base_vocab_size)

        assert model.base_vocab_size == base_vocab_size
        assert model.vocab_size == base_vocab_size + model.num_conditioning_tokens
        assert model.conditioning_offset == base_vocab_size

    def test_conditioning_token_ranges(self, test_config):
        """Test conditioning token ranges are properly defined."""
        model = ConditionedLoFiModel(test_config, 1000)

        # Check ranges don't overlap
        tempo_range = range(
            model.tempo_token_start, model.tempo_token_start + model.tempo_token_count
        )
        key_range = range(model.key_token_start, model.key_token_start + model.key_token_count)
        mood_range = range(model.mood_token_start, model.mood_token_start + model.mood_token_count)

        # Check no overlap
        assert not (set(tempo_range) & set(key_range))
        assert not (set(tempo_range) & set(mood_range))
        assert not (set(key_range) & set(mood_range))

    def test_get_tempo_token(self, test_config):
        """Test tempo token generation."""
        model = ConditionedLoFiModel(test_config, 1000)

        # Test various tempos
        test_tempos = [50, 75, 100, 125, 150, 175, 200]

        for tempo in test_tempos:
            token = model.get_tempo_token(tempo)

            assert (
                model.tempo_token_start <= token < model.tempo_token_start + model.tempo_token_count
            )

        # Test edge cases
        token_50 = model.get_tempo_token(50)  # Min
        token_200 = model.get_tempo_token(200)  # Max

        assert token_50 == model.tempo_token_start
        assert token_200 == model.tempo_token_start + model.tempo_token_count - 1

    def test_get_key_token(self, test_config):
        """Test key token generation."""
        model = ConditionedLoFiModel(test_config, 1000)

        # Test all 12 chromatic keys
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        tokens = []
        for key in keys:
            token = model.get_key_token(key)
            tokens.append(token)

            assert model.key_token_start <= token < model.key_token_start + model.key_token_count

        # All tokens should be unique
        assert len(set(tokens)) == len(keys)

        # Test minor keys
        token_am = model.get_key_token("Am")
        token_a = model.get_key_token("A")
        assert token_am == token_a  # Same root note

    def test_get_mood_token(self, test_config):
        """Test mood token generation."""
        model = ConditionedLoFiModel(test_config, 1000)

        # Test various moods
        moods = ["chill", "melancholic", "upbeat", "relaxed", "dreamy"]

        tokens = []
        for mood in moods:
            token = model.get_mood_token(mood)
            tokens.append(token)

            assert model.mood_token_start <= token < model.mood_token_start + model.mood_token_count

        # All tokens should be unique
        assert len(set(tokens)) == len(moods)

        # Test case insensitivity
        assert model.get_mood_token("CHILL") == model.get_mood_token("chill")

        # Test unknown mood (should return default)
        token_unknown = model.get_mood_token("unknown_mood")
        assert (
            model.mood_token_start
            <= token_unknown
            < model.mood_token_start + model.mood_token_count
        )

    def test_create_conditioning_prefix(self, test_config):
        """Test conditioning prefix creation."""
        model = ConditionedLoFiModel(test_config, 1000)

        prefix = model.create_conditioning_prefix(tempo=75, key="C", mood="chill")

        assert len(prefix) == 3
        assert all(isinstance(token, int) for token in prefix)

        # Check token ranges
        tempo_token, key_token, mood_token = prefix

        assert (
            model.tempo_token_start
            <= tempo_token
            < model.tempo_token_start + model.tempo_token_count
        )
        assert model.key_token_start <= key_token < model.key_token_start + model.key_token_count
        assert (
            model.mood_token_start <= mood_token < model.mood_token_start + model.mood_token_count
        )

    def test_conditioning_prefix_variations(self, test_config):
        """Test that different conditions produce different prefixes."""
        model = ConditionedLoFiModel(test_config, 1000)

        prefix1 = model.create_conditioning_prefix(75, "C", "chill")
        prefix2 = model.create_conditioning_prefix(85, "C", "chill")
        prefix3 = model.create_conditioning_prefix(75, "G", "chill")
        prefix4 = model.create_conditioning_prefix(75, "C", "upbeat")

        # Different tempos
        assert prefix1[0] != prefix2[0]

        # Different keys
        assert prefix1[1] != prefix3[1]

        # Different moods
        assert prefix1[2] != prefix4[2]

    def test_conditioned_generation(self, test_config, device):
        """Test generation with conditioning tokens."""
        model = ConditionedLoFiModel(test_config, 1000)
        model.to(device)

        # Create conditioning prefix
        prefix = model.create_conditioning_prefix(75, "C", "chill")
        input_ids = torch.tensor([prefix]).to(device)

        # Generate
        output = model.generate(
            input_ids,
            max_length=50,
            temperature=0.9,
        )

        assert output.shape[0] == 1
        assert output.shape[1] <= 50

        # First tokens should be conditioning tokens
        assert output[0, 0].item() in range(
            model.tempo_token_start, model.tempo_token_start + model.tempo_token_count
        )

    def test_vocab_size_extension(self, test_config):
        """Test that conditioned model extends vocabulary correctly."""
        base_vocab_size = 1000
        model = ConditionedLoFiModel(test_config, base_vocab_size)

        # Extended vocab should include conditioning tokens
        assert model.vocab_size > base_vocab_size
        assert model.vocab_size == base_vocab_size + model.num_conditioning_tokens

        # Model should use extended vocab
        assert model.gpt2_config.vocab_size == model.vocab_size
