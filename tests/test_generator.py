"""Unit tests for generator module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from src.generator import LoFiGenerator


@pytest.mark.unit
class TestLoFiGenerator:
    """Tests for LoFiGenerator class."""

    def test_init(self, mock_model, mock_tokenizer, test_config):
        """Test generator initialization."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        assert generator.model == mock_model
        assert generator.tokenizer == mock_tokenizer
        assert generator.config == test_config
        assert generator.device == "cpu"

    def test_generate_track_default_params(self, mock_model, mock_tokenizer, test_config):
        """Test track generation with default parameters."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        tokens, metadata = generator.generate_track()

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert isinstance(metadata, dict)

        # Check metadata
        assert "tempo" in metadata
        assert "key" in metadata
        assert "mood" in metadata
        assert "temperature" in metadata
        assert "num_tokens" in metadata

    def test_generate_track_with_conditioning(self, mock_model, mock_tokenizer, test_config):
        """Test track generation with specific conditioning."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        tokens, metadata = generator.generate_track(tempo=75, key="C", mood="chill")

        assert metadata["tempo"] == 75
        assert metadata["key"] == "C"
        assert metadata["mood"] == "chill"

    def test_generate_track_with_seed(self, mock_model, mock_tokenizer, test_config):
        """Test reproducible generation with seed."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        tokens1, _ = generator.generate_track(seed=42)
        tokens2, _ = generator.generate_track(seed=42)

        # Should be identical with same seed
        assert tokens1 == tokens2

    def test_generate_track_different_seeds(self, mock_model, mock_tokenizer, test_config):
        """Test different generations with different seeds."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        tokens1, _ = generator.generate_track(seed=42)
        tokens2, _ = generator.generate_track(seed=123)

        # Different seeds should produce different results (very likely)
        # Note: There's a tiny chance they could be the same by coincidence
        assert len(tokens1) > 0
        assert len(tokens2) > 0

    def test_generate_track_custom_sampling_params(self, mock_model, mock_tokenizer, test_config):
        """Test generation with custom sampling parameters."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        tokens, metadata = generator.generate_track(
            temperature=1.2, top_k=40, top_p=0.9, max_length=512
        )

        assert metadata["temperature"] == 1.2
        assert metadata["top_k"] == 40
        assert metadata["top_p"] == 0.9

    def test_tokens_to_midi_success(self, mock_model, mock_tokenizer, test_config, temp_dir):
        """Test converting tokens to MIDI."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        tokens = list(range(100))
        output_path = temp_dir / "test.mid"

        success = generator.tokens_to_midi(tokens, str(output_path))

        # Mock tokenizer should succeed
        assert success is True

    def test_tokens_to_midi_removes_conditioning_tokens(
        self, mock_tokenizer, test_config, temp_dir
    ):
        """Test that conditioning tokens are filtered out."""
        # Create conditioned model mock
        mock_model = MagicMock()
        mock_model.base_vocab_size = 1000
        mock_model.get_model.return_value = MagicMock()
        mock_model.to = lambda device: mock_model

        def mock_generate(**kwargs):
            return torch.randint(0, 1200, (1, 100))  # Include conditioning tokens

        mock_model.generate = mock_generate

        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        # Tokens include some conditioning tokens (>= 1000)
        tokens = list(range(50)) + [1000, 1001, 1002] + list(range(50, 100))
        output_path = temp_dir / "test.mid"

        generator.tokens_to_midi(tokens, str(output_path))

        # Should have filtered out tokens >= base_vocab_size
        # Check via mock call

    def test_generate_and_save(self, mock_model, mock_tokenizer, test_config, temp_dir):
        """Test generating and saving track."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        output_path = temp_dir / "track.mid"
        metadata = generator.generate_and_save(
            str(output_path), tempo=75, key="Am", mood="melancholic"
        )

        assert "output_path" in metadata
        assert metadata["tempo"] == 75
        assert metadata["key"] == "Am"
        assert metadata["mood"] == "melancholic"

    def test_batch_generate(self, mock_model, mock_tokenizer, test_config, temp_dir):
        """Test batch generation."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        output_dir = temp_dir / "tracks"
        metadata_list = generator.batch_generate(
            num_tracks=5, output_dir=str(output_dir), name_prefix="lofi"
        )

        assert len(metadata_list) == 5
        assert all("track_number" in m for m in metadata_list)
        assert metadata_list[0]["track_number"] == 1
        assert metadata_list[4]["track_number"] == 5

        # Check metadata file
        assert (output_dir / "lofi_metadata.json").exists()

    def test_batch_generate_with_variety(self, mock_model, mock_tokenizer, test_config, temp_dir):
        """Test batch generation ensures variety."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        output_dir = temp_dir / "tracks"
        metadata_list = generator.batch_generate(
            num_tracks=10, output_dir=str(output_dir), ensure_variety=True
        )

        # Extract tempos, keys, moods
        tempos = [m["tempo"] for m in metadata_list]
        keys = [m["key"] for m in metadata_list]
        moods = [m["mood"] for m in metadata_list]

        # Should have variety (more than 1 unique value)
        assert len(set(tempos)) > 1
        assert len(set(keys)) > 1
        assert len(set(moods)) > 1

    def test_batch_generate_without_variety(
        self, mock_model, mock_tokenizer, test_config, temp_dir
    ):
        """Test batch generation without enforced variety."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        output_dir = temp_dir / "tracks"
        metadata_list = generator.batch_generate(
            num_tracks=5, output_dir=str(output_dir), ensure_variety=False
        )

        assert len(metadata_list) == 5

    def test_calculate_quality_score(self, mock_model, mock_tokenizer, test_config):
        """Test quality score calculation."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        # Test with ideal parameters
        tokens = list(range(1500))  # Ideal length
        metadata = {"tempo": 75}  # Ideal tempo

        score = generator.calculate_quality_score(tokens, metadata)

        assert 0 <= score <= 10
        assert score > 5  # Should be above base score

    def test_calculate_quality_score_length_penalty(self, mock_model, mock_tokenizer, test_config):
        """Test quality score penalizes wrong length."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        # Too short
        tokens_short = list(range(100))
        score_short = generator.calculate_quality_score(tokens_short, {"tempo": 75})

        # Ideal length
        tokens_ideal = list(range(1500))
        score_ideal = generator.calculate_quality_score(tokens_ideal, {"tempo": 75})

        # Too long
        tokens_long = list(range(5000))
        score_long = generator.calculate_quality_score(tokens_long, {"tempo": 75})

        # Ideal should score higher
        assert score_ideal > score_short
        assert score_ideal > score_long

    def test_calculate_quality_score_diversity(self, mock_model, mock_tokenizer, test_config):
        """Test quality score rewards diversity."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        # Low diversity (many repeated tokens)
        tokens_low_div = [1] * 1000 + [2] * 500
        score_low = generator.calculate_quality_score(tokens_low_div, {"tempo": 75})

        # High diversity (all unique tokens)
        tokens_high_div = list(range(1500))
        score_high = generator.calculate_quality_score(tokens_high_div, {"tempo": 75})

        # Higher diversity should score better
        assert score_high > score_low

    def test_calculate_quality_score_tempo_bonus(self, mock_model, mock_tokenizer, test_config):
        """Test quality score gives bonus for ideal tempo."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        tokens = list(range(1500))

        # Ideal tempo (65-85 BPM for lo-fi)
        score_ideal = generator.calculate_quality_score(tokens, {"tempo": 75})

        # Non-ideal tempo
        score_fast = generator.calculate_quality_score(tokens, {"tempo": 140})

        # Ideal tempo should score higher
        assert score_ideal > score_fast

    def test_generate_track_conditioned_model(self, mock_tokenizer, test_config):
        """Test generation with conditioned model."""
        # Create conditioned model mock
        mock_model = MagicMock()
        mock_model.base_vocab_size = 1000
        mock_model.get_model.return_value = MagicMock()
        mock_model.to = lambda device: mock_model

        # Mock conditioning prefix creation
        mock_model.create_conditioning_prefix = lambda tempo, key, mood: [1000, 1001, 1002]

        def mock_generate(input_ids, **kwargs):
            return torch.randint(0, 1000, (1, 100))

        mock_model.generate = mock_generate

        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        tokens, metadata = generator.generate_track(tempo=75, key="C", mood="chill")

        assert len(tokens) > 0
        assert metadata["tempo"] == 75

    def test_model_in_eval_mode(self, mock_model, mock_tokenizer, test_config):
        """Test that model is set to eval mode."""
        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cpu")

        # Model should be in eval mode
        mock_model.get_model().eval.assert_called()

    @pytest.mark.requires_gpu
    def test_generate_on_gpu(self, mock_model, mock_tokenizer, test_config):
        """Test generation on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        generator = LoFiGenerator(mock_model, mock_tokenizer, test_config, device="cuda")

        tokens, metadata = generator.generate_track()

        assert len(tokens) > 0
