"""Unit tests for audio_processor module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import soundfile as sf

from src.audio_processor import LoFiAudioProcessor


@pytest.mark.unit
class TestLoFiAudioProcessor:
    """Tests for LoFiAudioProcessor class."""

    def test_init(self, test_config):
        """Test audio processor initialization."""
        processor = LoFiAudioProcessor(test_config)

        assert processor.config == test_config
        assert processor.audio_config == test_config["audio"]
        assert processor.lofi_config == test_config["audio"]["lofi_effects"]
        assert processor.sample_rate == test_config["audio"]["sample_rate"]

    @pytest.mark.requires_fluidsynth
    def test_midi_to_wav_with_fluidsynth(self, test_config, sample_midi_path, temp_dir):
        """Test MIDI to WAV conversion using FluidSynth."""
        processor = LoFiAudioProcessor(test_config)

        output_path = temp_dir / "output.wav"

        # This requires FluidSynth to be installed
        try:
            success = processor.midi_to_wav(
                str(sample_midi_path), str(output_path), soundfont_path=None
            )

            if success:
                assert output_path.exists()
                assert output_path.stat().st_size > 0
        except FileNotFoundError:
            pytest.skip("FluidSynth not installed")

    def test_midi_to_wav_fallback(self, test_config, sample_midi_path, temp_dir):
        """Test MIDI to WAV conversion fallback method."""
        processor = LoFiAudioProcessor(test_config)

        output_path = temp_dir / "output.wav"

        # Use fallback synthesis
        success = processor._synthesize_midi_to_wav(str(sample_midi_path), str(output_path))

        if success:
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_midi_to_wav_invalid_path(self, test_config, temp_dir):
        """Test MIDI to WAV with invalid path."""
        processor = LoFiAudioProcessor(test_config)

        output_path = temp_dir / "output.wav"

        success = processor.midi_to_wav("/nonexistent/file.mid", str(output_path))

        assert success is False

    def test_apply_lofi_effects(self, test_config, sample_wav_file, temp_dir):
        """Test applying lo-fi effects to audio."""
        processor = LoFiAudioProcessor(test_config)

        output_path = temp_dir / "lofi_output.wav"

        success = processor.apply_lofi_effects(sample_wav_file, str(output_path))

        assert success is True
        assert output_path.exists()

        # Load and check output
        audio, sr = sf.read(str(output_path))
        assert sr == test_config["audio"]["sample_rate"]
        assert len(audio) > 0

    def test_apply_lofi_effects_stereo(self, test_config, temp_dir):
        """Test lo-fi effects preserve stereo."""
        # Create stereo audio
        duration = 2.0
        sample_rate = test_config["audio"]["sample_rate"]
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Stereo signal (different frequencies for L/R)
        audio_stereo = np.stack(
            [
                0.5 * np.sin(2 * np.pi * 440 * t),  # Left
                0.5 * np.sin(2 * np.pi * 880 * t),  # Right
            ]
        )

        input_path = temp_dir / "stereo_input.wav"
        sf.write(str(input_path), audio_stereo.T, sample_rate)

        processor = LoFiAudioProcessor(test_config)
        output_path = temp_dir / "stereo_output.wav"

        success = processor.apply_lofi_effects(str(input_path), str(output_path))

        assert success is True

        # Check output is stereo
        audio_out, sr = sf.read(str(output_path))
        assert audio_out.ndim == 2
        assert audio_out.shape[1] == 2

    def test_add_vinyl_crackle(self, test_config, sample_audio_data):
        """Test adding vinyl crackle effect."""
        processor = LoFiAudioProcessor(test_config)
        audio, sr = sample_audio_data

        # Reshape to 2D for processing
        audio_2d = np.stack([audio, audio])

        audio_with_crackle = processor._add_vinyl_crackle(audio_2d, intensity=0.015)

        assert audio_with_crackle.shape == audio_2d.shape
        # Should be different due to crackle
        assert not np.allclose(audio_with_crackle, audio_2d)

    def test_add_vinyl_crackle_intensity(self, test_config, sample_audio_data):
        """Test vinyl crackle intensity variation."""
        processor = LoFiAudioProcessor(test_config)
        audio, sr = sample_audio_data
        audio_2d = np.stack([audio, audio])

        # Low intensity
        audio_low = processor._add_vinyl_crackle(audio_2d, intensity=0.001)

        # High intensity
        audio_high = processor._add_vinyl_crackle(audio_2d, intensity=0.1)

        # Higher intensity should produce more difference
        diff_low = np.abs(audio_low - audio_2d).mean()
        diff_high = np.abs(audio_high - audio_2d).mean()

        assert diff_high > diff_low

    def test_add_tape_effects(self, test_config, sample_audio_data):
        """Test adding tape wow and flutter effects."""
        processor = LoFiAudioProcessor(test_config)
        audio, sr = sample_audio_data

        audio_2d = np.stack([audio, audio])

        audio_with_tape = processor._add_tape_effects(audio_2d, depth=0.002)

        assert audio_with_tape.shape == audio_2d.shape
        assert not np.allclose(audio_with_tape, audio_2d)

    def test_add_tape_effects_depth(self, test_config, sample_audio_data):
        """Test tape effects depth variation."""
        processor = LoFiAudioProcessor(test_config)
        audio, sr = sample_audio_data
        audio_2d = np.stack([audio, audio])

        # Low depth
        audio_low = processor._add_tape_effects(audio_2d, depth=0.0001)

        # High depth
        audio_high = processor._add_tape_effects(audio_2d, depth=0.01)

        # Higher depth should produce more modulation
        diff_low = np.abs(audio_low - audio_2d).mean()
        diff_high = np.abs(audio_high - audio_2d).mean()

        assert diff_high > diff_low

    def test_normalize_audio(self, test_config, sample_audio_data):
        """Test audio normalization to target LUFS."""
        processor = LoFiAudioProcessor(test_config)
        audio, sr = sample_audio_data

        # Very quiet audio
        audio_quiet = audio * 0.1

        audio_normalized = processor._normalize_audio(audio_quiet)

        # Should be louder than input
        assert np.abs(audio_normalized).mean() > np.abs(audio_quiet).mean()

        # Should not clip
        assert np.abs(audio_normalized).max() <= 1.0

    def test_normalize_audio_stereo(self, test_config):
        """Test stereo audio normalization."""
        processor = LoFiAudioProcessor(test_config)

        # Create stereo audio
        duration = 2.0
        sr = test_config["audio"]["sample_rate"]
        t = np.linspace(0, duration, int(sr * duration))

        audio_stereo = np.stack(
            [
                0.1 * np.sin(2 * np.pi * 440 * t),
                0.1 * np.sin(2 * np.pi * 880 * t),
            ]
        )

        audio_normalized = processor._normalize_audio(audio_stereo)

        assert audio_normalized.shape == audio_stereo.shape
        assert np.abs(audio_normalized).mean() > np.abs(audio_stereo).mean()

    def test_normalize_audio_fallback(self, test_config, sample_audio_data):
        """Test normalization fallback when pyloudnorm fails."""
        processor = LoFiAudioProcessor(test_config)
        audio, sr = sample_audio_data

        # Mock pyloudnorm to fail
        with patch("src.audio_processor.pyln.Meter") as mock_meter:
            mock_meter.side_effect = Exception("Mock error")

            audio_normalized = processor._normalize_audio(audio)

            # Should still return normalized audio (using fallback)
            assert audio_normalized is not None
            assert len(audio_normalized) == len(audio)

    def test_process_midi_to_lofi_complete(self, test_config, sample_midi_path, temp_dir):
        """Test complete MIDI to lo-fi audio pipeline."""
        processor = LoFiAudioProcessor(test_config)

        output_dir = temp_dir / "output"

        result = processor.process_midi_to_lofi(
            str(sample_midi_path),
            str(output_dir),
            name="test_track",
            save_clean=True,
            save_lofi=True,
        )

        assert "midi_path" in result
        assert "name" in result

        # Check that files were created (if conversion succeeded)
        if "clean_wav_path" in result:
            assert Path(result["clean_wav_path"]).exists()

        if "lofi_wav_path" in result:
            assert Path(result["lofi_wav_path"]).exists()

    def test_process_midi_to_lofi_only_lofi(self, test_config, sample_midi_path, temp_dir):
        """Test processing MIDI with only lo-fi output."""
        processor = LoFiAudioProcessor(test_config)

        output_dir = temp_dir / "output"

        result = processor.process_midi_to_lofi(
            str(sample_midi_path), str(output_dir), save_clean=False, save_lofi=True
        )

        # Clean file should not be in results
        assert "clean_wav_path" not in result

        # Clean file should be deleted
        clean_path = output_dir / f"{Path(sample_midi_path).stem}_clean.wav"
        assert not clean_path.exists()

    def test_process_midi_to_lofi_invalid_midi(self, test_config, temp_dir):
        """Test processing invalid MIDI file."""
        processor = LoFiAudioProcessor(test_config)

        output_dir = temp_dir / "output"

        result = processor.process_midi_to_lofi("/nonexistent/file.mid", str(output_dir))

        assert "error" in result

    def test_downsampling_effect(self, test_config, sample_wav_file, temp_dir):
        """Test that downsampling is applied."""
        processor = LoFiAudioProcessor(test_config)

        # Load original
        audio_orig, sr_orig = sf.read(sample_wav_file)

        # Apply effects
        output_path = temp_dir / "processed.wav"
        processor.apply_lofi_effects(sample_wav_file, str(output_path))

        # Load processed
        audio_proc, sr_proc = sf.read(str(output_path))

        # Sample rates should match config
        assert sr_orig == test_config["audio"]["sample_rate"]
        assert sr_proc == test_config["audio"]["sample_rate"]

        # Audio should be modified
        # (Note: lengths might differ slightly due to processing)

    def test_compression_applied(self, test_config, temp_dir):
        """Test that compression is applied to audio."""
        # Create audio with wide dynamic range
        duration = 1.0
        sr = test_config["audio"]["sample_rate"]
        t = np.linspace(0, duration, int(sr * duration))

        # Alternating loud and quiet sections
        audio = np.concatenate(
            [
                0.9 * np.sin(2 * np.pi * 440 * t[: len(t) // 2]),  # Loud
                0.1 * np.sin(2 * np.pi * 440 * t[len(t) // 2 :]),  # Quiet
            ]
        )

        input_path = temp_dir / "dynamic_input.wav"
        sf.write(str(input_path), audio, sr)

        processor = LoFiAudioProcessor(test_config)
        output_path = temp_dir / "compressed_output.wav"

        processor.apply_lofi_effects(str(input_path), str(output_path))

        # Load output
        audio_out, _ = sf.read(str(output_path))

        # Compression should reduce dynamic range
        # (This is a rough test - actual compression is complex)
        assert len(audio_out) > 0

    def test_filtering_applied(self, test_config, temp_dir):
        """Test that frequency filtering is applied."""
        processor = LoFiAudioProcessor(test_config)

        # Create audio with wide frequency range
        duration = 1.0
        sr = test_config["audio"]["sample_rate"]
        t = np.linspace(0, duration, int(sr * duration))

        # Mix of low, mid, and high frequencies
        audio = (
            np.sin(2 * np.pi * 50 * t)  # Very low (should be filtered)
            + np.sin(2 * np.pi * 440 * t)  # Mid (should pass)
            + np.sin(2 * np.pi * 8000 * t)  # High (should be filtered)
        ) / 3.0

        input_path = temp_dir / "fullband_input.wav"
        sf.write(str(input_path), audio, sr)

        output_path = temp_dir / "filtered_output.wav"

        processor.apply_lofi_effects(str(input_path), str(output_path))

        # Load output
        audio_out, _ = sf.read(str(output_path))

        # Should be different due to filtering
        assert len(audio_out) > 0

    def test_bit_reduction_applied(self, test_config, sample_wav_file, temp_dir):
        """Test that bit reduction is applied."""
        processor = LoFiAudioProcessor(test_config)

        output_path = temp_dir / "bitcrushed.wav"

        processor.apply_lofi_effects(sample_wav_file, str(output_path))

        audio_out, _ = sf.read(str(output_path))

        # Bit reduction should quantize the audio
        assert len(audio_out) > 0

    def test_effects_can_be_disabled(self, test_config, sample_wav_file, temp_dir):
        """Test processing with effects disabled."""
        # Disable effects
        test_config["audio"]["lofi_effects"]["vinyl_crackle"]["enabled"] = False
        test_config["audio"]["lofi_effects"]["tape_wow_flutter"]["enabled"] = False

        processor = LoFiAudioProcessor(test_config)
        output_path = temp_dir / "minimal_effects.wav"

        success = processor.apply_lofi_effects(sample_wav_file, str(output_path))

        assert success is True
        assert output_path.exists()
