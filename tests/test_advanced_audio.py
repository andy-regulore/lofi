"""Tests for advanced audio processing."""

import numpy as np
import pytest


@pytest.fixture
def sample_audio():
    """Create sample audio for testing."""
    sample_rate = 44100
    duration = 1.0  # 1 second
    frequency = 440  # A4

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)

    return audio, sample_rate


def test_advanced_lofi_effects(sample_audio):
    """Test advanced lo-fi effects."""
    from src.advanced_audio import AdvancedLoFiEffects

    audio, sr = sample_audio
    processor = AdvancedLoFiEffects(sample_rate=sr)

    # Test vintage tape saturation
    saturated = processor.vintage_tape_saturation(audio, drive=1.5)
    assert saturated.shape == audio.shape
    assert not np.array_equal(saturated, audio)

    # Test wow/flutter
    modulated = processor.analog_wow_flutter(audio)
    assert modulated.shape == audio.shape

    # Test vinyl simulation
    vinyl = processor.vinyl_simulation(audio)
    assert vinyl.shape == audio.shape

    # Test vintage EQ
    eq_audio = processor.vintage_eq_curve(audio, style="lofi")
    assert eq_audio.shape == audio.shape


def test_lofi_presets(sample_audio):
    """Test lo-fi effects presets."""
    from src.advanced_audio import AdvancedLoFiEffects

    audio, sr = sample_audio
    processor = AdvancedLoFiEffects(sample_rate=sr)

    presets = ["classic", "heavy", "subtle", "vintage"]

    for preset in presets:
        processed = processor.apply_all_lofi_effects(audio, preset=preset)
        assert processed.shape == audio.shape
        assert not np.array_equal(processed, audio)


def test_spectral_processor(sample_audio):
    """Test spectral processing."""
    from src.advanced_audio import SpectralProcessor

    audio, sr = sample_audio
    processor = SpectralProcessor(sample_rate=sr)

    # Test spectral gate
    gated = processor.spectral_gate(audio, threshold_db=-40)
    assert gated.shape == audio.shape

    # Test spectral smearing
    smeared = processor.spectral_smearing(audio, amount=0.3)
    assert smeared.shape == audio.shape


def test_stem_separator(sample_audio):
    """Test stem separation."""
    from src.advanced_audio import StemSeparator

    audio, sr = sample_audio
    separator = StemSeparator()

    stems = separator.separate(audio, sample_rate=sr)

    assert "bass" in stems
    assert "mids" in stems
    assert "highs" in stems
    assert stems["bass"].shape == audio.shape


def test_professional_mixer(sample_audio):
    """Test professional mixing tools."""
    from src.advanced_audio import ProfessionalMixer

    audio, sr = sample_audio

    # Test stereo widener
    stereo_audio = np.array([audio, audio])  # Create stereo
    widened = ProfessionalMixer.stereo_widener(stereo_audio, width=1.5)

    assert widened.shape == stereo_audio.shape


def test_neural_vocoder():
    """Test neural vocoder."""
    from src.advanced_audio import NeuralVocoder

    vocoder = NeuralVocoder()

    # Test without model (should use Griffin-Lim fallback)
    mel_spec = np.random.randn(80, 100)  # Random mel-spectrogram
    audio = vocoder.synthesize(mel_spec)

    assert audio.ndim == 1
    assert len(audio) > 0
