"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import yaml


@pytest.fixture
def test_config() -> Dict:
    """Load test configuration."""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override for testing
    config['training']['device'] = 'cpu'
    config['training']['num_epochs'] = 2
    config['training']['batch_size'] = 2
    config['training']['fp16'] = False
    config['training']['dataloader_num_workers'] = 0

    return config


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    yield tmp_path
    # Cleanup handled automatically by pytest tmp_path


@pytest.fixture
def sample_midi_path(temp_dir):
    """Create a simple test MIDI file."""
    import pretty_midi

    # Create a simple MIDI file
    midi = pretty_midi.PrettyMIDI(initial_tempo=75)

    # Add piano track
    piano = pretty_midi.Instrument(program=0, is_drum=False, name='Piano')

    # Add some notes (C major scale)
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=i * 0.5,
            end=(i + 1) * 0.5
        )
        piano.notes.append(note)

    midi.instruments.append(piano)

    # Add drums
    drums = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')
    for i in range(16):
        note = pretty_midi.Note(
            velocity=100,
            pitch=36,  # Kick drum
            start=i * 0.25,
            end=(i + 0.1) * 0.25
        )
        drums.notes.append(note)

    midi.instruments.append(drums)

    # Save
    midi_path = temp_dir / 'test_track.mid'
    midi.write(str(midi_path))

    return str(midi_path)


@pytest.fixture
def sample_tokens():
    """Generate sample token sequences for testing."""
    np.random.seed(42)
    # Generate random token sequences
    sequences = [
        list(np.random.randint(0, 1000, size=512)) for _ in range(10)
    ]
    return sequences


@pytest.fixture
def mock_tokenizer(test_config):
    """Create a mock tokenizer for testing."""
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.get_vocab_size.return_value = 1000
    tokenizer.config = test_config
    tokenizer.token_config = test_config['tokenization']
    tokenizer.tokenizer_config = MagicMock()

    # Mock tokenization
    def mock_tokenize(midi_path, check_quality=True):
        return {
            'tokens': list(np.random.randint(0, 1000, size=512)),
            'metadata': {
                'tempo': 75,
                'duration': 30,
                'has_drums': True,
                'total_notes': 100,
                'note_density': 3.33,
                'key': 'C',
                'mood': 'chill',
                'instruments': [0],
                'num_tracks': 2,
            },
            'file_path': str(midi_path),
        }

    tokenizer.tokenize_midi = mock_tokenize
    tokenizer.tokens_to_midi = MagicMock()
    tokenizer.chunk_sequence = lambda tokens, chunk_size=None, overlap=None: [tokens]

    return tokenizer


@pytest.fixture
def mock_model(test_config):
    """Create a mock model for testing."""
    from unittest.mock import MagicMock
    import torch

    model = MagicMock()
    model.config = test_config
    model.model_config = test_config['model']
    model.vocab_size = 1000

    # Mock GPT2 model
    gpt2_model = MagicMock()
    gpt2_model.eval.return_value = None
    gpt2_model.parameters.return_value = [torch.randn(100, 100)]

    # Mock generation
    def mock_generate(input_ids, **kwargs):
        batch_size = input_ids.shape[0]
        max_length = kwargs.get('max_length', 1024)
        return torch.randint(0, 1000, (batch_size, max_length))

    gpt2_model.generate = mock_generate
    model.get_model.return_value = gpt2_model
    model.model = gpt2_model

    # Mock to() method
    model.to = lambda device: model
    model.generate = mock_generate

    return model


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    duration = 2.0  # seconds
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generate a simple sine wave
    frequency = 440  # A4
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    return audio, sample_rate


@pytest.fixture
def sample_wav_file(temp_dir, sample_audio_data):
    """Create a sample WAV file for testing."""
    import soundfile as sf

    audio, sample_rate = sample_audio_data
    wav_path = temp_dir / 'test_audio.wav'
    sf.write(str(wav_path), audio, sample_rate)

    return str(wav_path)


@pytest.fixture(scope='session')
def device():
    """Get the device to use for testing."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_wandb():
    """Mock wandb for testing."""
    mock = MagicMock()
    mock.init.return_value = None
    mock.log.return_value = None
    mock.finish.return_value = None
    return mock
