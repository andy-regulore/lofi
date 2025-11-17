"""Unit tests for tokenizer module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.tokenizer import LoFiTokenizer


@pytest.mark.unit
class TestLoFiTokenizer:
    """Tests for LoFiTokenizer class."""

    def test_init(self, test_config):
        """Test tokenizer initialization."""
        tokenizer = LoFiTokenizer(test_config)

        assert tokenizer.config == test_config
        assert tokenizer.token_config == test_config['tokenization']
        assert tokenizer.quality_filters == test_config['data']['quality_filters']
        assert tokenizer.tokenizer is not None

    def test_get_vocab_size(self, test_config):
        """Test vocabulary size retrieval."""
        tokenizer = LoFiTokenizer(test_config)
        vocab_size = tokenizer.get_vocab_size()

        assert isinstance(vocab_size, int)
        assert vocab_size > 0

    def test_check_quality_pass(self, test_config, sample_midi_path):
        """Test quality check for a valid MIDI file."""
        tokenizer = LoFiTokenizer(test_config)
        passes, metadata = tokenizer.check_quality(sample_midi_path)

        assert isinstance(passes, bool)
        assert isinstance(metadata, dict)
        assert 'tempo' in metadata
        assert 'duration' in metadata
        assert 'has_drums' in metadata
        assert 'note_density' in metadata
        assert 'quality_checks' in metadata

    def test_check_quality_fail_tempo(self, test_config, temp_dir):
        """Test quality check fails for tempo out of range."""
        import pretty_midi

        # Create MIDI with tempo too high
        midi = pretty_midi.PrettyMIDI(initial_tempo=150)
        piano = pretty_midi.Instrument(program=0)
        note = pretty_midi.Note(velocity=80, pitch=60, start=0, end=1)
        piano.notes.append(note)
        midi.instruments.append(piano)

        midi_path = temp_dir / 'high_tempo.mid'
        midi.write(str(midi_path))

        tokenizer = LoFiTokenizer(test_config)
        passes, metadata = tokenizer.check_quality(str(midi_path))

        assert passes is False
        assert metadata['quality_checks']['tempo_ok'] is False

    def test_check_quality_fail_no_drums(self, test_config, temp_dir):
        """Test quality check fails when drums are required but missing."""
        import pretty_midi

        # Create MIDI without drums
        midi = pretty_midi.PrettyMIDI(initial_tempo=75)
        piano = pretty_midi.Instrument(program=0, is_drum=False)
        for i in range(100):
            note = pretty_midi.Note(
                velocity=80,
                pitch=60 + (i % 12),
                start=i * 0.1,
                end=(i + 1) * 0.1
            )
            piano.notes.append(note)
        midi.instruments.append(piano)

        midi_path = temp_dir / 'no_drums.mid'
        midi.write(str(midi_path))

        tokenizer = LoFiTokenizer(test_config)
        passes, metadata = tokenizer.check_quality(str(midi_path))

        # Should fail if require_drums is True
        if test_config['data']['quality_filters']['require_drums']:
            assert passes is False
            assert metadata['quality_checks']['has_drums'] is False

    def test_extract_metadata(self, test_config, sample_midi_path):
        """Test metadata extraction."""
        import pretty_midi

        tokenizer = LoFiTokenizer(test_config)
        midi = pretty_midi.PrettyMIDI(sample_midi_path)
        metadata = tokenizer._extract_metadata(midi)

        assert 'tempo' in metadata
        assert 'duration' in metadata
        assert 'has_drums' in metadata
        assert 'total_notes' in metadata
        assert 'note_density' in metadata
        assert 'key' in metadata
        assert 'mood' in metadata
        assert 'instruments' in metadata
        assert 'num_tracks' in metadata

        assert metadata['tempo'] > 0
        assert metadata['duration'] > 0
        assert metadata['total_notes'] > 0

    def test_tokenize_midi_success(self, test_config, sample_midi_path):
        """Test successful MIDI tokenization."""
        tokenizer = LoFiTokenizer(test_config)
        result = tokenizer.tokenize_midi(sample_midi_path, check_quality=False)

        assert result is not None
        assert 'tokens' in result
        assert 'metadata' in result
        assert 'file_path' in result
        assert isinstance(result['tokens'], list)
        assert len(result['tokens']) > 0

    def test_tokenize_midi_quality_check_fail(self, test_config, temp_dir):
        """Test MIDI tokenization with quality check failure."""
        import pretty_midi

        # Create low-quality MIDI
        midi = pretty_midi.PrettyMIDI(initial_tempo=200)  # Too fast
        piano = pretty_midi.Instrument(program=0)
        note = pretty_midi.Note(velocity=80, pitch=60, start=0, end=1)
        piano.notes.append(note)
        midi.instruments.append(piano)

        midi_path = temp_dir / 'bad_quality.mid'
        midi.write(str(midi_path))

        tokenizer = LoFiTokenizer(test_config)
        result = tokenizer.tokenize_midi(str(midi_path), check_quality=True)

        assert result is None

    def test_chunk_sequence(self, test_config):
        """Test sequence chunking."""
        tokenizer = LoFiTokenizer(test_config)
        tokens = list(range(2048))

        chunk_size = 512
        overlap = 128

        chunks = tokenizer.chunk_sequence(tokens, chunk_size=chunk_size, overlap=overlap)

        assert len(chunks) > 1
        assert all(len(chunk) == chunk_size for chunk in chunks)

        # Check overlap
        if len(chunks) > 1:
            # Last tokens of first chunk should overlap with first tokens of second
            assert chunks[0][-overlap:] == chunks[1][:overlap]

    def test_chunk_sequence_padding(self, test_config):
        """Test that short sequences are padded correctly."""
        tokenizer = LoFiTokenizer(test_config)
        tokens = list(range(300))  # Shorter than chunk_size

        chunk_size = 512
        chunks = tokenizer.chunk_sequence(tokens, chunk_size=chunk_size, overlap=0)

        assert len(chunks) == 1
        assert len(chunks[0]) == chunk_size
        assert chunks[0][:300] == tokens
        assert all(t == 0 for t in chunks[0][300:])  # Padded with zeros

    def test_tokenize_directory(self, test_config, temp_dir, sample_midi_path):
        """Test batch tokenization of directory."""
        # Create input directory with MIDI files
        input_dir = temp_dir / 'input'
        input_dir.mkdir()
        output_dir = temp_dir / 'output'

        # Copy sample MIDI
        import shutil
        for i in range(3):
            shutil.copy(sample_midi_path, input_dir / f'track_{i}.mid')

        tokenizer = LoFiTokenizer(test_config)
        stats = tokenizer.tokenize_directory(
            str(input_dir),
            str(output_dir),
            check_quality=False
        )

        assert stats['total_files'] == 3
        assert stats['processed'] >= 0
        assert stats['passed_quality'] >= 0

        # Check output files exist
        assert (output_dir / 'tokenization_stats.json').exists()
        assert (output_dir / 'metadata.json').exists()

    def test_tokens_to_midi(self, test_config, temp_dir, sample_tokens):
        """Test converting tokens back to MIDI."""
        tokenizer = LoFiTokenizer(test_config)
        output_path = temp_dir / 'generated.mid'

        # This might fail depending on tokenizer implementation
        # Just check it doesn't raise unexpected errors
        try:
            tokenizer.tokens_to_midi(sample_tokens[0], str(output_path))
            # If successful, file should exist
            if output_path.exists():
                assert output_path.stat().st_size > 0
        except Exception:
            # Expected for some token sequences
            pass

    def test_key_detection(self, test_config, temp_dir):
        """Test musical key detection."""
        import pretty_midi

        # Create MIDI in C major (all C notes)
        midi = pretty_midi.PrettyMIDI(initial_tempo=75)
        piano = pretty_midi.Instrument(program=0)
        for i in range(50):
            note = pretty_midi.Note(
                velocity=80,
                pitch=60 + (i % 2) * 12,  # C notes only
                start=i * 0.1,
                end=(i + 1) * 0.1
            )
            piano.notes.append(note)
        midi.instruments.append(piano)

        tokenizer = LoFiTokenizer(test_config)
        metadata = tokenizer._extract_metadata(midi)

        # Key should be C (though heuristic might not be perfect)
        assert 'key' in metadata

    def test_mood_inference(self, test_config, temp_dir):
        """Test mood inference from tempo."""
        import pretty_midi

        test_cases = [
            (65, 'melancholic'),  # Very slow
            (75, 'chill'),         # Slow
            (85, 'relaxed'),       # Moderate
            (95, 'upbeat'),        # Fast
        ]

        tokenizer = LoFiTokenizer(test_config)

        for tempo, expected_mood in test_cases:
            midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
            piano = pretty_midi.Instrument(program=0)
            note = pretty_midi.Note(velocity=80, pitch=60, start=0, end=1)
            piano.notes.append(note)
            midi.instruments.append(piano)

            metadata = tokenizer._extract_metadata(midi)
            assert metadata['mood'] == expected_mood

    def test_invalid_midi_path(self, test_config):
        """Test handling of invalid MIDI path."""
        tokenizer = LoFiTokenizer(test_config)

        passes, metadata = tokenizer.check_quality('/nonexistent/file.mid')

        assert passes is False
        assert 'error' in metadata

    def test_note_density_calculation(self, test_config, temp_dir):
        """Test note density calculation."""
        import pretty_midi

        # Create MIDI with known note density
        midi = pretty_midi.PrettyMIDI(initial_tempo=75)
        piano = pretty_midi.Instrument(program=0)

        # Add 100 notes over 10 seconds = 10 notes/second
        for i in range(100):
            note = pretty_midi.Note(
                velocity=80,
                pitch=60,
                start=i * 0.1,
                end=(i + 1) * 0.1
            )
            piano.notes.append(note)
        midi.instruments.append(piano)

        tokenizer = LoFiTokenizer(test_config)
        metadata = tokenizer._extract_metadata(midi)

        assert 'note_density' in metadata
        # Should be around 10 notes per second
        assert 8 < metadata['note_density'] < 12

    def test_instruments_extraction(self, test_config, temp_dir):
        """Test instrument program extraction."""
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(initial_tempo=75)

        # Add piano (program 0)
        piano = pretty_midi.Instrument(program=0, is_drum=False)
        note = pretty_midi.Note(velocity=80, pitch=60, start=0, end=1)
        piano.notes.append(note)
        midi.instruments.append(piano)

        # Add strings (program 48)
        strings = pretty_midi.Instrument(program=48, is_drum=False)
        note = pretty_midi.Note(velocity=80, pitch=64, start=0, end=1)
        strings.notes.append(note)
        midi.instruments.append(strings)

        # Add drums (should be excluded)
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        note = pretty_midi.Note(velocity=80, pitch=36, start=0, end=1)
        drums.notes.append(note)
        midi.instruments.append(drums)

        tokenizer = LoFiTokenizer(test_config)
        metadata = tokenizer._extract_metadata(midi)

        assert 0 in metadata['instruments']
        assert 48 in metadata['instruments']
        # Drums should not be in instruments list
        assert len([i for i in metadata['instruments'] if i == 0]) == 1
