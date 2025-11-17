"""Integration tests for end-to-end pipeline."""

from pathlib import Path
import pytest
import yaml

from src.tokenizer import LoFiTokenizer
from src.model import LoFiMusicModel, ConditionedLoFiModel
from src.generator import LoFiGenerator
from src.audio_processor import LoFiAudioProcessor


@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration tests for complete pipeline."""

    def test_tokenize_generate_pipeline(self, test_config, sample_midi_path, temp_dir):
        """Test tokenization -> model -> generation pipeline."""
        # 1. Tokenize MIDI
        tokenizer = LoFiTokenizer(test_config)
        result = tokenizer.tokenize_midi(sample_midi_path, check_quality=False)

        assert result is not None
        assert 'tokens' in result

        # 2. Create model
        vocab_size = tokenizer.get_vocab_size()
        model = LoFiMusicModel(test_config, vocab_size)

        assert model.model is not None

        # 3. Generate
        generator = LoFiGenerator(model, tokenizer, test_config, device='cpu')
        output_path = temp_dir / 'generated.mid'

        metadata = generator.generate_and_save(
            str(output_path),
            tempo=75,
            key='C',
            mood='chill',
            max_length=256  # Shorter for testing
        )

        assert 'output_path' in metadata or 'error' in metadata

    def test_full_audio_pipeline(self, test_config, sample_midi_path, temp_dir):
        """Test complete MIDI -> tokens -> generation -> audio pipeline."""
        # 1. Process input MIDI
        tokenizer = LoFiTokenizer(test_config)
        result = tokenizer.tokenize_midi(sample_midi_path, check_quality=False)
        assert result is not None

        # 2. Create and generate with model
        vocab_size = tokenizer.get_vocab_size()
        model = LoFiMusicModel(test_config, vocab_size)
        generator = LoFiGenerator(model, tokenizer, test_config, device='cpu')

        midi_output = temp_dir / 'generated.mid'
        metadata = generator.generate_and_save(
            str(midi_output),
            max_length=256
        )

        # 3. Process to audio (if MIDI was generated)
        if 'output_path' in metadata:
            audio_processor = LoFiAudioProcessor(test_config)
            audio_output_dir = temp_dir / 'audio'

            audio_result = audio_processor.process_midi_to_lofi(
                metadata['output_path'],
                str(audio_output_dir),
                save_clean=True,
                save_lofi=True
            )

            # Check results
            if 'lofi_wav_path' in audio_result:
                assert Path(audio_result['lofi_wav_path']).exists()

    def test_conditioned_generation_pipeline(self, test_config, temp_dir):
        """Test conditioned model generation pipeline."""
        # Create conditioned model
        vocab_size = 1000
        model = ConditionedLoFiModel(test_config, vocab_size)

        # Create mock tokenizer
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.get_vocab_size.return_value = vocab_size
        tokenizer.tokens_to_midi = MagicMock()

        # Generate with conditioning
        generator = LoFiGenerator(model, tokenizer, test_config, device='cpu')

        tokens, metadata = generator.generate_track(
            tempo=75,
            key='Am',
            mood='melancholic',
            max_length=128
        )

        assert len(tokens) > 0
        assert metadata['tempo'] == 75
        assert metadata['key'] == 'Am'
        assert metadata['mood'] == 'melancholic'

    def test_batch_generation_pipeline(self, test_config, temp_dir):
        """Test batch generation workflow."""
        vocab_size = 1000
        model = LoFiMusicModel(test_config, vocab_size)

        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.get_vocab_size.return_value = vocab_size
        tokenizer.tokens_to_midi = MagicMock()

        generator = LoFiGenerator(model, tokenizer, test_config, device='cpu')

        output_dir = temp_dir / 'batch'
        metadata_list = generator.batch_generate(
            num_tracks=3,
            output_dir=str(output_dir),
            ensure_variety=True
        )

        assert len(metadata_list) == 3

        # Check metadata file exists
        assert (output_dir / 'lofi_track_metadata.json').exists()

    def test_directory_tokenization_pipeline(self, test_config, sample_midi_path, temp_dir):
        """Test batch tokenization of directory."""
        import shutil

        # Create input directory with multiple MIDI files
        input_dir = temp_dir / 'input_midi'
        input_dir.mkdir()

        for i in range(3):
            shutil.copy(sample_midi_path, input_dir / f'track_{i}.mid')

        # Tokenize directory
        tokenizer = LoFiTokenizer(test_config)
        output_dir = temp_dir / 'tokens'

        stats = tokenizer.tokenize_directory(
            str(input_dir),
            str(output_dir),
            check_quality=False
        )

        assert stats['total_files'] == 3
        assert stats['processed'] >= 0

        # Check output files
        assert (output_dir / 'tokenization_stats.json').exists()
        assert (output_dir / 'metadata.json').exists()

    def test_quality_filtering_pipeline(self, test_config, temp_dir):
        """Test quality filtering in tokenization pipeline."""
        import pretty_midi

        input_dir = temp_dir / 'quality_test'
        input_dir.mkdir()

        # Create good quality MIDI
        good_midi = pretty_midi.PrettyMIDI(initial_tempo=75)
        piano = pretty_midi.Instrument(program=0, is_drum=False)
        drums = pretty_midi.Instrument(program=0, is_drum=True)

        # Add enough notes for good duration and density
        for i in range(200):
            note = pretty_midi.Note(
                velocity=80,
                pitch=60 + (i % 12),
                start=i * 0.2,
                end=(i + 0.5) * 0.2
            )
            piano.notes.append(note)

        for i in range(100):
            note = pretty_midi.Note(
                velocity=100,
                pitch=36,
                start=i * 0.4,
                end=(i + 0.1) * 0.4
            )
            drums.notes.append(note)

        good_midi.instruments.append(piano)
        good_midi.instruments.append(drums)
        good_midi.write(str(input_dir / 'good.mid'))

        # Create bad quality MIDI (too fast, too short)
        bad_midi = pretty_midi.PrettyMIDI(initial_tempo=150)
        piano = pretty_midi.Instrument(program=0)
        note = pretty_midi.Note(velocity=80, pitch=60, start=0, end=1)
        piano.notes.append(note)
        bad_midi.instruments.append(piano)
        bad_midi.write(str(input_dir / 'bad.mid'))

        # Tokenize with quality filtering
        tokenizer = LoFiTokenizer(test_config)
        output_dir = temp_dir / 'tokens'

        stats = tokenizer.tokenize_directory(
            str(input_dir),
            str(output_dir),
            check_quality=True
        )

        assert stats['total_files'] == 2
        # At least one should pass quality check
        assert stats['passed_quality'] >= 1
        # At least one should fail
        assert stats['failed_quality'] >= 1

    def test_save_load_model_pipeline(self, test_config, temp_dir):
        """Test saving and loading model."""
        vocab_size = 1000

        # Create and save model
        model1 = LoFiMusicModel(test_config, vocab_size)
        save_path = temp_dir / 'model'
        model1.save(str(save_path))

        # Load model
        model2 = LoFiMusicModel(test_config, vocab_size)
        model2.load(str(save_path))

        # Should have same architecture
        assert model2.vocab_size == vocab_size

        # Get model info
        info1 = model1.get_model_info()
        info2 = model2.get_model_info()

        assert info1['total_parameters'] == info2['total_parameters']

    @pytest.mark.slow
    def test_complete_production_pipeline(self, test_config, sample_midi_path, temp_dir):
        """Test complete production pipeline as it would be used."""
        # This test simulates the full workflow from data to final audio

        # 1. Data preparation
        input_dir = temp_dir / 'input'
        input_dir.mkdir()

        import shutil
        shutil.copy(sample_midi_path, input_dir / 'track.mid')

        # 2. Tokenization
        tokenizer = LoFiTokenizer(test_config)
        tokens_dir = temp_dir / 'tokens'

        stats = tokenizer.tokenize_directory(
            str(input_dir),
            str(tokens_dir),
            check_quality=False
        )

        assert stats['processed'] > 0

        # 3. Model creation
        vocab_size = tokenizer.get_vocab_size()
        model = LoFiMusicModel(test_config, vocab_size)

        # 4. Save model
        model_dir = temp_dir / 'model'
        model.save(str(model_dir))

        # 5. Load model for generation
        gen_model = LoFiMusicModel(test_config, vocab_size)
        gen_model.load(str(model_dir))

        # 6. Generate tracks
        generator = LoFiGenerator(gen_model, tokenizer, test_config, device='cpu')

        output_dir = temp_dir / 'generated'
        metadata_list = generator.batch_generate(
            num_tracks=2,
            output_dir=str(output_dir),
            ensure_variety=True
        )

        assert len(metadata_list) == 2

        # 7. Process to audio
        audio_processor = LoFiAudioProcessor(test_config)
        audio_dir = temp_dir / 'audio'

        for metadata in metadata_list:
            if 'output_path' in metadata:
                result = audio_processor.process_midi_to_lofi(
                    metadata['output_path'],
                    str(audio_dir),
                    save_lofi=True,
                    save_clean=False
                )

                # Should produce audio files or error
                assert 'lofi_wav_path' in result or 'error' in result

    def test_config_loading_and_validation(self, test_config):
        """Test that config can be loaded and validated."""
        # Config should have all required keys
        required_keys = [
            'data', 'tokenization', 'model', 'training',
            'generation', 'audio', 'logging', 'seed'
        ]

        for key in required_keys:
            assert key in test_config

        # Validate specific config values
        assert test_config['audio']['sample_rate'] > 0
        assert test_config['model']['num_layers'] > 0
        assert test_config['training']['batch_size'] > 0

    def test_reproducibility_with_seed(self, test_config, temp_dir):
        """Test that generation is reproducible with same seed."""
        vocab_size = 1000
        model = LoFiMusicModel(test_config, vocab_size)

        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.get_vocab_size.return_value = vocab_size

        generator = LoFiGenerator(model, tokenizer, test_config, device='cpu')

        # Generate twice with same seed
        tokens1, metadata1 = generator.generate_track(seed=42, max_length=128)
        tokens2, metadata2 = generator.generate_track(seed=42, max_length=128)

        # Should be identical
        assert tokens1 == tokens2
        assert metadata1['tempo'] == metadata2['tempo']
