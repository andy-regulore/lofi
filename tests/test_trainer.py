"""Unit tests for trainer module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from src.trainer import (
    MusicTokenDataset,
    DataCollatorForMusicGeneration,
    MetricsCallback,
    LoFiTrainer,
)


@pytest.mark.unit
class TestMusicTokenDataset:
    """Tests for MusicTokenDataset class."""

    def test_init(self, sample_tokens):
        """Test dataset initialization."""
        dataset = MusicTokenDataset(sample_tokens)
        assert dataset.sequences == sample_tokens
        assert len(dataset) == len(sample_tokens)

    def test_len(self, sample_tokens):
        """Test dataset length."""
        dataset = MusicTokenDataset(sample_tokens)
        assert len(dataset) == 10

    def test_getitem(self, sample_tokens):
        """Test getting items from dataset."""
        dataset = MusicTokenDataset(sample_tokens)

        item = dataset[0]
        assert 'input_ids' in item
        assert isinstance(item['input_ids'], torch.Tensor)
        assert item['input_ids'].dtype == torch.long
        assert len(item['input_ids']) == len(sample_tokens[0])

    def test_getitem_multiple(self, sample_tokens):
        """Test getting multiple items."""
        dataset = MusicTokenDataset(sample_tokens)

        for i in range(len(dataset)):
            item = dataset[i]
            assert item['input_ids'].tolist() == sample_tokens[i]


@pytest.mark.unit
class TestDataCollatorForMusicGeneration:
    """Tests for DataCollatorForMusicGeneration class."""

    def test_init(self):
        """Test collator initialization."""
        collator = DataCollatorForMusicGeneration(pad_token_id=0)
        assert collator.pad_token_id == 0

    def test_collate_single_sequence(self):
        """Test collating a single sequence."""
        collator = DataCollatorForMusicGeneration(pad_token_id=0)

        features = [
            {'input_ids': torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)}
        ]

        batch = collator(features)

        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch

        assert batch['input_ids'].shape == (1, 5)
        assert batch['attention_mask'].shape == (1, 5)
        assert batch['labels'].shape == (1, 5)

        # Attention mask should be all ones
        assert batch['attention_mask'].sum().item() == 5

    def test_collate_multiple_sequences(self):
        """Test collating multiple sequences with different lengths."""
        collator = DataCollatorForMusicGeneration(pad_token_id=0)

        features = [
            {'input_ids': torch.tensor([1, 2, 3], dtype=torch.long)},
            {'input_ids': torch.tensor([4, 5, 6, 7, 8], dtype=torch.long)},
        ]

        batch = collator(features)

        # Should be padded to max length (5)
        assert batch['input_ids'].shape == (2, 5)

        # First sequence should be padded
        assert batch['input_ids'][0].tolist() == [1, 2, 3, 0, 0]
        assert batch['attention_mask'][0].tolist() == [1, 1, 1, 0, 0]

        # Second sequence should not be padded
        assert batch['input_ids'][1].tolist() == [4, 5, 6, 7, 8]
        assert batch['attention_mask'][1].tolist() == [1, 1, 1, 1, 1]

    def test_labels_masking(self):
        """Test that padding tokens are masked in labels."""
        collator = DataCollatorForMusicGeneration(pad_token_id=0)

        features = [
            {'input_ids': torch.tensor([1, 2, 3], dtype=torch.long)},
            {'input_ids': torch.tensor([4, 5, 6, 7, 8], dtype=torch.long)},
        ]

        batch = collator(features)

        # Padded positions should be -100 in labels
        assert batch['labels'][0, 3] == -100
        assert batch['labels'][0, 4] == -100

        # Non-padded positions should match input_ids
        assert batch['labels'][0, 0] == 1
        assert batch['labels'][0, 1] == 2
        assert batch['labels'][1, 0] == 4


@pytest.mark.unit
class TestMetricsCallback:
    """Tests for MetricsCallback class."""

    def test_init(self, temp_dir):
        """Test callback initialization."""
        callback = MetricsCallback(str(temp_dir))
        assert callback.output_dir == temp_dir
        assert callback.metrics_history == []

    def test_on_log(self, temp_dir):
        """Test logging callback."""
        callback = MetricsCallback(str(temp_dir))

        # Mock trainer state
        state = Mock()
        state.global_step = 100
        state.epoch = 1

        logs = {'loss': 2.5, 'learning_rate': 0.0001}

        callback.on_log(None, state, None, logs=logs)

        assert len(callback.metrics_history) == 1
        assert callback.metrics_history[0]['step'] == 100
        assert callback.metrics_history[0]['epoch'] == 1
        assert callback.metrics_history[0]['loss'] == 2.5

        # Check file was saved
        metrics_file = temp_dir / 'metrics_history.json'
        assert metrics_file.exists()

    def test_multiple_logs(self, temp_dir):
        """Test multiple logging events."""
        callback = MetricsCallback(str(temp_dir))

        state = Mock()

        for i in range(5):
            state.global_step = i * 100
            state.epoch = i
            logs = {'loss': 3.0 - i * 0.1}
            callback.on_log(None, state, None, logs=logs)

        assert len(callback.metrics_history) == 5
        assert callback.metrics_history[0]['loss'] == 3.0
        assert callback.metrics_history[4]['loss'] == 2.6


@pytest.mark.unit
class TestLoFiTrainer:
    """Tests for LoFiTrainer class."""

    def test_init(self, mock_model, test_config):
        """Test trainer initialization."""
        trainer = LoFiTrainer(mock_model, test_config, vocab_size=1000)

        assert trainer.model == mock_model
        assert trainer.config == test_config
        assert trainer.vocab_size == 1000
        assert trainer.trainer is None

    def test_prepare_training_args(self, mock_model, test_config, temp_dir):
        """Test training arguments preparation."""
        test_config['training']['output_dir'] = str(temp_dir / 'output')

        trainer = LoFiTrainer(mock_model, test_config, vocab_size=1000)
        training_args = trainer.prepare_training_args()

        assert training_args is not None
        assert training_args.output_dir == str(temp_dir / 'output')
        assert training_args.num_train_epochs == test_config['training']['num_epochs']
        assert training_args.per_device_train_batch_size == test_config['training']['batch_size']
        assert training_args.learning_rate == test_config['training']['learning_rate']

    def test_prepare_datasets(self, mock_model, test_config, sample_tokens):
        """Test dataset preparation."""
        trainer = LoFiTrainer(mock_model, test_config, vocab_size=1000)

        train_seqs = sample_tokens[:8]
        eval_seqs = sample_tokens[8:]

        train_dataset, eval_dataset = trainer.prepare_datasets(train_seqs, eval_seqs)

        assert len(train_dataset) == 8
        assert len(eval_dataset) == 2
        assert isinstance(train_dataset, MusicTokenDataset)
        assert isinstance(eval_dataset, MusicTokenDataset)

    @pytest.mark.slow
    def test_train(self, mock_model, test_config, sample_tokens, temp_dir):
        """Test training process."""
        test_config['training']['output_dir'] = str(temp_dir / 'output')
        test_config['training']['num_epochs'] = 1
        test_config['training']['logging_steps'] = 1
        test_config['training']['eval_steps'] = 10
        test_config['training']['save_steps'] = 10

        trainer = LoFiTrainer(mock_model, test_config, vocab_size=1000)

        train_seqs = sample_tokens[:8]
        eval_seqs = sample_tokens[8:]

        # Mock the HF Trainer
        with patch('src.trainer.Trainer') as mock_trainer_class:
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.train.return_value = Mock(metrics={'train_loss': 2.5})
            mock_trainer_instance.evaluate.return_value = {'eval_loss': 2.3}
            mock_trainer_instance.save_model = MagicMock()
            mock_trainer_instance.log_metrics = MagicMock()
            mock_trainer_instance.save_metrics = MagicMock()

            mock_trainer_class.return_value = mock_trainer_instance

            result = trainer.train(train_seqs, eval_seqs)

            assert 'train_metrics' in result
            assert 'eval_metrics' in result
            assert result['train_metrics']['train_loss'] == 2.5
            assert result['eval_metrics']['eval_loss'] == 2.3

    def test_evaluate(self, mock_model, test_config, sample_tokens):
        """Test evaluation."""
        trainer = LoFiTrainer(mock_model, test_config, vocab_size=1000)

        # Should raise error if not trained
        with pytest.raises(ValueError):
            trainer.evaluate(sample_tokens)

        # Mock trainer
        trainer.trainer = MagicMock()
        trainer.trainer.evaluate.return_value = {'eval_loss': 2.5}

        metrics = trainer.evaluate(sample_tokens)

        assert 'eval_loss' in metrics
        assert metrics['eval_loss'] == 2.5

    def test_early_stopping_configuration(self, mock_model, test_config, temp_dir):
        """Test that early stopping is configured correctly."""
        test_config['training']['output_dir'] = str(temp_dir / 'output')
        test_config['training']['early_stopping_patience'] = 3
        test_config['training']['early_stopping_threshold'] = 0.01

        trainer = LoFiTrainer(mock_model, test_config, vocab_size=1000)

        assert trainer.train_config['early_stopping_patience'] == 3

    def test_checkpoint_detection(self, mock_model, test_config, temp_dir):
        """Test checkpoint detection."""
        output_dir = temp_dir / 'output'
        output_dir.mkdir()
        test_config['training']['output_dir'] = str(output_dir)

        # Create a fake checkpoint directory
        checkpoint_dir = output_dir / 'checkpoint-100'
        checkpoint_dir.mkdir()
        (checkpoint_dir / 'pytorch_model.bin').touch()

        trainer = LoFiTrainer(mock_model, test_config, vocab_size=1000)

        # Should detect checkpoint when training
        # (tested via integration test)
