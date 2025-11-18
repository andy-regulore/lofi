"""Training module for lo-fi music generation model.

Handles model training with HuggingFace Trainer, including:
- Custom data collation
- Training metrics and logging
- Early stopping
- Checkpoint management
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import json

import torch
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MusicTokenDataset(Dataset):
    """Dataset for music token sequences."""

    def __init__(self, token_sequences):
        """Initialize dataset.

        Args:
            token_sequences: List of token ID lists
        """
        self.sequences = token_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long)
        }


class DataCollatorForMusicGeneration:
    """Data collator for causal language modeling with music tokens."""

    def __init__(self, pad_token_id: int = 0):
        """Initialize data collator.

        Args:
            pad_token_id: Token ID to use for padding
        """
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        """Collate batch of features.

        Args:
            features: List of dicts with 'input_ids' key

        Returns:
            Batch dictionary with input_ids, attention_mask, labels
        """
        # Get max length in batch
        max_length = max(len(f['input_ids']) for f in features)

        # Prepare batch tensors
        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)

        # Fill in sequences
        for i, feature in enumerate(features):
            seq_len = len(feature['input_ids'])
            input_ids[i, :seq_len] = feature['input_ids']
            attention_mask[i, :seq_len] = 1

        # For causal LM, labels are the same as input_ids
        # but we mask padding tokens with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class MetricsCallback(TrainerCallback):
    """Callback to track and save training metrics."""

    def __init__(self, output_dir: str):
        """Initialize metrics callback.

        Args:
            output_dir: Directory to save metrics
        """
        self.output_dir = Path(output_dir)
        self.metrics_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs."""
        if logs:
            self.metrics_history.append({
                'step': state.global_step,
                'epoch': state.epoch,
                **logs
            })

            # Save metrics
            metrics_file = self.output_dir / 'metrics_history.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)


class LoFiTrainer:
    """Trainer for lo-fi music generation model."""

    def __init__(self, model, config: Dict, vocab_size: int):
        """Initialize trainer.

        Args:
            model: LoFiMusicModel instance
            config: Configuration dictionary
            vocab_size: Size of vocabulary
        """
        self.model = model
        self.config = config
        self.train_config = config['training']
        self.vocab_size = vocab_size

        self.trainer = None
        self.training_args = None

    def prepare_training_args(self) -> TrainingArguments:
        """Prepare HuggingFace training arguments.

        Returns:
            TrainingArguments instance
        """
        output_dir = self.train_config['output_dir']

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.train_config['num_epochs'],
            per_device_train_batch_size=self.train_config['batch_size'],
            per_device_eval_batch_size=self.train_config['batch_size'],
            gradient_accumulation_steps=self.train_config['gradient_accumulation_steps'],
            learning_rate=self.train_config['learning_rate'],
            warmup_steps=self.train_config['warmup_steps'],
            weight_decay=self.train_config['weight_decay'],
            max_grad_norm=self.train_config['max_grad_norm'],
            logging_dir=f"{output_dir}/logs",
            logging_steps=self.train_config['logging_steps'],
            eval_strategy="steps",
            eval_steps=self.train_config['eval_steps'],
            save_strategy="steps",
            save_steps=self.train_config['save_steps'],
            save_total_limit=self.train_config['save_total_limit'],
            fp16=self.train_config['fp16'],
            dataloader_num_workers=self.train_config['dataloader_num_workers'],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="tensorboard" if self.config['logging']['tensorboard'] else "none",
            seed=self.config['seed'],
        )

        logger.info("Training arguments prepared")
        logger.info(f"  Effective batch size: {self.train_config['effective_batch_size']}")
        logger.info(f"  Total epochs: {self.train_config['num_epochs']}")
        logger.info(f"  Learning rate: {self.train_config['learning_rate']}")

        return self.training_args

    def prepare_datasets(self, train_sequences, eval_sequences):
        """Prepare training and evaluation datasets.

        Args:
            train_sequences: List of training token sequences
            eval_sequences: List of evaluation token sequences

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        train_dataset = MusicTokenDataset(train_sequences)
        eval_dataset = MusicTokenDataset(eval_sequences)

        logger.info(f"Training dataset: {len(train_dataset)} sequences")
        logger.info(f"Evaluation dataset: {len(eval_dataset)} sequences")

        return train_dataset, eval_dataset

    def train(self, train_sequences, eval_sequences, custom_callbacks=None):
        """Train the model.

        Args:
            train_sequences: List of training token sequences
            eval_sequences: List of evaluation token sequences
            custom_callbacks: Optional list of additional TrainerCallback instances

        Returns:
            Training metrics
        """
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets(
            train_sequences, eval_sequences
        )

        # Prepare training arguments
        training_args = self.prepare_training_args()

        # Data collator
        data_collator = DataCollatorForMusicGeneration(pad_token_id=0)

        # Callbacks
        callbacks = [
            MetricsCallback(training_args.output_dir),
        ]

        # Add custom callbacks if provided
        if custom_callbacks:
            callbacks.extend(custom_callbacks)

        # Add early stopping if configured
        if self.train_config.get('early_stopping_patience'):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.train_config['early_stopping_patience'],
                    early_stopping_threshold=self.train_config['early_stopping_threshold'],
                )
            )
            logger.info(f"Early stopping enabled with patience={self.train_config['early_stopping_patience']}")

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model.get_model(),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Check for existing checkpoint
        last_checkpoint = None
        if Path(training_args.output_dir).exists():
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint:
                logger.info(f"Found checkpoint: {last_checkpoint}")

        # Train
        logger.info("Starting training...")
        train_result = self.trainer.train(resume_from_checkpoint=last_checkpoint)

        # Save final model
        self.trainer.save_model()
        self.model.save(training_args.output_dir)

        # Save metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        # Final evaluation
        logger.info("Running final evaluation...")
        eval_metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", eval_metrics)
        self.trainer.save_metrics("eval", eval_metrics)

        logger.info("Training complete!")
        logger.info(f"  Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
        logger.info(f"  Final eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")

        # Check if target loss achieved
        target_loss = self.train_config.get('target_eval_loss', 2.5)
        if eval_metrics.get('eval_loss', float('inf')) < target_loss:
            logger.info(f"✓ Target evaluation loss ({target_loss}) achieved!")
        else:
            logger.warning(f"✗ Target evaluation loss ({target_loss}) not achieved")

        return {
            'train_metrics': metrics,
            'eval_metrics': eval_metrics,
        }

    def evaluate(self, eval_sequences):
        """Evaluate the model.

        Args:
            eval_sequences: List of evaluation token sequences

        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")

        eval_dataset = MusicTokenDataset(eval_sequences)
        metrics = self.trainer.evaluate(eval_dataset)

        logger.info("Evaluation results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        return metrics
