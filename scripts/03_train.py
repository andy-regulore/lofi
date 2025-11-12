#!/usr/bin/env python3
"""
Script 03: Train lo-fi music generation model

This script:
1. Loads the prepared dataset
2. Initializes the GPT-2 model (117M parameters)
3. Trains with HuggingFace Trainer
4. Saves trained model and metrics
"""

import sys
from pathlib import Path
import argparse
import yaml
import logging

import torch
from datasets import load_from_disk

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ConditionedLoFiModel
from src.trainer import LoFiTrainer
from src.tokenizer import LoFiTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train lo-fi music generation model')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        help='Directory containing dataset (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save model (overrides config)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint'
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dataset_dir = args.dataset_dir or config['data']['dataset_dir']
    output_dir = args.output_dir or config['training']['output_dir']

    # Update config with output dir
    config['training']['output_dir'] = output_dir

    logger.info("=" * 60)
    logger.info("LO-FI MUSIC MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Resume from checkpoint: {args.resume}")
    logger.info("")

    # Check dataset exists
    if not Path(dataset_dir).exists():
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        logger.info("\nPlease run 02_build_dataset.py first.")
        sys.exit(1)

    # Check GPU availability
    if not torch.cuda.is_available() and config['training']['device'] == 'cuda':
        logger.warning("CUDA not available! Training will be very slow on CPU.")
        logger.info("Consider using Google Colab or a cloud GPU instance.")
        config['training']['device'] = 'cpu'
        config['training']['fp16'] = False

    device = config['training']['device']
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info("")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_from_disk(dataset_dir)
    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Eval samples: {len(dataset['eval'])}")
    logger.info("")

    # Initialize tokenizer to get vocab size
    logger.info("Initializing tokenizer...")
    tokenizer = LoFiTokenizer(config)
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info("")

    # Initialize model
    logger.info("Initializing model...")
    model = ConditionedLoFiModel(config, vocab_size)
    model_info = model.get_model_info()

    logger.info("Model architecture:")
    logger.info(f"  Total parameters: {model_info['total_parameters']:,} ({model_info['total_parameters']/1e6:.1f}M)")
    logger.info(f"  Vocabulary size: {model_info['vocab_size']:,}")
    logger.info(f"  Embedding dim: {model_info['embedding_dim']}")
    logger.info(f"  Layers: {model_info['num_layers']}")
    logger.info(f"  Attention heads: {model_info['num_heads']}")
    logger.info(f"  Context length: {model_info['context_length']}")
    logger.info("")

    # Training configuration
    logger.info("Training configuration:")
    logger.info(f"  Epochs: {config['training']['num_epochs']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch size: {config['training']['effective_batch_size']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Warmup steps: {config['training']['warmup_steps']}")
    logger.info(f"  FP16 training: {config['training']['fp16']}")
    logger.info(f"  Target eval loss: {config['training']['target_eval_loss']}")
    logger.info("")

    # Estimate training time
    num_train_samples = len(dataset['train'])
    steps_per_epoch = num_train_samples // config['training']['effective_batch_size']
    total_steps = steps_per_epoch * config['training']['num_epochs']

    logger.info("Training estimates:")
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    logger.info(f"  Total training steps: {total_steps}")
    if device == 'cuda':
        logger.info(f"  Estimated time: 8-12 hours on RTX 3090")
    logger.info("")

    # Prepare sequences
    train_sequences = dataset['train']['input_ids']
    eval_sequences = dataset['eval']['input_ids']

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = LoFiTrainer(model, config, vocab_size)

    # Train
    logger.info("Starting training...")
    logger.info("=" * 60)

    results = trainer.train(train_sequences, eval_sequences)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    train_metrics = results['train_metrics']
    eval_metrics = results['eval_metrics']

    logger.info("Final metrics:")
    logger.info(f"  Train loss: {train_metrics.get('train_loss', 'N/A')}")
    logger.info(f"  Eval loss: {eval_metrics.get('eval_loss', 'N/A')}")
    logger.info(f"  Train runtime: {train_metrics.get('train_runtime', 0)/3600:.2f} hours")
    logger.info("")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("")

    # Check if target achieved
    target_loss = config['training']['target_eval_loss']
    if eval_metrics.get('eval_loss', float('inf')) < target_loss:
        logger.info(f"✓ SUCCESS: Target eval loss ({target_loss}) achieved!")
    else:
        logger.warning(f"✗ Target eval loss ({target_loss}) not achieved")
        logger.info("  Consider training for more epochs or adjusting hyperparameters")

    logger.info("")
    logger.info("Next step: Run 04_generate.py to generate lo-fi tracks")


if __name__ == '__main__':
    main()
