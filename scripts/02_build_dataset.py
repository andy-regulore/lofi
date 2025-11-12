#!/usr/bin/env python3
"""
Script 02: Build training dataset

This script:
1. Loads tokenized sequences
2. Chunks sequences into training samples
3. Creates HuggingFace dataset
4. Splits into train/eval sets
5. Saves dataset to disk
"""

import sys
from pathlib import Path
import argparse
import yaml
import logging
import json
from typing import List

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer import LoFiTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_tokenized_sequences(tokens_dir: str) -> List[dict]:
    """Load all tokenized sequences from directory.

    Args:
        tokens_dir: Directory containing tokenized JSON files

    Returns:
        List of tokenized sequence dictionaries
    """
    tokens_dir = Path(tokens_dir)
    token_files = list(tokens_dir.glob('*.json'))

    # Filter out metadata files
    token_files = [f for f in token_files if f.name not in ['tokenization_stats.json', 'metadata.json']]

    logger.info(f"Found {len(token_files)} tokenized files")

    sequences = []
    for token_file in token_files:
        with open(token_file, 'r') as f:
            data = json.load(f)
            sequences.append(data)

    return sequences


def chunk_all_sequences(sequences: List[dict], tokenizer: LoFiTokenizer) -> List[List[int]]:
    """Chunk all sequences into training samples.

    Args:
        sequences: List of tokenized sequence dictionaries
        tokenizer: LoFiTokenizer instance

    Returns:
        List of chunked token sequences
    """
    all_chunks = []

    for seq_data in sequences:
        tokens = seq_data['tokens']
        chunks = tokenizer.chunk_sequence(tokens)
        all_chunks.extend(chunks)

    return all_chunks


def main():
    parser = argparse.ArgumentParser(description='Build training dataset from tokenized sequences')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--tokens-dir',
        type=str,
        help='Directory containing tokenized data (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save dataset (overrides config)'
    )
    parser.add_argument(
        '--eval-split',
        type=float,
        default=0.1,
        help='Fraction of data to use for evaluation'
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    tokens_dir = args.tokens_dir or config['data']['tokens_dir']
    output_dir = args.output_dir or config['data']['dataset_dir']

    logger.info("=" * 60)
    logger.info("DATASET BUILDING")
    logger.info("=" * 60)
    logger.info(f"Tokens directory: {tokens_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Evaluation split: {args.eval_split * 100:.1f}%")
    logger.info("")

    # Check if tokens directory exists
    if not Path(tokens_dir).exists():
        logger.error(f"Tokens directory does not exist: {tokens_dir}")
        logger.info("\nPlease run 01_tokenize.py first.")
        sys.exit(1)

    # Initialize tokenizer (needed for chunking)
    logger.info("Initializing tokenizer...")
    tokenizer = LoFiTokenizer(config)

    # Load tokenized sequences
    logger.info("Loading tokenized sequences...")
    sequences = load_tokenized_sequences(tokens_dir)
    logger.info(f"Loaded {len(sequences)} sequences")

    if len(sequences) == 0:
        logger.error("No tokenized sequences found!")
        sys.exit(1)

    # Chunk sequences
    logger.info("Chunking sequences...")
    chunks = chunk_all_sequences(sequences, tokenizer)
    logger.info(f"Created {len(chunks)} training samples")

    # Split into train/eval
    logger.info(f"Splitting into train/eval ({1-args.eval_split:.1%}/{args.eval_split:.1%})...")
    train_chunks, eval_chunks = train_test_split(
        chunks,
        test_size=args.eval_split,
        random_state=config['seed']
    )

    logger.info(f"Train samples: {len(train_chunks)}")
    logger.info(f"Eval samples: {len(eval_chunks)}")

    # Create HuggingFace datasets
    logger.info("Creating HuggingFace datasets...")

    train_dataset = Dataset.from_dict({
        'input_ids': train_chunks
    })

    eval_dataset = Dataset.from_dict({
        'input_ids': eval_chunks
    })

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'eval': eval_dataset
    })

    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving dataset to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))

    # Save dataset info
    info = {
        'num_sequences': len(sequences),
        'num_chunks': len(chunks),
        'num_train': len(train_chunks),
        'num_eval': len(eval_chunks),
        'chunk_size': config['tokenization']['chunk_size'],
        'vocab_size': tokenizer.get_vocab_size(),
    }

    info_path = output_path / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total sequences: {len(sequences)}")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Train samples: {len(train_chunks)}")
    logger.info(f"Eval samples: {len(eval_chunks)}")
    logger.info(f"Dataset saved to: {output_path}")
    logger.info("")
    logger.info("Next step: Run 03_train.py to train the model")


if __name__ == '__main__':
    main()
