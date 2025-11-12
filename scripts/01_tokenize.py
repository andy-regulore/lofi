#!/usr/bin/env python3
"""
Script 01: Tokenize MIDI files

This script:
1. Loads MIDI files from data/midi directory
2. Applies quality filtering for lo-fi music
3. Tokenizes using MidiTok REMI tokenizer
4. Saves tokenized sequences and metadata
"""

import sys
from pathlib import Path
import argparse
import yaml
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer import LoFiTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Tokenize MIDI files for lo-fi music generation')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--midi-dir',
        type=str,
        help='Directory containing MIDI files (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save tokenized data (overrides config)'
    )
    parser.add_argument(
        '--no-quality-check',
        action='store_true',
        help='Disable quality filtering'
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    midi_dir = args.midi_dir or config['data']['midi_dir']
    output_dir = args.output_dir or config['data']['tokens_dir']

    logger.info("=" * 60)
    logger.info("MIDI TOKENIZATION")
    logger.info("=" * 60)
    logger.info(f"MIDI directory: {midi_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Quality filtering: {not args.no_quality_check}")
    logger.info("")

    # Check if MIDI directory exists
    if not Path(midi_dir).exists():
        logger.error(f"MIDI directory does not exist: {midi_dir}")
        logger.info("\nPlease add MIDI files to the data/midi directory.")
        logger.info("You can download lo-fi MIDI files from:")
        logger.info("  - https://freemidi.org/")
        logger.info("  - https://www.midiworld.com/")
        logger.info("  - Your own collection of lo-fi MIDI files")
        sys.exit(1)

    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = LoFiTokenizer(config)
    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    logger.info("")

    # Quality filter settings
    if not args.no_quality_check:
        filters = config['data']['quality_filters']
        logger.info("Quality filters:")
        logger.info(f"  Tempo range: {filters['min_tempo']}-{filters['max_tempo']} BPM")
        logger.info(f"  Duration range: {filters['min_duration']}-{filters['max_duration']} seconds")
        logger.info(f"  Require drums: {filters['require_drums']}")
        logger.info(f"  Note density: {filters['min_note_density']}-{filters['max_note_density']} notes/sec")
        logger.info("")

    # Tokenize directory
    logger.info("Starting tokenization...")
    stats = tokenizer.tokenize_directory(
        midi_dir=midi_dir,
        output_dir=output_dir,
        check_quality=not args.no_quality_check
    )

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("TOKENIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Passed quality check: {stats['passed_quality']}")
    logger.info(f"Failed quality check: {stats['failed_quality']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Success rate: {stats['passed_quality']/stats['total_files']*100:.1f}%" if stats['total_files'] > 0 else "N/A")
    logger.info("")
    logger.info(f"Tokenized data saved to: {output_dir}")
    logger.info("")

    if stats['passed_quality'] == 0:
        logger.warning("No files passed quality check!")
        logger.info("\nTips:")
        logger.info("  - Use --no-quality-check to disable filtering")
        logger.info("  - Adjust quality filters in config.yaml")
        logger.info("  - Ensure MIDI files are in lo-fi tempo range (60-95 BPM)")
        sys.exit(1)

    logger.info("Next step: Run 02_build_dataset.py to create training dataset")


if __name__ == '__main__':
    main()
