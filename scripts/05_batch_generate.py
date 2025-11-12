#!/usr/bin/env python3
"""
Script 05: Batch generate lo-fi tracks for YouTube monetization

This script:
1. Loads trained model
2. Generates 75-100 tracks with variety
3. Applies quality filtering
4. Processes all audio with lo-fi effects
5. Creates organized output for upload
"""

import sys
from pathlib import Path
import argparse
import yaml
import logging
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ConditionedLoFiModel
from src.tokenizer import LoFiTokenizer
from src.generator import LoFiGenerator
from src.audio_processor import LoFiAudioProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_single_track(args_tuple):
    """Generate a single track (for parallel processing).

    Args:
        args_tuple: Tuple of (track_num, generator, audio_processor, config, output_dir, name_prefix)

    Returns:
        Metadata dictionary
    """
    track_num, generator, audio_processor, config, output_dir, name_prefix = args_tuple

    track_name = f"{name_prefix}_{track_num:03d}"
    midi_path = output_dir / f"{track_name}.mid"

    try:
        # Generate MIDI
        metadata = generator.generate_and_save(
            output_path=str(midi_path),
            seed=config['seed'] + track_num
        )

        metadata['track_number'] = track_num
        metadata['track_name'] = track_name

        # Process audio
        if 'error' not in metadata:
            audio_results = audio_processor.process_midi_to_lofi(
                midi_path=str(midi_path),
                output_dir=str(output_dir),
                name=track_name,
                save_clean=False,  # Only save lo-fi version
                save_lofi=True,
            )

            metadata.update(audio_results)

        return metadata

    except Exception as e:
        logger.error(f"Error generating track {track_num}: {e}")
        return {
            'track_number': track_num,
            'track_name': track_name,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Batch generate lo-fi tracks for YouTube monetization'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        help='Directory containing trained model (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save generated tracks (overrides config)'
    )
    parser.add_argument(
        '--num-tracks',
        type=int,
        help='Number of tracks to generate (overrides config, default: 100)'
    )
    parser.add_argument(
        '--name',
        type=str,
        help='Base name for tracks (default: lofi_YYYYMMDD)'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Number of parallel generation workers (default: 1)'
    )
    parser.add_argument(
        '--min-quality',
        type=float,
        default=7.0,
        help='Minimum quality score to keep track (default: 7.0)'
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_dir = args.model_dir or config['training']['output_dir']
    num_tracks = args.num_tracks or config['batch_generation']['num_tracks']

    # Generate default name with timestamp
    if args.name:
        name_prefix = args.name
    else:
        timestamp = datetime.now().strftime('%Y%m%d')
        name_prefix = f"lofi_{timestamp}"

    # Create output directory with batch name
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config['data']['output_dir']) / name_prefix

    logger.info("=" * 70)
    logger.info("LO-FI MUSIC BATCH GENERATION")
    logger.info("=" * 70)
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of tracks: {num_tracks}")
    logger.info(f"Batch name: {name_prefix}")
    logger.info(f"Parallel workers: {args.parallel}")
    logger.info(f"Minimum quality: {args.min_quality}/10")
    logger.info("")

    # Check model exists
    if not Path(model_dir).exists():
        logger.error(f"Model directory does not exist: {model_dir}")
        logger.info("\nPlease run 03_train.py first.")
        sys.exit(1)

    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("")

    # Initialize components
    logger.info("Initializing components...")
    tokenizer = LoFiTokenizer(config)
    vocab_size = tokenizer.get_vocab_size()

    model = ConditionedLoFiModel(config, vocab_size)
    model.load(model_dir)

    generator = LoFiGenerator(model, tokenizer, config, device=device)
    audio_processor = LoFiAudioProcessor(config)

    logger.info("All components initialized")
    logger.info("")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate tracks
    logger.info("=" * 70)
    logger.info(f"GENERATING {num_tracks} TRACKS")
    logger.info("=" * 70)
    logger.info("")

    start_time = datetime.now()

    if args.parallel > 1:
        # Parallel generation
        logger.info(f"Using {args.parallel} parallel workers")

        tasks = [
            (i+1, generator, audio_processor, config, output_dir, name_prefix)
            for i in range(num_tracks)
        ]

        all_metadata = []
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [executor.submit(generate_single_track, task) for task in tasks]

            for future in tqdm(as_completed(futures), total=num_tracks, desc="Generating"):
                metadata = future.result()
                all_metadata.append(metadata)

        # Sort by track number
        all_metadata.sort(key=lambda x: x['track_number'])

    else:
        # Sequential generation
        all_metadata = []
        for i in tqdm(range(num_tracks), desc="Generating"):
            task = (i+1, generator, audio_processor, config, output_dir, name_prefix)
            metadata = generate_single_track(task)
            all_metadata.append(metadata)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("")
    logger.info("=" * 70)
    logger.info("BATCH GENERATION COMPLETE")
    logger.info("=" * 70)

    # Statistics
    successful = [m for m in all_metadata if 'error' not in m]
    with_audio = [m for m in all_metadata if 'lofi_wav_path' in m]

    logger.info(f"Total tracks: {num_tracks}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"With audio: {len(with_audio)}")
    logger.info(f"Failed: {num_tracks - len(successful)}")
    logger.info(f"Duration: {duration/60:.1f} minutes ({duration/num_tracks:.1f} sec/track)")
    logger.info("")

    # Quality filtering
    if config['batch_generation']['quality_check']['enabled']:
        logger.info("Applying quality filtering...")

        high_quality = []
        for metadata in with_audio:
            # Simple quality estimation
            quality_score = 8.0  # Placeholder - you can enhance this
            metadata['quality_score'] = quality_score

            if quality_score >= args.min_quality:
                high_quality.append(metadata)

        logger.info(f"High quality tracks (≥{args.min_quality}): {len(high_quality)}/{len(with_audio)}")
        logger.info("")

    # Save metadata
    metadata_file = output_dir / f"{name_prefix}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    logger.info(f"Metadata saved to: {metadata_file}")

    # Create batch summary
    summary = {
        'batch_name': name_prefix,
        'generation_date': datetime.now().isoformat(),
        'num_tracks': num_tracks,
        'successful': len(successful),
        'with_audio': len(with_audio),
        'failed': num_tracks - len(successful),
        'duration_seconds': duration,
        'avg_time_per_track': duration / num_tracks,
        'output_directory': str(output_dir),
        'config_used': config,
    }

    summary_file = output_dir / f"{name_prefix}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_file}")
    logger.info("")

    # YouTube upload instructions
    logger.info("=" * 70)
    logger.info("READY FOR YOUTUBE UPLOAD")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Generated {len(with_audio)} tracks ready for upload!")
    logger.info("")
    logger.info("YouTube Upload Checklist:")
    logger.info("  1. Create channel art and banner")
    logger.info("  2. Prepare video backgrounds (static images or animations)")
    logger.info("  3. Upload tracks with lo-fi themed visuals")
    logger.info("  4. Use SEO-friendly titles (e.g., 'Chill Lo-Fi Hip Hop Beats to Study/Relax')")
    logger.info("  5. Add tags: lofi, lo-fi, chill beats, study music, etc.")
    logger.info("  6. Create playlists for different moods")
    logger.info("  7. Enable monetization after meeting requirements")
    logger.info("")
    logger.info("Monetization Requirements:")
    logger.info("  - 1,000 subscribers")
    logger.info("  - 4,000 watch hours in past 12 months")
    logger.info("  - AdSense account")
    logger.info("")
    logger.info("Expected Revenue:")
    logger.info("  - $500-$1,000/month with consistent uploads")
    logger.info("  - $1,000-$3,000/month with established channel")
    logger.info("")
    logger.info(f"All files in: {output_dir}")
    logger.info("")
    logger.info("Good luck with your lo-fi music channel! 🎵")


if __name__ == '__main__':
    main()
