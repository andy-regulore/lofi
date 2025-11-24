"""MIDI Tokenization module for lo-fi music generation.

This module handles:
- MIDI file tokenization using MidiTok REMI tokenizer
- Quality filtering for lo-fi music (tempo, drums, density)
- Metadata extraction (tempo, key, mood, instruments)
- Sequence chunking for model training
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi
from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoFiTokenizer:
    """Tokenizer for lo-fi MIDI files with quality filtering."""

    def __init__(self, config: Dict):
        """Initialize the tokenizer.

        Args:
            config: Configuration dictionary with tokenization and quality filter settings
        """
        self.config = config
        self.token_config = config["tokenization"]
        self.quality_filters = config["data"]["quality_filters"]

        # Initialize MidiTok REMI tokenizer
        self.tokenizer_config = TokenizerConfig(
            num_velocities=self.token_config["velocity_bins"],
            use_chords=True,
            use_rests=True,
            use_tempos=True,
            use_time_signatures=True,
            use_programs=True,
            nb_tempos=self.token_config["tempo_bins"],
            tempo_range=(50, 200),
        )

        self.tokenizer = REMI(self.tokenizer_config)
        logger.info("Initialized REMI tokenizer")

    def check_quality(self, midi_path: str) -> Tuple[bool, Dict]:
        """Check if a MIDI file meets lo-fi quality standards.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Tuple of (passes_quality_check, metadata_dict)
        """
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))

            # Extract metadata
            metadata = self._extract_metadata(midi)

            # Quality checks
            checks = {
                "tempo_ok": self.quality_filters["min_tempo"]
                <= metadata["tempo"]
                <= self.quality_filters["max_tempo"],
                "duration_ok": self.quality_filters["min_duration"]
                <= metadata["duration"]
                <= self.quality_filters["max_duration"],
                "has_drums": (
                    metadata["has_drums"] if self.quality_filters["require_drums"] else True
                ),
                "density_ok": self.quality_filters["min_note_density"]
                <= metadata["note_density"]
                <= self.quality_filters["max_note_density"],
            }

            passes = all(checks.values())

            metadata["quality_checks"] = checks
            metadata["passes_quality"] = passes

            return passes, metadata

        except Exception as e:
            logger.warning(f"Error checking quality for {midi_path}: {e}")
            return False, {"error": str(e)}

    def _extract_metadata(self, midi: pretty_midi.PrettyMIDI) -> Dict:
        """Extract metadata from MIDI file.

        Args:
            midi: PrettyMIDI object

        Returns:
            Dictionary of metadata
        """
        # Get tempo (use first tempo change or default)
        tempo_changes = midi.get_tempo_changes()
        if len(tempo_changes[1]) > 0:
            tempo = float(tempo_changes[1][0])
        else:
            tempo = 120.0

        # Duration
        duration = midi.get_end_time()

        # Check for drums
        has_drums = any(inst.is_drum for inst in midi.instruments)

        # Count total notes and calculate density
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        note_density = total_notes / duration if duration > 0 else 0

        # Detect key (simple heuristic based on pitch class distribution)
        pitch_counts = np.zeros(12)
        for inst in midi.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    pitch_counts[note.pitch % 12] += 1

        # Map pitch class to likely key
        key_map = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        likely_key = key_map[np.argmax(pitch_counts)] if pitch_counts.sum() > 0 else "C"

        # Detect instruments
        instruments = [inst.program for inst in midi.instruments if not inst.is_drum]

        # Infer mood from tempo and key (simple heuristic)
        if tempo < 70:
            mood = "melancholic"
        elif tempo < 80:
            mood = "chill"
        elif tempo < 90:
            mood = "relaxed"
        else:
            mood = "upbeat"

        return {
            "tempo": tempo,
            "duration": duration,
            "has_drums": has_drums,
            "total_notes": total_notes,
            "note_density": note_density,
            "key": likely_key,
            "mood": mood,
            "instruments": instruments,
            "num_tracks": len(midi.instruments),
        }

    def tokenize_midi(self, midi_path: str, check_quality: bool = True) -> Optional[Dict]:
        """Tokenize a MIDI file.

        Args:
            midi_path: Path to MIDI file
            check_quality: Whether to check quality before tokenizing

        Returns:
            Dictionary with tokens and metadata, or None if quality check fails
        """
        # Quality check
        if check_quality:
            passes, metadata = self.check_quality(midi_path)
            if not passes:
                logger.debug(f"Skipping {midi_path} - quality check failed")
                return None
        else:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            metadata = self._extract_metadata(midi)

        # Tokenize with MidiTok
        try:
            tokens = self.tokenizer(str(midi_path))

            # Convert to list of token IDs
            if hasattr(tokens, "ids"):
                token_ids = tokens.ids
            elif isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = tokens.tolist()

            return {
                "tokens": token_ids,
                "metadata": metadata,
                "file_path": str(midi_path),
            }

        except Exception as e:
            logger.error(f"Error tokenizing {midi_path}: {e}")
            return None

    def chunk_sequence(
        self, tokens: List[int], chunk_size: int = None, overlap: int = None
    ) -> List[List[int]]:
        """Chunk a token sequence into fixed-size segments.

        Args:
            tokens: List of token IDs
            chunk_size: Size of each chunk (default from config)
            overlap: Overlap between chunks (default from config)

        Returns:
            List of token chunks
        """
        chunk_size = chunk_size or self.token_config["chunk_size"]
        overlap = overlap or self.token_config["overlap"]

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + chunk_size
            chunk = tokens[start:end]

            # Only keep chunks that are at least half the target size
            if len(chunk) >= chunk_size // 2:
                # Pad if necessary
                if len(chunk) < chunk_size:
                    chunk = chunk + [0] * (chunk_size - len(chunk))
                chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    def tokenize_directory(
        self, midi_dir: str, output_dir: str, check_quality: bool = True
    ) -> Dict:
        """Tokenize all MIDI files in a directory.

        Args:
            midi_dir: Directory containing MIDI files
            output_dir: Directory to save tokenized data
            check_quality: Whether to apply quality filtering

        Returns:
            Statistics dictionary
        """
        midi_dir = Path(midi_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all MIDI files
        midi_files = list(midi_dir.glob("**/*.mid")) + list(midi_dir.glob("**/*.midi"))
        logger.info(f"Found {len(midi_files)} MIDI files")

        stats = {
            "total_files": len(midi_files),
            "processed": 0,
            "passed_quality": 0,
            "failed_quality": 0,
            "errors": 0,
        }

        all_metadata = []

        for midi_file in midi_files:
            result = self.tokenize_midi(str(midi_file), check_quality=check_quality)

            if result is None:
                stats["failed_quality"] += 1
                continue

            stats["processed"] += 1
            stats["passed_quality"] += 1

            # Save tokens
            output_file = output_dir / f"{midi_file.stem}.json"
            with open(output_file, "w") as f:
                json.dump(result, f)

            all_metadata.append(result["metadata"])

        # Save statistics and metadata
        stats_file = output_dir / "tokenization_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(all_metadata, f, indent=2)

        logger.info(
            f"Tokenization complete: {stats['passed_quality']}/{stats['total_files']} files passed quality check"
        )

        return stats

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        return len(self.tokenizer.vocab)

    def tokens_to_midi(self, tokens: List[int], output_path: str):
        """Convert tokens back to MIDI.

        Args:
            tokens: List of token IDs
            output_path: Path to save MIDI file
        """
        try:
            midi = self.tokenizer.tokens_to_midi([tokens])
            midi.dump(output_path)
            logger.info(f"Saved MIDI to {output_path}")
        except Exception as e:
            logger.error(f"Error converting tokens to MIDI: {e}")
