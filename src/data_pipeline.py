"""Comprehensive data pipeline for music generation.

Features:
- MIDI data augmentation (transposition, tempo, time stretch)
- Automatic data cleaning and validation
- Genre classification
- Chord extraction and labeling
- Dataset statistics and analysis
- Data versioning support
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import hashlib

import numpy as np
from miditoolkit import MidiFile
import pretty_midi

logger = logging.getLogger(__name__)


class MIDIAugmentor:
    """Augment MIDI data for training."""

    def __init__(self, seed: int = 42):
        """Initialize augmentor.

        Args:
            seed: Random seed
        """
        self.rng = np.random.RandomState(seed)

    def transpose(
        self,
        midi: MidiFile,
        semitones: int,
    ) -> MidiFile:
        """Transpose MIDI by semitones.

        Args:
            midi: Input MIDI
            semitones: Number of semitones to transpose

        Returns:
            Transposed MIDI
        """
        # Create copy
        transposed = MidiFile()
        transposed.ticks_per_beat = midi.ticks_per_beat
        transposed.tempo_changes = midi.tempo_changes.copy()
        transposed.time_signature_changes = midi.time_signature_changes.copy()

        # Transpose instruments
        for inst in midi.instruments:
            new_inst = pretty_midi.Instrument(
                program=inst.program,
                is_drum=inst.is_drum,
                name=inst.name
            )

            for note in inst.notes:
                if not inst.is_drum:
                    new_pitch = note.pitch + semitones
                    # Keep in valid MIDI range
                    if 0 <= new_pitch <= 127:
                        new_note = pretty_midi.Note(
                            velocity=note.velocity,
                            pitch=new_pitch,
                            start=note.start,
                            end=note.end,
                        )
                        new_inst.notes.append(new_note)
                else:
                    # Don't transpose drums
                    new_inst.notes.append(note)

            transposed.instruments.append(new_inst)

        return transposed

    def change_tempo(
        self,
        midi: MidiFile,
        tempo_factor: float,
    ) -> MidiFile:
        """Change tempo by factor.

        Args:
            midi: Input MIDI
            tempo_factor: Tempo multiplier (e.g., 1.1 = 10% faster)

        Returns:
            MIDI with changed tempo
        """
        adjusted = MidiFile()
        adjusted.ticks_per_beat = midi.ticks_per_beat

        # Adjust tempo changes
        for tempo_change in midi.tempo_changes:
            new_tempo = pretty_midi.TempoChange(
                tempo=tempo_change.tempo * tempo_factor,
                time=tempo_change.time / tempo_factor,
            )
            adjusted.tempo_changes.append(new_tempo)

        # Adjust note timings
        for inst in midi.instruments:
            new_inst = pretty_midi.Instrument(
                program=inst.program,
                is_drum=inst.is_drum,
                name=inst.name
            )

            for note in inst.notes:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start / tempo_factor,
                    end=note.end / tempo_factor,
                )
                new_inst.notes.append(new_note)

            adjusted.instruments.append(new_inst)

        return adjusted

    def time_stretch(
        self,
        midi: MidiFile,
        stretch_factor: float,
    ) -> MidiFile:
        """Time stretch without changing pitch.

        Args:
            midi: Input MIDI
            stretch_factor: Stretch factor (>1 = slower, <1 = faster)

        Returns:
            Time-stretched MIDI
        """
        stretched = MidiFile()
        stretched.ticks_per_beat = midi.ticks_per_beat
        stretched.tempo_changes = midi.tempo_changes.copy()
        stretched.time_signature_changes = midi.time_signature_changes.copy()

        # Stretch note timings
        for inst in midi.instruments:
            new_inst = pretty_midi.Instrument(
                program=inst.program,
                is_drum=inst.is_drum,
                name=inst.name
            )

            for note in inst.notes:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start * stretch_factor,
                    end=note.end * stretch_factor,
                )
                new_inst.notes.append(new_note)

            stretched.instruments.append(new_inst)

        return stretched

    def add_velocity_variation(
        self,
        midi: MidiFile,
        variation: float = 0.1,
    ) -> MidiFile:
        """Add random velocity variation for humanization.

        Args:
            midi: Input MIDI
            variation: Variation amount (0-1)

        Returns:
            MIDI with varied velocities
        """
        humanized = MidiFile()
        humanized.ticks_per_beat = midi.ticks_per_beat
        humanized.tempo_changes = midi.tempo_changes.copy()
        humanized.time_signature_changes = midi.time_signature_changes.copy()

        for inst in midi.instruments:
            new_inst = pretty_midi.Instrument(
                program=inst.program,
                is_drum=inst.is_drum,
                name=inst.name
            )

            for note in inst.notes:
                # Add random variation
                velocity_change = self.rng.uniform(-variation, variation) * note.velocity
                new_velocity = int(np.clip(note.velocity + velocity_change, 1, 127))

                new_note = pretty_midi.Note(
                    velocity=new_velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                )
                new_inst.notes.append(new_note)

            humanized.instruments.append(new_inst)

        return humanized

    def augment(
        self,
        midi_path: str,
        output_dir: str,
        num_augmentations: int = 5,
        transpose_range: Tuple[int, int] = (-3, 3),
        tempo_range: Tuple[float, float] = (0.9, 1.1),
    ) -> List[str]:
        """Create augmented versions of MIDI.

        Args:
            midi_path: Input MIDI path
            output_dir: Output directory
            num_augmentations: Number of augmented versions
            transpose_range: Range of transposition (min, max semitones)
            tempo_range: Range of tempo change (min, max factor)

        Returns:
            List of augmented file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        midi = MidiFile(str(midi_path))
        base_name = Path(midi_path).stem

        augmented_paths = []

        for i in range(num_augmentations):
            # Random augmentations
            semitones = self.rng.randint(transpose_range[0], transpose_range[1] + 1)
            tempo_factor = self.rng.uniform(tempo_range[0], tempo_range[1])

            # Apply augmentations
            aug_midi = self.transpose(midi, semitones)
            aug_midi = self.change_tempo(aug_midi, tempo_factor)
            aug_midi = self.add_velocity_variation(aug_midi, variation=0.1)

            # Save
            output_path = output_dir / f"{base_name}_aug{i}_t{semitones}_tempo{tempo_factor:.2f}.mid"
            aug_midi.dump(str(output_path))
            augmented_paths.append(str(output_path))

        logger.info(f"Created {num_augmentations} augmentations for {midi_path}")

        return augmented_paths


class DataCleaner:
    """Clean and validate MIDI data."""

    def __init__(self):
        """Initialize data cleaner."""
        pass

    def validate_midi(self, midi_path: str) -> Tuple[bool, List[str]]:
        """Validate MIDI file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []

        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))

            # Check duration
            duration = midi.get_end_time()
            if duration < 10:
                issues.append("Too short (< 10 seconds)")
            if duration > 600:
                issues.append("Too long (> 10 minutes)")

            # Check for notes
            total_notes = sum(len(inst.notes) for inst in midi.instruments)
            if total_notes == 0:
                issues.append("No notes found")

            # Check for tempo
            tempo_changes = midi.get_tempo_changes()
            if len(tempo_changes[1]) == 0:
                issues.append("No tempo information")

            # Check for extreme note densities
            note_density = total_notes / duration
            if note_density > 50:
                issues.append(f"Very high note density: {note_density:.1f} notes/sec")

            # Check for stuck notes (very long notes)
            for inst in midi.instruments:
                for note in inst.notes:
                    note_duration = note.end - note.start
                    if note_duration > 30:  # 30 seconds
                        issues.append(f"Very long note detected: {note_duration:.1f}s")
                        break

        except Exception as e:
            issues.append(f"Error reading MIDI: {str(e)}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def fix_common_issues(
        self,
        midi_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Automatically fix common MIDI issues.

        Args:
            midi_path: Input MIDI path
            output_path: Output path (overwrites input if None)

        Returns:
            Tuple of (success, message)
        """
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            fixed = False

            # Remove empty instruments
            midi.instruments = [inst for inst in midi.instruments if len(inst.notes) > 0]
            if len(midi.instruments) == 0:
                return False, "No valid instruments after cleanup"

            # Fix stuck notes
            for inst in midi.instruments:
                for note in inst.notes:
                    if note.end - note.start > 10:  # Max 10 seconds
                        note.end = note.start + 10
                        fixed = True

            # Remove overlapping notes (same pitch)
            for inst in midi.instruments:
                sorted_notes = sorted(inst.notes, key=lambda n: n.start)
                cleaned_notes = []
                pitch_dict = {}

                for note in sorted_notes:
                    if note.pitch in pitch_dict:
                        prev_note = pitch_dict[note.pitch]
                        if note.start < prev_note.end:
                            # Overlapping - truncate previous
                            prev_note.end = note.start
                            fixed = True

                    cleaned_notes.append(note)
                    pitch_dict[note.pitch] = note

                inst.notes = cleaned_notes

            # Save
            if output_path is None:
                output_path = midi_path

            midi.write(str(output_path))

            message = "Fixed" if fixed else "No issues found"
            return True, message

        except Exception as e:
            return False, f"Error: {str(e)}"


class GenreClassifier:
    """Classify MIDI genre based on characteristics."""

    GENRE_SIGNATURES = {
        'lofi': {
            'tempo_range': (60, 95),
            'complexity_range': (0.3, 0.7),
            'has_drums': True,
            'instruments': [0, 1, 4, 24, 25, 32],  # Piano, keys, guitars, bass
        },
        'jazz': {
            'tempo_range': (100, 200),
            'complexity_range': (0.5, 0.9),
            'has_drums': True,
            'instruments': [0, 11, 64, 65, 66, 67, 68],  # Piano, sax, brass
        },
        'classical': {
            'tempo_range': (60, 140),
            'complexity_range': (0.4, 0.8),
            'has_drums': False,
            'instruments': [0, 40, 41, 42, 43, 44, 45],  # Strings, piano
        },
        'electronic': {
            'tempo_range': (110, 140),
            'complexity_range': (0.4, 0.8),
            'has_drums': True,
            'instruments': [80, 81, 82, 83, 84, 85],  # Synth sounds
        },
    }

    def __init__(self):
        """Initialize genre classifier."""
        pass

    def classify(self, midi_path: str) -> Dict[str, float]:
        """Classify MIDI genre.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Dictionary of genre probabilities
        """
        midi = pretty_midi.PrettyMIDI(str(midi_path))

        # Extract features
        tempo = midi.estimate_tempo() if hasattr(midi, 'estimate_tempo') else 120
        has_drums = any(inst.is_drum for inst in midi.instruments)
        instruments = set(inst.program for inst in midi.instruments if not inst.is_drum)

        # Calculate note density (complexity)
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        duration = midi.get_end_time()
        complexity = total_notes / duration / 10.0  # Normalize

        # Score each genre
        scores = {}

        for genre, signature in self.GENRE_SIGNATURES.items():
            score = 0.0

            # Tempo match
            if signature['tempo_range'][0] <= tempo <= signature['tempo_range'][1]:
                score += 0.3
            else:
                # Penalize distance
                distance = min(
                    abs(tempo - signature['tempo_range'][0]),
                    abs(tempo - signature['tempo_range'][1])
                )
                score += max(0, 0.3 - distance / 100)

            # Complexity match
            if signature['complexity_range'][0] <= complexity <= signature['complexity_range'][1]:
                score += 0.2
            else:
                score += max(0, 0.2 - abs(complexity - np.mean(signature['complexity_range'])))

            # Drums match
            if has_drums == signature['has_drums']:
                score += 0.2

            # Instrument match
            matching_instruments = len(instruments.intersection(signature['instruments']))
            score += 0.3 * (matching_instruments / max(len(signature['instruments']), 1))

            scores[genre] = score

        # Normalize to probabilities
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores


class ChordExtractor:
    """Extract chords from MIDI files."""

    def __init__(self):
        """Initialize chord extractor."""
        pass

    def extract_chords(
        self,
        midi_path: str,
        window_size: float = 0.5,
    ) -> List[Tuple[float, List[int], str]]:
        """Extract chord progression from MIDI.

        Args:
            midi_path: Path to MIDI file
            window_size: Window size in seconds for chord detection

        Returns:
            List of (time, notes, chord_name) tuples
        """
        midi = pretty_midi.PrettyMIDI(str(midi_path))

        chords = []
        duration = midi.get_end_time()

        # Slide window through MIDI
        time = 0
        while time < duration:
            # Get notes active in window
            active_notes = set()

            for inst in midi.instruments:
                if inst.is_drum:
                    continue

                for note in inst.notes:
                    if note.start <= time < note.end or (time <= note.start < time + window_size):
                        active_notes.add(note.pitch % 12)  # Pitch class

            if len(active_notes) >= 3:
                # Identify chord
                chord_name = self._identify_chord(list(active_notes))
                chords.append((time, sorted(active_notes), chord_name))

            time += window_size

        return chords

    def _identify_chord(self, pitch_classes: List[int]) -> str:
        """Identify chord name from pitch classes.

        Args:
            pitch_classes: List of pitch classes (0-11)

        Returns:
            Chord name
        """
        # Common chord patterns (intervals from root)
        CHORD_TYPES = {
            frozenset([0, 4, 7]): 'maj',
            frozenset([0, 3, 7]): 'min',
            frozenset([0, 3, 6]): 'dim',
            frozenset([0, 4, 8]): 'aug',
            frozenset([0, 4, 7, 11]): 'maj7',
            frozenset([0, 3, 7, 10]): 'min7',
            frozenset([0, 4, 7, 10]): '7',
            frozenset([0, 3, 6, 10]): 'm7b5',
            frozenset([0, 3, 6, 9]): 'dim7',
        }

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Try each possible root
        for root in pitch_classes:
            # Normalize to root
            normalized = frozenset((pc - root) % 12 for pc in pitch_classes)

            if normalized in CHORD_TYPES:
                return f"{note_names[root]}{CHORD_TYPES[normalized]}"

        return "unknown"


class DatasetStatistics:
    """Calculate comprehensive dataset statistics."""

    def __init__(self):
        """Initialize statistics calculator."""
        pass

    def calculate_statistics(
        self,
        midi_directory: str,
    ) -> Dict:
        """Calculate dataset statistics.

        Args:
            midi_directory: Directory containing MIDI files

        Returns:
            Statistics dictionary
        """
        midi_dir = Path(midi_directory)
        midi_files = list(midi_dir.glob('**/*.mid')) + list(midi_dir.glob('**/*.midi'))

        stats = {
            'total_files': len(midi_files),
            'valid_files': 0,
            'total_duration': 0.0,
            'total_notes': 0,
            'tempos': [],
            'note_densities': [],
            'durations': [],
            'instruments': {},
            'has_drums': 0,
            'genres': {},
        }

        cleaner = DataCleaner()
        genre_classifier = GenreClassifier()

        for midi_file in midi_files:
            try:
                is_valid, _ = cleaner.validate_midi(str(midi_file))
                if not is_valid:
                    continue

                stats['valid_files'] += 1

                midi = pretty_midi.PrettyMIDI(str(midi_file))

                # Duration
                duration = midi.get_end_time()
                stats['total_duration'] += duration
                stats['durations'].append(duration)

                # Notes
                total_notes = sum(len(inst.notes) for inst in midi.instruments)
                stats['total_notes'] += total_notes
                stats['note_densities'].append(total_notes / duration if duration > 0 else 0)

                # Tempo
                tempo_changes = midi.get_tempo_changes()
                if len(tempo_changes[1]) > 0:
                    stats['tempos'].append(float(tempo_changes[1][0]))

                # Instruments
                for inst in midi.instruments:
                    if inst.is_drum:
                        stats['has_drums'] += 1
                    else:
                        program = inst.program
                        stats['instruments'][program] = stats['instruments'].get(program, 0) + 1

                # Genre
                genre_probs = genre_classifier.classify(str(midi_file))
                top_genre = max(genre_probs.items(), key=lambda x: x[1])[0]
                stats['genres'][top_genre] = stats['genres'].get(top_genre, 0) + 1

            except Exception as e:
                logger.warning(f"Error processing {midi_file}: {e}")
                continue

        # Calculate averages
        if stats['valid_files'] > 0:
            stats['avg_duration'] = stats['total_duration'] / stats['valid_files']
            stats['avg_notes_per_file'] = stats['total_notes'] / stats['valid_files']
            stats['avg_tempo'] = np.mean(stats['tempos']) if stats['tempos'] else 0
            stats['avg_note_density'] = np.mean(stats['note_densities']) if stats['note_densities'] else 0

            # Percentiles
            if stats['durations']:
                stats['duration_percentiles'] = {
                    '25': float(np.percentile(stats['durations'], 25)),
                    '50': float(np.percentile(stats['durations'], 50)),
                    '75': float(np.percentile(stats['durations'], 75)),
                }

        return stats


class DataVersioning:
    """Track dataset versions and changes."""

    def __init__(self, dataset_dir: str):
        """Initialize data versioning.

        Args:
            dataset_dir: Dataset directory
        """
        self.dataset_dir = Path(dataset_dir)
        self.version_file = self.dataset_dir / '.dataset_version.json'
        self.versions = self._load_versions()

    def _load_versions(self) -> List[Dict]:
        """Load version history."""
        if self.version_file.exists():
            with open(self.version_file) as f:
                return json.load(f)
        return []

    def _save_versions(self):
        """Save version history."""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)

    def compute_hash(self, file_path: str) -> str:
        """Compute file hash.

        Args:
            file_path: Path to file

        Returns:
            MD5 hash
        """
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def create_version(
        self,
        version_name: str,
        description: str = "",
    ) -> Dict:
        """Create new dataset version.

        Args:
            version_name: Version name/number
            description: Version description

        Returns:
            Version metadata
        """
        midi_files = list(self.dataset_dir.glob('**/*.mid')) + list(self.dataset_dir.glob('**/*.midi'))

        version = {
            'version': version_name,
            'description': description,
            'timestamp': str(np.datetime64('now')),
            'num_files': len(midi_files),
            'files': {}
        }

        # Hash all files
        for midi_file in midi_files:
            rel_path = midi_file.relative_to(self.dataset_dir)
            version['files'][str(rel_path)] = self.compute_hash(str(midi_file))

        self.versions.append(version)
        self._save_versions()

        logger.info(f"Created dataset version: {version_name} ({len(midi_files)} files)")

        return version

    def compare_versions(
        self,
        version1: str,
        version2: str,
    ) -> Dict:
        """Compare two dataset versions.

        Args:
            version1: First version name
            version2: Second version name

        Returns:
            Comparison results
        """
        v1 = next((v for v in self.versions if v['version'] == version1), None)
        v2 = next((v for v in self.versions if v['version'] == version2), None)

        if not v1 or not v2:
            return {'error': 'Version not found'}

        files1 = set(v1['files'].keys())
        files2 = set(v2['files'].keys())

        added = files2 - files1
        removed = files1 - files2
        modified = set()

        for file in files1.intersection(files2):
            if v1['files'][file] != v2['files'][file]:
                modified.add(file)

        return {
            'added': list(added),
            'removed': list(removed),
            'modified': list(modified),
            'unchanged': len(files1.intersection(files2)) - len(modified),
        }
