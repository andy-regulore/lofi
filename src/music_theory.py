"""Music theory constraints and harmonic analysis."""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MusicTheoryEngine:
    """Apply music theory constraints to generation."""

    # Major scale intervals (semitones from root)
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

    # Minor scale intervals
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]

    # Common chord progressions (in scale degrees)
    COMMON_PROGRESSIONS = [
        [1, 4, 5, 1],  # I-IV-V-I
        [1, 5, 6, 4],  # I-V-vi-IV (very common in pop)
        [1, 6, 4, 5],  # I-vi-IV-V
        [2, 5, 1],  # ii-V-I (jazz)
        [1, 4, 1, 5],  # I-IV-I-V
    ]

    # Lo-fi specific progressions
    LOFI_PROGRESSIONS = [
        [6, 4, 1, 5],  # vi-IV-I-V (melancholic)
        [1, 3, 6, 4],  # I-iii-vi-IV
        [2, 5, 1, 6],  # ii-V-I-vi
    ]

    def __init__(self):
        """Initialize music theory engine."""
        self.key_signatures = self._initialize_key_signatures()

    def _initialize_key_signatures(self) -> Dict[str, Dict]:
        """Initialize key signature database.

        Returns:
            Dictionary mapping keys to their properties
        """
        keys = {}

        # Major keys
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for i, note in enumerate(note_names):
            keys[note] = {
                "root": i,
                "scale": [(i + interval) % 12 for interval in self.MAJOR_SCALE],
                "mode": "major",
            }

        # Minor keys
        for i, note in enumerate(note_names):
            keys[f"{note}m"] = {
                "root": i,
                "scale": [(i + interval) % 12 for interval in self.MINOR_SCALE],
                "mode": "minor",
            }

        return keys

    def validate_melody(
        self, notes: List[int], key: str = "C", allow_chromatic: bool = True
    ) -> Tuple[bool, List[int]]:
        """Validate melody against music theory rules.

        Args:
            notes: List of MIDI note numbers
            key: Musical key
            allow_chromatic: Allow chromatic (out-of-scale) notes

        Returns:
            Tuple of (is_valid, corrected_notes)
        """
        if key not in self.key_signatures:
            logger.warning(f"Unknown key: {key}, using C major")
            key = "C"

        scale_notes = set(self.key_signatures[key]["scale"])
        corrected_notes = []

        for note in notes:
            pitch_class = note % 12

            if pitch_class in scale_notes:
                corrected_notes.append(note)
            elif allow_chromatic:
                # Allow chromatic passing tones
                corrected_notes.append(note)
            else:
                # Snap to nearest scale note
                closest = min(scale_notes, key=lambda x: abs(x - pitch_class))
                corrected_note = (note // 12) * 12 + closest
                corrected_notes.append(corrected_note)

        is_valid = notes == corrected_notes
        return is_valid, corrected_notes

    def suggest_chord_progression(
        self, key: str = "C", num_chords: int = 4, style: str = "lofi"
    ) -> List[Tuple[int, str]]:
        """Suggest a chord progression.

        Args:
            key: Musical key
            num_chords: Number of chords in progression
            style: Style ('lofi', 'jazz', 'pop')

        Returns:
            List of (root_note, chord_quality) tuples
        """
        if key not in self.key_signatures:
            key = "C"

        key_info = self.key_signatures[key]
        root = key_info["root"]
        scale = key_info["scale"]
        mode = key_info["mode"]

        # Select progression pool based on style
        if style == "lofi":
            progressions = self.LOFI_PROGRESSIONS
        else:
            progressions = self.COMMON_PROGRESSIONS

        # Pick random progression
        progression = progressions[np.random.randint(len(progressions))]

        # Repeat/truncate to match num_chords
        while len(progression) < num_chords:
            progression = progression + progression
        progression = progression[:num_chords]

        # Convert scale degrees to actual notes
        chords = []
        for degree in progression:
            # Scale degrees are 1-indexed
            scale_index = (degree - 1) % len(scale)
            chord_root = scale[scale_index]

            # Determine chord quality based on scale degree
            if mode == "major":
                if degree in [1, 4, 5]:
                    quality = "major"
                elif degree in [2, 3, 6]:
                    quality = "minor"
                else:
                    quality = "diminished"
            else:  # minor mode
                if degree in [3, 6, 7]:
                    quality = "major"
                elif degree in [1, 4, 5]:
                    quality = "minor"
                else:
                    quality = "diminished"

            chords.append((chord_root, quality))

        return chords

    def analyze_harmony(self, notes: List[int]) -> Dict[str, any]:
        """Analyze harmonic content of a note sequence.

        Args:
            notes: List of MIDI note numbers

        Returns:
            Dictionary with harmony analysis
        """
        if not notes:
            return {"error": "No notes provided"}

        # Count pitch class occurrences
        pitch_classes = [note % 12 for note in notes]
        pitch_counts = np.bincount(pitch_classes, minlength=12)

        # Detect likely key
        likely_key = self._detect_key(pitch_counts)

        # Calculate consonance/dissonance
        consonance = self._calculate_consonance(notes)

        # Detect chord types
        chords = self._detect_chords(notes)

        return {
            "likely_key": likely_key,
            "pitch_class_distribution": pitch_counts.tolist(),
            "consonance_score": consonance,
            "detected_chords": chords,
            "total_notes": len(notes),
            "unique_pitches": len(set(notes)),
        }

    def _detect_key(self, pitch_counts: np.ndarray) -> str:
        """Detect most likely key from pitch class distribution.

        Args:
            pitch_counts: Count of each pitch class (0-11)

        Returns:
            Detected key string
        """
        best_key = "C"
        best_score = 0

        for key, key_info in self.key_signatures.items():
            scale_notes = set(key_info["scale"])

            # Score = sum of counts for in-scale notes
            score = sum(pitch_counts[note] for note in scale_notes)

            if score > best_score:
                best_score = score
                best_key = key

        return best_key

    def _calculate_consonance(self, notes: List[int]) -> float:
        """Calculate consonance score (0-1, higher = more consonant).

        Args:
            notes: List of MIDI note numbers

        Returns:
            Consonance score
        """
        if len(notes) < 2:
            return 1.0

        # Consonant intervals (in semitones)
        consonant_intervals = {0, 3, 4, 5, 7, 8, 9, 12}  # Perfect, major/minor 3rds, 6ths

        interval_scores = []

        # Check adjacent notes
        for i in range(len(notes) - 1):
            interval = abs(notes[i + 1] - notes[i]) % 12
            if interval in consonant_intervals:
                interval_scores.append(1.0)
            else:
                interval_scores.append(0.0)

        return np.mean(interval_scores) if interval_scores else 1.0

    def _detect_chords(self, notes: List[int], window: int = 4) -> List[str]:
        """Detect chords in note sequence.

        Args:
            notes: List of MIDI note numbers
            window: Window size for chord detection

        Returns:
            List of detected chord names
        """
        chords = []

        # Slide window through notes
        for i in range(0, len(notes) - window + 1, window):
            window_notes = notes[i : i + window]
            pitch_classes = set(note % 12 for note in window_notes)

            # Try to match chord patterns
            chord_name = self._identify_chord(pitch_classes)
            if chord_name:
                chords.append(chord_name)

        return chords

    def _identify_chord(self, pitch_classes: Set[int]) -> Optional[str]:
        """Identify chord from pitch classes.

        Args:
            pitch_classes: Set of pitch classes (0-11)

        Returns:
            Chord name or None
        """
        if len(pitch_classes) < 3:
            return None

        # Common chord templates (intervals from root)
        chord_types = {
            frozenset([0, 4, 7]): "major",
            frozenset([0, 3, 7]): "minor",
            frozenset([0, 3, 6]): "diminished",
            frozenset([0, 4, 8]): "augmented",
            frozenset([0, 4, 7, 11]): "maj7",
            frozenset([0, 3, 7, 10]): "min7",
            frozenset([0, 4, 7, 10]): "dom7",
        }

        # Try each possible root
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        for root in pitch_classes:
            # Normalize to root
            normalized = frozenset((pc - root) % 12 for pc in pitch_classes)

            if normalized in chord_types:
                chord_type = chord_types[normalized]
                return f"{note_names[root]}{chord_type}"

        return None


class RhythmEngine:
    """Generate and analyze rhythmic patterns."""

    # Common time signatures
    TIME_SIGNATURES = {
        "4/4": (4, 4),
        "3/4": (3, 4),
        "6/8": (6, 8),
        "5/4": (5, 4),
    }

    # Lo-fi rhythm patterns (as ratios of measure)
    LOFI_PATTERNS = [
        [0, 0.25, 0.5, 0.75],  # Quarter notes
        [0, 0.25, 0.5, 0.625, 0.75],  # With swing
        [0, 0.333, 0.667],  # Triplets
        [0, 0.25, 0.375, 0.5, 0.75, 0.875],  # Syncopated
    ]

    def generate_rhythm_pattern(
        self,
        time_signature: str = "4/4",
        complexity: float = 0.5,
        swing: float = 0.0,
    ) -> List[float]:
        """Generate a rhythm pattern.

        Args:
            time_signature: Time signature
            complexity: Rhythmic complexity (0-1)
            swing: Swing amount (0-1)

        Returns:
            List of note onset times (as fraction of measure)
        """
        if time_signature not in self.TIME_SIGNATURES:
            time_signature = "4/4"

        beats, unit = self.TIME_SIGNATURES[time_signature]

        # Start with base pattern
        base_pattern = self.LOFI_PATTERNS[0].copy()

        # Add complexity
        if complexity > 0.3:
            # Add eighth notes
            base_pattern.extend([0.125, 0.375, 0.625, 0.875])

        if complexity > 0.6:
            # Add sixteenth notes
            base_pattern.extend([0.0625, 0.1875, 0.3125, 0.5625, 0.6875, 0.8125])

        # Apply swing
        if swing > 0:
            swung_pattern = []
            for time in sorted(set(base_pattern)):
                if time % 0.25 == 0.125:  # Eighth note on offbeat
                    swung_time = time + (swing * 0.0833)  # Delay slightly
                    swung_pattern.append(swung_time)
                else:
                    swung_pattern.append(time)
            base_pattern = swung_pattern

        # Sort and remove duplicates
        pattern = sorted(set(base_pattern))
        pattern = [t for t in pattern if t < 1.0]

        return pattern

    def analyze_rhythm(self, note_times: List[float]) -> Dict[str, any]:
        """Analyze rhythmic characteristics.

        Args:
            note_times: List of note onset times (in seconds)

        Returns:
            Dictionary with rhythm analysis
        """
        if len(note_times) < 2:
            return {"error": "Not enough notes"}

        # Calculate inter-onset intervals
        iois = np.diff(sorted(note_times))

        # Detect tempo
        median_ioi = np.median(iois)
        tempo = 60.0 / median_ioi if median_ioi > 0 else 120.0

        # Calculate rhythmic regularity
        ioi_std = np.std(iois)
        regularity = 1.0 / (1.0 + ioi_std) if ioi_std > 0 else 1.0

        # Detect syncopation
        syncopation = self._detect_syncopation(iois)

        return {
            "tempo_bpm": tempo,
            "median_ioi": median_ioi,
            "ioi_std": ioi_std,
            "regularity": regularity,
            "syncopation_score": syncopation,
            "total_notes": len(note_times),
        }

    def _detect_syncopation(self, iois: np.ndarray) -> float:
        """Detect amount of syncopation in rhythm.

        Args:
            iois: Inter-onset intervals

        Returns:
            Syncopation score (0-1)
        """
        # Syncopation indicated by irregular IOI patterns
        if len(iois) < 2:
            return 0.0

        # Calculate coefficient of variation
        cv = np.std(iois) / (np.mean(iois) + 1e-8)

        # Normalize to 0-1
        syncopation = min(1.0, cv / 0.5)

        return syncopation
