"""
Advanced rhythm theory and generation.

This module provides sophisticated rhythm capabilities including:
- Polyrhythms and complex time signatures
- African and Latin rhythm patterns
- Groove quantization and humanization
- Syncopation analysis and generation
- Rhythmic motif development
- Metric modulation

Author: Claude
License: MIT
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TimeSignature:
    """Represents a time signature."""

    numerator: int  # Beats per measure
    denominator: int  # Beat unit (4 = quarter note, 8 = eighth note)

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    def measure_duration(self, tempo_bpm: float) -> float:
        """Calculate measure duration in seconds."""
        beat_duration = 60.0 / tempo_bpm  # Duration of quarter note
        beats_per_measure = self.numerator * (4 / self.denominator)
        return beat_duration * beats_per_measure

    def is_compound(self) -> bool:
        """Check if time signature is compound (numerator divisible by 3)."""
        return self.numerator % 3 == 0 and self.numerator > 3

    def is_odd(self) -> bool:
        """Check if time signature has odd numerator."""
        return self.numerator % 2 == 1


@dataclass
class RhythmicEvent:
    """Single rhythmic event."""

    onset: float  # Time in beats
    duration: float  # Duration in beats
    velocity: int  # MIDI velocity (0-127)
    pitch: Optional[int] = None  # MIDI pitch (optional)


class Polyrhythm:
    """Generate and analyze polyrhythms."""

    @staticmethod
    def generate(ratio: Tuple[int, int], duration: float) -> Tuple[List[float], List[float]]:
        """
        Generate polyrhythm with given ratio.

        Args:
            ratio: (a, b) for a:b polyrhythm (e.g., (3, 2) for 3 against 2)
            duration: Total duration in beats

        Returns:
            Tuple of (onsets_a, onsets_b) as lists of beat positions
        """
        a, b = ratio

        # Generate onsets for each rhythm
        onsets_a = [i * (duration / a) for i in range(a)]
        onsets_b = [i * (duration / b) for i in range(b)]

        return onsets_a, onsets_b

    @staticmethod
    def detect_polyrhythm(
        onsets_a: List[float], onsets_b: List[float]
    ) -> Optional[Tuple[int, int]]:
        """
        Detect if two rhythm patterns form a polyrhythm.

        Args:
            onsets_a: First rhythm onset times
            onsets_b: Second rhythm onset times

        Returns:
            Detected ratio or None
        """
        if not onsets_a or not onsets_b:
            return None

        # Calculate inter-onset intervals
        ioi_a = np.diff(onsets_a)
        ioi_b = np.diff(onsets_b)

        # Check if IOIs are regular
        if np.std(ioi_a) > 0.01 or np.std(ioi_b) > 0.01:
            return None  # Not regular rhythms

        # Calculate ratio
        ratio_float = np.mean(ioi_b) / np.mean(ioi_a)

        # Try to find simple ratio
        for denom in range(2, 10):
            numer = round(ratio_float * denom)
            if abs(ratio_float - numer / denom) < 0.01:
                return (numer, denom)

        return None

    @staticmethod
    def get_common_polyrhythms() -> Dict[str, Tuple[int, int]]:
        """Return dictionary of common polyrhythms."""
        return {
            "hemiola": (3, 2),
            "triplet_duplet": (3, 2),
            "four_against_three": (4, 3),
            "five_against_four": (5, 4),
            "seven_against_four": (7, 4),
            "three_against_four": (3, 4),
            "five_against_three": (5, 3),
        }


class OddMeter:
    """Handle odd/complex time signatures."""

    # Common odd meter patterns
    COMMON_ODD_METERS = {
        "5/4": [(2, 3), (3, 2)],  # Groupings: 2+3 or 3+2
        "7/8": [(2, 2, 3), (3, 2, 2), (2, 3, 2)],  # 2+2+3, 3+2+2, 2+3+2
        "7/4": [(3, 4), (4, 3), (2, 2, 3)],
        "9/8": [(2, 2, 2, 3), (3, 3, 3)],  # 2+2+2+3 or 3+3+3
        "11/8": [(3, 3, 3, 2), (2, 3, 3, 3)],
        "13/8": [(3, 3, 3, 4), (4, 3, 3, 3)],
    }

    @staticmethod
    def generate_pattern(
        time_sig: TimeSignature, grouping: Optional[List[int]] = None
    ) -> List[float]:
        """
        Generate onset pattern for odd meter.

        Args:
            time_sig: TimeSignature object
            grouping: Beat grouping (e.g., [2, 3] for 5/4)

        Returns:
            List of onset times in beats
        """
        sig_str = str(time_sig)

        # Get default grouping if not provided
        if grouping is None:
            groupings = OddMeter.COMMON_ODD_METERS.get(sig_str)
            if groupings:
                grouping = groupings[0]
            else:
                # Default: split evenly
                grouping = [time_sig.numerator]

        # Generate onsets based on grouping
        onsets = [0.0]
        current_time = 0.0

        beat_unit = 4.0 / time_sig.denominator  # Convert to quarter notes

        for group_size in grouping:
            current_time += group_size * beat_unit
            if current_time < time_sig.numerator * beat_unit:
                onsets.append(current_time)

        return onsets

    @staticmethod
    def analyze_grouping(onsets: List[float]) -> List[int]:
        """
        Analyze beat grouping from onset pattern.

        Args:
            onsets: List of onset times

        Returns:
            Detected grouping
        """
        if len(onsets) < 2:
            return []

        # Calculate inter-onset intervals
        iois = np.diff(onsets)

        # Round to nearest integer (in eighth notes)
        grouping = [round(ioi * 2) for ioi in iois]

        return grouping


class AfricanRhythms:
    """African rhythm patterns and clave rhythms."""

    # Standard clave patterns (in 16th note grid)
    CLAVES = {
        "son_clave_2_3": [0, 3, 6, 10, 12],  # 2-3 son clave
        "son_clave_3_2": [0, 2, 6, 8, 11],  # 3-2 son clave
        "rumba_clave_2_3": [0, 3, 6, 10, 11],
        "rumba_clave_3_2": [0, 1, 5, 7, 10],
        "bossa_nova": [0, 3, 6, 10, 13],
        "afro_cuban_6_8": [0, 3, 6, 9],
    }

    # West African bell patterns
    BELL_PATTERNS = {
        "gankogui": [0, 2, 3, 5, 7, 9, 10],  # 7-stroke Ewe pattern
        "standard_pattern": [0, 3, 5, 6, 9, 10, 11],  # 12-pulse pattern
        "fume_fume": [0, 2, 4, 6, 9, 11],
        "kpanlogo": [0, 3, 5, 7, 9, 10],
    }

    @staticmethod
    def generate_clave(pattern_name: str, num_measures: int = 1) -> List[float]:
        """
        Generate clave pattern.

        Args:
            pattern_name: Name of clave pattern
            num_measures: Number of measures to generate

        Returns:
            List of onset times in beats
        """
        if pattern_name not in AfricanRhythms.CLAVES:
            raise ValueError(f"Unknown clave pattern: {pattern_name}")

        # Get pattern (in 16th note positions over 2 measures)
        pattern = AfricanRhythms.CLAVES[pattern_name]

        # Convert to beat positions (assuming 4/4, 16th = 0.25 beats)
        onsets_per_cycle = [pos * 0.25 for pos in pattern]

        # Repeat for num_measures (each cycle is 2 measures)
        onsets = []
        cycle_duration = 8.0  # 2 measures of 4/4
        for cycle in range((num_measures + 1) // 2):
            for onset in onsets_per_cycle:
                onsets.append(onset + cycle * cycle_duration)

        # Trim to exact num_measures
        measure_duration = 4.0
        onsets = [o for o in onsets if o < num_measures * measure_duration]

        return onsets

    @staticmethod
    def generate_bell_pattern(pattern_name: str, num_cycles: int = 1) -> List[float]:
        """
        Generate West African bell pattern.

        Args:
            pattern_name: Name of bell pattern
            num_cycles: Number of cycles to generate

        Returns:
            List of onset times in beats
        """
        if pattern_name not in AfricanRhythms.BELL_PATTERNS:
            raise ValueError(f"Unknown bell pattern: {pattern_name}")

        pattern = AfricanRhythms.BELL_PATTERNS[pattern_name]

        # Convert to beat positions (12-pulse = 3 beats)
        cycle_duration = 3.0
        onsets_per_cycle = [pos * (cycle_duration / 12) for pos in pattern]

        # Repeat for num_cycles
        onsets = []
        for cycle in range(num_cycles):
            for onset in onsets_per_cycle:
                onsets.append(onset + cycle * cycle_duration)

        return onsets


class GrooveEngine:
    """Generate and manipulate groove patterns."""

    @staticmethod
    def quantize(onsets: List[float], grid: float, strength: float = 1.0) -> List[float]:
        """
        Quantize onsets to grid.

        Args:
            onsets: Original onset times
            grid: Grid size in beats (e.g., 0.25 for 16th notes)
            strength: Quantization strength (0=none, 1=full)

        Returns:
            Quantized onsets
        """
        quantized = []
        for onset in onsets:
            # Find nearest grid position
            grid_pos = round(onset / grid) * grid
            # Interpolate based on strength
            new_onset = onset * (1 - strength) + grid_pos * strength
            quantized.append(new_onset)

        return quantized

    @staticmethod
    def humanize(
        onsets: List[float], timing_variation: float = 0.02, velocity_variation: float = 10
    ) -> Tuple[List[float], List[int]]:
        """
        Add human-like timing and velocity variations.

        Args:
            onsets: Original onset times
            timing_variation: Max timing deviation in beats (e.g., 0.02 = ±20ms at 120 BPM)
            velocity_variation: Max velocity deviation

        Returns:
            Tuple of (humanized_onsets, velocities)
        """
        humanized_onsets = []
        velocities = []

        for onset in onsets:
            # Add timing variation
            deviation = np.random.uniform(-timing_variation, timing_variation)
            humanized_onsets.append(onset + deviation)

            # Add velocity variation (centered around 80)
            base_velocity = 80
            vel_dev = np.random.uniform(-velocity_variation, velocity_variation)
            velocity = int(np.clip(base_velocity + vel_dev, 1, 127))
            velocities.append(velocity)

        return humanized_onsets, velocities

    @staticmethod
    def swing(onsets: List[float], swing_amount: float = 0.5) -> List[float]:
        """
        Apply swing to eighth note pattern.

        Args:
            onsets: Original onset times (assuming straight eighths)
            swing_amount: Amount of swing (0=straight, 0.5=triplet, 1.0=extreme)

        Returns:
            Swung onsets
        """
        swung = []

        for i, onset in enumerate(onsets):
            # Determine if this is an "off-beat" eighth note
            beat_pos = onset % 1.0  # Position within beat

            if abs(beat_pos - 0.5) < 0.1:  # Off-beat eighth
                # Delay by swing amount
                delay = swing_amount * 0.167  # Max delay is 1/6 beat (triplet feel)
                swung.append(onset + delay)
            else:
                swung.append(onset)

        return swung

    @staticmethod
    def create_groove(pattern_type: str, num_beats: int = 4) -> Dict[str, List[float]]:
        """
        Create complete groove pattern with multiple instruments.

        Args:
            pattern_type: 'funk', 'jazz', 'rock', 'latin'
            num_beats: Number of beats to generate

        Returns:
            Dictionary mapping instrument to onset lists
        """
        groove = {}

        if pattern_type == "funk":
            # Kick: on 1 and 3, some syncopation
            groove["kick"] = [0, 2, 3.5]
            # Snare: on 2 and 4
            groove["snare"] = [1, 3]
            # Hi-hat: sixteenth notes
            groove["hihat"] = [i * 0.25 for i in range(16)]

        elif pattern_type == "jazz":
            # Ride: swing pattern
            groove["ride"] = [i * 0.5 for i in range(8)]
            groove["ride"] = GrooveEngine.swing(groove["ride"], swing_amount=0.5)
            # Hi-hat: on 2 and 4
            groove["hihat"] = [1, 3]
            # Kick: sparse, syncopated
            groove["kick"] = [0, 2.5, 3.5]

        elif pattern_type == "rock":
            # Kick: on 1 and 3
            groove["kick"] = [0, 2]
            # Snare: on 2 and 4
            groove["snare"] = [1, 3]
            # Hi-hat: eighth notes
            groove["hihat"] = [i * 0.5 for i in range(8)]

        elif pattern_type == "latin":
            # Use clave as basis
            clave_onsets = AfricanRhythms.generate_clave("son_clave_2_3", num_measures=1)
            groove["clave"] = clave_onsets
            # Tumbao bass pattern
            groove["bass"] = [0, 0.5, 1.5, 2.5, 3, 3.5]

        # Extend to requested number of beats
        for inst in groove:
            pattern = groove[inst]
            extended = []
            pattern_length = 4.0  # Assuming 4-beat patterns
            num_repeats = int(np.ceil(num_beats / pattern_length))
            for rep in range(num_repeats):
                for onset in pattern:
                    extended.append(onset + rep * pattern_length)
            # Trim to exact length
            groove[inst] = [o for o in extended if o < num_beats]

        return groove


class Syncopation:
    """Analyze and generate syncopation."""

    @staticmethod
    def calculate_syncopation_score(onsets: List[float], time_sig: TimeSignature) -> float:
        """
        Calculate syncopation score for rhythm pattern.
        Uses Longuet-Higgins and Lee model.

        Args:
            onsets: Onset times in beats
            time_sig: Time signature

        Returns:
            Syncopation score (0-1, higher = more syncopated)
        """
        if not onsets:
            return 0.0

        # Define metrical hierarchy (weights for beat positions)
        beat_duration = 4.0 / time_sig.denominator
        hierarchy = {}

        # Downbeat has highest weight
        hierarchy[0.0] = 1.0

        # Strong beats
        for i in range(time_sig.numerator):
            beat_pos = i * beat_duration
            if i % 2 == 0:
                hierarchy[beat_pos] = 0.8
            else:
                hierarchy[beat_pos] = 0.6

        # Weak positions (between beats)
        for i in range(time_sig.numerator * 2):
            off_beat_pos = i * beat_duration / 2
            if off_beat_pos not in hierarchy:
                hierarchy[off_beat_pos] = 0.3

        # Calculate syncopation
        total_sync = 0.0
        for onset in onsets:
            # Find nearest metrical position
            beat_pos = onset % (time_sig.numerator * beat_duration)

            # Get weight (or interpolate)
            nearest_pos = min(hierarchy.keys(), key=lambda x: abs(x - beat_pos))
            weight = hierarchy[nearest_pos]

            # Syncopation = 1 - weight
            total_sync += 1.0 - weight

        # Normalize
        return total_sync / len(onsets) if onsets else 0.0

    @staticmethod
    def add_syncopation(base_onsets: List[float], amount: float = 0.3) -> List[float]:
        """
        Add syncopation to rhythm pattern.

        Args:
            base_onsets: Original onsets (on-beat)
            amount: Amount of syncopation to add (0-1)

        Returns:
            Modified onsets with syncopation
        """
        syncopated = list(base_onsets)

        # Number of notes to shift
        num_to_shift = int(len(base_onsets) * amount)

        # Randomly select notes to shift off-beat
        indices = np.random.choice(len(base_onsets), size=num_to_shift, replace=False)

        for idx in indices:
            # Shift slightly off-beat (between -0.2 and -0.1 beats)
            shift = np.random.uniform(-0.2, -0.1)
            syncopated[idx] += shift

        return sorted(syncopated)


class MetricModulation:
    """Handle metric modulation (tempo relationships)."""

    @staticmethod
    def calculate_new_tempo(
        old_tempo: float, old_note_value: Fraction, new_note_value: Fraction
    ) -> float:
        """
        Calculate new tempo after metric modulation.

        Example: Quarter note = dotted quarter note
        old_note_value = Fraction(1, 4)  # Quarter
        new_note_value = Fraction(3, 8)  # Dotted quarter

        Args:
            old_tempo: Original tempo in BPM
            old_note_value: Old note value as fraction of whole note
            new_note_value: New note value as fraction of whole note

        Returns:
            New tempo in BPM
        """
        # old_tempo * old_note_value = new_tempo * new_note_value
        return old_tempo * (float(old_note_value) / float(new_note_value))

    @staticmethod
    def common_modulations() -> Dict[str, Tuple[float, str]]:
        """
        Return common metric modulations.

        Returns:
            Dict mapping name to (tempo_ratio, description)
        """
        return {
            "quarter_to_dotted_quarter": (2 / 3, "Quarter = dotted quarter (slower)"),
            "quarter_to_eighth": (2.0, "Quarter = eighth (faster)"),
            "quarter_to_triplet": (3 / 2, "Quarter = triplet quarter"),
            "eighth_to_quarter": (0.5, "Eighth = quarter (slower)"),
            "triplet_to_duplet": (2 / 3, "Triplet quarter = regular quarter"),
        }


# Example usage and testing
if __name__ == "__main__":
    print("=== Polyrhythm Generation ===")
    onsets_3, onsets_2 = Polyrhythm.generate((3, 2), duration=4.0)
    print(f"3 against 2: {onsets_3} vs {onsets_2}")

    print("\n=== Odd Meter Patterns ===")
    time_sig_5_4 = TimeSignature(5, 4)
    pattern = OddMeter.generate_pattern(time_sig_5_4, grouping=[2, 3])
    print(f"5/4 (2+3): {pattern}")

    print("\n=== Clave Patterns ===")
    son_clave = AfricanRhythms.generate_clave("son_clave_2_3", num_measures=2)
    print(f"Son clave (2-3): {son_clave}")

    print("\n=== Groove Generation ===")
    funk_groove = GrooveEngine.create_groove("funk", num_beats=4)
    print("Funk groove:")
    for inst, onsets in funk_groove.items():
        print(f"  {inst}: {onsets}")

    print("\n=== Syncopation ===")
    straight = [0, 1, 2, 3]
    syncopated = Syncopation.add_syncopation(straight, amount=0.5)
    print(f"Original: {straight}")
    print(f"Syncopated: {syncopated}")

    score = Syncopation.calculate_syncopation_score(syncopated, TimeSignature(4, 4))
    print(f"Syncopation score: {score:.2f}")

    print("\n=== Humanization ===")
    mechanical = [0, 0.5, 1.0, 1.5, 2.0]
    human_onsets, velocities = GrooveEngine.humanize(mechanical)
    print(f"Mechanical: {mechanical}")
    print(f"Humanized: {[f'{o:.3f}' for o in human_onsets]}")
    print(f"Velocities: {velocities}")

    print("\n=== Swing ===")
    straight_eighths = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    swung = GrooveEngine.swing(straight_eighths, swing_amount=0.5)
    print(f"Straight: {straight_eighths}")
    print(f"Swung: {[f'{o:.3f}' for o in swung]}")

    print("\n=== Metric Modulation ===")
    old_tempo = 120
    new_tempo = MetricModulation.calculate_new_tempo(
        old_tempo, Fraction(1, 4), Fraction(3, 8)  # Quarter  # Dotted quarter
    )
    print(f"Old tempo: {old_tempo} BPM")
    print(f"New tempo (quarter = dotted quarter): {new_tempo} BPM")
