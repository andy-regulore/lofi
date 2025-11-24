"""
Advanced orchestration and instrumentation engine.

This module provides professional orchestration capabilities including:
- Instrument characteristics and ranges
- Voice spacing and doubling rules
- Orchestral balance and blend
- Automatic arrangement from melody
- Section-based orchestration
- MIDI program number mappings

Author: Claude
License: MIT
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class InstrumentFamily(Enum):
    """Instrument family classifications."""

    STRINGS = "strings"
    WOODWINDS = "woodwinds"
    BRASS = "brass"
    PERCUSSION = "percussion"
    KEYBOARDS = "keyboards"
    GUITARS = "guitars"
    BASS = "bass"
    SYNTH = "synth"


@dataclass
class InstrumentCharacteristics:
    """Characteristics of a musical instrument."""

    name: str
    midi_program: int
    family: InstrumentFamily
    range_low: int  # MIDI note number
    range_high: int  # MIDI note number
    comfortable_low: int  # Sweet spot low
    comfortable_high: int  # Sweet spot high
    timbre_tags: List[str]  # e.g., ['bright', 'mellow', 'percussive']
    dynamic_range: float  # 0-1, ability to play soft to loud
    agility: float  # 0-1, ability to play fast passages
    sustain: float  # 0-1, ability to sustain notes
    blend_score: float  # 0-1, how well it blends with others
    solo_score: float  # 0-1, how well it works as solo instrument
    typical_roles: List[str]  # e.g., ['melody', 'harmony', 'bass', 'rhythm']


class InstrumentDatabase:
    """Database of instrument characteristics."""

    def __init__(self):
        """Initialize with comprehensive instrument database."""
        self.instruments = self._build_database()

    def _build_database(self) -> Dict[str, InstrumentCharacteristics]:
        """Build comprehensive instrument database."""
        instruments = {}

        # === KEYBOARDS ===
        instruments["acoustic_grand_piano"] = InstrumentCharacteristics(
            name="Acoustic Grand Piano",
            midi_program=0,
            family=InstrumentFamily.KEYBOARDS,
            range_low=21,  # A0
            range_high=108,  # C8
            comfortable_low=28,  # E1
            comfortable_high=96,  # C7
            timbre_tags=["bright", "percussive", "resonant"],
            dynamic_range=0.95,
            agility=0.9,
            sustain=0.8,
            blend_score=0.7,
            solo_score=0.95,
            typical_roles=["melody", "harmony", "bass", "rhythm"],
        )

        instruments["electric_piano"] = InstrumentCharacteristics(
            name="Electric Piano",
            midi_program=4,
            family=InstrumentFamily.KEYBOARDS,
            range_low=28,
            range_high=103,
            comfortable_low=36,
            comfortable_high=96,
            timbre_tags=["mellow", "warm", "percussive"],
            dynamic_range=0.85,
            agility=0.85,
            sustain=0.6,
            blend_score=0.8,
            solo_score=0.85,
            typical_roles=["melody", "harmony", "rhythm"],
        )

        # === GUITARS ===
        instruments["acoustic_guitar_nylon"] = InstrumentCharacteristics(
            name="Acoustic Guitar (nylon)",
            midi_program=24,
            family=InstrumentFamily.GUITARS,
            range_low=40,  # E2
            range_high=84,  # C6
            comfortable_low=40,
            comfortable_high=79,
            timbre_tags=["warm", "mellow", "plucked"],
            dynamic_range=0.7,
            agility=0.8,
            sustain=0.4,
            blend_score=0.75,
            solo_score=0.9,
            typical_roles=["melody", "harmony", "rhythm"],
        )

        instruments["acoustic_guitar_steel"] = InstrumentCharacteristics(
            name="Acoustic Guitar (steel)",
            midi_program=25,
            family=InstrumentFamily.GUITARS,
            range_low=40,
            range_high=84,
            comfortable_low=40,
            comfortable_high=79,
            timbre_tags=["bright", "crisp", "plucked"],
            dynamic_range=0.75,
            agility=0.8,
            sustain=0.5,
            blend_score=0.75,
            solo_score=0.9,
            typical_roles=["melody", "harmony", "rhythm"],
        )

        instruments["electric_guitar_clean"] = InstrumentCharacteristics(
            name="Electric Guitar (clean)",
            midi_program=27,
            family=InstrumentFamily.GUITARS,
            range_low=40,
            range_high=88,
            comfortable_low=40,
            comfortable_high=84,
            timbre_tags=["bright", "clear", "sustained"],
            dynamic_range=0.8,
            agility=0.85,
            sustain=0.7,
            blend_score=0.8,
            solo_score=0.95,
            typical_roles=["melody", "harmony", "rhythm"],
        )

        # === BASS ===
        instruments["acoustic_bass"] = InstrumentCharacteristics(
            name="Acoustic Bass",
            midi_program=32,
            family=InstrumentFamily.BASS,
            range_low=28,  # E1
            range_high=67,  # G4
            comfortable_low=28,
            comfortable_high=55,
            timbre_tags=["warm", "deep", "resonant"],
            dynamic_range=0.8,
            agility=0.7,
            sustain=0.6,
            blend_score=0.9,
            solo_score=0.6,
            typical_roles=["bass"],
        )

        instruments["electric_bass_finger"] = InstrumentCharacteristics(
            name="Electric Bass (finger)",
            midi_program=33,
            family=InstrumentFamily.BASS,
            range_low=28,
            range_high=67,
            comfortable_low=28,
            comfortable_high=60,
            timbre_tags=["warm", "punchy", "percussive"],
            dynamic_range=0.85,
            agility=0.8,
            sustain=0.5,
            blend_score=0.9,
            solo_score=0.7,
            typical_roles=["bass", "rhythm"],
        )

        # === STRINGS ===
        instruments["violin"] = InstrumentCharacteristics(
            name="Violin",
            midi_program=40,
            family=InstrumentFamily.STRINGS,
            range_low=55,  # G3
            range_high=103,  # G7
            comfortable_low=55,
            comfortable_high=93,
            timbre_tags=["bright", "expressive", "singing"],
            dynamic_range=0.95,
            agility=0.9,
            sustain=0.95,
            blend_score=0.9,
            solo_score=0.95,
            typical_roles=["melody", "harmony"],
        )

        instruments["cello"] = InstrumentCharacteristics(
            name="Cello",
            midi_program=42,
            family=InstrumentFamily.STRINGS,
            range_low=36,  # C2
            range_high=84,  # C6
            comfortable_low=36,
            comfortable_high=76,
            timbre_tags=["warm", "rich", "expressive"],
            dynamic_range=0.95,
            agility=0.8,
            sustain=0.95,
            blend_score=0.85,
            solo_score=0.9,
            typical_roles=["melody", "harmony", "bass"],
        )

        instruments["string_ensemble"] = InstrumentCharacteristics(
            name="String Ensemble",
            midi_program=48,
            family=InstrumentFamily.STRINGS,
            range_low=36,
            range_high=96,
            comfortable_low=40,
            comfortable_high=88,
            timbre_tags=["lush", "warm", "sustained"],
            dynamic_range=0.9,
            agility=0.7,
            sustain=0.95,
            blend_score=0.95,
            solo_score=0.7,
            typical_roles=["harmony", "pad"],
        )

        # === BRASS ===
        instruments["trumpet"] = InstrumentCharacteristics(
            name="Trumpet",
            midi_program=56,
            family=InstrumentFamily.BRASS,
            range_low=55,  # F#3
            range_high=94,  # A#6
            comfortable_low=60,
            comfortable_high=84,
            timbre_tags=["bright", "brilliant", "powerful"],
            dynamic_range=0.9,
            agility=0.85,
            sustain=0.9,
            blend_score=0.7,
            solo_score=0.9,
            typical_roles=["melody", "harmony"],
        )

        instruments["trombone"] = InstrumentCharacteristics(
            name="Trombone",
            midi_program=57,
            family=InstrumentFamily.BRASS,
            range_low=40,  # E2
            range_high=79,  # G5
            comfortable_low=45,
            comfortable_high=72,
            timbre_tags=["warm", "smooth", "powerful"],
            dynamic_range=0.9,
            agility=0.6,
            sustain=0.9,
            blend_score=0.8,
            solo_score=0.8,
            typical_roles=["harmony", "bass"],
        )

        # === WOODWINDS ===
        instruments["flute"] = InstrumentCharacteristics(
            name="Flute",
            midi_program=73,
            family=InstrumentFamily.WOODWINDS,
            range_low=60,  # C4
            range_high=96,  # C7
            comfortable_low=65,
            comfortable_high=89,
            timbre_tags=["bright", "airy", "pure"],
            dynamic_range=0.85,
            agility=0.95,
            sustain=0.7,
            blend_score=0.85,
            solo_score=0.9,
            typical_roles=["melody"],
        )

        instruments["clarinet"] = InstrumentCharacteristics(
            name="Clarinet",
            midi_program=71,
            family=InstrumentFamily.WOODWINDS,
            range_low=50,  # D3
            range_high=94,  # A#6
            comfortable_low=55,
            comfortable_high=86,
            timbre_tags=["warm", "mellow", "expressive"],
            dynamic_range=0.9,
            agility=0.9,
            sustain=0.8,
            blend_score=0.9,
            solo_score=0.9,
            typical_roles=["melody", "harmony"],
        )

        instruments["saxophone"] = InstrumentCharacteristics(
            name="Alto Sax",
            midi_program=65,
            family=InstrumentFamily.WOODWINDS,
            range_low=49,  # C#3
            range_high=81,  # A5
            comfortable_low=53,
            comfortable_high=77,
            timbre_tags=["warm", "expressive", "jazzy"],
            dynamic_range=0.9,
            agility=0.85,
            sustain=0.85,
            blend_score=0.75,
            solo_score=0.95,
            typical_roles=["melody", "harmony"],
        )

        # === SYNTHS ===
        instruments["synth_pad"] = InstrumentCharacteristics(
            name="Synth Pad",
            midi_program=88,
            family=InstrumentFamily.SYNTH,
            range_low=24,
            range_high=96,
            comfortable_low=36,
            comfortable_high=84,
            timbre_tags=["lush", "atmospheric", "sustained"],
            dynamic_range=0.7,
            agility=0.5,
            sustain=0.98,
            blend_score=0.95,
            solo_score=0.6,
            typical_roles=["pad", "harmony"],
        )

        instruments["synth_lead"] = InstrumentCharacteristics(
            name="Synth Lead",
            midi_program=80,
            family=InstrumentFamily.SYNTH,
            range_low=36,
            range_high=96,
            comfortable_low=48,
            comfortable_high=84,
            timbre_tags=["bright", "cutting", "electronic"],
            dynamic_range=0.8,
            agility=0.95,
            sustain=0.9,
            blend_score=0.6,
            solo_score=0.95,
            typical_roles=["melody", "lead"],
        )

        return instruments

    def get_instrument(self, name: str) -> Optional[InstrumentCharacteristics]:
        """Get instrument by name."""
        return self.instruments.get(name)

    def get_by_family(self, family: InstrumentFamily) -> List[InstrumentCharacteristics]:
        """Get all instruments in a family."""
        return [inst for inst in self.instruments.values() if inst.family == family]

    def get_by_role(self, role: str) -> List[InstrumentCharacteristics]:
        """Get instruments suitable for a role."""
        return [inst for inst in self.instruments.values() if role in inst.typical_roles]


class VoiceSpacing:
    """Voice spacing and doubling rules."""

    @staticmethod
    def check_spacing(notes: List[int]) -> Dict[str, any]:
        """
        Check voice spacing quality.

        Args:
            notes: MIDI note numbers in the chord (sorted low to high)

        Returns:
            Dictionary with spacing analysis
        """
        if len(notes) < 2:
            return {"is_good": True, "issues": []}

        issues = []
        sorted_notes = sorted(notes)
        intervals = [sorted_notes[i + 1] - sorted_notes[i] for i in range(len(sorted_notes) - 1)]

        # Check for large gaps (>12 semitones) in lower voices
        for i, interval in enumerate(intervals[:-1]):  # Not the highest voice
            if interval > 12:
                issues.append(f"Large gap ({interval} semitones) between voices {i} and {i+1}")

        # Check for very close spacing in low register
        lowest_note = sorted_notes[0]
        if lowest_note < 48:  # Below C3
            close_intervals = [iv for iv in intervals[:2] if iv < 3]
            if close_intervals:
                issues.append("Very close spacing in low register (muddy)")

        # Calculate overall spacing quality
        avg_interval = np.mean(intervals)
        spacing_quality = 1.0 - (len(issues) * 0.2)

        return {
            "is_good": len(issues) == 0,
            "issues": issues,
            "intervals": intervals,
            "avg_interval": avg_interval,
            "quality_score": max(0.0, spacing_quality),
        }

    @staticmethod
    def optimize_spacing(notes: List[int], target_register: str = "mid") -> List[int]:
        """
        Optimize voice spacing by octave displacement.

        Args:
            notes: MIDI note numbers
            target_register: 'low', 'mid', or 'high'

        Returns:
            Optimized note list
        """
        if len(notes) < 2:
            return notes

        # Define target ranges
        target_ranges = {"low": (36, 60), "mid": (48, 72), "high": (60, 84)}

        target_low, target_high = target_ranges[target_register]

        # Normalize to pitch classes
        pitch_classes = [note % 12 for note in notes]

        # Place in target register
        optimized = []
        for pc in pitch_classes:
            # Find best octave
            best_note = pc
            while best_note < target_low:
                best_note += 12
            while best_note > target_high:
                best_note -= 12
            optimized.append(best_note)

        # Sort and check spacing
        optimized = sorted(optimized)

        # Ensure no unisons
        final = [optimized[0]]
        for note in optimized[1:]:
            if note == final[-1]:
                note += 12  # Raise by octave
            final.append(note)

        return final

    @staticmethod
    def double_notes(notes: List[int], doubling_strategy: str = "root_fifth") -> List[int]:
        """
        Double notes according to orchestration principles.

        Args:
            notes: MIDI note numbers of chord
            doubling_strategy: 'root', 'root_fifth', 'all', 'octaves'

        Returns:
            Notes with doubling applied
        """
        if len(notes) == 0:
            return notes

        result = list(notes)
        root = notes[0]  # Assume root is lowest note

        if doubling_strategy == "root":
            # Double root in higher octave
            result.append(root + 12)

        elif doubling_strategy == "root_fifth":
            # Double root and fifth (if present)
            result.append(root + 12)
            # Check for fifth (7 semitones above root)
            fifth_pc = (root + 7) % 12
            for note in notes:
                if note % 12 == fifth_pc:
                    result.append(note + 12)
                    break

        elif doubling_strategy == "all":
            # Double all notes
            result.extend([n + 12 for n in notes])

        elif doubling_strategy == "octaves":
            # Double each note at octave
            result.extend([n + 12 for n in notes])

        return sorted(result)


class OrchestrationEngine:
    """Main orchestration engine."""

    def __init__(self):
        """Initialize orchestration engine."""
        self.instrument_db = InstrumentDatabase()
        self.voice_spacing = VoiceSpacing()

    def arrange_melody(
        self,
        melody_notes: List[int],
        arrangement_style: str = "full",
        instrumentation: Optional[List[str]] = None,
    ) -> Dict[str, List[int]]:
        """
        Arrange a melody for multiple instruments.

        Args:
            melody_notes: MIDI note numbers of melody
            arrangement_style: 'minimal', 'full', 'orchestral'
            instrumentation: List of instrument names, or None for auto

        Returns:
            Dictionary mapping instrument names to note lists
        """
        if instrumentation is None:
            instrumentation = self._select_instruments(arrangement_style)

        arrangement = {}

        for inst_name in instrumentation:
            inst = self.instrument_db.get_instrument(inst_name)
            if inst is None:
                continue

            # Transpose melody to comfortable range
            transposed = self._transpose_to_range(
                melody_notes, inst.comfortable_low, inst.comfortable_high
            )

            arrangement[inst_name] = transposed

        return arrangement

    def _select_instruments(self, style: str) -> List[str]:
        """Select instruments based on arrangement style."""
        if style == "minimal":
            return ["acoustic_grand_piano", "acoustic_bass"]

        elif style == "full":
            return ["acoustic_grand_piano", "electric_bass_finger", "string_ensemble", "synth_pad"]

        elif style == "orchestral":
            return ["violin", "cello", "flute", "clarinet", "trumpet", "trombone"]

        else:
            return ["acoustic_grand_piano"]

    def _transpose_to_range(self, notes: List[int], low: int, high: int) -> List[int]:
        """Transpose notes to fit within instrument range."""
        if not notes:
            return notes

        # Calculate current center
        current_center = np.mean(notes)
        target_center = (low + high) / 2

        # Calculate octave shift
        shift = round((target_center - current_center) / 12) * 12

        # Apply shift
        transposed = [n + shift for n in notes]

        # Ensure all notes in range
        while any(n < low for n in transposed):
            transposed = [n + 12 for n in transposed]

        while any(n > high for n in transposed):
            transposed = [n - 12 for n in transposed]

        return transposed

    def check_balance(self, arrangement: Dict[str, List[int]]) -> Dict[str, float]:
        """
        Check orchestral balance of arrangement.

        Args:
            arrangement: Dict mapping instrument names to note lists

        Returns:
            Dictionary with balance scores
        """
        scores = {}

        # Check register coverage
        all_notes = []
        for notes in arrangement.values():
            all_notes.extend(notes)

        if not all_notes:
            return {"overall_balance": 0.0}

        low_count = sum(1 for n in all_notes if n < 48)
        mid_count = sum(1 for n in all_notes if 48 <= n < 72)
        high_count = sum(1 for n in all_notes if n >= 72)

        total = low_count + mid_count + high_count

        # Ideal distribution: 30% low, 40% mid, 30% high
        low_balance = 1.0 - abs(low_count / total - 0.3)
        mid_balance = 1.0 - abs(mid_count / total - 0.4)
        high_balance = 1.0 - abs(high_count / total - 0.3)

        scores["register_balance"] = (low_balance + mid_balance + high_balance) / 3

        # Check family balance
        families = []
        for inst_name in arrangement.keys():
            inst = self.instrument_db.get_instrument(inst_name)
            if inst:
                families.append(inst.family.value)

        family_diversity = len(set(families)) / len(families) if families else 0
        scores["family_diversity"] = family_diversity

        # Overall balance
        scores["overall_balance"] = (scores["register_balance"] + scores["family_diversity"]) / 2

        return scores

    def suggest_instrumentation(
        self, melody_range: Tuple[int, int], genre: str = "lofi", num_instruments: int = 4
    ) -> List[str]:
        """
        Suggest instrumentation based on context.

        Args:
            melody_range: (low, high) MIDI note numbers
            genre: Musical genre
            num_instruments: Number of instruments to suggest

        Returns:
            List of suggested instrument names
        """
        genre_preferences = {
            "lofi": [
                "acoustic_grand_piano",
                "electric_bass_finger",
                "synth_pad",
                "electric_guitar_clean",
            ],
            "jazz": ["acoustic_grand_piano", "acoustic_bass", "saxophone", "electric_guitar_clean"],
            "classical": ["violin", "cello", "flute", "clarinet"],
            "electronic": [
                "synth_lead",
                "synth_pad",
                "electric_bass_finger",
                "acoustic_grand_piano",
            ],
        }

        preferred = genre_preferences.get(genre, ["acoustic_grand_piano", "electric_bass_finger"])

        # Filter by range compatibility
        low, high = melody_range
        suggestions = []

        for inst_name in preferred:
            inst = self.instrument_db.get_instrument(inst_name)
            if inst is None:
                continue

            # Check if melody fits in instrument range
            if inst.range_low <= low and inst.range_high >= high:
                suggestions.append(inst_name)

        # Fill up to num_instruments
        while len(suggestions) < num_instruments:
            suggestions.append("acoustic_grand_piano")

        return suggestions[:num_instruments]


class SATBVoicing:
    """SATB (Soprano-Alto-Tenor-Bass) voice leading."""

    # Standard SATB ranges
    SOPRANO_RANGE = (60, 81)  # C4-A5
    ALTO_RANGE = (53, 74)  # F3-D5
    TENOR_RANGE = (48, 69)  # C3-A4
    BASS_RANGE = (40, 62)  # E2-D4

    @staticmethod
    def create_satb_voicing(chord_notes: List[int]) -> Dict[str, int]:
        """
        Create SATB voicing from chord notes.

        Args:
            chord_notes: Pitch classes (0-11) or MIDI notes

        Returns:
            Dictionary with SATB voices
        """
        # Normalize to pitch classes
        pitch_classes = [note % 12 for note in chord_notes]

        # Place bass (root)
        bass_pc = pitch_classes[0]
        bass = bass_pc + 12 * 3  # Place around E2-D4 range
        while bass < SATBVoicing.BASS_RANGE[0]:
            bass += 12
        while bass > SATBVoicing.BASS_RANGE[1]:
            bass -= 12

        # Place tenor (3rd or 5th)
        tenor_pc = pitch_classes[1] if len(pitch_classes) > 1 else pitch_classes[0]
        tenor = tenor_pc + 12 * 4
        while tenor < SATBVoicing.TENOR_RANGE[0]:
            tenor += 12
        while tenor > SATBVoicing.TENOR_RANGE[1]:
            tenor -= 12

        # Place alto
        alto_pc = pitch_classes[2] if len(pitch_classes) > 2 else pitch_classes[0]
        alto = alto_pc + 12 * 4
        while alto < SATBVoicing.ALTO_RANGE[0]:
            alto += 12
        while alto > SATBVoicing.ALTO_RANGE[1]:
            alto -= 12

        # Place soprano (melody note, usually highest)
        soprano_pc = pitch_classes[-1]
        soprano = soprano_pc + 12 * 5
        while soprano < SATBVoicing.SOPRANO_RANGE[0]:
            soprano += 12
        while soprano > SATBVoicing.SOPRANO_RANGE[1]:
            soprano -= 12

        return {"soprano": soprano, "alto": alto, "tenor": tenor, "bass": bass}


# Example usage and testing
if __name__ == "__main__":
    # Initialize engine
    engine = OrchestrationEngine()

    # Test instrument database
    print("=== Instrument Database ===")
    piano = engine.instrument_db.get_instrument("acoustic_grand_piano")
    print(f"Piano range: {piano.range_low}-{piano.range_high}")
    print(f"Piano roles: {piano.typical_roles}")

    # Test arrangement
    print("\n=== Melody Arrangement ===")
    melody = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
    arrangement = engine.arrange_melody(melody, arrangement_style="full")
    for inst, notes in arrangement.items():
        print(f"{inst}: {notes[:4]}...")

    # Test balance
    print("\n=== Balance Check ===")
    balance = engine.check_balance(arrangement)
    print(f"Register balance: {balance['register_balance']:.2f}")
    print(f"Overall balance: {balance['overall_balance']:.2f}")

    # Test voice spacing
    print("\n=== Voice Spacing ===")
    chord = [60, 64, 67, 72]  # C major chord
    spacing = VoiceSpacing.check_spacing(chord)
    print(f"Spacing quality: {spacing['quality_score']:.2f}")
    print(f"Intervals: {spacing['intervals']}")

    # Test SATB
    print("\n=== SATB Voicing ===")
    satb = SATBVoicing.create_satb_voicing([0, 4, 7])  # C major
    print(
        f"Soprano: {satb['soprano']}, Alto: {satb['alto']}, Tenor: {satb['tenor']}, Bass: {satb['bass']}"
    )

    # Test instrumentation suggestion
    print("\n=== Instrumentation Suggestion ===")
    suggestions = engine.suggest_instrumentation((60, 84), genre="lofi")
    print(f"Suggested instruments: {suggestions}")
