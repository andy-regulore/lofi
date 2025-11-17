"""Advanced music theory for sophisticated harmony and composition.

Extensions beyond basic music theory:
- Jazz harmony (extensions, alterations, substitutions)
- Voice leading rules
- Modal interchange
- Secondary dominants
- Chord voicings
- Reharmonization techniques
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

logger = logging.getLogger(__name__)


class JazzHarmony:
    """Advanced jazz harmony theory."""

    # Chord extensions and alterations
    EXTENSIONS = {
        'maj7': [0, 4, 7, 11],
        'min7': [0, 3, 7, 10],
        'dom7': [0, 4, 7, 10],
        'min7b5': [0, 3, 6, 10],  # Half-diminished
        'dim7': [0, 3, 6, 9],
        'maj9': [0, 4, 7, 11, 14],
        'min9': [0, 3, 7, 10, 14],
        'dom9': [0, 4, 7, 10, 14],
        'maj11': [0, 4, 7, 11, 14, 17],
        'dom11': [0, 4, 7, 10, 14, 17],
        'maj13': [0, 4, 7, 11, 14, 17, 21],
        'dom13': [0, 4, 7, 10, 14, 17, 21],
        # Altered dominants
        'dom7b9': [0, 4, 7, 10, 13],
        'dom7#9': [0, 4, 7, 10, 15],
        'dom7b5': [0, 4, 6, 10],
        'dom7#5': [0, 4, 8, 10],
        'dom7alt': [0, 4, 6, 10, 13, 15],  # Altered scale
    }

    # Common jazz progressions
    JAZZ_PROGRESSIONS = [
        # ii-V-I (most important)
        ['min7', 'dom7', 'maj7'],
        # iii-VI-ii-V-I
        ['min7', 'dom7', 'min7', 'dom7', 'maj7'],
        # Coltrane changes (Giant Steps)
        ['maj7', 'dom7', 'maj7', 'dom7', 'maj7'],
        # Rhythm changes (I Got Rhythm)
        ['maj7', 'dom7', 'min7', 'dom7'],
        # Minor ii-V-i
        ['min7b5', 'dom7', 'min7'],
    ]

    def __init__(self):
        """Initialize jazz harmony engine."""
        pass

    def get_chord_voicing(
        self,
        chord_root: int,
        chord_type: str,
        voicing_style: str = 'close',
        bass_note: Optional[int] = None,
    ) -> List[int]:
        """Get specific chord voicing.

        Args:
            chord_root: Root note (0-11)
            chord_type: Chord type from EXTENSIONS
            voicing_style: 'close', 'open', 'drop2', 'drop3', 'spread'
            bass_note: Optional bass note for slash chords

        Returns:
            List of MIDI note numbers for voicing
        """
        if chord_type not in self.EXTENSIONS:
            chord_type = 'maj7'  # Default

        # Get basic chord tones
        intervals = self.EXTENSIONS[chord_type]
        base_octave = 60  # Middle C

        # Create basic voicing
        notes = [(chord_root + interval) % 12 + base_octave for interval in intervals]

        # Apply voicing style
        if voicing_style == 'close':
            # All notes within an octave
            notes = [n if n >= base_octave else n + 12 for n in notes]

        elif voicing_style == 'open':
            # Spread out over 2 octaves
            for i in range(1, len(notes), 2):
                notes[i] += 12

        elif voicing_style == 'drop2':
            # Drop second-highest note by an octave
            if len(notes) >= 2:
                notes = sorted(notes)
                notes[-2] -= 12

        elif voicing_style == 'drop3':
            # Drop third-highest note by an octave
            if len(notes) >= 3:
                notes = sorted(notes)
                notes[-3] -= 12

        elif voicing_style == 'spread':
            # Wide voicing
            for i, note in enumerate(notes):
                notes[i] = note + (i * 7)  # Spread by fifths

        # Add bass note if specified (slash chord)
        if bass_note is not None:
            notes.insert(0, bass_note % 12 + base_octave - 12)

        return sorted(notes)

    def get_chord_substitutions(
        self,
        chord_root: int,
        chord_type: str,
    ) -> List[Tuple[int, str]]:
        """Get possible chord substitutions.

        Args:
            chord_root: Root note
            chord_type: Original chord type

        Returns:
            List of (new_root, new_type) substitutions
        """
        substitutions = []

        # Tritone substitution (for dominants)
        if 'dom' in chord_type:
            tritone_root = (chord_root + 6) % 12
            substitutions.append((tritone_root, chord_type))

        # Relative major/minor
        if 'min' in chord_type:
            relative_major = (chord_root + 3) % 12
            substitutions.append((relative_major, chord_type.replace('min', 'maj')))
        elif 'maj' in chord_type:
            relative_minor = (chord_root + 9) % 12
            substitutions.append((relative_minor, chord_type.replace('maj', 'min')))

        # Diminished substitution
        substitutions.append(((chord_root + 1) % 12, 'dim7'))

        return substitutions

    def reharmonize_progression(
        self,
        progression: List[Tuple[int, str]],
        style: str = 'jazz',
    ) -> List[Tuple[int, str]]:
        """Reharmonize a chord progression.

        Args:
            progression: Original progression [(root, type), ...]
            style: Reharmonization style ('jazz', 'modal', 'chromatic')

        Returns:
            Reharmonized progression
        """
        reharmonized = []

        for i, (root, chord_type) in enumerate(progression):
            if style == 'jazz':
                # Add ii-V before I chords
                if 'maj7' in chord_type and i > 0:
                    # Add ii-V leading to this chord
                    ii_root = (root + 2) % 12
                    v_root = (root + 7) % 12
                    reharmonized.append((ii_root, 'min7'))
                    reharmonized.append((v_root, 'dom7'))

            elif style == 'modal':
                # Use modal interchange
                if 'maj' in chord_type:
                    # Borrow from parallel minor
                    reharmonized.append((root, chord_type.replace('maj', 'min')))
                else:
                    reharmonized.append((root, chord_type))

            elif style == 'chromatic':
                # Add chromatic approach chords
                if i < len(progression) - 1:
                    next_root = progression[i + 1][0]
                    # Add chromatic approach
                    approach = (next_root - 1) % 12
                    reharmonized.append((root, chord_type))
                    reharmonized.append((approach, 'dom7'))
                else:
                    reharmonized.append((root, chord_type))

        return reharmonized


class VoiceLeading:
    """Voice leading rules for smooth melodic motion."""

    def __init__(self):
        """Initialize voice leading analyzer."""
        pass

    def analyze_voice_leading(
        self,
        chord1: List[int],
        chord2: List[int],
    ) -> Dict[str, any]:
        """Analyze voice leading between two chords.

        Args:
            chord1: First chord notes
            chord2: Second chord notes

        Returns:
            Analysis with smoothness score and issues
        """
        # Calculate total voice motion
        if len(chord1) != len(chord2):
            # Pad to same length
            max_len = max(len(chord1), len(chord2))
            chord1 = chord1 + [chord1[-1]] * (max_len - len(chord1))
            chord2 = chord2 + [chord2[-1]] * (max_len - len(chord2))

        # Find optimal voice mapping (minimal motion)
        import itertools
        min_motion = float('inf')
        best_mapping = None

        for perm in itertools.permutations(range(len(chord2))):
            motion = sum(abs(chord1[i] - chord2[j]) for i, j in enumerate(perm))
            if motion < min_motion:
                min_motion = motion
                best_mapping = perm

        # Calculate individual voice motions
        voice_motions = [chord2[best_mapping[i]] - chord1[i] for i in range(len(chord1))]

        # Detect issues
        issues = []

        # Parallel fifths/octaves
        for i in range(len(chord1) - 1):
            for j in range(i + 1, len(chord1)):
                interval1 = abs(chord1[j] - chord1[i])
                interval2 = abs(chord2[best_mapping[j]] - chord2[best_mapping[i]])

                if interval1 % 12 == 7 and interval2 % 12 == 7:
                    issues.append(f"Parallel fifths: voices {i} and {j}")
                if interval1 % 12 == 0 and interval2 % 12 == 0:
                    issues.append(f"Parallel octaves: voices {i} and {j}")

        # Large leaps
        for i, motion in enumerate(voice_motions):
            if abs(motion) > 12:  # Larger than octave
                issues.append(f"Large leap in voice {i}: {motion} semitones")

        # Smoothness score (inverse of total motion)
        smoothness = 1.0 / (1.0 + min_motion / len(chord1))

        return {
            'total_motion': min_motion,
            'average_motion': min_motion / len(chord1),
            'voice_motions': voice_motions,
            'smoothness_score': smoothness,
            'issues': issues,
            'has_issues': len(issues) > 0,
        }

    def optimize_voice_leading(
        self,
        chord_progression: List[List[int]],
    ) -> List[List[int]]:
        """Optimize voice leading for entire progression.

        Args:
            chord_progression: List of chords (each a list of notes)

        Returns:
            Optimized progression with smooth voice leading
        """
        if len(chord_progression) < 2:
            return chord_progression

        optimized = [chord_progression[0]]

        for i in range(1, len(chord_progression)):
            prev_chord = optimized[-1]
            current_chord = chord_progression[i]

            # Find best voicing of current chord
            best_voicing = None
            best_smoothness = -1

            # Try different octave displacements
            for displacement in range(-2, 3):
                test_voicing = [note + displacement * 12 for note in current_chord]

                analysis = self.analyze_voice_leading(prev_chord, test_voicing)

                if analysis['smoothness_score'] > best_smoothness and not analysis['has_issues']:
                    best_smoothness = analysis['smoothness_score']
                    best_voicing = test_voicing

            if best_voicing is None:
                best_voicing = current_chord

            optimized.append(best_voicing)

        return optimized


class ModalInterchange:
    """Modal interchange and borrowed chords."""

    # Modes and their characteristic intervals
    MODES = {
        'ionian': [0, 2, 4, 5, 7, 9, 11],      # Major
        'dorian': [0, 2, 3, 5, 7, 9, 10],
        'phrygian': [0, 1, 3, 5, 7, 8, 10],
        'lydian': [0, 2, 4, 6, 7, 9, 11],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'aeolian': [0, 2, 3, 5, 7, 8, 10],     # Natural minor
        'locrian': [0, 1, 3, 5, 6, 8, 10],
        # Additional scales
        'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
        'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
        'blues': [0, 3, 5, 6, 7, 10],
        'whole_tone': [0, 2, 4, 6, 8, 10],
        'diminished': [0, 2, 3, 5, 6, 8, 9, 11],  # Half-whole
    }

    def __init__(self):
        """Initialize modal interchange engine."""
        pass

    def get_borrowed_chords(
        self,
        key: int,
        source_mode: str = 'ionian',
        target_mode: str = 'aeolian',
    ) -> List[Tuple[int, str]]:
        """Get chords borrowed from parallel mode.

        Args:
            key: Key root note
            source_mode: Original mode
            target_mode: Mode to borrow from

        Returns:
            List of borrowed chord (root, quality) pairs
        """
        if source_mode not in self.MODES or target_mode not in self.MODES:
            return []

        source_scale = [(key + interval) % 12 for interval in self.MODES[source_mode]]
        target_scale = [(key + interval) % 12 for interval in self.MODES[target_mode]]

        # Find notes in target but not in source
        borrowed_notes = set(target_scale) - set(source_scale)

        # Build chords using borrowed notes
        borrowed_chords = []

        for note in borrowed_notes:
            # Determine chord quality based on target scale
            idx = target_scale.index(note)

            # Simple chord quality determination
            third_idx = (idx + 2) % len(target_scale)
            third = target_scale[third_idx]
            third_interval = (third - note) % 12

            if third_interval == 4:
                quality = 'major'
            elif third_interval == 3:
                quality = 'minor'
            else:
                quality = 'diminished'

            borrowed_chords.append((note, quality))

        return borrowed_chords

    def suggest_modal_substitution(
        self,
        original_note: int,
        key: int,
        target_mode: str = 'dorian',
    ) -> List[int]:
        """Suggest modal substitutions for a note.

        Args:
            original_note: Original note
            key: Key root
            target_mode: Target mode for substitution

        Returns:
            List of substitute notes
        """
        if target_mode not in self.MODES:
            return [original_note]

        scale = [(key + interval) % 12 for interval in self.MODES[target_mode]]

        # Find closest scale notes
        note_class = original_note % 12
        substitutes = []

        if note_class not in scale:
            # Find nearest scale notes
            distances = [(abs((note_class - s) % 12), s) for s in scale]
            distances.sort()

            # Return two closest
            substitutes = [s for _, s in distances[:2]]
        else:
            substitutes = [note_class]

        # Convert back to original octave
        octave = original_note // 12
        substitutes = [s + octave * 12 for s in substitutes]

        return substitutes


class SecondaryDominants:
    """Secondary dominant and diminished chords."""

    def __init__(self):
        """Initialize secondary dominants engine."""
        pass

    def get_secondary_dominant(
        self,
        target_chord_root: int,
        target_chord_type: str = 'major',
    ) -> Tuple[int, str]:
        """Get secondary dominant for target chord.

        Args:
            target_chord_root: Root of target chord
            target_chord_type: Type of target chord

        Returns:
            (root, type) of secondary dominant
        """
        # V of target = perfect fifth above
        dominant_root = (target_chord_root + 7) % 12

        # Always use dom7 for secondary dominants
        return (dominant_root, 'dom7')

    def get_secondary_diminished(
        self,
        target_chord_root: int,
    ) -> Tuple[int, str]:
        """Get secondary diminished (vii° of target).

        Args:
            target_chord_root: Root of target chord

        Returns:
            (root, type) of secondary diminished
        """
        # vii° of target = half step below
        dim_root = (target_chord_root - 1) % 12
        return (dim_root, 'dim7')

    def add_secondary_dominants(
        self,
        progression: List[Tuple[int, str]],
        frequency: float = 0.5,
    ) -> List[Tuple[int, str]]:
        """Add secondary dominants to progression.

        Args:
            progression: Original progression
            frequency: Probability of adding secondary dominant (0-1)

        Returns:
            Enhanced progression with secondary dominants
        """
        enhanced = []

        for i, (root, chord_type) in enumerate(progression):
            # Randomly add secondary dominant before this chord
            if i > 0 and np.random.random() < frequency:
                sec_dom = self.get_secondary_dominant(root, chord_type)
                enhanced.append(sec_dom)

            enhanced.append((root, chord_type))

        return enhanced


class Reharmonization:
    """Advanced reharmonization techniques."""

    def __init__(self):
        """Initialize reharmonization engine."""
        self.jazz = JazzHarmony()
        self.voice_leading = VoiceLeading()
        self.modal = ModalInterchange()
        self.secondary = SecondaryDominants()

    def reharmonize_melody(
        self,
        melody_notes: List[int],
        original_chords: List[Tuple[int, str]],
        style: str = 'jazz',
        complexity: float = 0.5,
    ) -> List[Tuple[int, str]]:
        """Reharmonize melody with new chords.

        Args:
            melody_notes: Melody note sequence
            original_chords: Original chord progression
            style: Reharmonization style
            complexity: Complexity level (0-1)

        Returns:
            New chord progression
        """
        new_chords = []

        for i, (root, chord_type) in enumerate(original_chords):
            # Get melody notes for this chord
            notes_per_chord = len(melody_notes) // len(original_chords)
            chord_melody = melody_notes[i * notes_per_chord:(i + 1) * notes_per_chord]

            if not chord_melody:
                new_chords.append((root, chord_type))
                continue

            # Choose reharmonization based on style and complexity
            if style == 'jazz' and complexity > 0.6:
                # Use altered dominants and extensions
                if 'dom' in chord_type:
                    new_type = np.random.choice(['dom7#9', 'dom7b9', 'dom7alt'])
                    new_chords.append((root, new_type))
                else:
                    # Add extensions
                    if complexity > 0.8:
                        new_type = chord_type.replace('7', '13')
                    elif complexity > 0.7:
                        new_type = chord_type.replace('7', '11')
                    else:
                        new_type = chord_type.replace('7', '9')
                    new_chords.append((root, new_type))

            elif style == 'modal' and complexity > 0.5:
                # Use modal interchange
                borrowed = self.modal.get_borrowed_chords(root)
                if borrowed and np.random.random() < complexity:
                    new_chords.append(borrowed[0])
                else:
                    new_chords.append((root, chord_type))

            elif style == 'chromatic' and complexity > 0.4:
                # Add chromatic passing chords
                new_chords.append((root, chord_type))
                if i < len(original_chords) - 1 and np.random.random() < complexity:
                    next_root = original_chords[i + 1][0]
                    passing = (root + next_root) // 2
                    new_chords.append((passing, 'dim7'))

            else:
                new_chords.append((root, chord_type))

        return new_chords

    def apply_all_techniques(
        self,
        progression: List[Tuple[int, str]],
        melody: Optional[List[int]] = None,
        complexity: float = 0.7,
    ) -> List[Tuple[int, str]]:
        """Apply multiple reharmonization techniques.

        Args:
            progression: Original progression
            melody: Optional melody
            complexity: Complexity level

        Returns:
            Fully reharmonized progression
        """
        # Start with original
        result = progression.copy()

        # Add secondary dominants
        if complexity > 0.5:
            result = self.secondary.add_secondary_dominants(result, frequency=complexity * 0.6)

        # Jazz reharmonization
        if complexity > 0.6:
            result = self.jazz.reharmonize_progression(result, style='jazz')

        # Modal substitutions
        if complexity > 0.7 and melody:
            result = self.reharmonize_melody(melody, result, style='modal', complexity=complexity)

        return result
