"""
Simple MIDI Generator for LoFi Beats
Creates MIDI files with realistic lofi chord progressions and melodies
"""

import mido
from mido import Message, MidiFile, MidiTrack
import numpy as np
from typing import List, Tuple, Optional


class SimpleMidiGenerator:
    """Generate simple but musical MIDI files for lofi beats"""

    # Key to MIDI note mappings (root notes)
    KEY_MAP = {
        'C': 60, 'C#': 61, 'Db': 61,
        'D': 62, 'D#': 63, 'Eb': 63,
        'E': 64,
        'F': 65, 'F#': 66, 'Gb': 66,
        'G': 67, 'G#': 68, 'Ab': 68,
        'A': 69, 'A#': 70, 'Bb': 70,
        'B': 71,
        'Cm': 60, 'C#m': 61, 'Dm': 62, 'D#m': 63, 'Em': 64,
        'Fm': 65, 'F#m': 66, 'Gm': 67, 'G#m': 68, 'Am': 69, 'A#m': 70, 'Bm': 71
    }

    # Chord progressions for different moods
    PROGRESSIONS = {
        'chill': [
            [0, 4, 7],      # I (major)
            [-5, -1, 2],    # IV (major)
            [-3, 0, 4],     # vi (minor)
            [-5, -1, 2],    # IV (major)
        ],
        'melancholic': [
            [0, 3, 7],      # i (minor)
            [-5, -2, 2],    # iv (minor)
            [-3, 0, 4],     # VI (major)
            [-5, -2, 2],    # iv (minor)
        ],
        'upbeat': [
            [0, 4, 7],      # I
            [-3, 0, 4],     # vi
            [-5, -1, 2],    # IV
            [-7, -3, 0],    # V
        ],
        'dreamy': [
            [0, 4, 7, 11],  # Imaj7
            [-3, 0, 4, 7],  # vi7
            [-5, -1, 2, 7], # IVmaj7
            [-3, 0, 4, 7],  # vi7
        ],
        'relaxed': [
            [0, 4, 7],      # I
            [2, 5, 9],      # ii
            [-5, -1, 2],    # IV
            [0, 4, 7],      # I
        ]
    }

    def __init__(self, key: str = 'C', mood: str = 'chill', tempo: int = 75):
        """
        Initialize MIDI generator

        Args:
            key: Musical key (e.g., 'C', 'Am', 'F#')
            mood: Mood/style (chill, melancholic, upbeat, dreamy, relaxed)
            tempo: BPM
        """
        self.key = key
        self.root_note = self.KEY_MAP.get(key, 69)  # Default to A
        self.mood = mood if mood in self.PROGRESSIONS else 'chill'
        self.tempo = tempo

    def generate(self, duration: int = 180, output_path: Optional[str] = None) -> pretty_midi.PrettyMIDI:
        """
        Generate a complete MIDI file

        Args:
            duration: Duration in seconds
            output_path: Optional path to save MIDI file

        Returns:
            PrettyMIDI object
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.tempo)

        # Create instruments
        piano = pretty_midi.Instrument(program=0, name='Piano')  # Acoustic Grand Piano
        bass = pretty_midi.Instrument(program=33, name='Bass')   # Electric Bass (finger)

        # Calculate timing
        beat_duration = 60.0 / self.tempo
        bar_duration = beat_duration * 4
        num_bars = int(duration / bar_duration)

        progression = self.PROGRESSIONS[self.mood]

        # Generate chords and bass
        for bar in range(num_bars):
            chord_index = bar % len(progression)
            chord_intervals = progression[chord_index]
            start_time = bar * bar_duration

            # Add chord notes (held for full bar)
            for interval in chord_intervals:
                note_number = self.root_note + interval + 12  # Octave up for chords
                note = pretty_midi.Note(
                    velocity=60 + np.random.randint(-10, 10),
                    pitch=note_number,
                    start=start_time,
                    end=start_time + bar_duration * 0.95  # Slight gap
                )
                piano.notes.append(note)

            # Add bass note (root of chord, octave down)
            bass_note = self.root_note + chord_intervals[0] - 12
            bass_pattern = self._generate_bass_pattern(bass_note, start_time, bar_duration, beat_duration)
            bass.notes.extend(bass_pattern)

            # Add melody notes occasionally
            if bar % 2 == 1 or self.mood == 'upbeat':
                melody_notes = self._generate_melody(chord_intervals, start_time, bar_duration, beat_duration)
                piano.notes.extend(melody_notes)

        midi.instruments.append(piano)
        midi.instruments.append(bass)

        if output_path:
            midi.write(output_path)

        return midi

    def _generate_bass_pattern(
        self,
        root: int,
        start_time: float,
        bar_duration: float,
        beat_duration: float
    ) -> List[pretty_midi.Note]:
        """Generate bass pattern for a bar"""
        notes = []

        # Simple pattern: root on beat 1 and 3
        for beat in [0, 2]:
            note = pretty_midi.Note(
                velocity=80 + np.random.randint(-5, 5),
                pitch=root,
                start=start_time + beat * beat_duration,
                end=start_time + beat * beat_duration + beat_duration * 0.8
            )
            notes.append(note)

        return notes

    def _generate_melody(
        self,
        chord_intervals: List[int],
        start_time: float,
        bar_duration: float,
        beat_duration: float
    ) -> List[pretty_midi.Note]:
        """Generate simple melody over chord"""
        notes = []

        # Use pentatonic scale based on chord
        scale = [0, 2, 4, 7, 9]  # Major pentatonic relative to root

        # Generate 4-8 melody notes per bar
        num_notes = np.random.randint(4, 9)

        for i in range(num_notes):
            # Random timing within bar
            note_start = start_time + np.random.uniform(0, bar_duration - beat_duration * 0.5)
            note_duration = beat_duration * np.random.choice([0.25, 0.5, 0.75, 1.0])

            # Choose note from scale, 2 octaves up
            scale_degree = np.random.choice(scale)
            pitch = self.root_note + scale_degree + 24

            # Occasionally use chord tones for more consonance
            if np.random.random() < 0.4:
                pitch = self.root_note + np.random.choice(chord_intervals) + 24

            note = pretty_midi.Note(
                velocity=50 + np.random.randint(-10, 20),
                pitch=pitch,
                start=note_start,
                end=min(note_start + note_duration, start_time + bar_duration)
            )
            notes.append(note)

        return notes


def generate_lofi_midi(
    key: str = 'Am',
    mood: str = 'chill',
    tempo: int = 75,
    duration: int = 180,
    output_path: str = 'output.mid'
) -> str:
    """
    Convenience function to generate a lofi MIDI file

    Args:
        key: Musical key
        mood: Mood/style
        tempo: BPM
        duration: Duration in seconds
        output_path: Where to save the MIDI file

    Returns:
        Path to generated MIDI file
    """
    generator = SimpleMidiGenerator(key=key, mood=mood, tempo=tempo)
    generator.generate(duration=duration, output_path=output_path)
    return output_path


if __name__ == '__main__':
    # Test generation
    print("Generating test MIDI file...")
    generate_lofi_midi(
        key='Am',
        mood='chill',
        tempo=75,
        duration=30,
        output_path='test_lofi.mid'
    )
    print("✅ Generated test_lofi.mid")
