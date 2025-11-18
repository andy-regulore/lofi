"""
Batch Audio Generation - Generate Multiple Tracks
No video processing - perfect for building your music library fast
"""

import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import sys

# Import only what we need
from src.metadata_generator import MetadataGenerator
from src.copyright_protection import CopyrightDatabase, CopyrightProtector

# Configuration
NUM_TRACKS = 10  # Change this to generate more/fewer tracks
MOODS = ['chill', 'melancholic', 'upbeat', 'relaxed', 'dreamy']
KEYS = ['C', 'Am', 'F', 'G', 'Dm', 'Em']

print("🎵 LoFi Music Empire - Batch Audio Generation")
print("=" * 60)
print(f"\n📊 Generating {NUM_TRACKS} tracks...")
print("✅ Audio-only mode (no video processing)\n")

# Load config
with open('config.json') as f:
    config = json.load(f)

# Setup
output_dir = Path('output')
(output_dir / 'audio').mkdir(parents=True, exist_ok=True)
(output_dir / 'metadata').mkdir(parents=True, exist_ok=True)

# Initialize modules
metadata_gen = MetadataGenerator()
copyright_db = CopyrightDatabase('data/copyright.db')
copyright_protector = CopyrightProtector(copyright_db)

# Track generation
successful_tracks = []

for i in range(NUM_TRACKS):
    print(f"\n{'=' * 60}")
    print(f"TRACK {i+1}/{NUM_TRACKS}")
    print("=" * 60)

    # Vary the parameters for each track
    mood = MOODS[i % len(MOODS)]
    key = KEYS[i % len(KEYS)]

    # Generate audio
    track_id = f"track_{int(time.time())}_{i+1:03d}"
    audio_path = output_dir / 'audio' / f"{track_id}.wav"

    print(f"\n🎼 Generating audio...")
    print(f"  Mood: {mood}")
    print(f"  Key: {key}")
    print(f"  Duration: 180s")

    try:
        # Generate simple lofi-style audio
        sample_rate = 44100
        duration = 180
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Different chord progressions based on mood
        if mood == 'chill':
            freqs = [261.63, 196.00, 220.00, 174.61]  # C, G, Am, F
        elif mood == 'melancholic':
            freqs = [220.00, 174.61, 196.00, 146.83]  # Am, F, G, D
        elif mood == 'upbeat':
            freqs = [261.63, 293.66, 196.00, 261.63]  # C, D, G, C
        else:
            freqs = [261.63, 220.00, 174.61, 196.00]  # C, Am, F, G

        audio = np.zeros(len(t))
        beats_per_bar = 4
        beat_duration = duration / (beats_per_bar * 4)

        for j, freq in enumerate(freqs):
            start_idx = int(j * beat_duration * 4 * sample_rate)
            end_idx = int((j + 1) * beat_duration * 4 * sample_rate)
            if end_idx > len(t):
                end_idx = len(t)

            segment = t[start_idx:end_idx]
            # Add some variation to each track
            volume = 0.15 + (i * 0.01)
            audio[start_idx:end_idx] += volume * np.sin(2 * np.pi * freq * segment)
            audio[start_idx:end_idx] += (volume * 0.7) * np.sin(2 * np.pi * freq * 1.5 * segment)
            audio[start_idx:end_idx] += (volume * 0.5) * np.sin(2 * np.pi * freq * 1.25 * segment)

        # Vinyl crackle
        audio += (0.02 + i * 0.002) * np.random.randn(len(audio))

        # Fade in/out
        fade_samples = int(0.1 * sample_rate)
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.7

        # Save
        sf.write(str(audio_path), audio, sample_rate)
        print(f"  ✅ Generated: {audio_path.name}")

        # Copyright check
        melody_notes = [60, 62, 64, 65, 67, 65, 64, 62]
        melody_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        chords = ["C", "G", "Am", "F"]

        report = copyright_protector.check_composition(
            melody_notes=melody_notes,
            melody_times=melody_times,
            chords=chords,
            chord_key=key
        )
        print(f"  Copyright: {report.risk_level.value} ({report.max_similarity * 100:.1f}% similarity)")

        # Generate metadata
        metadata = metadata_gen.generate_complete_metadata(
            mood=mood,
            style='lofi',
            use_case='study',
            bpm=75,
            key=key,
            duration=180
        )

        # Save metadata
        metadata_path = output_dir / 'metadata' / f"{track_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        print(f"  Title: {metadata.title}")
        print(f"  ✅ Track {i+1} complete!")

        successful_tracks.append({
            'track_id': track_id,
            'audio_path': str(audio_path),
            'title': metadata.title,
            'mood': mood,
            'key': key
        })

        # Small delay to ensure unique timestamps
        time.sleep(0.1)

    except Exception as e:
        print(f"  ❌ Error generating track {i+1}: {e}")
        continue

# Final summary
print("\n" + "=" * 60)
print("✅ BATCH GENERATION COMPLETE!")
print("=" * 60)
print(f"\n📊 Successfully generated: {len(successful_tracks)}/{NUM_TRACKS} tracks")
print(f"\n📂 All files saved to:")
print(f"   Audio: {(output_dir / 'audio').absolute()}")
print(f"   Metadata: {(output_dir / 'metadata').absolute()}")

if successful_tracks:
    print(f"\n🎵 Generated tracks:")
    for track in successful_tracks:
        print(f"   • {track['title']} ({track['mood']}, {track['key']})")

print(f"\n▶️  Open folder: start {(output_dir / 'audio').absolute()}")
print("\n💡 Edit NUM_TRACKS at the top of this script to generate more!")
