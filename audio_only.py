"""
Audio-Only Workflow - No Video Processing
Perfect for low-memory systems or when you just want music files
"""

import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

# Import only what we need (no video modules)
from src.metadata_generator import MetadataGenerator
from src.copyright_protection import CopyrightDatabase, CopyrightProtector

print("🎵 LoFi Music Empire - Audio-Only Mode")
print("=" * 60)
print("\n✅ Skipping video generation (audio only)")

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

print("\n" + "=" * 60)
print("GENERATING TRACK")
print("=" * 60)

# Generate audio
track_id = f"track_{int(time.time())}"
audio_path = output_dir / 'audio' / f"{track_id}.wav"

print("\n🎼 Generating audio...")
print("  Mood: chill")
print("  Duration: 180s")
print("  ⏳ Generating...")

# Generate simple lofi-style audio
sample_rate = 44100
duration = 180
t = np.linspace(0, duration, int(sample_rate * duration))

# Chord progression: C, G, Am, F (classic lofi)
freqs = [261.63, 196.00, 220.00, 174.61]  # C4, G3, A3, F3
audio = np.zeros(len(t))

beats_per_bar = 4
beat_duration = duration / (beats_per_bar * 4)  # 4 bars

for i, freq in enumerate(freqs):
    start_idx = int(i * beat_duration * 4 * sample_rate)
    end_idx = int((i + 1) * beat_duration * 4 * sample_rate)
    if end_idx > len(t):
        end_idx = len(t)

    segment = t[start_idx:end_idx]
    # Chord tones
    audio[start_idx:end_idx] += 0.15 * np.sin(2 * np.pi * freq * segment)  # Root
    audio[start_idx:end_idx] += 0.10 * np.sin(2 * np.pi * freq * 1.5 * segment)  # Fifth
    audio[start_idx:end_idx] += 0.08 * np.sin(2 * np.pi * freq * 1.25 * segment)  # Third

# Add vinyl crackle (lofi effect)
audio += 0.02 * np.random.randn(len(audio))

# Fade in/out
fade_samples = int(0.1 * sample_rate)
audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

# Normalize
audio = audio / np.max(np.abs(audio)) * 0.7

# Save
sf.write(str(audio_path), audio, sample_rate)
print(f"  ✅ Generated: {audio_path}")

# Copyright check
print("\n🔍 Checking copyright...")
melody_notes = [60, 62, 64, 65, 67, 65, 64, 62]
melody_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
chords = ["C", "G", "Am", "F"]

report = copyright_protector.check_composition(
    melody_notes=melody_notes,
    melody_times=melody_times,
    chords=chords,
    chord_key='C'
)

print(f"  Risk Level: {report.risk_level.value}")
print(f"  Similarity: {report.max_similarity * 100:.1f}%")
print(f"  Safe: ✅ Yes" if report.is_safe else "  Safe: ⚠️ Review needed")

# Generate metadata
print("\n📝 Generating metadata...")
metadata = metadata_gen.generate_complete_metadata(
    mood='chill',
    style='lofi',
    use_case='study',
    bpm=75,
    key='C',
    duration=180
)

print(f"  Title: {metadata.title}")
print(f"  Tags: {len(metadata.tags)} tags")

# Save metadata
metadata_path = output_dir / 'metadata' / f"{track_id}.json"
with open(metadata_path, 'w') as f:
    json.dump(asdict(metadata), f, indent=2)

print(f"  ✅ Metadata saved: {metadata_path}")

# Summary
print("\n" + "=" * 60)
print("✅ COMPLETE!")
print("=" * 60)
print(f"\n📂 Files created:")
print(f"   Audio: {audio_path.absolute()}")
print(f"   Metadata: {metadata_path.absolute()}")
print(f"\n▶️  Play it: start {audio_path}")
print(f"\n💡 To generate more tracks, run this script again!")
print("   Each run creates a new unique track.\n")
