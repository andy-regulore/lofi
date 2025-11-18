"""
Enhanced Audio Generation with LoFi Effects
Much better quality than the basic version!
"""

import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import asdict

# Import the good stuff
from src.metadata_generator import MetadataGenerator
from src.copyright_protection import CopyrightDatabase, CopyrightProtector
from src.lofi_effects import LoFiEffectsChain
from src.ambient_sounds import AmbientSoundGenerator

# ============================================================================
# CONFIGURATION - EDIT THESE TO CHANGE YOUR MUSIC!
# ============================================================================

MOOD = 'chill'           # Options: chill, melancholic, upbeat, relaxed, dreamy
THEME = 'urban_chill'    # Options: rain, cafe, urban_chill, nature, plain
LOFI_PRESET = 'medium'   # Options: light, medium, heavy (how much lofi effect)
DURATION = 180           # Seconds
KEY = 'Am'               # Musical key

print("🎵 LoFi Music Empire - Enhanced Audio Generation")
print("=" * 60)
print(f"\n🎨 Settings:")
print(f"   Mood: {MOOD}")
print(f"   Theme: {THEME}")
print(f"   LoFi Effect: {LOFI_PRESET}")
print(f"   Duration: {DURATION}s")
print(f"   Key: {KEY}\n")

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
lofi_effects = LoFiEffectsChain()
ambient_gen = AmbientSoundGenerator()

print("\n" + "=" * 60)
print("GENERATING ENHANCED TRACK")
print("=" * 60)

# Generate audio
track_id = f"track_{int(time.time())}"
audio_path = output_dir / 'audio' / f"{track_id}.wav"

print("\n🎼 Step 1: Generating base music...")

# Generate base music with better harmony
sample_rate = 44100
t = np.linspace(0, DURATION, int(sample_rate * DURATION))

# Different chord progressions by mood
chord_progressions = {
    'chill': ([261.63, 196.00, 220.00, 174.61], "C-G-Am-F"),
    'melancholic': ([220.00, 174.61, 196.00, 146.83], "Am-F-G-D"),
    'upbeat': ([261.63, 293.66, 196.00, 261.63], "C-D-G-C"),
    'relaxed': ([174.61, 261.63, 196.00, 220.00], "F-C-G-Am"),
    'dreamy': ([220.00, 196.00, 174.61, 261.63], "Am-G-F-C")
}

freqs, prog_name = chord_progressions.get(MOOD, chord_progressions['chill'])
print(f"   Progression: {prog_name}")

audio = np.zeros(len(t))
beats_per_bar = 4
beat_duration = DURATION / (beats_per_bar * 4)

# Add chords with more harmonics
for i, freq in enumerate(freqs):
    start_idx = int(i * beat_duration * 4 * sample_rate)
    end_idx = int((i + 1) * beat_duration * 4 * sample_rate)
    if end_idx > len(t):
        end_idx = len(t)

    segment = t[start_idx:end_idx]

    # Root note
    audio[start_idx:end_idx] += 0.20 * np.sin(2 * np.pi * freq * segment)
    # Fifth
    audio[start_idx:end_idx] += 0.15 * np.sin(2 * np.pi * freq * 1.5 * segment)
    # Third
    audio[start_idx:end_idx] += 0.12 * np.sin(2 * np.pi * freq * 1.25 * segment)
    # Octave (adds depth)
    audio[start_idx:end_idx] += 0.08 * np.sin(2 * np.pi * freq * 0.5 * segment)

    # Add subtle melody on top
    melody_freq = freq * 2  # One octave up
    melody = 0.10 * np.sin(2 * np.pi * melody_freq * segment + np.sin(segment * 0.5))
    audio[start_idx:end_idx] += melody

# Normalize before effects
audio = audio / np.max(np.abs(audio))

print("   ✅ Base music generated")

# Step 2: Add ambient sounds based on theme
print(f"\n🌧️  Step 2: Adding {THEME} ambience...")

if THEME == 'rain':
    ambient = ambient_gen.generate_rain(DURATION, intensity='medium', include_thunder=False)
    mix_level = 0.15
elif THEME == 'cafe':
    ambient = ambient_gen.generate_cafe(DURATION, crowd_level='medium')
    mix_level = 0.12
elif THEME == 'urban_chill':
    # Mix of city sounds - distant traffic, occasional horn
    ambient = np.random.randn(len(audio)) * 0.03  # Base city noise
    # Add occasional car pass (low frequency rumble)
    for _ in range(5):
        pos = np.random.randint(0, len(audio) - sample_rate * 3)
        car_sound = np.sin(2 * np.pi * 80 * np.arange(sample_rate * 3) / sample_rate)
        car_sound *= np.exp(-np.arange(sample_rate * 3) / sample_rate)  # Fade out
        if pos + len(car_sound) <= len(ambient):
            ambient[pos:pos+len(car_sound)] += car_sound * 0.05
    mix_level = 0.10
elif THEME == 'nature':
    ambient = ambient_gen.generate_nature(DURATION, environment='forest')
    mix_level = 0.15
else:  # plain - no ambient
    ambient = np.zeros(len(audio))
    mix_level = 0.0

# Mix ambient with music
if mix_level > 0:
    # Ensure same length
    if len(ambient) > len(audio):
        ambient = ambient[:len(audio)]
    elif len(ambient) < len(audio):
        ambient = np.pad(ambient, (0, len(audio) - len(ambient)))

    audio = audio * (1 - mix_level) + ambient * mix_level
    print(f"   ✅ Added {THEME} ambience (mix: {mix_level*100:.0f}%)")
else:
    print(f"   ⏭️  Skipped ambient (plain theme)")

# Step 3: Apply LoFi effects
print(f"\n🎛️  Step 3: Applying LoFi effects ({LOFI_PRESET})...")

audio_lofi = lofi_effects.process_full_chain(audio, preset=LOFI_PRESET, sample_rate=sample_rate)
print("   ✅ Vinyl crackle added")
print("   ✅ Bit crushing applied")
print("   ✅ Wow & flutter added")
print("   ✅ Tape saturation applied")

# Fade in/out
fade_samples = int(0.5 * sample_rate)  # 0.5 second fade
audio_lofi[:fade_samples] *= np.linspace(0, 1, fade_samples)
audio_lofi[-fade_samples:] *= np.linspace(1, 0, fade_samples)

# Final normalize
audio_lofi = audio_lofi / np.max(np.abs(audio_lofi)) * 0.75

# Save
sf.write(str(audio_path), audio_lofi, sample_rate)
print(f"\n💾 Step 4: Saved audio")
print(f"   File: {audio_path}")
print(f"   Size: {audio_path.stat().st_size / 1024 / 1024:.2f} MB")

# Copyright check
print("\n🔍 Step 5: Copyright check...")
melody_notes = [60, 62, 64, 65, 67, 65, 64, 62]
melody_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
chords = ["C", "G", "Am", "F"]

report = copyright_protector.check_composition(
    melody_notes=melody_notes,
    melody_times=melody_times,
    chords=chords,
    chord_key=KEY
)

print(f"   Risk Level: {report.risk_level.value}")
print(f"   Similarity: {report.max_similarity * 100:.1f}%")
print(f"   Safe: ✅ Yes" if report.is_safe else "   Safe: ⚠️ Review needed")

# Generate metadata
print("\n📝 Step 6: Generating metadata...")
metadata = metadata_gen.generate_complete_metadata(
    mood=MOOD,
    style='lofi',
    use_case='study',
    bpm=75,
    key=KEY,
    duration=DURATION
)

print(f"   Title: {metadata.title}")
print(f"   Description: {metadata.description[:100]}...")
print(f"   Tags: {len(metadata.tags)} tags")

# Save metadata
metadata_path = output_dir / 'metadata' / f"{track_id}.json"
with open(metadata_path, 'w') as f:
    json.dump(asdict(metadata), f, indent=2)

# Summary
print("\n" + "=" * 60)
print("✅ ENHANCED TRACK COMPLETE!")
print("=" * 60)
print(f"\n📂 Files:")
print(f"   🎵 Audio: {audio_path.absolute()}")
print(f"   📄 Metadata: {metadata_path.absolute()}")
print(f"\n▶️  Play it: start {audio_path}")
print(f"\n💡 To customize:")
print(f"   • Edit settings at the top of this script")
print(f"   • Try different MOOD, THEME, LOFI_PRESET combinations")
print(f"   • Experiment with DURATION and KEY\n")
