"""
Simple audio-only generation script
"""
import json
import time
from pathlib import Path

# Create output directory
output_dir = Path('output/audio')
output_dir.mkdir(parents=True, exist_ok=True)

print("🎵 LoFi Music Empire - Audio Only Generation")
print("=" * 60)

# Generate placeholder audio (in production, use your actual model)
print("\n🎼 Generating audio track...")
print("  Mood: chill")
print("  Duration: 180s")
print("  ⏳ Generating...")

# Simulate generation
import numpy as np
import soundfile as sf

# Generate simple sine wave as placeholder
sample_rate = 44100
duration = 180
t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note

# Add some variation (simple lofi-ish sound)
audio += 0.1 * np.sin(2 * np.pi * 220 * t)  # A3
audio += 0.05 * np.random.randn(len(audio))  # Noise

# Save
track_id = f"track_{int(time.time())}"
output_path = output_dir / f"{track_id}.wav"
sf.write(output_path, audio, sample_rate)

print(f"  ✅ Generated: {output_path}")
print(f"\n🎉 Success! Audio track created.")
print(f"\n📂 File location: {output_path.absolute()}")
print(f"\n▶️  Play it: start {output_path}")
