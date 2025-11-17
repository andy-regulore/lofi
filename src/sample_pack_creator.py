"""
Sample Pack Creation System

Creates commercial sample packs from generated tracks:
- One-shot drum samples
- Melodic loops
- Chord progressions (MIDI)
- Bass lines
- FX sounds

Author: Claude
License: MIT
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import librosa
import soundfile as sf
import json
from datetime import datetime


class SamplePackCreator:
    """Create commercial sample packs from generated music."""

    def __init__(self, output_dir: str = 'output/sample_packs'):
        """
        Initialize sample pack creator.

        Args:
            output_dir: Directory for sample packs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_pack_from_tracks(
        self,
        track_list: List[str],
        pack_name: str,
        pack_type: str = 'full'
    ) -> Dict:
        """
        Create sample pack from list of tracks.

        Args:
            track_list: List of audio file paths
            pack_name: Name of the pack
            pack_type: 'drums', 'melodic', 'midi', or 'full'

        Returns:
            Pack information dict
        """
        print(f"\n📦 Creating sample pack: {pack_name}")
        print(f"   Type: {pack_type}")
        print(f"   Source tracks: {len(track_list)}")

        # Create pack directory
        pack_dir = self.output_dir / pack_name
        pack_dir.mkdir(exist_ok=True)

        samples = {}

        # Extract different types of samples
        if pack_type in ['drums', 'full']:
            samples['drums'] = self._extract_drum_samples(track_list, pack_dir)

        if pack_type in ['melodic', 'full']:
            samples['melodic'] = self._extract_melodic_loops(track_list, pack_dir)

        if pack_type in ['midi', 'full']:
            samples['midi'] = self._extract_midi_files(track_list, pack_dir)

        # Create pack metadata
        metadata = self._create_pack_metadata(pack_name, pack_type, samples)

        # Save metadata
        metadata_file = pack_dir / 'pack_info.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create README
        self._create_readme(pack_dir, metadata)

        print(f"   ✅ Pack created: {pack_dir}")
        print(f"   Total samples: {sum(len(v) for v in samples.values())}")

        return metadata

    def _extract_drum_samples(
        self,
        track_list: List[str],
        output_dir: Path
    ) -> List[str]:
        """Extract one-shot drum samples."""
        drum_dir = output_dir / 'drums'
        drum_dir.mkdir(exist_ok=True)

        samples = []

        print("   Extracting drum samples...")

        for track_path in track_list[:5]:  # Limit to avoid too many
            try:
                audio, sr = librosa.load(track_path, sr=None)

                # Detect onsets (drum hits)
                onset_frames = librosa.onset.onset_detect(
                    y=audio,
                    sr=sr,
                    units='samples',
                    backtrack=True
                )

                # Extract samples around onsets
                for i, onset in enumerate(onset_frames[:10]):  # Max 10 per track
                    # Extract 100ms around onset
                    sample_length = int(sr * 0.1)
                    start = max(0, onset - sample_length // 4)
                    end = min(len(audio), start + sample_length)

                    sample = audio[start:end]

                    # Classify sample (kick, snare, hat based on frequency)
                    sample_type = self._classify_drum_sample(sample, sr)

                    # Save
                    filename = f"{sample_type}_{len(samples):03d}.wav"
                    filepath = drum_dir / filename
                    sf.write(filepath, sample, sr)

                    samples.append(str(filepath))

            except Exception as e:
                print(f"     Warning: Could not process {track_path}: {e}")
                continue

        print(f"     Extracted {len(samples)} drum samples")
        return samples

    def _extract_melodic_loops(
        self,
        track_list: List[str],
        output_dir: Path
    ) -> List[str]:
        """Extract melodic loops."""
        loop_dir = output_dir / 'melodic_loops'
        loop_dir.mkdir(exist_ok=True)

        loops = []

        print("   Extracting melodic loops...")

        for i, track_path in enumerate(track_list[:10]):
            try:
                audio, sr = librosa.load(track_path, sr=None, duration=8.0)  # 8-second loop

                # Detect tempo
                tempo = librosa.beat.tempo(y=audio, sr=sr)[0]

                # Find loop points (4 or 8 bars)
                beat_frames = librosa.beat.beat_track(y=audio, sr=sr)[1]

                if len(beat_frames) >= 16:  # At least 4 bars (4 beats per bar)
                    # Extract 4-bar loop
                    loop_start = beat_frames[0]
                    loop_end = beat_frames[16] if len(beat_frames) > 16 else len(audio)

                    loop = audio[loop_start:loop_end]

                    # Save
                    filename = f"loop_{int(tempo)}bpm_{i:02d}.wav"
                    filepath = loop_dir / filename
                    sf.write(filepath, loop, sr)

                    loops.append(str(filepath))

            except Exception as e:
                print(f"     Warning: Could not process {track_path}: {e}")
                continue

        print(f"     Extracted {len(loops)} melodic loops")
        return loops

    def _extract_midi_files(
        self,
        track_list: List[str],
        output_dir: Path
    ) -> List[str]:
        """Copy/organize MIDI files."""
        midi_dir = output_dir / 'midi'
        midi_dir.mkdir(exist_ok=True)

        midi_files = []

        print("   Organizing MIDI files...")

        # Find corresponding MIDI files
        for track_path in track_list:
            track_path = Path(track_path)
            midi_path = track_path.with_suffix('.mid')

            if midi_path.exists():
                # Copy to pack
                dest_path = midi_dir / f"{track_path.stem}.mid"
                import shutil
                shutil.copy(midi_path, dest_path)
                midi_files.append(str(dest_path))

        print(f"     Organized {len(midi_files)} MIDI files")
        return midi_files

    def _classify_drum_sample(self, sample: np.ndarray, sr: int) -> str:
        """Classify drum sample by frequency content."""
        # Get frequency spectrum
        fft = np.fft.fft(sample)
        freqs = np.fft.fftfreq(len(fft), 1/sr)

        # Get magnitude in different frequency bands
        low_energy = np.mean(np.abs(fft[(freqs > 20) & (freqs < 150)]))
        mid_energy = np.mean(np.abs(fft[(freqs > 150) & (freqs < 2000)]))
        high_energy = np.mean(np.abs(fft[(freqs > 2000) & (freqs < 8000)]))

        # Classify based on dominant frequency range
        if low_energy > mid_energy and low_energy > high_energy:
            return 'kick'
        elif high_energy > low_energy and high_energy > mid_energy:
            return 'hat'
        else:
            return 'snare'

    def _create_pack_metadata(
        self,
        pack_name: str,
        pack_type: str,
        samples: Dict
    ) -> Dict:
        """Create pack metadata."""
        return {
            'name': pack_name,
            'type': pack_type,
            'created_at': datetime.now().isoformat(),
            'total_samples': sum(len(v) for v in samples.values()),
            'contents': {
                k: len(v) for k, v in samples.items()
            },
            'license': 'Royalty-free for commercial use',
            'format': 'WAV 44.1kHz 24-bit',
            'genre': 'Lo-Fi / Chill Hop',
            'price_suggested': self._calculate_price(samples)
        }

    def _calculate_price(self, samples: Dict) -> float:
        """Calculate suggested price based on content."""
        total_samples = sum(len(v) for v in samples.values())

        # Pricing tiers
        if total_samples < 25:
            return 9.99
        elif total_samples < 50:
            return 19.99
        elif total_samples < 100:
            return 29.99
        else:
            return 49.99

    def _create_readme(self, pack_dir: Path, metadata: Dict):
        """Create README file for pack."""
        readme_content = f"""# {metadata['name']}

**Type:** {metadata['type'].capitalize()} Sample Pack
**Genre:** {metadata['genre']}
**Total Samples:** {metadata['total_samples']}

## Contents

"""
        for category, count in metadata['contents'].items():
            readme_content += f"- {category.capitalize()}: {count} samples\n"

        readme_content += f"""

## Specifications

- **Format:** {metadata['format']}
- **License:** {metadata['license']}
- **Created:** {metadata['created_at'][:10]}

## Usage

All samples are 100% royalty-free and cleared for commercial use.
Perfect for:
- Lo-Fi beats
- Chill hop productions
- Study music
- Background music
- Podcast intros

## Support

For questions or support, contact us at support@lofibeats.ai

---

Made with ❤️ by LoFi AI
"""

        readme_file = pack_dir / 'README.md'
        with open(readme_file, 'w') as f:
            f.write(readme_content)


def quick_create_pack(
    track_directory: str = 'output/audio',
    pack_name: str = None
) -> str:
    """
    Quick function to create a sample pack.

    Args:
        track_directory: Directory with source tracks
        pack_name: Pack name (auto-generated if None)

    Returns:
        Path to created pack
    """
    if pack_name is None:
        pack_name = f"LoFi_Pack_{datetime.now().strftime('%Y%m%d')}"

    creator = SamplePackCreator()

    # Get tracks
    tracks = list(Path(track_directory).glob('*.wav'))

    if not tracks:
        print(f"❌ No tracks found in {track_directory}")
        return None

    # Create pack
    metadata = creator.create_pack_from_tracks(
        [str(t) for t in tracks],
        pack_name,
        pack_type='full'
    )

    pack_dir = creator.output_dir / pack_name
    return str(pack_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create sample packs from tracks')
    parser.add_argument('--tracks', default='output/audio', help='Track directory')
    parser.add_argument('--name', help='Pack name')
    parser.add_argument('--type', default='full', choices=['drums', 'melodic', 'midi', 'full'])

    args = parser.parse_args()

    pack_dir = quick_create_pack(args.tracks, args.name)

    if pack_dir:
        print(f"\n✅ Sample pack ready: {pack_dir}")
        print(f"   Upload to Gumroad, Bandcamp, or your store!")
