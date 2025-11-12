"""Audio processing module for lo-fi effects.

Handles:
- MIDI to WAV conversion
- Lo-fi audio effects (vinyl, tape, bit reduction, filtering)
- Audio normalization for YouTube/Spotify
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import subprocess
import os

import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from pedalboard import Pedalboard, Compressor, LowpassFilter, HighpassFilter, Bitcrush
import pyloudnorm as pyln

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoFiAudioProcessor:
    """Audio processor for creating lo-fi sound."""

    def __init__(self, config: Dict):
        """Initialize audio processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.audio_config = config['audio']
        self.lofi_config = self.audio_config['lofi_effects']
        self.sample_rate = self.audio_config['sample_rate']

    def midi_to_wav(
        self,
        midi_path: str,
        output_path: str,
        soundfont_path: Optional[str] = None
    ) -> bool:
        """Convert MIDI to WAV using FluidSynth.

        Args:
            midi_path: Path to MIDI file
            output_path: Path to save WAV file
            soundfont_path: Path to soundfont file (uses default if None)

        Returns:
            True if successful, False otherwise
        """
        soundfont = soundfont_path or self.audio_config.get('soundfont_path')

        if not soundfont or not Path(soundfont).exists():
            logger.warning(f"Soundfont not found at {soundfont}, using synthesized audio")
            return self._synthesize_midi_to_wav(midi_path, output_path)

        try:
            # Use FluidSynth to render MIDI
            cmd = [
                'fluidsynth',
                '-ni',  # Non-interactive
                soundfont,
                midi_path,
                '-F', output_path,
                '-r', str(self.sample_rate),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                logger.info(f"Converted MIDI to WAV: {output_path}")
                return True
            else:
                logger.error(f"FluidSynth error: {result.stderr}")
                return False

        except FileNotFoundError:
            logger.warning("FluidSynth not found, using fallback synthesis")
            return self._synthesize_midi_to_wav(midi_path, output_path)
        except Exception as e:
            logger.error(f"Error converting MIDI to WAV: {e}")
            return False

    def _synthesize_midi_to_wav(self, midi_path: str, output_path: str) -> bool:
        """Synthesize MIDI to WAV using pretty_midi (fallback).

        Args:
            midi_path: Path to MIDI file
            output_path: Path to save WAV file

        Returns:
            True if successful, False otherwise
        """
        try:
            import pretty_midi

            midi = pretty_midi.PrettyMIDI(str(midi_path))
            audio = midi.fluidsynth(fs=self.sample_rate)

            # Normalize
            audio = audio / np.max(np.abs(audio))

            sf.write(output_path, audio, self.sample_rate)
            logger.info(f"Synthesized MIDI to WAV: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error synthesizing MIDI: {e}")
            return False

    def apply_lofi_effects(
        self,
        input_path: str,
        output_path: str
    ) -> bool:
        """Apply lo-fi effects to audio file.

        Effects applied:
        - Sample rate reduction
        - Low-pass and high-pass filtering
        - Bit reduction
        - Vinyl crackle
        - Tape wow/flutter
        - Compression

        Args:
            input_path: Path to input WAV file
            output_path: Path to save processed WAV file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=False)

            # Ensure stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio])

            logger.info("Applying lo-fi effects...")

            # 1. Downsample for vintage sound
            downsample_rate = self.lofi_config['downsample_rate']
            if downsample_rate < self.sample_rate:
                audio = librosa.resample(
                    audio,
                    orig_sr=self.sample_rate,
                    target_sr=downsample_rate
                )
                audio = librosa.resample(
                    audio,
                    orig_sr=downsample_rate,
                    target_sr=self.sample_rate
                )

            # 2. Apply filters using pedalboard
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=self.lofi_config['highpass_cutoff']),
                LowpassFilter(cutoff_frequency_hz=self.lofi_config['lowpass_cutoff']),
                Bitcrush(bit_depth=self.lofi_config['bit_depth']),
                Compressor(
                    threshold_db=self.lofi_config['compression']['threshold_db'],
                    ratio=self.lofi_config['compression']['ratio'],
                ),
            ])

            audio = board(audio, self.sample_rate)

            # 3. Add vinyl crackle
            if self.lofi_config['vinyl_crackle']['enabled']:
                audio = self._add_vinyl_crackle(
                    audio,
                    intensity=self.lofi_config['vinyl_crackle']['intensity']
                )

            # 4. Add tape wow/flutter
            if self.lofi_config['tape_wow_flutter']['enabled']:
                audio = self._add_tape_effects(
                    audio,
                    depth=self.lofi_config['tape_wow_flutter']['depth']
                )

            # 5. Final normalization
            audio = self._normalize_audio(audio)

            # Save
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)

            sf.write(output_path, audio.T, self.sample_rate)
            logger.info(f"Saved lo-fi audio to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error applying lo-fi effects: {e}")
            return False

    def _add_vinyl_crackle(self, audio: np.ndarray, intensity: float = 0.015) -> np.ndarray:
        """Add vinyl crackle noise.

        Args:
            audio: Audio array
            intensity: Crackle intensity

        Returns:
            Audio with vinyl crackle
        """
        # Generate crackle (sparse impulse noise)
        crackle = np.random.poisson(0.001, audio.shape) * np.random.randn(*audio.shape)
        crackle = crackle * intensity

        return audio + crackle

    def _add_tape_effects(self, audio: np.ndarray, depth: float = 0.002) -> np.ndarray:
        """Add tape wow and flutter (pitch modulation).

        Args:
            audio: Audio array
            depth: Modulation depth

        Returns:
            Audio with tape effects
        """
        # Create low-frequency modulation (wow and flutter)
        duration = audio.shape[-1] / self.sample_rate
        t = np.linspace(0, duration, audio.shape[-1])

        # Wow (slow pitch variation, ~0.5-2 Hz)
        wow = np.sin(2 * np.pi * 0.7 * t) * depth

        # Flutter (faster pitch variation, ~5-10 Hz)
        flutter = np.sin(2 * np.pi * 6 * t) * (depth * 0.3)

        # Combined modulation
        modulation = 1 + wow + flutter

        # Apply modulation (simple amplitude modulation as approximation)
        if audio.ndim == 2:
            modulation = modulation[np.newaxis, :]

        return audio * modulation

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target LUFS.

        Args:
            audio: Audio array

        Returns:
            Normalized audio
        """
        try:
            # Initialize loudness meter
            meter = pyln.Meter(self.sample_rate)

            # Measure loudness
            if audio.ndim == 1:
                loudness = meter.integrated_loudness(audio)
            else:
                loudness = meter.integrated_loudness(audio.T)

            # Normalize
            target_lufs = self.audio_config['target_lufs']
            if audio.ndim == 1:
                audio_normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
            else:
                audio_normalized = pyln.normalize.loudness(audio.T, loudness, target_lufs).T

            # Peak limiting
            peak = np.max(np.abs(audio_normalized))
            target_peak = 10 ** (self.audio_config['true_peak_max'] / 20)

            if peak > target_peak:
                audio_normalized = audio_normalized * (target_peak / peak)

            logger.info(f"Normalized audio: {loudness:.1f} LUFS -> {target_lufs} LUFS")

            return audio_normalized

        except Exception as e:
            logger.warning(f"Error normalizing audio: {e}, using simple normalization")
            # Fallback to simple peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                return audio * 0.9 / peak
            return audio

    def process_midi_to_lofi(
        self,
        midi_path: str,
        output_dir: str,
        name: Optional[str] = None,
        save_clean: bool = True,
        save_lofi: bool = True,
    ) -> Dict:
        """Complete processing pipeline: MIDI -> WAV -> Lo-fi WAV.

        Args:
            midi_path: Path to MIDI file
            output_dir: Directory to save output files
            name: Base name for output files (uses MIDI filename if None)
            save_clean: Save clean WAV file
            save_lofi: Save lo-fi processed WAV file

        Returns:
            Dictionary with output paths and status
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if name is None:
            name = Path(midi_path).stem

        results = {
            'midi_path': str(midi_path),
            'name': name,
        }

        # Convert MIDI to WAV
        clean_wav_path = output_dir / f"{name}_clean.wav"
        success = self.midi_to_wav(str(midi_path), str(clean_wav_path))

        if not success:
            results['error'] = 'Failed to convert MIDI to WAV'
            return results

        if save_clean:
            results['clean_wav_path'] = str(clean_wav_path)

        # Apply lo-fi effects
        if save_lofi:
            lofi_wav_path = output_dir / f"{name}_lofi.wav"
            success = self.apply_lofi_effects(str(clean_wav_path), str(lofi_wav_path))

            if success:
                results['lofi_wav_path'] = str(lofi_wav_path)
            else:
                results['lofi_error'] = 'Failed to apply lo-fi effects'

        # Remove clean WAV if not requested
        if not save_clean and clean_wav_path.exists():
            clean_wav_path.unlink()

        return results
