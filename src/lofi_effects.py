"""
LoFi Effects Chain - Authentic Vintage Sound Processing

Implements professional LoFi effects including:
- Vinyl crackle/pops
- Bit crushing
- Wow & flutter (pitch modulation)
- Tape saturation
- Analog warmth

Author: Claude
License: MIT
"""

from typing import Optional, Tuple

import librosa
import numpy as np
from scipy import signal


class LoFiEffectsChain:
    """Complete LoFi effects processing chain for authentic vintage sound."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize LoFi effects chain.

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate

    def add_vinyl_crackle(
        self,
        audio: np.ndarray,
        crackle_amount: float = 0.015,
        pop_frequency: float = 0.5,
        pop_intensity: float = 0.3,
    ) -> np.ndarray:
        """
        Add realistic vinyl crackle and pops.

        Args:
            audio: Input audio array
            crackle_amount: Amount of continuous crackle (0.0-1.0)
            pop_frequency: How often pops occur (pops per second)
            pop_intensity: Intensity of pops (0.0-1.0)

        Returns:
            Audio with vinyl noise
        """
        # Generate continuous crackle (filtered white noise)
        crackle = np.random.randn(len(audio)) * crackle_amount

        # Filter crackle to sound more realistic (high-pass + band-pass)
        # High-pass to remove low rumble
        b_hp, a_hp = signal.butter(2, 1000 / (self.sample_rate / 2), "high")
        crackle = signal.filtfilt(b_hp, a_hp, crackle)

        # Band-pass for characteristic vinyl hiss
        b_bp, a_bp = signal.butter(
            2, [3000 / (self.sample_rate / 2), 8000 / (self.sample_rate / 2)], "band"
        )
        crackle = signal.filtfilt(b_bp, a_bp, crackle)

        # Generate random pops
        num_pops = int(len(audio) / self.sample_rate * pop_frequency)
        pop_positions = np.random.randint(0, len(audio), num_pops)

        for pos in pop_positions:
            # Create pop envelope (quick attack, slower decay)
            pop_length = int(self.sample_rate * 0.01)  # 10ms pop
            if pos + pop_length < len(audio):
                pop_env = np.exp(-np.linspace(0, 5, pop_length))
                pop_signal = np.random.randn(pop_length) * pop_intensity * pop_env
                crackle[pos : pos + pop_length] += pop_signal

        # Mix with original audio
        return audio + crackle

    def add_bit_crushing(self, audio: np.ndarray, bit_depth: int = 12) -> np.ndarray:
        """
        Apply bit crushing for digital lo-fi grit.

        Args:
            audio: Input audio array
            bit_depth: Target bit depth (8-16, lower = more crushed)

        Returns:
            Bit-crushed audio
        """
        # Normalize to -1 to 1
        audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)

        # Quantize to target bit depth
        levels = 2**bit_depth
        crushed = np.round(audio_norm * levels) / levels

        # Restore original amplitude
        return crushed * np.max(np.abs(audio))

    def add_wow_flutter(
        self, audio: np.ndarray, rate: float = 0.3, depth: float = 5.0
    ) -> np.ndarray:
        """
        Add wow and flutter (pitch modulation from tape speed variations).

        Args:
            audio: Input audio array
            rate: Modulation rate in Hz (0.1-1.0 for wow, 2-10 for flutter)
            depth: Modulation depth in cents (1-10 cents typical)

        Returns:
            Audio with pitch modulation
        """
        # Create time array
        t = np.arange(len(audio)) / self.sample_rate

        # Generate LFO (Low Frequency Oscillator)
        # Use combination of sine waves for realistic variation
        lfo = (
            np.sin(2 * np.pi * rate * t)
            + 0.5 * np.sin(2 * np.pi * rate * 1.7 * t)
            + 0.3 * np.sin(2 * np.pi * rate * 2.3 * t)
        )

        # Normalize LFO
        lfo = lfo / np.max(np.abs(lfo))

        # Convert depth from cents to pitch ratio
        # 1 cent = 1/100 of a semitone
        pitch_variation = 2 ** (lfo * depth / 1200)

        # Apply pitch modulation via resampling
        # This simulates tape speed variations
        indices = np.cumsum(pitch_variation)
        indices = indices / indices[-1] * (len(audio) - 1)

        # Interpolate to get modulated signal
        modulated = np.interp(np.arange(len(audio)), indices, audio)

        return modulated

    def add_tape_saturation(
        self, audio: np.ndarray, drive: float = 2.0, mix: float = 0.5
    ) -> np.ndarray:
        """
        Apply tape saturation for analog warmth and harmonic distortion.

        Args:
            audio: Input audio array
            drive: Amount of saturation (1.0-5.0)
            mix: Dry/wet mix (0.0-1.0)

        Returns:
            Saturated audio
        """
        # Apply soft clipping (hyperbolic tangent)
        # This creates harmonic distortion similar to tape
        saturated = np.tanh(audio * drive) / np.tanh(drive)

        # Apply even-harmonic emphasis (characteristic of tape)
        # This is a simple approximation using asymmetric clipping
        saturated = saturated + 0.1 * (saturated**2) * np.sign(saturated)

        # Mix with dry signal
        return (1 - mix) * audio + mix * saturated

    def add_analog_warmth(self, audio: np.ndarray, warmth: float = 0.3) -> np.ndarray:
        """
        Add analog warmth via subtle low-end boost and high-end roll-off.

        Args:
            audio: Input audio array
            warmth: Amount of warmth (0.0-1.0)

        Returns:
            Warmed audio
        """
        # Low shelf boost (80-200 Hz)
        b_low, a_low = signal.butter(1, 200 / (self.sample_rate / 2), "low")
        low_boost = signal.filtfilt(b_low, a_low, audio) * warmth * 0.3

        # High shelf cut (8000+ Hz)
        b_high, a_high = signal.butter(1, 8000 / (self.sample_rate / 2), "high")
        high_cut = signal.filtfilt(b_high, a_high, audio) * (1 - warmth * 0.5)

        # Combine
        return audio + low_boost - (audio - high_cut)

    def add_downsampling(self, audio: np.ndarray, target_rate: int = 22050) -> np.ndarray:
        """
        Downsample and upsample to reduce fidelity (lo-fi effect).

        Args:
            audio: Input audio array
            target_rate: Intermediate sample rate

        Returns:
            Audio with reduced fidelity
        """
        # Downsample
        downsampled = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=target_rate)

        # Upsample back to original rate
        upsampled = librosa.resample(downsampled, orig_sr=target_rate, target_sr=self.sample_rate)

        return upsampled

    def process_full_chain(
        self, audio: np.ndarray, preset: str = "medium", custom_params: Optional[dict] = None
    ) -> np.ndarray:
        """
        Apply complete LoFi effects chain with presets.

        Args:
            audio: Input audio array
            preset: 'light', 'medium', or 'heavy'
            custom_params: Optional dict with custom parameters

        Returns:
            Fully processed LoFi audio
        """
        # Preset configurations
        presets = {
            "light": {
                "bit_depth": 14,
                "vinyl_crackle": 0.008,
                "vinyl_pops": 0.3,
                "wow_rate": 0.2,
                "wow_depth": 2.0,
                "saturation_drive": 1.5,
                "saturation_mix": 0.3,
                "warmth": 0.2,
                "downsample": None,
            },
            "medium": {
                "bit_depth": 12,
                "vinyl_crackle": 0.015,
                "vinyl_pops": 0.5,
                "wow_rate": 0.3,
                "wow_depth": 5.0,
                "saturation_drive": 2.0,
                "saturation_mix": 0.5,
                "warmth": 0.3,
                "downsample": 32000,
            },
            "heavy": {
                "bit_depth": 10,
                "vinyl_crackle": 0.025,
                "vinyl_pops": 0.8,
                "wow_rate": 0.5,
                "wow_depth": 8.0,
                "saturation_drive": 3.0,
                "saturation_mix": 0.7,
                "warmth": 0.5,
                "downsample": 22050,
            },
        }

        # Get parameters
        params = presets.get(preset, presets["medium"])
        if custom_params:
            params.update(custom_params)

        # Apply effects in order
        processed = audio.copy()

        # 1. Tape saturation (first for analog warmth)
        processed = self.add_tape_saturation(
            processed, drive=params["saturation_drive"], mix=params["saturation_mix"]
        )

        # 2. Wow & flutter
        processed = self.add_wow_flutter(
            processed, rate=params["wow_rate"], depth=params["wow_depth"]
        )

        # 3. Bit crushing
        processed = self.add_bit_crushing(processed, bit_depth=params["bit_depth"])

        # 4. Downsampling (if specified)
        if params["downsample"]:
            processed = self.add_downsampling(processed, target_rate=params["downsample"])

        # 5. Vinyl crackle
        processed = self.add_vinyl_crackle(
            processed, crackle_amount=params["vinyl_crackle"], pop_frequency=params["vinyl_pops"]
        )

        # 6. Analog warmth (last for final tone shaping)
        processed = self.add_analog_warmth(processed, warmth=params["warmth"])

        # Normalize to prevent clipping
        max_val = np.max(np.abs(processed))
        if max_val > 0.95:
            processed = processed * 0.95 / max_val

        return processed


def apply_lofi_effects(
    audio_path: str, output_path: str, preset: str = "medium", sample_rate: int = 44100
) -> None:
    """
    Convenience function to apply LoFi effects to an audio file.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save processed audio
        preset: 'light', 'medium', or 'heavy'
        sample_rate: Target sample rate
    """
    import soundfile as sf

    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=False)

    # Handle stereo
    if audio.ndim == 2:
        # Process each channel
        effects = LoFiEffectsChain(sample_rate=sr)
        processed_left = effects.process_full_chain(audio[0], preset=preset)
        processed_right = effects.process_full_chain(audio[1], preset=preset)
        processed = np.stack([processed_left, processed_right])
    else:
        # Mono
        effects = LoFiEffectsChain(sample_rate=sr)
        processed = effects.process_full_chain(audio, preset=preset)

    # Save
    sf.write(output_path, processed.T if audio.ndim == 2 else processed, sr)
    print(f"✅ LoFi effects applied: {output_path}")


if __name__ == "__main__":
    # Demo usage
    import argparse

    parser = argparse.ArgumentParser(description="Apply LoFi effects to audio file")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument(
        "--preset",
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Effect intensity preset",
    )

    args = parser.parse_args()

    apply_lofi_effects(args.input, args.output, preset=args.preset)
