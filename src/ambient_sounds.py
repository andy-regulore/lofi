"""
Ambient Sound Library for LoFi Tracks

Adds atmospheric soundscapes to LoFi music:
- Rain (light, medium, heavy)
- Café ambience (chatter, espresso machine)
- Nature (birds, wind, waves)
- City (distant traffic, urban atmosphere)

Author: Claude
License: MIT
"""

import numpy as np
from scipy import signal
import librosa
from pathlib import Path
from typing import Optional, Dict, Tuple
import soundfile as sf


class AmbientSoundGenerator:
    """Generate realistic ambient sounds procedurally."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize ambient sound generator.

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate

    def generate_rain(
        self,
        duration: float,
        intensity: str = 'medium',
        include_thunder: bool = False
    ) -> np.ndarray:
        """
        Generate realistic rain sound.

        Args:
            duration: Duration in seconds
            intensity: 'light', 'medium', or 'heavy'
            include_thunder: Add occasional thunder

        Returns:
            Rain audio array
        """
        samples = int(duration * self.sample_rate)

        # Rain is basically filtered noise with specific characteristics
        intensities = {
            'light': {
                'density': 0.3,
                'pitch_range': (2000, 8000),
                'amplitude': 0.1
            },
            'medium': {
                'density': 0.5,
                'pitch_range': (1500, 6000),
                'amplitude': 0.15
            },
            'heavy': {
                'density': 0.8,
                'pitch_range': (1000, 5000),
                'amplitude': 0.25
            }
        }

        params = intensities.get(intensity, intensities['medium'])

        # Generate base noise
        rain = np.random.randn(samples) * params['amplitude']

        # Apply band-pass filter for rain frequency range
        low, high = params['pitch_range']
        b, a = signal.butter(
            4,
            [low / (self.sample_rate / 2), high / (self.sample_rate / 2)],
            'band'
        )
        rain = signal.filtfilt(b, a, rain)

        # Add density variation (rain comes in waves)
        t = np.linspace(0, duration, samples)
        density_lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)  # Slow variation
        rain = rain * density_lfo

        # Add occasional heavy drops
        num_drops = int(duration * params['density'] * 10)
        for _ in range(num_drops):
            pos = np.random.randint(0, samples - 1000)
            drop_env = np.exp(-np.linspace(0, 10, 1000))
            drop_sound = np.random.randn(1000) * 0.3 * drop_env
            rain[pos:pos+1000] += drop_sound

        # Add thunder if requested
        if include_thunder:
            rain = self._add_thunder(rain, duration)

        return rain

    def generate_cafe_ambience(
        self,
        duration: float,
        busyness: str = 'medium'
    ) -> np.ndarray:
        """
        Generate café ambience.

        Args:
            duration: Duration in seconds
            busyness: 'quiet', 'medium', or 'busy'

        Returns:
            Café ambience audio
        """
        samples = int(duration * self.sample_rate)

        # Base ambient hum
        ambience = np.random.randn(samples) * 0.03

        # Low-pass filter for background murmur
        b, a = signal.butter(2, 500 / (self.sample_rate / 2), 'low')
        ambience = signal.filtfilt(b, a, ambience)

        # Add speech-like bursts (chatter)
        busyness_levels = {
            'quiet': 0.5,
            'medium': 1.0,
            'busy': 2.0
        }

        burst_rate = busyness_levels.get(busyness, 1.0)
        num_bursts = int(duration * burst_rate)

        for _ in range(num_bursts):
            pos = np.random.randint(0, samples - 5000)
            burst_length = np.random.randint(1000, 5000)

            # Create speech-like burst (filtered noise with envelope)
            burst = np.random.randn(burst_length) * 0.08
            burst_env = signal.windows.hann(burst_length)
            burst = burst * burst_env

            # Filter to speech range
            b_speech, a_speech = signal.butter(2, [200 / (self.sample_rate / 2),
                                                     3000 / (self.sample_rate / 2)], 'band')
            burst = signal.filtfilt(b_speech, a_speech, burst)

            if pos + len(burst) < samples:
                ambience[pos:pos+len(burst)] += burst

        # Add occasional cup/dish sounds
        num_dishes = int(duration * 0.3)
        for _ in range(num_dishes):
            pos = np.random.randint(0, samples - 500)
            dish_sound = np.random.randn(500) * 0.15
            dish_env = np.exp(-np.linspace(0, 8, 500))
            dish_sound = dish_sound * dish_env

            # High-pass for clinking sound
            b_hp, a_hp = signal.butter(2, 2000 / (self.sample_rate / 2), 'high')
            dish_sound = signal.filtfilt(b_hp, a_hp, dish_sound)

            if pos + len(dish_sound) < samples:
                ambience[pos:pos+len(dish_sound)] += dish_sound

        return ambience

    def generate_nature_sounds(
        self,
        duration: float,
        environment: str = 'forest'
    ) -> np.ndarray:
        """
        Generate nature sounds.

        Args:
            duration: Duration in seconds
            environment: 'forest', 'beach', or 'wind'

        Returns:
            Nature sounds audio
        """
        samples = int(duration * self.sample_rate)

        if environment == 'forest':
            return self._generate_forest_sounds(duration, samples)
        elif environment == 'beach':
            return self._generate_beach_sounds(duration, samples)
        elif environment == 'wind':
            return self._generate_wind_sounds(duration, samples)
        else:
            return self._generate_forest_sounds(duration, samples)

    def _generate_forest_sounds(self, duration: float, samples: int) -> np.ndarray:
        """Generate forest ambience with birds."""
        # Base forest ambience (gentle wind through leaves)
        ambience = np.random.randn(samples) * 0.04
        b, a = signal.butter(2, [200 / (self.sample_rate / 2),
                                  2000 / (self.sample_rate / 2)], 'band')
        ambience = signal.filtfilt(b, a, ambience)

        # Add bird chirps
        num_birds = int(duration * 2)  # ~2 chirps per second
        for _ in range(num_birds):
            pos = np.random.randint(0, samples - 2000)

            # Create chirp (frequency sweep)
            chirp_duration = np.random.uniform(0.1, 0.3)
            chirp_samples = int(chirp_duration * self.sample_rate)
            t = np.linspace(0, chirp_duration, chirp_samples)

            f0 = np.random.uniform(2000, 4000)
            f1 = f0 + np.random.uniform(-500, 500)

            chirp = np.sin(2 * np.pi * (f0 + (f1 - f0) * t) * t)
            chirp_env = signal.windows.hann(chirp_samples)
            chirp = chirp * chirp_env * 0.1

            if pos + len(chirp) < samples:
                ambience[pos:pos+len(chirp)] += chirp

        return ambience

    def _generate_beach_sounds(self, duration: float, samples: int) -> np.ndarray:
        """Generate beach waves."""
        ambience = np.zeros(samples)

        # Waves (periodic filtered noise)
        wave_period = 5.0  # seconds between waves
        num_waves = int(duration / wave_period)

        for i in range(num_waves):
            wave_start = int(i * wave_period * self.sample_rate)
            wave_length = int(3 * self.sample_rate)  # 3 second wave

            if wave_start + wave_length < samples:
                # Create wave sound
                wave = np.random.randn(wave_length) * 0.15

                # Filter for wave character
                b, a = signal.butter(2, 800 / (self.sample_rate / 2), 'low')
                wave = signal.filtfilt(b, a, wave)

                # Envelope (swell and crash)
                env = np.concatenate([
                    np.linspace(0, 1, wave_length // 3),  # Build up
                    np.linspace(1, 0.3, wave_length // 3),  # Crash
                    np.linspace(0.3, 0, wave_length // 3)  # Fade
                ])
                wave = wave * env

                ambience[wave_start:wave_start+wave_length] += wave

        return ambience

    def _generate_wind_sounds(self, duration: float, samples: int) -> np.ndarray:
        """Generate wind ambience."""
        # Wind is low-frequency filtered noise with slow variations
        wind = np.random.randn(samples) * 0.12

        # Low-pass filter
        b, a = signal.butter(3, 400 / (self.sample_rate / 2), 'low')
        wind = signal.filtfilt(b, a, wind)

        # Add slow amplitude variations
        t = np.linspace(0, duration, samples)
        lfo1 = np.sin(2 * np.pi * 0.05 * t)
        lfo2 = np.sin(2 * np.pi * 0.13 * t)
        modulation = 0.5 + 0.3 * lfo1 + 0.2 * lfo2
        wind = wind * modulation

        return wind

    def _add_thunder(self, rain: np.ndarray, duration: float) -> np.ndarray:
        """Add occasional thunder to rain."""
        num_thunder = max(1, int(duration / 30))  # One every 30 seconds

        for _ in range(num_thunder):
            pos = np.random.randint(0, len(rain) - self.sample_rate * 2)
            thunder_length = int(self.sample_rate * 1.5)

            # Thunder is low-frequency rumble
            thunder = np.random.randn(thunder_length) * 0.4
            b, a = signal.butter(2, 200 / (self.sample_rate / 2), 'low')
            thunder = signal.filtfilt(b, a, thunder)

            # Envelope
            thunder_env = np.concatenate([
                np.linspace(0, 1, thunder_length // 4),
                np.ones(thunder_length // 2),
                np.linspace(1, 0, thunder_length // 4)
            ])
            thunder = thunder * thunder_env

            rain[pos:pos+thunder_length] += thunder

        return rain


class AmbientSoundMixer:
    """Mix ambient sounds with music tracks."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize mixer.

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.generator = AmbientSoundGenerator(sample_rate)

    def add_ambient_to_track(
        self,
        audio: np.ndarray,
        ambient_type: str,
        mix_level: float = 0.15,
        **ambient_params
    ) -> np.ndarray:
        """
        Add ambient sound to music track.

        Args:
            audio: Music audio array
            ambient_type: 'rain', 'cafe', or 'nature'
            mix_level: Volume of ambient (0.0-1.0)
            **ambient_params: Additional parameters for ambient generation

        Returns:
            Mixed audio
        """
        duration = len(audio) / self.sample_rate

        # Generate appropriate ambient sound
        if ambient_type == 'rain':
            ambient = self.generator.generate_rain(duration, **ambient_params)
        elif ambient_type == 'cafe':
            ambient = self.generator.generate_cafe_ambience(duration, **ambient_params)
        elif ambient_type == 'nature':
            ambient = self.generator.generate_nature_sounds(duration, **ambient_params)
        else:
            print(f"Unknown ambient type: {ambient_type}")
            return audio

        # Ensure same length
        if len(ambient) > len(audio):
            ambient = ambient[:len(audio)]
        elif len(ambient) < len(audio):
            ambient = np.pad(ambient, (0, len(audio) - len(ambient)))

        # Mix with music
        mixed = audio + (ambient * mix_level)

        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 0.95:
            mixed = mixed * 0.95 / max_val

        return mixed


def add_ambient_to_file(
    input_path: str,
    output_path: str,
    ambient_type: str = 'rain',
    mix_level: float = 0.15,
    **ambient_params
) -> None:
    """
    Convenience function to add ambient to audio file.

    Args:
        input_path: Input audio file
        output_path: Output audio file
        ambient_type: Type of ambient sound
        mix_level: Mix volume
        **ambient_params: Additional parameters
    """
    # Load audio
    audio, sr = librosa.load(input_path, sr=None, mono=False)

    # Create mixer
    mixer = AmbientSoundMixer(sample_rate=sr)

    # Handle stereo
    if audio.ndim == 2:
        # Mix both channels
        mixed_left = mixer.add_ambient_to_track(
            audio[0], ambient_type, mix_level, **ambient_params
        )
        mixed_right = mixer.add_ambient_to_track(
            audio[1], ambient_type, mix_level, **ambient_params
        )
        mixed = np.stack([mixed_left, mixed_right])
    else:
        # Mono
        mixed = mixer.add_ambient_to_track(
            audio, ambient_type, mix_level, **ambient_params
        )

    # Save
    sf.write(output_path, mixed.T if audio.ndim == 2 else mixed, sr)
    print(f"✅ Ambient sound added: {output_path}")


# Preset combinations

AMBIENT_PRESETS = {
    'rainy_day': {
        'ambient_type': 'rain',
        'intensity': 'medium',
        'include_thunder': False,
        'mix_level': 0.12
    },
    'storm': {
        'ambient_type': 'rain',
        'intensity': 'heavy',
        'include_thunder': True,
        'mix_level': 0.18
    },
    'cafe': {
        'ambient_type': 'cafe',
        'busyness': 'medium',
        'mix_level': 0.10
    },
    'forest': {
        'ambient_type': 'nature',
        'environment': 'forest',
        'mix_level': 0.12
    },
    'beach': {
        'ambient_type': 'nature',
        'environment': 'beach',
        'mix_level': 0.15
    },
    'windy': {
        'ambient_type': 'nature',
        'environment': 'wind',
        'mix_level': 0.10
    }
}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Add ambient sounds to audio')
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('output', help='Output audio file')
    parser.add_argument('--preset', choices=list(AMBIENT_PRESETS.keys()),
                        default='rainy_day', help='Ambient preset')

    args = parser.parse_args()

    # Get preset params
    params = AMBIENT_PRESETS[args.preset]

    # Add ambient
    add_ambient_to_file(args.input, args.output, **params)
