"""Advanced audio mastering with AI-driven processing."""

import logging
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import scipy.signal as signal
from scipy import interpolate

logger = logging.getLogger(__name__)


class AIAudioMaster:
    """AI-driven audio mastering chain."""

    def __init__(self, sample_rate: int = 44100):
        """Initialize audio master.

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate

    def auto_master(
        self,
        audio: np.ndarray,
        target_loudness: float = -14.0,
        target_style: str = 'lofi',
    ) -> Tuple[np.ndarray, Dict]:
        """Automatically master audio to target style.

        Args:
            audio: Input audio array
            target_loudness: Target LUFS
            target_style: Mastering style ('lofi', 'clean', 'aggressive')

        Returns:
            Tuple of (mastered_audio, processing_info)
        """
        logger.info(f"Auto-mastering with target: {target_style}")

        processing_info = {}

        # 1. Analyze audio
        analysis = self._analyze_audio(audio)
        processing_info['input_analysis'] = analysis

        # 2. Apply AI-driven EQ
        audio, eq_info = self._ai_eq(audio, target_style, analysis)
        processing_info['eq'] = eq_info

        # 3. Apply multiband compression
        audio, comp_info = self._multiband_compression(audio, target_style)
        processing_info['compression'] = comp_info

        # 4. Apply stereo enhancement
        if audio.ndim == 2:
            audio, stereo_info = self._stereo_enhancement(audio, target_style)
            processing_info['stereo'] = stereo_info

        # 5. Apply saturation/warmth
        audio, sat_info = self._saturation(audio, target_style)
        processing_info['saturation'] = sat_info

        # 6. Final limiting and loudness normalization
        audio, limiter_info = self._final_limiting(audio, target_loudness)
        processing_info['limiter'] = limiter_info

        logger.info("Auto-mastering complete")
        return audio, processing_info

    def _analyze_audio(self, audio: np.ndarray) -> Dict:
        """Analyze audio characteristics.

        Args:
            audio: Audio array

        Returns:
            Analysis dictionary
        """
        # RMS and peak levels
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))

        # Spectral analysis
        stft = librosa.stft(audio if audio.ndim == 1 else audio[0])
        spec_mean = np.mean(np.abs(stft), axis=1)

        # Frequency bands energy
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        low_energy = np.mean(spec_mean[freqs < 250])
        mid_energy = np.mean(spec_mean[(freqs >= 250) & (freqs < 4000)])
        high_energy = np.mean(spec_mean[freqs >= 4000])

        # Crest factor
        crest_factor = 20 * np.log10(peak / (rms + 1e-8))

        return {
            'rms': float(rms),
            'peak': float(peak),
            'crest_factor': float(crest_factor),
            'low_energy': float(low_energy),
            'mid_energy': float(mid_energy),
            'high_energy': float(high_energy),
        }

    def _ai_eq(
        self,
        audio: np.ndarray,
        style: str,
        analysis: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply AI-driven EQ based on audio analysis.

        Args:
            audio: Audio array
            style: Target style
            analysis: Audio analysis from _analyze_audio

        Returns:
            Tuple of (eq'd_audio, eq_info)
        """
        # Determine EQ curve based on style and analysis
        if style == 'lofi':
            # Lo-fi: boost lows and mids, cut highs, add character
            eq_curve = {
                60: 2.0,      # Sub bass boost
                120: 1.5,     # Bass boost
                250: 1.0,     # Low-mids
                1000: 0.5,    # Mids (slight cut for lo-fi character)
                3000: -1.0,   # High-mids cut
                8000: -2.5,   # Highs cut (for lo-fi warmth)
                12000: -3.0,  # Air frequencies cut
            }
        elif style == 'clean':
            # Clean: balanced with slight enhancements
            eq_curve = {
                60: 0.5,
                250: 0.0,
                1000: 0.5,
                3000: 1.0,
                8000: 1.5,
                12000: 1.0,
            }
        else:  # aggressive
            # Aggressive: enhanced everything
            eq_curve = {
                60: 3.0,
                250: 1.5,
                1000: 2.0,
                3000: 2.5,
                8000: 2.0,
                12000: 1.5,
            }

        # Apply parametric EQ
        audio_eq = audio.copy()

        for freq, gain_db in eq_curve.items():
            if gain_db != 0:
                audio_eq = self._parametric_eq_band(
                    audio_eq,
                    freq,
                    gain_db,
                    q=1.0
                )

        eq_info = {
            'style': style,
            'eq_curve': eq_curve,
        }

        return audio_eq, eq_info

    def _parametric_eq_band(
        self,
        audio: np.ndarray,
        freq: float,
        gain_db: float,
        q: float = 1.0,
    ) -> np.ndarray:
        """Apply parametric EQ at specific frequency.

        Args:
            audio: Audio array
            freq: Center frequency (Hz)
            gain_db: Gain in dB
            q: Q factor (bandwidth)

        Returns:
            EQ'd audio
        """
        # Design bell filter
        w0 = 2 * np.pi * freq / self.sample_rate
        A = np.sqrt(10 ** (gain_db / 20))
        alpha = np.sin(w0) / (2 * q)

        # Bell filter coefficients
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        # Apply filter
        if audio.ndim == 1:
            return signal.filtfilt(b, a, audio)
        else:
            return np.array([signal.filtfilt(b, a, ch) for ch in audio])

    def _multiband_compression(
        self,
        audio: np.ndarray,
        style: str,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply multiband compression.

        Args:
            audio: Audio array
            style: Target style

        Returns:
            Tuple of (compressed_audio, compression_info)
        """
        # Split into frequency bands
        bands = self._split_frequency_bands(audio, [250, 2000, 8000])

        # Compression settings per band based on style
        if style == 'lofi':
            settings = [
                {'threshold': -20, 'ratio': 3.0, 'attack': 0.01, 'release': 0.1},  # Low
                {'threshold': -18, 'ratio': 2.5, 'attack': 0.005, 'release': 0.08}, # Mid
                {'threshold': -15, 'ratio': 2.0, 'attack': 0.001, 'release': 0.05},  # High
            ]
        else:
            settings = [
                {'threshold': -15, 'ratio': 2.0, 'attack': 0.01, 'release': 0.1},
                {'threshold': -12, 'ratio': 1.5, 'attack': 0.005, 'release': 0.08},
                {'threshold': -10, 'ratio': 1.5, 'attack': 0.001, 'release': 0.05},
            ]

        # Compress each band
        compressed_bands = []
        for band, setting in zip(bands, settings):
            compressed = self._compress_band(band, **setting)
            compressed_bands.append(compressed)

        # Recombine bands
        audio_compressed = sum(compressed_bands)

        comp_info = {
            'num_bands': len(bands),
            'settings': settings,
        }

        return audio_compressed, comp_info

    def _split_frequency_bands(
        self,
        audio: np.ndarray,
        crossover_freqs: List[float],
    ) -> List[np.ndarray]:
        """Split audio into frequency bands.

        Args:
            audio: Audio array
            crossover_freqs: Crossover frequencies

        Returns:
            List of band audio arrays
        """
        bands = []

        # Design Linkwitz-Riley crossover filters
        # For simplicity, using butterworth filters here
        order = 4

        # Low band
        sos_low = signal.butter(order, crossover_freqs[0], 'low', fs=self.sample_rate, output='sos')
        if audio.ndim == 1:
            low = signal.sosfiltfilt(sos_low, audio)
        else:
            low = np.array([signal.sosfiltfilt(sos_low, ch) for ch in audio])
        bands.append(low)

        # Mid bands
        for i in range(len(crossover_freqs) - 1):
            sos_band = signal.butter(
                order,
                [crossover_freqs[i], crossover_freqs[i + 1]],
                'bandpass',
                fs=self.sample_rate,
                output='sos'
            )
            if audio.ndim == 1:
                band = signal.sosfiltfilt(sos_band, audio)
            else:
                band = np.array([signal.sosfiltfilt(sos_band, ch) for ch in audio])
            bands.append(band)

        # High band
        sos_high = signal.butter(
            order,
            crossover_freqs[-1],
            'high',
            fs=self.sample_rate,
            output='sos'
        )
        if audio.ndim == 1:
            high = signal.sosfiltfilt(sos_high, audio)
        else:
            high = np.array([signal.sosfiltfilt(sos_high, ch) for ch in audio])
        bands.append(high)

        return bands

    def _compress_band(
        self,
        audio: np.ndarray,
        threshold: float,
        ratio: float,
        attack: float,
        release: float,
    ) -> np.ndarray:
        """Apply compression to an audio band.

        Args:
            audio: Audio array
            threshold: Threshold in dB
            ratio: Compression ratio
            attack: Attack time in seconds
            release: Release time in seconds

        Returns:
            Compressed audio
        """
        # Simple feed-forward compressor
        # Convert to dB
        audio_db = 20 * np.log10(np.abs(audio) + 1e-8)

        # Calculate gain reduction
        gain_reduction = np.zeros_like(audio_db)
        above_threshold = audio_db > threshold

        gain_reduction[above_threshold] = (audio_db[above_threshold] - threshold) * (1 - 1/ratio)

        # Apply envelope (simplified)
        # In production, would use proper attack/release envelope

        # Convert back to linear
        gain_linear = 10 ** (-gain_reduction / 20)

        # Apply gain
        return audio * gain_linear

    def _stereo_enhancement(
        self,
        audio: np.ndarray,
        style: str,
    ) -> Tuple[np.ndarray, Dict]:
        """Enhance stereo width.

        Args:
            audio: Stereo audio array (2, N)
            style: Target style

        Returns:
            Tuple of (enhanced_audio, stereo_info)
        """
        if audio.shape[0] != 2:
            return audio, {'error': 'Not stereo'}

        # Mid-side processing
        mid = (audio[0] + audio[1]) / 2
        side = (audio[0] - audio[1]) / 2

        # Adjust width based on style
        if style == 'lofi':
            width_factor = 1.2  # Subtle widening
        else:
            width_factor = 1.5  # More width

        # Enhanced stereo
        left = mid + side * width_factor
        right = mid - side * width_factor

        enhanced = np.array([left, right])

        stereo_info = {
            'width_factor': width_factor,
        }

        return enhanced, stereo_info

    def _saturation(
        self,
        audio: np.ndarray,
        style: str,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply saturation for warmth.

        Args:
            audio: Audio array
            style: Target style

        Returns:
            Tuple of (saturated_audio, saturation_info)
        """
        # Saturation amount based on style
        if style == 'lofi':
            drive = 1.5  # Moderate warmth
        else:
            drive = 1.0  # Clean

        # Soft clipping saturation
        saturated = np.tanh(audio * drive) / np.tanh(drive)

        sat_info = {
            'drive': drive,
        }

        return saturated, sat_info

    def _final_limiting(
        self,
        audio: np.ndarray,
        target_loudness: float,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply final limiting and loudness normalization.

        Args:
            audio: Audio array
            target_loudness: Target LUFS

        Returns:
            Tuple of (limited_audio, limiter_info)
        """
        # Measure current loudness (simplified - should use pyloudnorm)
        rms = np.sqrt(np.mean(audio ** 2))
        current_db = 20 * np.log10(rms + 1e-8)

        # Calculate required gain
        gain_db = target_loudness - current_db
        gain_linear = 10 ** (gain_db / 20)

        # Apply gain
        audio_gained = audio * gain_linear

        # Apply brick-wall limiter at -0.1 dB
        ceiling = 10 ** (-0.1 / 20)
        audio_limited = np.clip(audio_gained, -ceiling, ceiling)

        limiter_info = {
            'target_loudness': target_loudness,
            'gain_applied_db': gain_db,
            'ceiling_db': -0.1,
        }

        return audio_limited, limiter_info
