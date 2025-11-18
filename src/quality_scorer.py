"""Real music quality scoring system for generated tracks."""

import logging
from typing import Dict, List, Tuple

import librosa
import numpy as np
from scipy import signal, stats

logger = logging.getLogger(__name__)


class MusicQualityScorer:
    """Advanced quality scoring for generated MIDI and audio."""

    def __init__(self, config: Dict):
        """Initialize quality scorer.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def score_midi_tokens(self, tokens: List[int], metadata: Dict) -> float:
        """Score MIDI token sequence quality.

        Args:
            tokens: Token sequence
            metadata: Generation metadata

        Returns:
            Quality score (0-10)
        """
        scores = []

        # 1. Length score (prefer 1000-2000 tokens for ~2-3 min tracks)
        length_score = self._score_length(len(tokens), target=1500, tolerance=500)
        scores.append(('length', length_score, 1.5))

        # 2. Diversity score (unique token ratio)
        diversity_score = self._score_diversity(tokens)
        scores.append(('diversity', diversity_score, 2.0))

        # 3. Repetition score (check for good repetition patterns)
        repetition_score = self._score_repetition(tokens)
        scores.append(('repetition', repetition_score, 1.5))

        # 4. Tempo appropriateness for lo-fi (65-85 BPM is ideal)
        tempo = metadata.get('tempo', 75)
        tempo_score = self._score_tempo(tempo, ideal_min=65, ideal_max=85)
        scores.append(('tempo', tempo_score, 1.0))

        # 5. Token distribution (should be relatively uniform, not skewed)
        distribution_score = self._score_token_distribution(tokens)
        scores.append(('distribution', distribution_score, 1.0))

        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        total_weight = sum(weight for _, _, weight in scores)
        final_score = (total_score / total_weight) * 10

        # Log breakdown
        logger.debug(f"Quality score breakdown:")
        for name, score, weight in scores:
            logger.debug(f"  {name}: {score:.3f} (weight: {weight})")
        logger.debug(f"  Final: {final_score:.2f}/10")

        return min(10.0, max(0.0, final_score))

    def score_audio(self, audio_path: str) -> float:
        """Score audio file quality.

        Args:
            audio_path: Path to audio file

        Returns:
            Quality score (0-10)
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)

            scores = []

            # 1. Dynamic range score
            dynamic_range = self._score_dynamic_range(y)
            scores.append(('dynamic_range', dynamic_range, 1.5))

            # 2. Spectral centroid variance (timbral variety)
            spectral_score = self._score_spectral_features(y, sr)
            scores.append(('spectral', spectral_score, 2.0))

            # 3. Zero-crossing rate (rhythmic complexity)
            zcr_score = self._score_zero_crossings(y)
            scores.append(('zero_crossings', zcr_score, 1.0))

            # 4. Loudness consistency
            loudness_score = self._score_loudness(y)
            scores.append(('loudness', loudness_score, 1.5))

            # 5. Frequency balance (low-fi should have balanced spectrum)
            balance_score = self._score_frequency_balance(y, sr)
            scores.append(('frequency_balance', balance_score, 1.5))

            # Calculate weighted average
            total_score = sum(score * weight for _, score, weight in scores)
            total_weight = sum(weight for _, _, weight in scores)
            final_score = (total_score / total_weight) * 10

            logger.debug(f"Audio quality score: {final_score:.2f}/10")
            return min(10.0, max(0.0, final_score))

        except Exception as e:
            logger.error(f"Error scoring audio: {e}")
            return 5.0  # Return neutral score on error

    @staticmethod
    def _score_length(length: int, target: int, tolerance: int) -> float:
        """Score based on length proximity to target.

        Returns score between 0 and 1.
        """
        diff = abs(length - target)
        if diff <= tolerance:
            return 1.0 - (diff / tolerance) * 0.5
        else:
            # Exponential decay beyond tolerance
            excess = diff - tolerance
            return 0.5 * np.exp(-excess / tolerance)

    @staticmethod
    def _score_diversity(tokens: List[int]) -> float:
        """Score token diversity (unique ratio).

        Returns score between 0 and 1.
        """
        if not tokens:
            return 0.0

        unique_ratio = len(set(tokens)) / len(tokens)

        # Ideal diversity is around 0.3-0.7 (some repetition is good for music)
        if 0.3 <= unique_ratio <= 0.7:
            return 1.0
        elif unique_ratio < 0.3:
            # Too repetitive
            return unique_ratio / 0.3
        else:
            # Too diverse (lacks cohesion)
            return 1.0 - (unique_ratio - 0.7) / 0.3

    @staticmethod
    def _score_repetition(tokens: List[int], pattern_length: int = 16) -> float:
        """Score repetition patterns (musical structure).

        Returns score between 0 and 1.
        """
        if len(tokens) < pattern_length * 2:
            return 0.5  # Not enough data

        # Check for repeating patterns
        patterns = {}
        for i in range(len(tokens) - pattern_length):
            pattern = tuple(tokens[i : i + pattern_length])
            patterns[pattern] = patterns.get(pattern, 0) + 1

        if not patterns:
            return 0.5

        # Good music has some repeating patterns but not too many
        max_repeats = max(patterns.values())
        num_unique_patterns = len(patterns)

        # Ideal: 2-4 repetitions of common patterns
        repeat_score = 1.0 - abs(max_repeats - 3) / 5.0
        repeat_score = max(0.0, min(1.0, repeat_score))

        return repeat_score

    @staticmethod
    def _score_tempo(tempo: float, ideal_min: float, ideal_max: float) -> float:
        """Score tempo appropriateness.

        Returns score between 0 and 1.
        """
        if ideal_min <= tempo <= ideal_max:
            return 1.0
        elif tempo < ideal_min:
            # Too slow
            return max(0.0, 1.0 - (ideal_min - tempo) / ideal_min)
        else:
            # Too fast
            return max(0.0, 1.0 - (tempo - ideal_max) / ideal_max)

    @staticmethod
    def _score_token_distribution(tokens: List[int]) -> float:
        """Score uniformity of token distribution.

        Returns score between 0 and 1.
        """
        if not tokens:
            return 0.0

        # Calculate histogram
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        counts = list(token_counts.values())

        # Use coefficient of variation (std/mean)
        # Lower CV = more uniform distribution
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        if mean_count == 0:
            return 0.0

        cv = std_count / mean_count

        # Ideal CV is around 0.5-1.5 (some variation but not extreme)
        if 0.5 <= cv <= 1.5:
            return 1.0
        elif cv < 0.5:
            return cv / 0.5
        else:
            return max(0.0, 1.0 - (cv - 1.5) / 1.5)

    @staticmethod
    def _score_dynamic_range(audio: np.ndarray) -> float:
        """Score dynamic range of audio.

        Returns score between 0 and 1.
        """
        # Calculate RMS energy in dB
        rms = librosa.feature.rms(y=audio)[0]
        rms_db = librosa.amplitude_to_db(rms)

        # Calculate dynamic range (difference between loud and soft parts)
        dynamic_range = np.percentile(rms_db, 95) - np.percentile(rms_db, 5)

        # Good dynamic range is 20-40 dB
        if 20 <= dynamic_range <= 40:
            return 1.0
        elif dynamic_range < 20:
            return dynamic_range / 20
        else:
            return max(0.0, 1.0 - (dynamic_range - 40) / 40)

    @staticmethod
    def _score_spectral_features(audio: np.ndarray, sr: int) -> float:
        """Score spectral features (timbral variety).

        Returns score between 0 and 1.
        """
        # Calculate spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

        # Variance in spectral centroid indicates timbral variety
        centroid_std = np.std(spectral_centroids)

        # Normalize by sample rate
        normalized_std = centroid_std / (sr / 2)

        # Good variety is around 0.1-0.3 normalized
        if 0.1 <= normalized_std <= 0.3:
            return 1.0
        elif normalized_std < 0.1:
            return normalized_std / 0.1
        else:
            return max(0.0, 1.0 - (normalized_std - 0.3) / 0.3)

    @staticmethod
    def _score_zero_crossings(audio: np.ndarray) -> float:
        """Score zero-crossing rate (rhythmic complexity).

        Returns score between 0 and 1.
        """
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        mean_zcr = np.mean(zcr)

        # Good ZCR is around 0.05-0.15
        if 0.05 <= mean_zcr <= 0.15:
            return 1.0
        elif mean_zcr < 0.05:
            return mean_zcr / 0.05
        else:
            return max(0.0, 1.0 - (mean_zcr - 0.15) / 0.15)

    @staticmethod
    def _score_loudness(audio: np.ndarray) -> float:
        """Score loudness consistency.

        Returns score between 0 and 1.
        """
        # Calculate RMS energy over time
        rms = librosa.feature.rms(y=audio)[0]

        # Coefficient of variation
        cv = np.std(rms) / (np.mean(rms) + 1e-8)

        # Good consistency has CV around 0.3-0.7
        if 0.3 <= cv <= 0.7:
            return 1.0
        elif cv < 0.3:
            return cv / 0.3
        else:
            return max(0.0, 1.0 - (cv - 0.7) / 0.7)

    @staticmethod
    def _score_frequency_balance(audio: np.ndarray, sr: int) -> float:
        """Score frequency spectrum balance.

        Returns score between 0 and 1.
        """
        # Compute STFT
        D = librosa.stft(audio)
        mag = np.abs(D)

        # Divide spectrum into bands: low, mid, high
        freqs = librosa.fft_frequencies(sr=sr)
        low_band = mag[freqs < 500, :].mean()
        mid_band = mag[(freqs >= 500) & (freqs < 4000), :].mean()
        high_band = mag[freqs >= 4000, :].mean()

        # Calculate balance (ratio of energies)
        total_energy = low_band + mid_band + high_band + 1e-8

        low_ratio = low_band / total_energy
        mid_ratio = mid_band / total_energy
        high_ratio = high_band / total_energy

        # Good balance for lo-fi: more low and mid, less high
        # Ideal: low ~0.4, mid ~0.45, high ~0.15
        score = 1.0 - (
            abs(low_ratio - 0.40) + abs(mid_ratio - 0.45) + abs(high_ratio - 0.15)
        ) / 2.0

        return max(0.0, min(1.0, score))
