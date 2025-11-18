"""Comprehensive music analysis and evaluation tools.

Features:
- Music Information Retrieval (MIR) metrics
- Perplexity and likelihood metrics
- Human evaluation framework
- A/B testing infrastructure
- Quality metrics dashboard
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
from pathlib import Path

import librosa
import pretty_midi

logger = logging.getLogger(__name__)


class MIRMetrics:
    """Music Information Retrieval metrics."""

    def __init__(self, sample_rate: int = 44100):
        """Initialize MIR metrics calculator.

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate

    def compute_all_metrics(
        self,
        audio_path: str,
        midi_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute all MIR metrics.

        Args:
            audio_path: Path to audio file
            midi_path: Optional path to MIDI file

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Spectral metrics
        metrics.update(self._compute_spectral_metrics(audio))

        # Rhythm metrics
        metrics.update(self._compute_rhythm_metrics(audio))

        # Tonal metrics
        metrics.update(self._compute_tonal_metrics(audio))

        # Dynamics metrics
        metrics.update(self._compute_dynamics_metrics(audio))

        # MIDI metrics (if available)
        if midi_path:
            metrics.update(self._compute_midi_metrics(midi_path))

        return metrics

    def _compute_spectral_metrics(self, audio: np.ndarray) -> Dict[str, float]:
        """Compute spectral metrics.

        Args:
            audio: Audio array

        Returns:
            Spectral metrics
        """
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        centroid_mean = float(np.mean(centroid))
        centroid_std = float(np.std(centroid))

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        rolloff_mean = float(np.mean(rolloff))

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        contrast_mean = float(np.mean(contrast))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = float(np.mean(zcr))

        return {
            'spectral_centroid_mean': centroid_mean,
            'spectral_centroid_std': centroid_std,
            'spectral_rolloff_mean': rolloff_mean,
            'spectral_contrast_mean': contrast_mean,
            'zero_crossing_rate_mean': zcr_mean,
        }

    def _compute_rhythm_metrics(self, audio: np.ndarray) -> Dict[str, float]:
        """Compute rhythm metrics.

        Args:
            audio: Audio array

        Returns:
            Rhythm metrics
        """
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)

        # Beat strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        beat_strength = float(np.mean(onset_env[beats])) if len(beats) > 0 else 0.0

        # Rhythmic regularity (consistency of beat intervals)
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            rhythm_regularity = float(1.0 / (1.0 + np.std(beat_intervals)))
        else:
            rhythm_regularity = 0.0

        return {
            'tempo': float(tempo),
            'num_beats': len(beats),
            'beat_strength': beat_strength,
            'rhythm_regularity': rhythm_regularity,
        }

    def _compute_tonal_metrics(self, audio: np.ndarray) -> Dict[str, float]:
        """Compute tonal/harmonic metrics.

        Args:
            audio: Audio array

        Returns:
            Tonal metrics
        """
        # Chroma features
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate)

        # Tonal centroid (key strength)
        tonal_centroid = float(np.mean(np.max(chroma, axis=0)))

        # Harmonic-to-noise ratio
        harmonic, percussive = librosa.effects.hpss(audio)
        hnr = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-6))

        # Key clarity (how clear the key is)
        chroma_std = float(np.mean(np.std(chroma, axis=1)))

        return {
            'tonal_centroid': tonal_centroid,
            'harmonic_to_noise_ratio': hnr,
            'key_clarity': chroma_std,
        }

    def _compute_dynamics_metrics(self, audio: np.ndarray) -> Dict[str, float]:
        """Compute dynamics metrics.

        Args:
            audio: Audio array

        Returns:
            Dynamics metrics
        """
        # RMS energy
        rms = librosa.feature.rms(y=audio)
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        # Dynamic range
        dynamic_range = float(20 * np.log10(np.max(np.abs(audio)) / (rms_mean + 1e-8)))

        # Loudness variation
        loudness_variation = rms_std / (rms_mean + 1e-8)

        return {
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'dynamic_range_db': dynamic_range,
            'loudness_variation': float(loudness_variation),
        }

    def _compute_midi_metrics(self, midi_path: str) -> Dict[str, float]:
        """Compute MIDI-specific metrics.

        Args:
            midi_path: Path to MIDI file

        Returns:
            MIDI metrics
        """
        midi = pretty_midi.PrettyMIDI(str(midi_path))

        # Pitch range
        all_pitches = []
        for inst in midi.instruments:
            if not inst.is_drum:
                all_pitches.extend([note.pitch for note in inst.notes])

        if all_pitches:
            pitch_range = max(all_pitches) - min(all_pitches)
            pitch_variety = len(set(all_pitches)) / len(all_pitches)
        else:
            pitch_range = 0
            pitch_variety = 0

        # Note density
        duration = midi.get_end_time()
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        note_density = total_notes / duration if duration > 0 else 0

        # Polyphony (average notes playing simultaneously)
        time_steps = np.arange(0, duration, 0.1)
        polyphony_counts = []

        for t in time_steps:
            count = 0
            for inst in midi.instruments:
                for note in inst.notes:
                    if note.start <= t < note.end:
                        count += 1
            polyphony_counts.append(count)

        avg_polyphony = float(np.mean(polyphony_counts)) if polyphony_counts else 0

        return {
            'pitch_range': pitch_range,
            'pitch_variety': float(pitch_variety),
            'note_density': float(note_density),
            'avg_polyphony': avg_polyphony,
            'duration': float(duration),
        }


class PerplexityCalculator:
    """Calculate perplexity and likelihood metrics for music."""

    def __init__(self, model, tokenizer):
        """Initialize perplexity calculator.

        Args:
            model: Trained music generation model
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def calculate_perplexity(
        self,
        tokens: List[int],
    ) -> float:
        """Calculate perplexity of token sequence.

        Lower perplexity = model is more confident

        Args:
            tokens: Token sequence

        Returns:
            Perplexity value
        """
        import torch

        self.model.eval()

        with torch.no_grad():
            tokens_tensor = torch.tensor([tokens], device=self.model.device)

            outputs = self.model.get_model()(tokens_tensor)
            logits = outputs.logits

            # Shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tokens_tensor[:, 1:].contiguous()

            # Calculate cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Perplexity = exp(loss)
            perplexity = torch.exp(loss).item()

        return perplexity

    def calculate_conditional_perplexity(
        self,
        tokens: List[int],
        conditioning: Dict,
    ) -> float:
        """Calculate perplexity given conditioning.

        Args:
            tokens: Token sequence
            conditioning: Conditioning information (tempo, key, mood)

        Returns:
            Conditional perplexity
        """
        # Create conditioning prefix
        if hasattr(self.model, 'create_conditioning_prefix'):
            prefix = self.model.create_conditioning_prefix(
                tempo=conditioning.get('tempo', 75),
                key=conditioning.get('key', 'C'),
                mood=conditioning.get('mood', 'chill'),
            )
            conditioned_tokens = prefix + tokens
        else:
            conditioned_tokens = tokens

        return self.calculate_perplexity(conditioned_tokens)


class HumanEvaluationFramework:
    """Framework for collecting human evaluations."""

    def __init__(self, output_file: str):
        """Initialize evaluation framework.

        Args:
            output_file: Path to save evaluations
        """
        self.output_file = output_file
        self.evaluations = []

        # Load existing
        if Path(output_file).exists():
            with open(output_file) as f:
                self.evaluations = json.load(f)

    def add_evaluation(
        self,
        track_id: str,
        scores: Dict[str, int],
        comments: Optional[str] = None,
        evaluator_id: Optional[str] = None,
    ):
        """Add human evaluation.

        Args:
            track_id: Track identifier
            scores: Scores for different aspects (1-10 scale)
                    e.g., {'melody': 8, 'harmony': 7, 'rhythm': 9, 'overall': 8}
            comments: Optional comments
            evaluator_id: Optional evaluator identifier
        """
        evaluation = {
            'track_id': track_id,
            'scores': scores,
            'comments': comments,
            'evaluator_id': evaluator_id,
            'timestamp': str(np.datetime64('now')),
        }

        self.evaluations.append(evaluation)
        self._save()

    def _save(self):
        """Save evaluations."""
        with open(self.output_file, 'w') as f:
            json.dump(self.evaluations, f, indent=2)

    def get_statistics(
        self,
        track_id: Optional[str] = None,
    ) -> Dict:
        """Get evaluation statistics.

        Args:
            track_id: Optional track ID to filter by

        Returns:
            Statistics dictionary
        """
        evals = self.evaluations
        if track_id:
            evals = [e for e in evals if e['track_id'] == track_id]

        if not evals:
            return {}

        # Collect all scores
        all_scores = {}
        for eval_data in evals:
            for aspect, score in eval_data['scores'].items():
                if aspect not in all_scores:
                    all_scores[aspect] = []
                all_scores[aspect].append(score)

        # Calculate statistics
        stats = {}
        for aspect, scores in all_scores.items():
            stats[aspect] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'median': float(np.median(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'count': len(scores),
            }

        return stats


class ABTestingFramework:
    """A/B testing for comparing models or configurations."""

    def __init__(self, experiment_name: str, output_dir: str):
        """Initialize A/B testing framework.

        Args:
            experiment_name: Name of experiment
            output_dir: Directory to save results
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.output_dir / f"{experiment_name}_results.json"
        self.results = []

        # Load existing
        if self.results_file.exists():
            with open(self.results_file) as f:
                self.results = json.load(f)

    def record_comparison(
        self,
        variant_a_id: str,
        variant_b_id: str,
        winner: str,  # 'a', 'b', or 'tie'
        metrics: Optional[Dict[str, float]] = None,
        evaluator_id: Optional[str] = None,
    ):
        """Record A/B comparison result.

        Args:
            variant_a_id: ID of variant A
            variant_b_id: ID of variant B
            winner: Which variant won
            metrics: Optional objective metrics
            evaluator_id: Optional evaluator ID
        """
        result = {
            'variant_a': variant_a_id,
            'variant_b': variant_b_id,
            'winner': winner,
            'metrics': metrics or {},
            'evaluator_id': evaluator_id,
            'timestamp': str(np.datetime64('now')),
        }

        self.results.append(result)
        self._save()

    def _save(self):
        """Save results."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def get_win_rates(self) -> Dict[str, Dict[str, float]]:
        """Calculate win rates for each variant.

        Returns:
            Win rate statistics
        """
        variant_stats = {}

        for result in self.results:
            for variant_key in ['variant_a', 'variant_b']:
                variant_id = result[variant_key]

                if variant_id not in variant_stats:
                    variant_stats[variant_id] = {
                        'wins': 0,
                        'losses': 0,
                        'ties': 0,
                        'total': 0,
                    }

                variant_stats[variant_id]['total'] += 1

                # Determine outcome
                if result['winner'] == 'tie':
                    variant_stats[variant_id]['ties'] += 1
                elif (result['winner'] == 'a' and variant_key == 'variant_a') or \
                     (result['winner'] == 'b' and variant_key == 'variant_b'):
                    variant_stats[variant_id]['wins'] += 1
                else:
                    variant_stats[variant_id]['losses'] += 1

        # Calculate rates
        for variant_id, stats in variant_stats.items():
            total = stats['total']
            if total > 0:
                stats['win_rate'] = stats['wins'] / total
                stats['loss_rate'] = stats['losses'] / total
                stats['tie_rate'] = stats['ties'] / total

        return variant_stats

    def statistical_significance(
        self,
        variant_a_id: str,
        variant_b_id: str,
        alpha: float = 0.05,
    ) -> Dict:
        """Test statistical significance of difference.

        Uses binomial test.

        Args:
            variant_a_id: First variant ID
            variant_b_id: Second variant ID
            alpha: Significance level

        Returns:
            Test results
        """
        # Count direct comparisons
        a_wins = 0
        b_wins = 0
        ties = 0

        for result in self.results:
            if result['variant_a'] == variant_a_id and result['variant_b'] == variant_b_id:
                if result['winner'] == 'a':
                    a_wins += 1
                elif result['winner'] == 'b':
                    b_wins += 1
                else:
                    ties += 1

        total_comparisons = a_wins + b_wins + ties

        if total_comparisons == 0:
            return {'error': 'No direct comparisons found'}

        # Binomial test (ignoring ties)
        trials = a_wins + b_wins
        if trials == 0:
            return {'error': 'All comparisons were ties'}

        from scipy import stats
        p_value = stats.binom_test(a_wins, trials, 0.5)

        return {
            'a_wins': a_wins,
            'b_wins': b_wins,
            'ties': ties,
            'total_comparisons': total_comparisons,
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'winner': variant_a_id if a_wins > b_wins else (variant_b_id if b_wins > a_wins else 'tie'),
        }


class QualityDashboard:
    """Dashboard for monitoring generation quality."""

    def __init__(self, output_dir: str):
        """Initialize quality dashboard.

        Args:
            output_dir: Directory for dashboard data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.output_dir / "quality_metrics.json"
        self.metrics_history = []

        # Load existing
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                self.metrics_history = json.load(f)

    def record_generation(
        self,
        generation_id: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None,
    ):
        """Record generation metrics.

        Args:
            generation_id: Generation identifier
            metrics: Quality metrics
            metadata: Optional metadata
        """
        record = {
            'generation_id': generation_id,
            'timestamp': str(np.datetime64('now')),
            'metrics': metrics,
            'metadata': metadata or {},
        }

        self.metrics_history.append(record)
        self._save()

    def _save(self):
        """Save metrics."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def get_summary_statistics(
        self,
        metric_name: Optional[str] = None,
        time_window: Optional[int] = None,
    ) -> Dict:
        """Get summary statistics.

        Args:
            metric_name: Optional specific metric to analyze
            time_window: Optional time window in hours

        Returns:
            Summary statistics
        """
        if not self.metrics_history:
            return {}

        # Filter by time window if specified
        records = self.metrics_history
        if time_window:
            cutoff = np.datetime64('now') - np.timedelta64(time_window, 'h')
            records = [r for r in records if np.datetime64(r['timestamp']) >= cutoff]

        if not records:
            return {}

        # Collect metrics
        if metric_name:
            values = [r['metrics'].get(metric_name) for r in records if metric_name in r['metrics']]

            if not values:
                return {}

            return {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'percentile_25': float(np.percentile(values, 25)),
                'percentile_75': float(np.percentile(values, 75)),
            }
        else:
            # All metrics
            all_metric_names = set()
            for record in records:
                all_metric_names.update(record['metrics'].keys())

            summary = {}
            for name in all_metric_names:
                summary[name] = self.get_summary_statistics(name, time_window)

            return summary

    def detect_quality_drift(
        self,
        metric_name: str,
        window_size: int = 100,
        threshold: float = 0.1,
    ) -> Dict:
        """Detect quality drift over time.

        Args:
            metric_name: Metric to monitor
            window_size: Window size for comparison
            threshold: Threshold for drift detection (relative change)

        Returns:
            Drift detection results
        """
        if len(self.metrics_history) < window_size * 2:
            return {'error': 'Insufficient data'}

        # Get recent and baseline values
        all_values = [r['metrics'].get(metric_name) for r in self.metrics_history if metric_name in r['metrics']]

        if len(all_values) < window_size * 2:
            return {'error': 'Insufficient metric data'}

        baseline = all_values[:window_size]
        recent = all_values[-window_size:]

        baseline_mean = np.mean(baseline)
        recent_mean = np.mean(recent)

        if baseline_mean == 0:
            return {'error': 'Baseline mean is zero'}

        relative_change = (recent_mean - baseline_mean) / baseline_mean

        drift_detected = abs(relative_change) > threshold

        return {
            'baseline_mean': float(baseline_mean),
            'recent_mean': float(recent_mean),
            'relative_change': float(relative_change),
            'drift_detected': drift_detected,
            'threshold': threshold,
        }
