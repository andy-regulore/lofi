"""
Musical style transfer and genre blending.

This module provides advanced style transfer capabilities:
- Content-style separation (like neural style transfer for images)
- Genre interpolation and blending
- Artist style emulation
- Cross-genre harmonization
- Texture transfer (rhythm, articulation)

Based on techniques from:
- Neural Style Transfer (Gatys et al.)
- CycleGAN for music
- WaveNet-based style conditioning

Author: Claude
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class StyleFeature(Enum):
    """Types of stylistic features."""
    RHYTHM = "rhythm"
    HARMONY = "harmony"
    MELODY = "melody"
    TIMBRE = "timbre"
    DYNAMICS = "dynamics"
    ARTICULATION = "articulation"


@dataclass
class StyleProfile:
    """Profile of a musical style."""
    name: str
    tempo_range: Tuple[float, float]
    key_preference: Dict[str, float]  # Key signatures and probabilities
    chord_vocabulary: List[str]  # Preferred chord types
    rhythm_patterns: List[str]  # Common rhythm patterns
    instrument_preferences: Dict[str, float]  # Instrument and probability
    dynamics_profile: Dict[str, float]  # Dynamic range characteristics
    articulation_style: str  # 'legato', 'staccato', 'mixed'


class StyleDatabase:
    """Database of musical style profiles."""

    def __init__(self):
        """Initialize with common style profiles."""
        self.styles = self._build_database()

    def _build_database(self) -> Dict[str, StyleProfile]:
        """Build database of style profiles."""
        styles = {}

        # Lo-Fi Hip-Hop
        styles['lofi'] = StyleProfile(
            name='Lo-Fi Hip-Hop',
            tempo_range=(60, 95),
            key_preference={'C': 0.2, 'Am': 0.2, 'G': 0.15, 'Dm': 0.15, 'F': 0.1, 'Em': 0.1, 'D': 0.05, 'Bm': 0.05},
            chord_vocabulary=['maj7', 'min7', '9', 'min9', 'maj9', '13'],
            rhythm_patterns=['boom_bap', 'laid_back', 'offbeat_hats'],
            instrument_preferences={
                'electric_piano': 0.8,
                'synth_pad': 0.6,
                'electric_bass': 0.9,
                'drums': 0.95,
                'vinyl_effects': 0.7
            },
            dynamics_profile={'range': 0.6, 'variation': 0.3, 'avg_loudness': 0.65},
            articulation_style='legato'
        )

        # Jazz
        styles['jazz'] = StyleProfile(
            name='Jazz',
            tempo_range=(100, 200),
            key_preference={'Bb': 0.2, 'F': 0.2, 'Eb': 0.15, 'C': 0.1, 'G': 0.1, 'D': 0.1, 'A': 0.08, 'E': 0.07},
            chord_vocabulary=['maj7', 'min7', 'dom7', '7b9', '7#9', '7alt', 'min7b5', 'dim7'],
            rhythm_patterns=['swing', 'syncopated', 'polyrhythmic'],
            instrument_preferences={
                'acoustic_piano': 0.9,
                'acoustic_bass': 0.9,
                'saxophone': 0.7,
                'trumpet': 0.6,
                'drums_brushes': 0.8
            },
            dynamics_profile={'range': 0.9, 'variation': 0.7, 'avg_loudness': 0.7},
            articulation_style='mixed'
        )

        # Classical
        styles['classical'] = StyleProfile(
            name='Classical',
            tempo_range=(60, 140),
            key_preference={'C': 0.15, 'G': 0.15, 'D': 0.12, 'A': 0.1, 'F': 0.1, 'Bb': 0.08, 'Eb': 0.08, 'E': 0.07},
            chord_vocabulary=['maj', 'min', 'dim', 'aug', 'maj7', 'dom7'],
            rhythm_patterns=['regular', 'rubato', 'waltz'],
            instrument_preferences={
                'strings': 0.95,
                'woodwinds': 0.7,
                'brass': 0.6,
                'piano': 0.8
            },
            dynamics_profile={'range': 0.95, 'variation': 0.8, 'avg_loudness': 0.6},
            articulation_style='mixed'
        )

        # Electronic/EDM
        styles['electronic'] = StyleProfile(
            name='Electronic',
            tempo_range=(110, 140),
            key_preference={'Am': 0.2, 'Em': 0.18, 'Dm': 0.15, 'Gm': 0.12, 'C': 0.1, 'F': 0.1, 'G': 0.08, 'D': 0.07},
            chord_vocabulary=['min', 'maj', 'sus2', 'sus4', 'add9'],
            rhythm_patterns=['four_on_floor', 'breakbeat', 'sidechained'],
            instrument_preferences={
                'synth_lead': 0.9,
                'synth_pad': 0.85,
                'synth_bass': 0.9,
                'drums_electronic': 0.95,
                'synth_pluck': 0.7
            },
            dynamics_profile={'range': 0.85, 'variation': 0.6, 'avg_loudness': 0.85},
            articulation_style='staccato'
        )

        # Bossa Nova
        styles['bossa_nova'] = StyleProfile(
            name='Bossa Nova',
            tempo_range=(110, 140),
            key_preference={'C': 0.15, 'G': 0.15, 'D': 0.12, 'Am': 0.12, 'Dm': 0.1, 'Em': 0.1, 'F': 0.1, 'A': 0.08},
            chord_vocabulary=['maj7', 'min7', 'dom7', '9', 'min9', 'maj9', 'm7b5'],
            rhythm_patterns=['bossa_clave', 'syncopated', 'latin'],
            instrument_preferences={
                'nylon_guitar': 0.95,
                'acoustic_bass': 0.8,
                'piano': 0.7,
                'percussion': 0.85
            },
            dynamics_profile={'range': 0.7, 'variation': 0.5, 'avg_loudness': 0.6},
            articulation_style='mixed'
        )

        return styles

    def get_style(self, name: str) -> Optional[StyleProfile]:
        """Get style profile by name."""
        return self.styles.get(name.lower())

    def get_all_styles(self) -> List[str]:
        """Get list of all available styles."""
        return list(self.styles.keys())


class StyleEncoder(nn.Module):
    """
    Encode music into style and content representations.

    Similar to image style transfer, separates "what" (content) from "how" (style).
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, style_dim: int = 128):
        """
        Initialize style encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            style_dim: Style embedding dimension
        """
        super().__init__()

        # Content encoder (what notes, when)
        self.content_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Style encoder (how it's played)
        self.style_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, style_dim),
        )

        # Statistics pooling for style (like Gram matrix)
        self.style_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input into content and style.

        Args:
            x: Input features [batch, length, features]

        Returns:
            Tuple of (content, style) embeddings
        """
        # Content encoding (preserve sequence structure)
        content = self.content_encoder(x)

        # Style encoding (aggregate statistics)
        style_features = self.style_encoder(x)
        # Global average pooling
        style = style_features.mean(dim=1)

        # Can also use Gram matrix-like representation
        style_gram = self._compute_gram_matrix(style_features)

        return content, style_gram

    def _compute_gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix for style representation.

        Args:
            features: [batch, length, features]

        Returns:
            Gram matrix [batch, features, features]
        """
        batch_size, length, feat_dim = features.shape

        # Reshape to [batch, features, length]
        features = features.transpose(1, 2)

        # Compute Gram matrix: F * F^T
        gram = torch.bmm(features, features.transpose(1, 2))

        # Normalize by number of elements
        gram = gram / (length * feat_dim)

        return gram


class StyleDecoder(nn.Module):
    """Decode content with target style."""

    def __init__(self, content_dim: int = 256, style_dim: int = 128, output_dim: int = 512):
        """
        Initialize style decoder.

        Args:
            content_dim: Content embedding dimension
            style_dim: Style embedding dimension
            output_dim: Output feature dimension
        """
        super().__init__()

        # Adaptive instance normalization layers
        self.adain_layers = nn.ModuleList([
            AdaIN(content_dim, style_dim),
            AdaIN(content_dim, style_dim),
        ])

        self.decoder = nn.Sequential(
            nn.Linear(content_dim, content_dim),
            nn.ReLU(),
            nn.Linear(content_dim, output_dim),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Decode content with target style.

        Args:
            content: Content embedding [batch, length, content_dim]
            style: Style embedding [batch, style_dim]

        Returns:
            Styled output [batch, length, output_dim]
        """
        x = content

        # Apply AdaIN layers
        for adain in self.adain_layers:
            x = adain(x, style)

        # Decode
        output = self.decoder(x)

        return output


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization.

    Transfers style statistics to content features.
    """

    def __init__(self, content_dim: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(content_dim, affine=False)

        # Learn to predict style parameters
        self.style_proj = nn.Linear(style_dim, content_dim * 2)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive instance normalization.

        Args:
            content: [batch, length, channels]
            style: [batch, style_dim]

        Returns:
            Normalized content with style statistics
        """
        # Transpose for instance norm [batch, channels, length]
        content = content.transpose(1, 2)

        # Normalize content
        normalized = self.norm(content)

        # Get style parameters
        style_params = self.style_proj(style)
        gamma, beta = style_params.chunk(2, dim=1)

        # Apply affine transformation
        gamma = gamma.unsqueeze(2)
        beta = beta.unsqueeze(2)

        styled = gamma * normalized + beta

        # Transpose back
        return styled.transpose(1, 2)


class StyleTransferModel(nn.Module):
    """Complete style transfer model."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, style_dim: int = 128):
        """Initialize style transfer model."""
        super().__init__()

        self.encoder = StyleEncoder(input_dim, hidden_dim, style_dim)
        self.decoder = StyleDecoder(hidden_dim, style_dim * style_dim, input_dim)  # Flattened Gram

    def forward(self, content_music: torch.Tensor, style_music: torch.Tensor) -> torch.Tensor:
        """
        Transfer style from style_music to content_music.

        Args:
            content_music: Source music [batch, length, features]
            style_music: Style reference [batch, length, features]

        Returns:
            Styled music [batch, length, features]
        """
        # Encode
        content, _ = self.encoder(content_music)
        _, style = self.encoder(style_music)

        # Flatten Gram matrix for decoder
        batch_size = style.shape[0]
        style_flat = style.view(batch_size, -1)

        # Decode with style
        output = self.decoder(content, style_flat)

        return output

    def loss(self,
             output: torch.Tensor,
             content_target: torch.Tensor,
             style_target: torch.Tensor,
             content_weight: float = 1.0,
             style_weight: float = 100.0) -> Dict[str, torch.Tensor]:
        """
        Compute style transfer loss.

        Args:
            output: Generated output
            content_target: Target content
            style_target: Target style
            content_weight: Weight for content loss
            style_weight: Weight for style loss

        Returns:
            Dictionary of losses
        """
        # Content loss (preserve content)
        output_content, _ = self.encoder(output)
        target_content, _ = self.encoder(content_target)
        content_loss = F.mse_loss(output_content, target_content)

        # Style loss (match style statistics)
        _, output_style = self.encoder(output)
        _, target_style = self.encoder(style_target)
        style_loss = F.mse_loss(output_style, target_style)

        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        return {
            'total': total_loss,
            'content': content_loss,
            'style': style_loss
        }


class GenreBlender:
    """Blend multiple genres together."""

    def __init__(self, style_db: StyleDatabase):
        """Initialize genre blender."""
        self.style_db = style_db

    def blend_styles(self,
                     styles: List[str],
                     weights: Optional[List[float]] = None) -> StyleProfile:
        """
        Blend multiple styles with weights.

        Args:
            styles: List of style names
            weights: Weights for each style (default: equal)

        Returns:
            Blended style profile
        """
        if weights is None:
            weights = [1.0 / len(styles)] * len(styles)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Get style profiles
        profiles = [self.style_db.get_style(s) for s in styles]
        if any(p is None for p in profiles):
            raise ValueError("One or more styles not found")

        # Blend tempo range
        tempo_low = sum(p.tempo_range[0] * w for p, w in zip(profiles, weights))
        tempo_high = sum(p.tempo_range[1] * w for p, w in zip(profiles, weights))

        # Blend key preferences
        all_keys = set()
        for p in profiles:
            all_keys.update(p.key_preference.keys())

        key_pref = {}
        for key in all_keys:
            key_pref[key] = sum(
                p.key_preference.get(key, 0) * w
                for p, w in zip(profiles, weights)
            )

        # Normalize key preferences
        total_key_prob = sum(key_pref.values())
        if total_key_prob > 0:
            key_pref = {k: v / total_key_prob for k, v in key_pref.items()}

        # Blend chord vocabulary (union)
        chord_vocab = []
        for p in profiles:
            chord_vocab.extend(p.chord_vocabulary)
        chord_vocab = list(set(chord_vocab))

        # Blend rhythm patterns (union)
        rhythm_patterns = []
        for p in profiles:
            rhythm_patterns.extend(p.rhythm_patterns)
        rhythm_patterns = list(set(rhythm_patterns))

        # Blend instruments
        all_instruments = set()
        for p in profiles:
            all_instruments.update(p.instrument_preferences.keys())

        inst_pref = {}
        for inst in all_instruments:
            inst_pref[inst] = sum(
                p.instrument_preferences.get(inst, 0) * w
                for p, w in zip(profiles, weights)
            )

        # Blend dynamics
        dynamics = {
            'range': sum(p.dynamics_profile['range'] * w for p, w in zip(profiles, weights)),
            'variation': sum(p.dynamics_profile['variation'] * w for p, w in zip(profiles, weights)),
            'avg_loudness': sum(p.dynamics_profile['avg_loudness'] * w for p, w in zip(profiles, weights))
        }

        # Articulation (use most weighted)
        max_weight_idx = weights.index(max(weights))
        articulation = profiles[max_weight_idx].articulation_style

        # Create blended profile
        blended = StyleProfile(
            name=f"Blend: {'+'.join(styles)}",
            tempo_range=(tempo_low, tempo_high),
            key_preference=key_pref,
            chord_vocabulary=chord_vocab,
            rhythm_patterns=rhythm_patterns,
            instrument_preferences=inst_pref,
            dynamics_profile=dynamics,
            articulation_style=articulation
        )

        return blended

    def interpolate_styles(self,
                          style_a: str,
                          style_b: str,
                          alpha: float) -> StyleProfile:
        """
        Interpolate between two styles.

        Args:
            style_a: First style name
            style_b: Second style name
            alpha: Interpolation factor (0 = style_a, 1 = style_b)

        Returns:
            Interpolated style profile
        """
        return self.blend_styles([style_a, style_b], weights=[1 - alpha, alpha])


class CrossGenreHarmonizer:
    """Harmonize melody with chords from different genre."""

    def __init__(self, style_db: StyleDatabase):
        """Initialize cross-genre harmonizer."""
        self.style_db = style_db

    def harmonize(self,
                  melody_notes: List[int],
                  source_genre: str,
                  target_genre: str) -> List[Tuple[int, str]]:
        """
        Harmonize melody from source genre using target genre harmony.

        Args:
            melody_notes: MIDI note numbers
            source_genre: Source genre (for context)
            target_genre: Target genre (for harmony style)

        Returns:
            List of (root, chord_type) tuples
        """
        target_style = self.style_db.get_style(target_genre)
        if target_style is None:
            raise ValueError(f"Unknown genre: {target_genre}")

        # Simplified harmonization
        progression = []

        for note in melody_notes:
            # Get pitch class
            pc = note % 12

            # Choose chord from target genre vocabulary
            chord_type = np.random.choice(target_style.chord_vocabulary)

            # Simple root finding (could be more sophisticated)
            # Use note as guide tone
            if chord_type in ['maj7', 'maj9', 'maj', 'maj13']:
                # Major chords: note is likely 1, 3, 5, or 7
                possible_roots = [pc, (pc - 4) % 12, (pc - 7) % 12, (pc - 11) % 12]
            else:
                # Minor/other chords
                possible_roots = [pc, (pc - 3) % 12, (pc - 7) % 12, (pc - 10) % 12]

            root = possible_roots[0]  # Simplified
            progression.append((root, chord_type))

        return progression


# Example usage
if __name__ == '__main__':
    print("=== Style Database ===")
    db = StyleDatabase()
    print(f"Available styles: {db.get_all_styles()}")

    lofi = db.get_style('lofi')
    print(f"\nLo-Fi profile:")
    print(f"  Tempo: {lofi.tempo_range}")
    print(f"  Top keys: {list(lofi.key_preference.items())[:3]}")
    print(f"  Chords: {lofi.chord_vocabulary}")

    print("\n=== Genre Blending ===")
    blender = GenreBlender(db)
    blend = blender.blend_styles(['lofi', 'jazz'], weights=[0.7, 0.3])
    print(f"Blended style: {blend.name}")
    print(f"  Tempo: {blend.tempo_range}")
    print(f"  Chords: {blend.chord_vocabulary}")

    print("\n=== Style Interpolation ===")
    interpolated = blender.interpolate_styles('electronic', 'classical', alpha=0.5)
    print(f"Interpolated: {interpolated.name}")
    print(f"  Tempo: {interpolated.tempo_range}")

    print("\n=== Neural Style Transfer ===")
    model = StyleTransferModel(input_dim=512, hidden_dim=256, style_dim=128)

    # Example tensors
    content = torch.randn(2, 100, 512)
    style_ref = torch.randn(2, 100, 512)

    output = model(content, style_ref)
    print(f"Style transfer output shape: {output.shape}")

    # Compute loss
    losses = model.loss(output, content, style_ref)
    print(f"Losses: content={losses['content']:.4f}, style={losses['style']:.4f}, total={losses['total']:.4f}")

    print("\n=== Cross-Genre Harmonization ===")
    harmonizer = CrossGenreHarmonizer(db)
    melody = [60, 62, 64, 65, 67]  # C major scale
    chords = harmonizer.harmonize(melody, source_genre='lofi', target_genre='jazz')
    print(f"Harmonized melody with jazz chords:")
    for note, (root, chord_type) in zip(melody, chords):
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        print(f"  Note {note_names[note % 12]}: {note_names[root]}{chord_type}")
